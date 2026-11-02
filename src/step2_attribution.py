"""Step 2: RQ3-style attribution on the selected Concurrency failures, using the repo's own
pipeline unchanged: cig.CustomIntegratedGradients -> BERT_Arch.get_embeddings ->
sum over embedding dim -> detokenization.combine_tokens (subword merge) -> rank.

Caveat carried over from the artifact review: cig.py's attribute() returns the raw
gradient d(output[target])/d(embeddings), not an integrated path -- that IS what RQ3
computed in the artifact, so it is used here as-is to stay faithful to the method.
"""
import re
import numpy as np
import pandas as pd
import torch

from analysis_common import NAMES, DEV, load_model, load_test, predict
from cig import CustomIntegratedGradients
from detokenization import combine_tokens

SELECTED = [0, 2, 16, 7, 12, 4, 6, 8]   # indices into the 17 Concurrency tests (Step 1)

CONC = re.compile(r"synchron|lock|thread|join|latch|countdown|atomic|executor|concurr|race|"
                  r"volatile|semaphore|barrier|future|submit|parallel|interrupt|mutex|notify|"
                  r"worker|pool|scheduler|schedul|runnable|callable", re.I)
TIME = re.compile(r"sleep|timeout|delay|timeunit|millis|seconds|duration|nanotime|"
                  r"currenttime|timer|\btime\b|poll|eventually|awaitility|retry|backoff|"
                  r"tick|clock", re.I)
AWAIT = re.compile(r"await", re.I)


def tag(word):
    if AWAIT.search(word):
        return "ambig"      # await: CountDownLatch.await (conc) vs Awaitility (async-wait)
    if CONC.search(word):
        return "CONC"
    if TIME.search(word):
        return "time"
    return "-"


def attribute(model, tok, code, target):
    """Repo pipeline: raw gradient wrt embeddings, summed per token, merged per word."""
    enc = tok.batch_encode_plus([code], max_length=512, padding="max_length", truncation=True)
    ids = torch.tensor(enc["input_ids"]).to(DEV)
    msk = torch.tensor(enc["attention_mask"]).to(DEV)
    emb = model.get_embeddings(ids).clone().detach().float().requires_grad_(True)
    grads = CustomIntegratedGradients(model).attribute(inputs=(emb, msk.float()), target=target)
    per_tok = grads[0].sum(dim=2).squeeze(0).detach().cpu().numpy()
    tokens = tok.convert_ids_to_tokens(ids[0].cpu().numpy())
    df = pd.DataFrame({"Token": tokens, "Attribution": per_tok,
                       "Predicted_class": target, "True_class": 1,
                       "Test_Pred_logit": 0.0, "Test_confidence_score": 0.0})
    df = df[df.Token != "<pad>"]
    words = combine_tokens(df)
    words = words[~words.Token.isin(["<s>", "</s>", ""])]
    words["Attribution"] = words["Attribution"].astype(float)
    return words.sort_values("Attribution", ascending=False)


def main():
    tok, model = load_model()
    te = load_test(2)
    conc = te[te.category == 1].reset_index(drop=True)
    preds, probs = predict(model, tok, conc.full_code)

    rows = []
    for i in SELECTED:
        code, pred = conc.full_code[i], int(preds[i])
        top_pred = attribute(model, tok, code, pred).head(5)
        top_conc = attribute(model, tok, code, 1).head(5)

        print("=" * 100)
        print(f"CASE #{i}  {conc.project[i]} :: {conc.test_name[i]}")
        print(f"   predicted = {NAMES[pred]}  (P={probs[i,pred]:.2f})   P(Concurrency)={probs[i,1]:.2f}"
              f"{'   <-- CORRECT' if pred==1 else ''}")
        print(f"\n   top-5 tokens driving the prediction toward **{NAMES[pred]}**:")
        for _, r in top_pred.iterrows():
            print(f"      {r.Attribution:>8.3f}  {r.Token:<28} [{tag(r.Token)}]")
        if pred != 1:
            print(f"\n   top-5 tokens that would push toward **Concurrency** (target=1):")
            for _, r in top_conc.iterrows():
                print(f"      {r.Attribution:>8.3f}  {r.Token:<28} [{tag(r.Token)}]")
        for _, r in top_pred.iterrows():
            rows.append({"case": i, "target": NAMES[pred], "token": r.Token,
                         "attribution": r.Attribution, "tag": tag(r.Token)})

    out = pd.DataFrame(rows)
    out.to_csv("../results/step2_attributions.csv", index=False)
    print("\n" + "=" * 100)
    print("SUMMARY over the top-5 tokens driving each WRONG prediction:")
    wrong = out[out.target != "Concurrency"]
    for t, n in wrong.tag.value_counts().items():
        print(f"   {n:>2} tokens tagged [{t}]")
    print("\nsaved -> results/step2_attributions.csv")


if __name__ == "__main__":
    main()
