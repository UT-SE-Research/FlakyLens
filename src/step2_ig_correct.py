"""Step 2 (corrected): true Integrated Gradients on the selected Concurrency failures.

The repo's cig.py path is broken (see step2_attribution.py output): BERT_Arch.get_embeddings
already applies position embeddings + LayerNorm, and forward(is_embedding=True) applies them
again, so the model collapses to ~uniform and the gradients are noise (~1e-9).

Here we hook captum's LayerIntegratedGradients on model.bert.embeddings and drive the
model through its NORMAL input_ids path, so attributions explain the prediction the model
actually makes. Baseline = same length, all <pad> except <s> and </s>. 50 steps.
"""
import re
import numpy as np
import pandas as pd
import torch
from captum.attr import LayerIntegratedGradients

from analysis_common import NAMES, DEV, load_model, load_test, predict
from detokenization import combine_tokens
from step2_attribution import SELECTED, tag


def ig_words(model, tok, code, target, n_steps=50):
    enc = tok.batch_encode_plus([code], max_length=512, padding="max_length", truncation=True)
    ids = torch.tensor(enc["input_ids"]).to(DEV)
    msk = torch.tensor(enc["attention_mask"]).to(DEV)
    base = torch.full_like(ids, tok.pad_token_id)
    base[0, 0], base[0, int(msk[0].sum()) - 1] = tok.cls_token_id, tok.sep_token_id

    def fwd(input_ids, attention_mask):
        return model(input_ids, attention_mask)          # log-probs, normal path

    lig = LayerIntegratedGradients(fwd, model.bert.embeddings)
    attr, delta = lig.attribute(inputs=ids, baselines=base, additional_forward_args=(msk,),
                                target=target, n_steps=n_steps, internal_batch_size=8,
                                return_convergence_delta=True)
    per_tok = attr.sum(dim=2).squeeze(0).detach().cpu().numpy()
    tokens = tok.convert_ids_to_tokens(ids[0].cpu().numpy())
    df = pd.DataFrame({"Token": tokens, "Attribution": per_tok, "Predicted_class": target,
                       "True_class": 1, "Test_Pred_logit": 0.0, "Test_confidence_score": 0.0})
    df = df[df.Token != "<pad>"]
    words = combine_tokens(df)
    words = words[~words.Token.isin(["<s>", "</s>", ""])].copy()
    words["Attribution"] = words["Attribution"].astype(float)
    return words.sort_values("Attribution", ascending=False), float(abs(delta.item()))


def main():
    tok, model = load_model()
    te = load_test(2)
    conc = te[te.category == 1].reset_index(drop=True)
    preds, probs = predict(model, tok, conc.full_code)

    rows = []
    for i in SELECTED:
        code, pred = conc.full_code[i], int(preds[i])
        top_pred, d1 = ig_words(model, tok, code, pred)
        print("=" * 100)
        print(f"CASE #{i}  {conc.project[i]} :: {conc.test_name[i]}")
        print(f"   predicted = {NAMES[pred]} (P={probs[i,pred]:.2f})   P(Concurrency)={probs[i,1]:.2f}"
              f"{'   <-- CORRECT' if pred == 1 else ''}    [IG convergence delta {d1:.3f}]")
        print(f"\n   top-5 tokens driving the prediction toward **{NAMES[pred]}**:")
        for _, r in top_pred.head(5).iterrows():
            print(f"      {r.Attribution:>8.3f}  {r.Token:<28} [{tag(r.Token)}]")
            rows.append({"case": i, "direction": f"toward {NAMES[pred]}", "token": r.Token,
                         "attribution": r.Attribution, "tag": tag(r.Token)})
        print(f"   top-3 tokens pushing AGAINST {NAMES[pred]}:")
        for _, r in top_pred.tail(3).iloc[::-1].iterrows():
            print(f"      {r.Attribution:>8.3f}  {r.Token:<28} [{tag(r.Token)}]")
        if pred != 1:
            top_conc, _ = ig_words(model, tok, code, 1)
            print(f"   top-5 tokens that support **Concurrency** (target=1):")
            for _, r in top_conc.head(5).iterrows():
                print(f"      {r.Attribution:>8.3f}  {r.Token:<28} [{tag(r.Token)}]")
                rows.append({"case": i, "direction": "toward Concurrency", "token": r.Token,
                             "attribution": r.Attribution, "tag": tag(r.Token)})

    out = pd.DataFrame(rows)
    out.to_csv("../results/step2_ig_attributions.csv", index=False)
    print("\n" + "=" * 100)
    wrong = out[(out.direction != "toward Concurrency") & (out.case != 8)]
    print("SUMMARY — tags of the top-5 tokens driving each WRONG prediction (7 cases, 35 tokens):")
    for t, n in wrong.tag.value_counts().items():
        print(f"   {n:>2}  [{t}]")
    print("\nsaved -> results/step2_ig_attributions.csv")


if __name__ == "__main__":
    main()
