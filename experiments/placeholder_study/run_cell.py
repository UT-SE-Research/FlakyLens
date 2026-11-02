"""Step 4 - train + evaluate one cell of the 2x2.

Cell = {original | placeholder} x {same_project | project_disjoint} x seed.

Training recipe is identical to the "Balanced" configuration used throughout the earlier
work (undersample non-flaky to 800, oversample each flaky class to 160, focal loss
gamma=2 with balanced class weights, AdamW 1e-5, batch 8, max_len 512, fp16,
length-grouped dynamic padding), so numbers stay comparable with the 59.12% baseline.
Both best-epoch (validation macro-F1) and last-epoch metrics are recorded, because
validation here is same-project in both settings and is known to misrank.
"""
import argparse, json, os, sys, time
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.utils.class_weight import compute_class_weight

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from codebert_model import BERT_Arch                                    # noqa: E402
from run_one_fold import FocalLoss, set_seed, make_dynamic_loader, run_eval   # noqa: E402

NAMES = ["Async Wait", "Concurrency", "Time", "Unordered Collections",
         "Test Order Dependency", "Non-flaky"]
DEV = torch.device("cuda")


def rebalance(df, majority, minority, seed):
    parts = []
    for c, g in df.groupby("category"):
        if c == 5 and len(g) > majority:
            g = g.sample(n=majority, random_state=seed)
        elif c != 5 and len(g) < minority:
            g = pd.concat([g, g.sample(n=minority - len(g), replace=True, random_state=seed)])
        parts.append(g)
    return pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def metrics(y, p):
    cm = confusion_matrix(y, p, labels=list(range(6)))
    per = f1_score(y, p, labels=list(range(6)), average=None, zero_division=0) * 100
    rep = classification_report(y, p, labels=list(range(6)), target_names=NAMES,
                                digits=4, zero_division=0, output_dict=True)
    return {
        "accuracy": float(accuracy_score(y, p)) * 100,
        "macro_f1": float(per.mean()),
        "per_class_f1": {n: float(v) for n, v in zip(NAMES, per)},
        "per_class_precision": {n: rep[n]["precision"] * 100 for n in NAMES},
        "per_class_recall": {n: rep[n]["recall"] * 100 for n in NAMES},
        "support": {n: int(rep[n]["support"]) for n in NAMES},
        "confusion_matrix": cm.tolist(),
        "conc_to_async_count": int(cm[1][0]),
        "conc_total": int(cm[1].sum()),
        "conc_to_async_rate": float(cm[1][0] / max(cm[1].sum(), 1)) * 100,
        "conc_correct": int(cm[1][1]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, choices=["original", "placeholder"])
    ap.add_argument("--setting", required=True, choices=["same_project", "project_disjoint"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--gamma", type=float, default=2.0)
    ap.add_argument("--undersample-majority", type=int, default=800)
    ap.add_argument("--oversample-minority", type=int, default=160)
    ap.add_argument("--base-model", default="local_models/codebert-base")
    ap.add_argument("--root", default="experiments/placeholder_study")
    a = ap.parse_args()

    cell = f"{a.corpus}__{a.setting}__seed{a.seed}"
    outdir, ckptdir = f"{a.root}/results", f"{a.root}/checkpoints"
    os.makedirs(outdir, exist_ok=True); os.makedirs(ckptdir, exist_ok=True)
    ckpt = f"{ckptdir}/{cell}.pt"
    set_seed(a.seed)

    d = f"{a.root}/data/splits/{a.corpus}/{a.setting}"
    tr, va, te = (pd.read_csv(f"{d}/{p}.csv") for p in ("train", "valid", "test"))
    tr_bal = rebalance(tr, a.undersample_majority, a.oversample_minority, a.seed)
    print(f"[{cell}] train {len(tr)} -> rebalanced {len(tr_bal)} "
          f"{tr_bal.category.value_counts().sort_index().to_dict()} | valid {len(va)} | test {len(te)}",
          flush=True)

    from transformers import AutoConfig, AutoModel, AutoTokenizer
    cfg = AutoConfig.from_pretrained(a.base_model, return_dict=False,
                                     output_hidden_states=True, local_files_only=True)
    tok = AutoTokenizer.from_pretrained(a.base_model, local_files_only=True)
    enc = lambda s: tok.batch_encode_plus(list(s), max_length=a.max_length,
                                          truncation=True)["input_ids"]
    loaders = [make_dynamic_loader(enc(df.full_code), df.category.tolist(), a.batch_size,
                                   tok.pad_token_id, sh, a.seed)
               for df, sh in ((tr_bal, True), (va, False), (te, False))]
    train_loader, valid_loader, test_loader = loaders

    cw = compute_class_weight("balanced", classes=np.unique(tr_bal.category),
                              y=tr_bal.category.values)
    crit = FocalLoss(alpha=torch.tensor(cw, dtype=torch.float).to(DEV), gamma=a.gamma)
    model = BERT_Arch(AutoModel.from_pretrained(a.base_model, config=cfg,
                                                local_files_only=True), 6).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    best, bad, hist, t0 = -1.0, 0, [], time.time()
    for ep in range(1, a.epochs + 1):
        model.train(); opt.zero_grad(set_to_none=True)
        for sid, msk, y in train_loader:
            sid, msk, y = sid.to(DEV), msk.to(DEV), y.to(DEV)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = crit(model(sid, msk), y)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
        vp, vl = run_eval(model, valid_loader, DEV, True)
        vf = f1_score(vl, vp, average="macro", zero_division=0)
        hist.append({"epoch": ep, "valid_macro_f1": float(vf)})
        star = ""
        if vf > best:
            best, bad, star = vf, 0, "  <- best"
            torch.save(model.state_dict(), ckpt)
        else:
            bad += 1
        print(f"[{cell}] epoch {ep:>2}/{a.epochs} valid_macro_f1={vf:.4f}{star}", flush=True)
        if bad >= a.patience:
            print(f"[{cell}] early stop at epoch {ep}", flush=True); break

    last_p, last_y = run_eval(model, test_loader, DEV, True)     # last epoch
    model.load_state_dict(torch.load(ckpt))
    best_p, best_y = run_eval(model, test_loader, DEV, True)     # best-by-validation

    out = {"cell": cell, "corpus": a.corpus, "setting": a.setting, "seed": a.seed,
           "config": vars(a), "epochs_run": len(hist), "minutes": round((time.time()-t0)/60, 2),
           "best_valid_macro_f1": best * 100, "history": hist,
           "test_best_epoch": metrics(best_y, best_p),
           "test_last_epoch": metrics(last_y, last_p)}
    json.dump(out, open(f"{outdir}/{cell}.json", "w"), indent=2)
    m = out["test_best_epoch"]
    print(f"[{cell}] DONE macro-F1={m['macro_f1']:.2f}%  acc={m['accuracy']:.2f}%  "
          f"Conc->Async={m['conc_to_async_count']}/{m['conc_total']} "
          f"({m['conc_to_async_rate']:.1f}%)  Conc-F1={m['per_class_f1']['Concurrency']:.2f}%",
          flush=True)


if __name__ == "__main__":
    main()
