"""Single-fold train+eval driver for FlakyLens.

Mirrors the protocol in Bert_train_per_project.py / Testing_per_project.py
(CodeBERT-base, focal loss w/ balanced class weights, AdamW, max_len 512,
early stopping on validation macro-F1) but runs ONE project group so it fits
a short time budget, and never regenerates the shipped project splits.

Everything is driven by flags so baseline and ablation runs are identical
except for the one knob under test.
"""
import argparse
import gc
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, f1_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler

from codebert_model import BERT_Arch
from utils import codebert_model_define

DATA_DIR = "FlakyLens_Categorization_PerProject-Data"
CATEGORY_NAMES = ["Async", "Conc", "Time", "UC", "OD", "Non-flaky"]
# Paper's reported 4-fold averages, for reference in the printout.
PAPER_F1 = [58.37, 35.92, 72.73, 73.63, 64.35, 100.00]


class FocalLoss(nn.Module):
    """Verbatim from Bert_train_per_project.py."""

    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction="none", weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def encode(tokenizer, texts, max_length):
    tok = tokenizer.batch_encode_plus(
        list(texts), max_length=max_length, padding="max_length", truncation=True
    )
    return torch.tensor(tok["input_ids"]), torch.tensor(tok["attention_mask"])


def make_loader(seq, mask, y, batch_size, shuffle):
    ds = TensorDataset(seq, mask, y)
    sampler = RandomSampler(ds) if shuffle else SequentialSampler(ds)
    return DataLoader(ds, sampler=sampler, batch_size=batch_size)


class RaggedDataset(torch.utils.data.Dataset):
    """Holds unpadded token id lists so batches can be padded to their own max."""

    def __init__(self, ids, labels):
        self.ids = ids
        self.labels = labels

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        return self.ids[i], self.labels[i]


def pad_collate(batch, pad_id):
    """Pad to the longest sequence *in this batch* rather than to 512."""
    seqs, labels = zip(*batch)
    n = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), n), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), n), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = torch.tensor(s, dtype=torch.long)
        mask[i, :len(s)] = 1
    return ids, mask, torch.tensor(labels)


def make_dynamic_loader(ids, labels, batch_size, pad_id, shuffle, seed=0):
    """Length-grouped batching: sort by length so each batch is homogeneous, then
    shuffle the batch order. Same examples and same batch size as the padded loader,
    just far less wasted compute on padding."""
    order = np.argsort([len(s) for s in ids], kind="stable")
    batches = [order[i:i + batch_size].tolist()
               for i in range(0, len(order), batch_size)]
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(batches)
    ds = RaggedDataset(ids, labels)
    return DataLoader(ds, batch_sampler=batches,
                      collate_fn=lambda b: pad_collate(b, pad_id))


def run_eval(model, loader, device, amp):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in loader:
            sent_id, mask, y = [t.to(device, non_blocking=True) for t in batch]
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                out = model(sent_id, mask)
            preds.append(out.float().detach().cpu().numpy())
            labels.append(y.detach().cpu().numpy())
    preds = np.concatenate(preds, axis=0)
    labels = np.concatenate(labels, axis=0)
    return np.argmax(preds, axis=1), labels


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fold", type=int, default=1)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--gamma", type=float, default=2.0, help="focal loss gamma")
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--amp", action="store_true", help="fp16 autocast (big speedup on Ada)")
    p.add_argument("--tag", type=str, default="baseline")
    p.add_argument("--max-train", type=int, default=0,
                   help="if >0, cap training rows (stratified) to speed up a smoke test")
    p.add_argument("--undersample-majority", type=int, default=0,
                   help="if >0, cap class 5 (non-flaky) training rows to this many")
    p.add_argument("--oversample-minority", type=int, default=0,
                   help="if >0, duplicate each class 0-4 up to this many training rows")
    p.add_argument("--adversarial-augment", action="store_true",
                   help="add 5 perturbed variants of every minority row using the repo's own "
                        "perturbation functions, keeping the true label (adversarial training, "
                        "as in the commented-out boosting_noisy_data_for_train path)")
    p.add_argument("--augment-minority", type=int, default=0,
                   help="if >0, grow each class 0-4 to this many rows with label-preserving "
                        "variants (real augmentation, not duplication)")
    p.add_argument("--valid-subsample", type=int, default=0,
                   help="if >0, stratified-subsample the validation set (speed only)")
    p.add_argument("--dynamic-padding", action="store_true",
                   help="length-grouped batches padded to the batch max instead of 512; "
                        "pure speed optimisation, same examples and batch size")
    p.add_argument("--base-model", type=str, default="",
                   help="local directory holding the encoder; avoids any Hub lookup. "
                        "Empty falls back to the repo's codebert_model_define().")
    p.add_argument("--resume", action="store_true",
                   help="continue from <tag>_state.pt if it exists (model+optimizer+epoch)")
    args = p.parse_args()

    device = torch.device("cuda")
    set_seed(args.seed)
    os.makedirs("../models", exist_ok=True)
    os.makedirs("../results/single_fold", exist_ok=True)
    ckpt = f"../models/fold{args.fold}_{args.tag}.pt"

    train_df = pd.read_csv(f"{DATA_DIR}/data_split/train_set_{args.fold}.csv")
    valid_df = pd.read_csv(f"{DATA_DIR}/data_split/valid_set_{args.fold}.csv")
    test_df = pd.read_csv(f"{DATA_DIR}/test_set_{args.fold}.csv")

    if args.max_train:
        train_df = (train_df.groupby("category", group_keys=False)
                    .apply(lambda g: g.sample(
                        n=max(1, int(round(args.max_train * len(g) / len(train_df)))),
                        random_state=args.seed))
                    .reset_index(drop=True))

    # --- class rebalancing (training set only; valid/test are never touched) ---
    if args.undersample_majority or args.oversample_minority or args.augment_minority:
        before = train_df["category"].value_counts().sort_index().to_dict()
        parts = []
        for cls, grp in train_df.groupby("category"):
            if cls == 5 and args.undersample_majority and len(grp) > args.undersample_majority:
                grp = grp.sample(n=args.undersample_majority, random_state=args.seed)
            elif cls != 5 and args.augment_minority and len(grp) < args.augment_minority:
                # real label-preserving variants, not copies
                from augment import augment_to
                codes = augment_to(grp["full_code"].tolist(), args.augment_minority,
                                   seed=args.seed + int(cls))
                grp = pd.DataFrame({"full_code": codes, "category": cls})
                print(f"  class {cls}: augmented -> {len(grp)} rows, "
                      f"{grp['full_code'].nunique()} distinct")
            elif cls != 5 and args.oversample_minority and len(grp) < args.oversample_minority:
                # duplicate with replacement up to the target count
                extra = grp.sample(n=args.oversample_minority - len(grp),
                                   replace=True, random_state=args.seed)
                grp = pd.concat([grp, extra])
            parts.append(grp)
        train_df = (pd.concat(parts)
                    .sample(frac=1.0, random_state=args.seed)
                    .reset_index(drop=True))
        print(f"rebalanced train: {before}")
        print(f"              ->  {train_df['category'].value_counts().sort_index().to_dict()}")

    if args.adversarial_augment:
        # Mirrors boosting_noisy_data_for_train(): one perturbed copy of every non-flaky-5
        # row per perturbation type, true label kept. The perturbations inject other
        # categories' signature code; training through that is the point -- the model must
        # learn to ignore surface tokens rather than key off them.
        from perturbation import (deadcode_insertion, printStatement_insertion,
                                  variableRenaming_insertion, multi_comment_insertion,
                                  single_comment_insertion)
        funcs = [deadcode_insertion, printStatement_insertion, variableRenaming_insertion,
                 multi_comment_insertion, single_comment_insertion]
        mino = train_df[train_df["category"] != 5]
        extra = []
        for fn in funcs:
            xs = mino["full_code"].copy().reset_index(drop=True)
            ys = mino["category"].copy().reset_index(drop=True)
            try:
                out = fn(xs, ys)
                extra.append(pd.DataFrame({"full_code": list(out), "category": list(ys)}))
            except Exception as e:
                print(f"  {fn.__name__} failed ({type(e).__name__}), skipped")
        train_df = (pd.concat([train_df] + extra)
                    .sample(frac=1.0, random_state=args.seed).reset_index(drop=True))
        print(f"adversarial augment: +{sum(len(e) for e in extra)} perturbed minority rows "
              f"from {len(extra)} perturbation types -> {len(train_df)} total")

    if args.valid_subsample and args.valid_subsample < len(valid_df):
        valid_df = (valid_df.groupby("category", group_keys=False)
                    .apply(lambda g: g.sample(
                        n=max(1, int(round(args.valid_subsample * len(g) / len(valid_df)))),
                        random_state=args.seed))
                    .reset_index(drop=True))

    print(f"fold {args.fold}: train={len(train_df)} valid={len(valid_df)} test={len(test_df)}")
    print("train class counts:", train_df["category"].value_counts().sort_index().to_dict())
    print("test  class counts:", test_df["category"].value_counts().sort_index().to_dict())

    if args.base_model:
        # Same construction as codebert_model_define(), but from a local path so no
        # network call happens (a DNS blip killed two earlier runs).
        from transformers import AutoConfig, AutoModel, AutoTokenizer
        cfg = AutoConfig.from_pretrained(args.base_model, return_dict=False,
                                         output_hidden_states=True, local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(args.base_model, local_files_only=True)
        auto_model = AutoModel.from_pretrained(args.base_model, config=cfg,
                                               local_files_only=True)
        print(f"encoder: {args.base_model} (local, offline)")
    else:
        _, tokenizer, auto_model = codebert_model_define()
    tr_y = torch.tensor(train_df["category"].values)
    va_y = torch.tensor(valid_df["category"].values)
    te_y = torch.tensor(test_df["category"].values)

    if args.dynamic_padding:
        pad_id = tokenizer.pad_token_id
        enc = lambda s: tokenizer.batch_encode_plus(
            list(s), max_length=args.max_length, truncation=True)["input_ids"]
        tr_ids, va_ids, te_ids = (enc(train_df["full_code"]), enc(valid_df["full_code"]),
                                  enc(test_df["full_code"]))
        avg = np.mean([len(s) for s in tr_ids])
        print(f"dynamic padding on: mean train length {avg:.0f} tokens "
              f"(vs {args.max_length} fixed) -> ~{args.max_length / avg:.1f}x less padding")
        train_loader = make_dynamic_loader(tr_ids, tr_y.tolist(), args.batch_size,
                                           pad_id, True, args.seed)
        valid_loader = make_dynamic_loader(va_ids, va_y.tolist(), args.batch_size,
                                           pad_id, False)
        test_loader = make_dynamic_loader(te_ids, te_y.tolist(), args.batch_size,
                                          pad_id, False)
    else:
        tr_seq, tr_mask = encode(tokenizer, train_df["full_code"], args.max_length)
        va_seq, va_mask = encode(tokenizer, valid_df["full_code"], args.max_length)
        te_seq, te_mask = encode(tokenizer, test_df["full_code"], args.max_length)
        train_loader = make_loader(tr_seq, tr_mask, tr_y, args.batch_size, True)
        valid_loader = make_loader(va_seq, va_mask, va_y, args.batch_size, False)
        test_loader = make_loader(te_seq, te_mask, te_y, args.batch_size, False)

    classes = np.unique(tr_y.numpy())
    cw = compute_class_weight(class_weight="balanced", classes=classes, y=tr_y.numpy())
    weights = torch.tensor(cw, dtype=torch.float).to(device)
    criterion = FocalLoss(alpha=weights, gamma=args.gamma)
    print(f"class weights: {np.round(cw, 3).tolist()}  gamma={args.gamma}  lr={args.lr}")

    model = BERT_Arch(auto_model, 6).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    best_f1, bad_epochs, history = -1.0, 0, []
    start_epoch = 1
    state_path = f"../models/fold{args.fold}_{args.tag}_state.pt"
    if args.resume and os.path.exists(state_path):
        st = torch.load(state_path)
        model.load_state_dict(st["model"])
        optimizer.load_state_dict(st["optimizer"])
        scaler.load_state_dict(st["scaler"])
        best_f1, bad_epochs, history = st["best_f1"], st["bad_epochs"], st["history"]
        start_epoch = st["epoch"] + 1
        print(f"resumed from {state_path}: continuing at epoch {start_epoch}, "
              f"best valid macro-F1 so far {best_f1:.4f}")

    t_start = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        t0 = time.time()
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader):
            sent_id, mask, y = [t.to(device, non_blocking=True) for t in batch]
            with torch.autocast("cuda", dtype=torch.float16, enabled=args.amp):
                out = model(sent_id, mask)
                loss = criterion(out, y) / args.grad_accum
            scaler.scale(loss).backward()
            if (step + 1) % args.grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        train_s = time.time() - t0

        vp, vl = run_eval(model, valid_loader, device, args.amp)
        vf1 = f1_score(vl, vp, average="macro")
        history.append({"epoch": epoch, "valid_macro_f1": float(vf1),
                        "train_s": round(train_s, 1)})
        star = ""
        if vf1 > best_f1:
            best_f1, bad_epochs, star = vf1, 0, "  <- best, saved"
            torch.save(model.state_dict(), ckpt)
        else:
            bad_epochs += 1
        # full training state every epoch so an interrupted run can be resumed
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": epoch, "best_f1": best_f1,
                    "bad_epochs": bad_epochs, "history": history}, state_path)

        peak = torch.cuda.max_memory_allocated() / 1024 ** 3
        print(f"epoch {epoch:>2}/{args.epochs}  valid_macro_f1={vf1:.4f}  "
              f"train={train_s:.0f}s  peak_vram={peak:.1f}GB{star}", flush=True)
        if bad_epochs >= args.patience:
            print(f"early stopping at epoch {epoch}")
            break

    total_train_time = time.time() - t_start
    print(f"\ntraining done in {total_train_time/60:.1f} min; best valid macro-F1 = {best_f1:.4f}")

    model.load_state_dict(torch.load(ckpt))
    tp, tl = run_eval(model, test_loader, device, args.amp)

    print(f"\n=== fold {args.fold} / {args.tag} — held-out test set ===")
    print(classification_report(tl, tp, labels=list(range(6)),
                                target_names=CATEGORY_NAMES, digits=4, zero_division=0))

    per_class = f1_score(tl, tp, labels=list(range(6)), average=None, zero_division=0) * 100
    macro = float(np.mean(per_class))
    print(f"{'category':<12}{'this run':>10}{'paper (4-fold)':>17}{'delta':>9}")
    for name, got, ref in zip(CATEGORY_NAMES, per_class, PAPER_F1):
        print(f"{name:<12}{got:>9.2f}%{ref:>16.2f}%{got - ref:>+8.2f}")
    print(f"{'MACRO F1':<12}{macro:>9.2f}%{65.79:>16.2f}%{macro - 65.79:>+8.2f}")

    out = {
        "tag": args.tag, "fold": args.fold,
        "args": vars(args),
        "best_valid_macro_f1": best_f1,
        "test_macro_f1": macro,
        "test_per_class_f1": {n: float(v) for n, v in zip(CATEGORY_NAMES, per_class)},
        "epochs_run": len(history),
        "train_minutes": round(total_train_time / 60, 2),
        "history": history,
    }
    path = f"../results/single_fold/fold{args.fold}_{args.tag}.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {path}")

    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
