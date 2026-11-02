"""Step 3 - build the 2x2 split families.

Design: the comparison is only interpretable if the two settings differ in ONE thing -
whether the test projects were seen during training. So we hold constant:

  * total data        : all 8,574 tests in both settings
  * split sizes       : 5,114 train / 1,279 valid / 2,181 test
  * test class support: identical per-class counts (Async 24, Conc 17, Time 15, UC 4,
                        OD 43, Non-flaky 2,078) so confusion counts compare directly

and vary only the partitioning rule:

  project_disjoint : the repo's fold-2 split; test projects never appear in train
  same_project     : stratified random re-partition; test projects also appear in train

The placeholder variants reuse the SAME id partitions, so original vs placeholder differ
only in `full_code`.

Validation is a same-project 20% slice of train in BOTH settings - that is what the paper
does, and keeping the selection rule identical across all four cells stops checkpoint
selection becoming a confound. Step 4 also reports last-epoch numbers as a cross-check.
"""
import argparse
import csv
import json
import os
import random
from collections import Counter, defaultdict

csv.field_size_limit(10 ** 9)
NAMES = ["Async Wait", "Concurrency", "Time", "Unordered Collections",
         "Test Order Dependency", "Non-flaky"]
BASE = "src/FlakyLens_Categorization_PerProject-Data"
OUT = "experiments/placeholder_study/data/splits"
SEED = 42


def read(p):
    return list(csv.DictReader(open(p)))


def write(rows, path, fields):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def dist(rows):
    c = Counter(int(r["category"]) for r in rows)
    return [c.get(i, 0) for i in range(6)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=2)
    args = ap.parse_args()

    orig = read("FlakeBench/FlakeBench_dataset.csv")
    place_rows = read("experiments/placeholder_study/data/FlakeBench_placeholder.csv")
    fields = list(orig[0].keys())

    # NOTE: FlakeBench's `id` column is NOT unique - 66 values are shared by rows from
    # different projects AND different categories (see logs/step3_id_diagnosis.log).
    # Joining on it corrupts labels. placeholderize.py preserved row order, so we key on
    # row position instead, which is exactly 1:1.
    assert len(orig) == len(place_rows), "row-count mismatch between corpora"
    for a, b in zip(orig, place_rows):
        assert a["project"] == b["project"] and a["category"] == b["category"], \
            "row order diverged between corpora"
    by_id = {k: r for k, r in enumerate(orig)}
    place = {k: r for k, r in enumerate(place_rows)}

    # ---------------- project-disjoint: fold-2 semantics, rebuilt by project ------------
    # Derived from project membership rather than row matching, so the duplicate-id
    # defect cannot affect it. This reproduces the repo's fold-2 partition exactly.
    pd_te_rows = read(f"{BASE}/test_set_{args.fold}.csv")
    test_projects = set(r["project"] for r in pd_te_rows)
    pd_test = [k for k in by_id if by_id[k]["project"] in test_projects]
    pool = [k for k in by_id if k not in set(pd_test)]

    rng = random.Random(SEED)
    n_valid = len(read(f"{BASE}/data_split/valid_set_{args.fold}.csv"))
    by_cls = defaultdict(list)
    for k in pool:
        by_cls[int(by_id[k]["category"])].append(k)
    pd_valid = []
    frac = n_valid / len(pool)
    for c in range(6):
        g = sorted(by_cls[c]); rng.shuffle(g)
        pd_valid += g[:round(len(g) * frac)]     # stratified, like the repo's 80/20
    pd_valid = pd_valid[:n_valid]
    pd_train = [k for k in pool if k not in set(pd_valid)]
    pd_ids = {"train": pd_train, "valid": pd_valid, "test": pd_test}
    target = dist([by_id[k] for k in pd_test])   # per-class quota for the same-project test

    # ---------------- same-project: stratified re-partition of everything ---------------
    rng = random.Random(SEED)
    by_cls = defaultdict(list)
    for k in by_id:
        by_cls[int(by_id[k]["category"])].append(k)
    sp_test = []
    for c in range(6):
        g = sorted(by_cls[c]); rng.shuffle(g)
        sp_test += g[:target[c]]                 # identical per-class support to PD test
    rest = [k for k in sorted(by_id) if k not in set(sp_test)]
    rng.shuffle(rest)
    sp_valid, sp_train = rest[:len(pd_valid)], rest[len(pd_valid):]
    sp_ids = {"train": sp_train, "valid": sp_valid, "test": sp_test}

    # ---------------- materialise 2x2 -------------------------------------------------
    report = {}
    for corpus, src in (("original", by_id), ("placeholder", place)):
        for setting, ids in (("project_disjoint", pd_ids), ("same_project", sp_ids)):
            for part in ("train", "valid", "test"):
                rows = [src[i] for i in ids[part]]
                write(rows, f"{OUT}/{corpus}/{setting}/{part}.csv", fields)
            report[(corpus, setting)] = {p: [src[i] for i in ids[p]] for p in ids}

    # ---------------- verification ----------------------------------------------------
    print("=" * 94)
    print("VERIFICATION")
    ok = True
    for setting, ids in (("project_disjoint", pd_ids), ("same_project", sp_ids)):
        P = lambda p: set(by_id[i]["project"] for i in ids[p])
        tr_te = P("train") & P("test")
        frac = len(tr_te) / len(P("test"))
        print(f"\n  [{setting}]")
        print(f"    projects: train={len(P('train')):>3}  valid={len(P('valid')):>3}  test={len(P('test')):>3}")
        print(f"    train-test project overlap: {len(tr_te)} of {len(P('test'))} test projects "
              f"({100*frac:.1f}%)")
        if setting == "project_disjoint" and len(tr_te) != 0:
            ok = False; print("    !! LEAKAGE in project-disjoint split")
        if setting == "same_project" and frac < 0.9:
            ok = False; print("    !! same-project split has too little overlap")
        # no row may appear in two parts
        allids = [i for p in ids for i in ids[p]]
        if len(allids) != len(set(allids)):
            ok = False; print("    !! duplicate rows across parts")
        print(f"    rows: train={len(ids['train'])}  valid={len(ids['valid'])}  test={len(ids['test'])}"
              f"  total={len(allids)}")

    # id sets identical across corpora
    for setting in ("project_disjoint", "same_project"):
        a = {p: [ (r["project"],r["test_name"]) for r in report[("original", setting)][p] ] for p in ("train","valid","test")}
        b = {p: [ (r["project"],r["test_name"]) for r in report[("placeholder", setting)][p] ] for p in ("train","valid","test")}
        same = all(a[p] == b[p] for p in a)
        print(f"\n  row sets identical original vs placeholder [{setting}]: {same}")
        if not same:
            ok = False
    print(f"\n  ALL CHECKS PASSED: {ok}")

    # ---------------- class distribution ----------------------------------------------
    print("\n" + "=" * 94)
    print("CLASS DISTRIBUTION (identical for original and placeholder by construction)")
    print(f"{'':<26}{'project-disjoint':>26}   {'same-project':>26}")
    print(f"{'class':<26}{'train':>8}{'valid':>8}{'test':>8}   {'train':>8}{'valid':>8}{'test':>8}")
    for c in range(6):
        row = f"{NAMES[c]:<26}"
        for ids in (pd_ids, sp_ids):
            for p in ("train", "valid", "test"):
                row += f"{sum(1 for i in ids[p] if int(by_id[i]['category'])==c):>8}"
            row += "   " if ids is pd_ids else ""
        print(row)
    print(f"{'TOTAL':<26}" + f"{len(pd_ids['train']):>8}{len(pd_ids['valid']):>8}{len(pd_ids['test']):>8}   "
          f"{len(sp_ids['train']):>8}{len(sp_ids['valid']):>8}{len(sp_ids['test']):>8}")

    cfg = {"seed": SEED, "fold": args.fold,
           "sizes": {k: len(v) for k, v in pd_ids.items()},
           "test_class_quota": dict(zip(NAMES, target)),
           "note": "same_project test matches project_disjoint test per-class counts exactly"}
    json.dump(cfg, open(f"{OUT}/config.json", "w"), indent=2)
    print(f"\nwrote 12 csv files under {OUT}/  + config.json")


if __name__ == "__main__":
    main()
