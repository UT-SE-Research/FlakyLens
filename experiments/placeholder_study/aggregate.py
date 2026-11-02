"""Step 4 aggregation - mean +/- std over the 3 seeds for each cell of the 2x2."""
import glob, json
import numpy as np

NAMES = ["Async Wait", "Concurrency", "Time", "Unordered Collections",
         "Test Order Dependency", "Non-flaky"]
SHORT = ["Async", "Conc", "Time", "UC", "OD", "NonFl"]
CELLS = [("original", "same_project"), ("original", "project_disjoint"),
         ("placeholder", "same_project"), ("placeholder", "project_disjoint")]

runs = [json.load(open(f)) for f in sorted(glob.glob("experiments/placeholder_study/results/*__*__seed*.json"))]
print(f"runs loaded: {len(runs)}\n")


def cell(c, s, which="test_best_epoch"):
    return [r[which] for r in runs if r["corpus"] == c and r["setting"] == s]


def ms(v):
    return f"{np.mean(v):.2f} ± {np.std(v, ddof=1):.2f}"


print("=" * 104)
print("2x2 SUMMARY  (test set, best-epoch checkpoint, mean ± std over seeds 42/1337/2024)")
print("=" * 104)
print(f"{'corpus':<12}{'setting':<18}{'n':>3}{'macro-F1':>16}{'accuracy':>16}"
      f"{'Conc->Async':>16}{'rate %':>16}{'Conc F1':>16}")
for c, s in CELLS:
    m = cell(c, s)
    print(f"{c:<12}{s:<18}{len(m):>3}"
          f"{ms([x['macro_f1'] for x in m]):>16}{ms([x['accuracy'] for x in m]):>16}"
          f"{ms([x['conc_to_async_count'] for x in m]):>16}"
          f"{ms([x['conc_to_async_rate'] for x in m]):>16}"
          f"{ms([x['per_class_f1']['Concurrency'] for x in m]):>16}")

print("\n" + "=" * 104)
print("PER-CLASS F1 (mean ± std)")
print("=" * 104)
print(f"{'class':<24}" + "".join(f"{c[:4]+'/'+s[:4]:>20}" for c, s in CELLS))
for n in NAMES:
    print(f"{n:<24}" + "".join(f"{ms([x['per_class_f1'][n] for x in cell(c,s)]):>20}" for c, s in CELLS))

print("\n" + "=" * 104)
print("PER-CLASS PRECISION / RECALL (mean over seeds)")
print("=" * 104)
for c, s in CELLS:
    m = cell(c, s)
    print(f"\n[{c} / {s}]   support: " +
          ", ".join(f"{SHORT[i]}={m[0]['support'][NAMES[i]]}" for i in range(6)))
    print(f"   {'class':<24}{'precision':>12}{'recall':>12}{'f1':>12}")
    for n in NAMES:
        print(f"   {n:<24}{np.mean([x['per_class_precision'][n] for x in m]):>11.2f}%"
              f"{np.mean([x['per_class_recall'][n] for x in m]):>11.2f}%"
              f"{np.mean([x['per_class_f1'][n] for x in m]):>11.2f}%")

print("\n" + "=" * 104)
print("CONFUSION MATRICES (summed over the 3 seeds; rows=true, cols=pred)")
print("=" * 104)
for c, s in CELLS:
    cm = np.sum([np.array(x["confusion_matrix"]) for x in cell(c, s)], axis=0)
    print(f"\n[{c} / {s}]")
    print("          " + "".join(f"{x:>8}" for x in SHORT))
    for i, n in enumerate(SHORT):
        print(f"   {n:>6} " + "".join(f"{v:>8}" for v in cm[i]))

print("\n" + "=" * 104)
print("KEY CONTRASTS (macro-F1 / Conc->Async rate / Concurrency F1)")
print("=" * 104)


def delta(a, b, label):
    A, B = cell(*a), cell(*b)
    for key, name in (("macro_f1", "macro-F1"), ("conc_to_async_rate", "Conc->Async rate"),
                      ("per_class_f1", "Concurrency F1")):
        f = (lambda x: x["per_class_f1"]["Concurrency"]) if key == "per_class_f1" else (lambda x: x[key])
        d = np.mean([f(x) for x in A]) - np.mean([f(x) for x in B])
        pooled = np.sqrt(np.var([f(x) for x in A], ddof=1) / 3 + np.var([f(x) for x in B], ddof=1) / 3)
        print(f"   {label:<52}{name:<20}{d:>+8.2f}   (±{pooled:.2f} SE)")
    print()


delta(("original", "same_project"), ("original", "project_disjoint"),
      "ORIGINAL: same-project minus project-disjoint")
delta(("placeholder", "project_disjoint"), ("original", "project_disjoint"),
      "PROJECT-DISJOINT: placeholder minus original")
delta(("placeholder", "same_project"), ("original", "same_project"),
      "SAME-PROJECT: placeholder minus original")

print("=" * 104)
print("LAST-EPOCH CROSS-CHECK (macro-F1 / Concurrency F1), guards against selection artefacts")
print("=" * 104)
for c, s in CELLS:
    b, l = cell(c, s, "test_best_epoch"), cell(c, s, "test_last_epoch")
    print(f"   {c:<12}{s:<18}best {ms([x['macro_f1'] for x in b]):>16}   "
          f"last {ms([x['macro_f1'] for x in l]):>16}   |   "
          f"Conc best {ms([x['per_class_f1']['Concurrency'] for x in b]):>14}   "
          f"last {ms([x['per_class_f1']['Concurrency'] for x in l]):>14}")
