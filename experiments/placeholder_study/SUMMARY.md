# Does FlakyLens classify flakiness, or project identity?

Identifier-anonymisation and information-sufficiency study on FlakeBench fold 2.

**Headline.** Replacing project-specific identifiers with canonical placeholders takes
Concurrency F1 from **0.00% (3/3 seeds) to 36.37 ± 22.06%** under project-disjoint
evaluation, and cuts the Concurrency→Async-Wait confusion from **92% to 67%**. It costs
**5.07 macro-F1 points**, concentrated in Time and Unordered Collections. The benefit
appears *only* in the project-disjoint setting, which is the signature of removing a
shortcut rather than making the task easier.

---

## 1. Setup

| | |
|---|---|
| Model | `microsoft/codebert-base` + MLP head (the repo's `BERT_Arch`), unchanged |
| Data | FlakeBench, 8,574 tests, 98 projects, 6 classes (280 flaky / 8,294 non-flaky) |
| Fold | fold 2 only |
| Recipe | focal loss γ=2 + balanced class weights, AdamW 1e-5, batch 8, max_len 512, fp16, length-grouped dynamic padding, non-flaky undersampled to 800 / flaky oversampled to 160 |
| Schedule | max 20 epochs, early stop patience 8 on validation macro-F1 |
| Seeds | 42, 1337, 2024 |
| Hardware | 1× RTX 4060 Laptop (8 GB), 131 GPU-minutes total |

**Placeholder transform** (`placeholderize.py`): `javalang` tokenises each method;
every `Identifier` is classified CLASS/METHOD/VAR/PKG (AST where it parses — 93.5% — else
a positional heuristic) and rewritten to `CLASS_1`, `METHOD_2`, `VAR_3`, `PKG_1`,
consistent within a test and independent across tests. A ~300-entry allowlist preserves
JDK, `java.util.concurrent`, locks/atomics, JUnit, Mockito/Hamcrest/Awaitility and the
concurrency-wait verb vocabulary.

- 265,920 tokens replaced — 31.0 per test, **79.7% of all identifiers**
- distinct CamelCase identifiers across the corpus: 7,835 → 2,357 (**69.9% removed**)
- concurrency/wait vocabulary: 2,632 occurrences preserved; **3 of 8,508 tests lost any**

**Splits.** Data volume, split sizes and *test-set per-class support* are held identical
across settings, so confusion counts compare one-to-one. Only the partitioning rule varies.

| | train | valid | test | train–test project overlap |
|---|---|---|---|---|
| project-disjoint | 5,114 | 1,279 | 2,181 | **0 of 25 (0%)** |
| same-project | 5,114 | 1,279 | 2,181 | **96 of 97 (99%)** |

Test support in both: Async 24, Concurrency 17, Time 15, UC 4, OD 43, Non-flaky 2,078.

---

## 2. The 2×2

Test set, best-epoch checkpoint, mean ± std over 3 seeds.

| corpus | setting | macro-F1 | accuracy | Conc→Async rate | **Concurrency F1** |
|---|---|---|---|---|---|
| original | same-project | 64.37 ± 2.08 | 98.21 ± 0.05 | 80.4 ± 18.9 | 24.43 ± 18.63 |
| original | project-disjoint | 62.31 ± 4.26 | 98.21 ± 0.26 | **92.2 ± 3.4** | **0.00 ± 0.00** |
| placeholder | same-project | 63.14 ± 3.54 | 98.24 ± 0.17 | 70.6 ± 10.2 | 21.52 ± 6.42 |
| placeholder | project-disjoint | 57.24 ± 3.12 | 97.91 ± 0.15 | **66.7 ± 26.5** | **36.37 ± 22.06** |

### Contrasts

| Contrast | macro-F1 | Conc→Async rate | Concurrency F1 |
|---|---|---|---|
| original: same-project − project-disjoint | +2.06 (±2.73 SE) | −11.8 (±11.1) | **+24.43 (±10.8)** |
| project-disjoint: placeholder − original | **−5.07 (±3.05)** | −25.5 (±15.4) | **+36.37 (±12.7)** |
| same-project: placeholder − original | −1.22 (±2.37) | −9.8 (±12.4) | −2.91 (±11.4) |

### Concurrency precision / recall

| corpus | setting | precision | recall | F1 |
|---|---|---|---|---|
| original | project-disjoint | **0.00%** | 0.00% | 0.00% |
| original | same-project | 40.00% | 19.61% | 24.43% |
| placeholder | same-project | 55.56% | 13.73% | 21.52% |
| placeholder | project-disjoint | **78.89%** | 27.45% | 36.37% |

### Confusion matrices (summed over 3 seeds; rows = true)

```
[original / project_disjoint]          [placeholder / project_disjoint]
         Async Conc Time  UC   OD              Async Conc Time  UC   OD
 Async      56    0    2   0   14       Async     60    1    1   2    8
  Conc      47    0    0   0    4        Conc     34   14    0   2    1
  Time      10    0   29   0    6        Time     18    5   12   5    5
    UC       0    0    0  11    1          UC      1    0    0  11    0
    OD      20    0    3   9   97          OD     24    0    0  29   76
```

---

## 3. Findings

### F1. Identifier anonymisation repairs the Concurrency collapse on unseen projects

Under project-disjoint evaluation the original corpus gets **zero of 51** Concurrency
predictions right across three seeds; the placeholder corpus gets **14 of 51**. Precision
goes 0.00% → **78.89%**: when the anonymised model does say "Concurrency" on unfamiliar
code, it is usually right. The effect holds in 3/3 seeds and survives the last-epoch
cross-check (0.00 → 20.56), so it is not a checkpoint-selection artefact.

Evidence strength: **+36.4 at ~2.9 SE** — the only contrast in the study that clears 2 SE.

### F2. The gap between settings is invisible in macro-F1 and lives entirely in the rare classes

Same-project vs project-disjoint moves macro-F1 by **+2.06** (inside noise) but
Concurrency F1 by **+24.43**. On original text the model identifies **no** Concurrency test
on unseen projects and some on seen projects. Because 2,078 of 2,181 test cases are
non-flaky, accuracy is 98.2% in every cell and macro-F1 is nearly blind to the effect.
Reporting either metric alone conceals the reliance on project familiarity.

### F3. The benefit is specific to project-disjoint — it is shortcut removal, not easier text

Placeholders help when test projects are unseen (+36.4 Concurrency F1) and do nothing when
they are seen (−2.9, well inside noise). A "placeholders raise signal-to-noise" explanation
predicts a benefit in both settings, so the interaction rules it out. What the transform
removes is a crutch that only fails on unfamiliar projects.

### F4. There is a real cost, and it is not uniform

macro-F1 −5.07 under project-disjoint, concentrated in:

| class | original | placeholder | Δ |
|---|---|---|---|
| Time | 73.06 ± 15.08 | 41.40 ± 1.22 | **−31.7** |
| Unordered Collections | 69.40 ± 9.53 | 38.52 ± 10.14 | **−30.9** |
| Test Order Dependency | 76.72 ± 6.72 | 68.98 ± 3.07 | −7.7 |
| Async Wait | 54.69 ± 1.66 | 58.18 ± 11.13 | +3.5 |
| Concurrency | 0.00 ± 0.00 | 36.37 ± 22.06 | **+36.4** |

UC precision collapses to 25.6%, driven by OD→UC errors rising to 29. So anonymisation is
a **trade**, not a free win: project vocabulary carries genuine signal for Time and UC
(often domain-specific timing and collection types) while acting as a misleading shortcut
for Concurrency.

### F5. Concurrency is the *best*-cued class, and cue-free tests are not the failure cases

This **corrects an earlier claim of mine** that ~1/3 of Concurrency tests contain no
concurrency information.

| class | cue-free (broad lexicon) |
|---|---|
| **Concurrency** | **18.9% (7/37)** — lowest of all six |
| Async Wait | 34.2% |
| Time | 78.8% |
| Unordered Collections | 90.2% |
| Test Order Dependency | 89.2% |
| Non-flaky | 90.0% |

(Narrow lexicon — JDK primitives only — gives 35% for Concurrency, 39% for Async Wait.)

Predictions on 204 clean test-set observations:

| group | n | →Async | correct |
|---|---|---|---|
| cue-bearing | 132 | 76.5% | 15.2% |
| cue-free | 72 | 77.8% | 16.7% |

**No difference.** Cue-free tests are not the ones being misclassified. And the most
striking single cell: on original text with unseen projects, cue-free Concurrency tests go
to Async Wait **100% of the time**; after anonymisation the same tests are correct **38.1%**
of the time — the best Concurrency accuracy anywhere in the study. Tests without an
explicit primitive benefit *more* from anonymisation, because they were being judged almost
entirely on framework vocabulary.

### F6. Where the concurrency lives, for the 13 narrow-lexicon cue-free tests

| Location | n | Example |
|---|---|---|
| Code under test — starts a real server/cluster/daemon | 5 | ignite `startGrids(3)`; cassandra `Cluster.build(2)` |
| Code under test — multi-client / replica interaction | 3 | opensearch `NUM_THREADS = scaledRandomIntBetween(100,120)` |
| Library's threading model | 2 | RxJava `Schedulers.computation()`; reactive-grpc `Mono` chain |
| Class-level field | 2 | graylog — `scheduler` field passed to `new KafkaJournal(...)` |
| Unclear / possibly mislabeled | 1 | atlasdb `extraSweepersGiveUpAfterFailing...` |

For **11 of 13** the concurrency is real but outside the method. Method-only input is
insufficient for those — but since they are a minority *and* no worse-predicted than the
rest, insufficient input is **not** the main cause of the Conc→Async collapse.

---

## 4. Artifact issues found

1. **`id` is not a unique key.** 66 `id` values in `FlakeBench_dataset.csv` are shared by
   two or more rows spanning **different projects and different categories** (e.g. `id=281`
   is both a soot Unordered-Collections test and a hadoop Non-flaky test). The repo's own
   pipeline reads columns positionally and never joins on `id`, so **published results are
   unaffected** — but any downstream join on `id` silently mislabels data. We hit this and
   rekeyed on row position.
2. **Surrounding code is not shipped.** `full_code` is the test method only; there is no
   setup/teardown, no class fields, no helpers, no code under test. This bounds what any
   method-only classifier can achieve and made the F6 categorisation inferential.
3. **README column semantics are swapped** — it documents `label` as numeric 0–5 and
   `category` as the string; the CSV is the other way round (code is correct).

---

## 5. Limitations

1. **Variance dominates.** Concurrency F1 std is ±18–22 on means of 21–36. Only F1 clears
   ~2.9 SE; F2 and F3 sit at 1–2 SE and are **suggestive, not established**. Three seeds is
   thin for a class with 17 test instances where one prediction moves F1 ~6 points.
2. **One fold.** Fold 2 only. Fold-to-fold variation on this dataset is large (earlier
   4-fold work spanned 46.7–62.1% macro-F1), so these numbers should not be read as
   dataset-level.
3. **String literals were not anonymised**, deliberately — rewriting them is not cleanly
   semantics-preserving. Consequently ~16% of tests still contain a project-name token
   inside a string (down from 29%). The measured effect is therefore a *lower bound* on
   what full anonymisation would show.
4. **The transform removes two things at once**: project identifiers *and* test-method
   names, which carry semantic hints (`...Concurrently`, `...Race`). These are not separated.
5. **Not comparable to earlier numbers** in this project: the validation slice is re-derived
   with seed 42 rather than the repo's `random_state=49`, so 62.31 ± 4.26% is this study's
   own baseline, not an improvement over prior work.
6. Validation is same-project in both settings (matching the paper). Best-epoch Concurrency
   numbers are consistently higher than last-epoch, confirming that same-project validation
   inflates the rare class; the direction of every finding is unchanged.

---

## 6. Proposed next experiments

| # | Experiment | Cost | Tests |
|---|---|---|---|
| 1 | Repeat the 2×2 with **10 seeds on all 4 folds** | ~12 h on a 5090 | whether F1–F3 survive proper statistics — the single most valuable follow-up |
| 2 | **Ablate the transform**: identifiers only vs method names only vs both | ~2 h | separates project identity from semantic naming hints (limitation 4) |
| 3 | **Per-class anonymisation**: apply placeholders at inference only for classes it helps | ~1 h | whether the F4 trade can be avoided rather than accepted |
| 4 | **Add class-level context**: prepend fields + `@Before`/`@After` for the 13 cue-free tests | needs repo scraping | directly tests F6's sufficiency claim |
| 5 | **Anonymise string literals too** | ~1 h | closes the lower-bound gap in limitation 3 |

Recommended order: 1, then 2. Experiment 1 decides whether any of this is real; 2 is the
cheapest way to sharpen the claim if it is.

---

## 7. Reproducing

```bash
cd ~/projects/FlakyLens
uv venv --python 3.9 .venv && uv pip install --python .venv/bin/python -r requirements.txt
SNAP=$(ls -d ~/.cache/huggingface/hub/models--microsoft--codebert-base/snapshots/*/ | head -1)
mkdir -p local_models/codebert-base && cp -L "$SNAP"* local_models/codebert-base/

.venv/bin/python experiments/placeholder_study/placeholderize.py
.venv/bin/python experiments/placeholder_study/build_splits.py
for c in original placeholder; do for s in project_disjoint same_project; do for d in 42 1337 2024; do
  .venv/bin/python experiments/placeholder_study/run_cell.py --corpus $c --setting $s --seed $d \
    --epochs 20 --patience 8 --root experiments/placeholder_study
done; done; done
.venv/bin/python experiments/placeholder_study/aggregate.py
```

Scripts: `placeholderize.py`, `build_splits.py`, `run_cell.py`, `aggregate.py`.
Outputs: `data/`, `results/` (12 run JSONs + Step 5 artefacts), `logs/`, `checkpoints/`
(5.7 GB, reproducible — safe to delete).
All original repo files and results are untouched; everything here is additive.
