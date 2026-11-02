# Placeholder study — paused mid Step 4

Paused 2026-09-29 after 11 of 12 runs (power outage). Steps 1-3 complete and approved.

## Progress

| cell | seeds done | status |
|---|---|---|
| original / project_disjoint | 42, 1337, 2024 | ✅ complete |
| original / same_project | 42, 1337, 2024 | ✅ complete |
| placeholder / project_disjoint | 42, 1337, 2024 | ✅ complete |
| placeholder / same_project | 42, 1337 | ⬜ **seed 2024 outstanding** |

116 GPU-minutes spent. **One run left (~11 min):** placeholder / same_project / seed 2024.

## Resume command

Runs only the 9 outstanding cells; the 3 finished ones are skipped, so nothing is
recomputed and nothing is overwritten.

```bash
cd ~/projects/FlakyLens
for corpus in original placeholder; do
  for setting in project_disjoint same_project; do
    for seed in 42 1337 2024; do
      f="experiments/placeholder_study/results/${corpus}__${setting}__seed${seed}.json"
      [ -f "$f" ] && { echo "skip $corpus/$setting/$seed"; continue; }
      echo "##### $corpus / $setting / seed $seed"
      .venv/bin/python experiments/placeholder_study/run_cell.py \
        --corpus $corpus --setting $setting --seed $seed \
        --epochs 20 --patience 8 --root experiments/placeholder_study
    done
  done
done 2>&1 | grep -viE "futurewarning|deprecat|warnings.warn|weights of the model|newly initialized|You should probably" \
  | tee -a experiments/placeholder_study/logs/step4_all_runs.log
```

## Results so far — original / project_disjoint (best-epoch, test set)

| seed | macro-F1 | accuracy | Conc→Async | Conc F1 | epochs | min |
|---|---|---|---|---|---|---|
| 42 | 66.46% | 98.49% | 15/17 (88.2%) | 0.00% | 15 | 10.7 |
| 1337 | 57.96% | 97.98% | 16/17 (94.1%) | 0.00% | 20 | 13.9 |
| 2024 | 62.52% | 98.17% | 16/17 (94.1%) | 0.00% | 11 | 7.9 |
| **mean ± std** | **62.31 ± 4.25%** | 98.21 ± 0.26% | **15.7/17 (92.2%)** | **0.00%** | | |

Macro-F1 swings 8.5 points on seed alone, but **Conc→Async is ~92% in every seed and
Concurrency F1 is 0.00% in every seed** — the quantity this study measures is far more
stable than the headline score. The earlier surviving checkpoint (Conc F1 10.53%,
Conc→Async 70.6%) now looks like a favourable draw rather than typical behaviour.

## Environment notes

- `local_models/codebert-base` is gitignored and did **not** survive the repo wipe. It was
  restored from the HF cache. If it goes missing again:
  ```bash
  SNAP=$(ls -d ~/.cache/huggingface/hub/models--microsoft--codebert-base/snapshots/*/ | head -1)
  mkdir -p local_models/codebert-base && cp -L "$SNAP"* local_models/codebert-base/
  ```
- Two orphan checkpoints from the interrupted 4th run and the smoke test were deleted.
- `experiments/placeholder_study/` is 2.4 GB, almost all of it checkpoints
  (~478 MB each). The 12 finished runs will total ~5.7 GB.

## Known caveats carried into Step 4

1. These project-disjoint numbers are **not** directly comparable with the earlier 59.12%
   checkpoint: the validation slice is re-derived with seed 42 rather than the repo's
   `random_state=49`, so the train/valid partition differs slightly. Treat 62.31 ± 4.25%
   as this study's own baseline, not as an improvement over prior work.
2. FlakeBench's `id` column is **not unique** — 66 values are shared across rows from
   different projects and different categories. All joins here key on row position.
   The repo's own pipeline reads columns positionally, so the paper is unaffected.
3. String literals were deliberately left untouched by the placeholder transform, so
   ~16% of tests still contain a project-name token inside a string. This is the study's
   main limitation.

## Next after Step 4

Step 5 (information-sufficiency analysis) and Step 6 (SUMMARY.md + email draft).
