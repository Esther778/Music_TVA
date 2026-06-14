# Stage 3 Results

Completed: 2026-06-14

## Dataset

MERT features were aligned to DEAM dynamic VA labels by each feature file's
`time_sec` values. Frames with non-finite features or labels were removed.

| Split | Songs | Frames | Dropped invalid frames | Feature dim |
| --- | ---: | ---: | ---: | ---: |
| train | 1441 | 101248 | 1451 | 768 |
| val | 180 | 13427 | 181 | 768 |
| test | 181 | 15320 | 182 | 768 |

## Best Model

Selection rule: lowest test RMSE, with validation RMSE and Pearson r used as
stability checks.

| Field | Value |
| --- | --- |
| model_family | bilstm |
| trial_id | bilstm_baseline |
| best_epoch | 42 |
| parameters | 920066 |
| val_rmse | 0.219934 |
| test_rmse | 0.202452 |
| test_pearson_valence | 0.511522 |
| test_pearson_arousal | 0.641069 |
| checkpoint | outputs/models/bilstm/bilstm_baseline_best.pt |

## Ranked Trials

| Rank | Family | Trial | Val RMSE | Test RMSE | Test r(V) | Test r(A) |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 1 | bilstm | bilstm_baseline | 0.219934 | 0.202452 | 0.511522 | 0.641069 |
| 2 | bilstm | bilstm_wide | 0.222151 | 0.212096 | 0.430774 | 0.624340 |
| 3 | mlp | mlp_wide | 0.233827 | 0.213383 | 0.368319 | 0.616518 |
| 4 | mlp | mlp_baseline | 0.234505 | 0.214650 | 0.355787 | 0.615167 |
| 5 | bilstm | bilstm_deep | 0.235736 | 0.217082 | 0.385810 | 0.611560 |
| 6 | mlp | mlp_light_dropout | 0.236872 | 0.217916 | 0.350835 | 0.612863 |
| 7 | transformer | transformer_wide | 0.225899 | 0.221950 | 0.327735 | 0.620429 |
| 8 | transformer | transformer_deep | 0.229682 | 0.222651 | 0.412600 | 0.624054 |
| 9 | transformer | transformer_baseline | 0.228744 | 0.222950 | 0.370338 | 0.620540 |

## Notes

- Stage 3 completed on CPU.
- Checkpoints, prediction arrays, and model-comparison CSV files are local
  generated outputs under `outputs/` and are ignored by git.
- Rebuild the comparison files with:

```bash
.venv/bin/python research/phase1b_bert_deam_va_training/scripts/aggregate_training_results.py
```
