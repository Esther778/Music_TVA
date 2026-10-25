# Stage 3 Results

Last updated: 2026-06-16

## Dataset

MERT features were aligned to DEAM dynamic VA labels by each feature file's
`time_sec` values. Frames with non-finite features or labels were removed.

| Split | Songs | Frames | Dropped invalid frames | Feature dim |
| --- | ---: | ---: | ---: | ---: |
| train | 1441 | 101248 | 1451 | 768 |
| val | 180 | 13427 | 181 | 768 |
| test | 181 | 15320 | 182 | 768 |

## Best Model

Selection rule: lowest validation RMSE. Test metrics are reported only after
model selection and are not used for tuning.

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
| val_mean_ccc | 0.627696 |
| test_mean_ccc | 0.538263 |
| checkpoint | outputs/models/bilstm/bilstm_baseline_best.pt |

## Ranked Trials

| Rank | Family | Trial | Val RMSE | Val CCC | Test RMSE | Test CCC | Test r(V) | Test r(A) |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | bilstm | bilstm_baseline | 0.219934 | 0.627696 | 0.202452 | 0.538263 | 0.511522 | 0.641069 |
| 2 | bilstm | bilstm_wide | 0.222151 | 0.601676 | 0.212096 | 0.496496 | 0.430774 | 0.624340 |
| 3 | bilstm | bilstm_h128_d035 | 0.224221 | 0.601310 | 0.207778 | 0.504572 | 0.456160 | 0.634006 |
| 4 | transformer | transformer_wide | 0.225899 | 0.588435 | 0.221950 | 0.447763 | 0.327735 | 0.620429 |
| 5 | bilstm | bilstm_h96_wd5e4 | 0.226601 | 0.605443 | 0.206791 | 0.535974 | 0.488385 | 0.635803 |
| 6 | transformer | transformer_baseline | 0.228744 | 0.581232 | 0.222950 | 0.472710 | 0.370338 | 0.620540 |
| 7 | bilstm | bilstm_h64_d035 | 0.228948 | 0.592385 | 0.215911 | 0.485719 | 0.434412 | 0.593785 |
| 8 | transformer | transformer_deep | 0.229682 | 0.608335 | 0.222651 | 0.505742 | 0.412600 | 0.624054 |
| 9 | bilstm | bilstm_h96_valence_weight | 0.229913 | 0.579883 | 0.205333 | 0.532501 | 0.496761 | 0.632016 |
| 10 | bilstm | bilstm_h96_lr3e4 | 0.231452 | 0.568230 | 0.207012 | 0.522579 | 0.476013 | 0.633054 |
| 11 | bilstm | bilstm_h96_d035 | 0.231509 | 0.568768 | 0.205154 | 0.527726 | 0.488750 | 0.638460 |
| 12 | bilstm | bilstm_h96_smooth005 | 0.231515 | 0.568486 | 0.205111 | 0.527348 | 0.488407 | 0.638902 |
| 13 | mlp | mlp_h256_valence_weight | 0.232948 | 0.525726 | 0.208067 | 0.446329 | 0.411544 | 0.636056 |
| 14 | mlp | mlp_h128_d035 | 0.233411 | 0.522396 | 0.206266 | 0.458847 | 0.430149 | 0.640970 |
| 15 | mlp | mlp_h512_d035_lr3e4 | 0.233482 | 0.519872 | 0.210158 | 0.435209 | 0.394345 | 0.620763 |
| 16 | mlp | mlp_h256_d035 | 0.233527 | 0.514441 | 0.208009 | 0.444714 | 0.411191 | 0.630851 |
| 17 | mlp | mlp_wide | 0.233827 | 0.526689 | 0.213383 | 0.433187 | 0.368319 | 0.616518 |
| 18 | mlp | mlp_h256_d05 | 0.234412 | 0.511895 | 0.206751 | 0.441927 | 0.412213 | 0.644199 |
| 19 | mlp | mlp_baseline | 0.234505 | 0.532141 | 0.214650 | 0.429464 | 0.355787 | 0.615167 |
| 20 | bilstm | bilstm_deep | 0.235736 | 0.568300 | 0.217082 | 0.469829 | 0.385810 | 0.611560 |
| 21 | mlp | mlp_light_dropout | 0.236872 | 0.543182 | 0.217916 | 0.439781 | 0.350835 | 0.612863 |

## Notes

- Stage 3 completed on CPU.
- All three model families remain in scope for RMSE reduction: MLP, BiLSTM, and
  Transformer.
- Current best candidate is provisional until the new RMSE-reduction trials and
  multi-seed checks are complete.
- First MLP RMSE-reduction round completed on 2026-06-16. The best MLP changed
  from `mlp_wide` (`val_rmse=0.233827`) to `mlp_h256_valence_weight`
  (`val_rmse=0.232948`). This is a small MLP-family improvement, but it does
  not surpass the current BiLSTM baseline (`val_rmse=0.219934`).
- First BiLSTM RMSE-reduction round completed on 2026-06-16. The best new
  BiLSTM trial was `bilstm_h128_d035` (`val_rmse=0.224221`), but it did not
  surpass the existing `bilstm_baseline`.
- The current practical teacher model remains `bilstm_baseline`. Further
  exhaustive search is paused for now by user direction.
- OpenL3+MLP song-level validation completed on 2026-06-21 using the exact same
  split. Its three-seed ensemble obtained `val_rmse=0.222275` and
  `test_rmse=0.208309`, so the current MERT+BiLSTM baseline remains ahead by
  validation RMSE. Matching MERT multi-seed verification is the next required
  check before final Teacher selection.
- Checkpoints, prediction arrays, and model-comparison CSV files are local
  generated outputs under `outputs/` and are ignored by git.
- Rebuild the comparison files with:

```bash
.venv/bin/python research/phase1b_bert_deam_va_training/scripts/aggregate_training_results.py
```
