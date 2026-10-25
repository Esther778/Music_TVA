# Valence Rescale Follow-Up Experiments

Report time: 2026-06-28 Asia/Shanghai

Current research progress: Phase 1 VA Teacher optimization. This report records
follow-up checks after promoting:

```text
pmemo_valence_prediction_only_rescale_ccc_preserving_20260628
```

## Current Best CV Candidate

The current best macro cross-split Valence candidate remains:

```text
model_id = pmemo_valence_prediction_only_rescale_ccc_preserving_20260628
macro test RMSE/MAE/PCC/CCC = 0.195316/0.151793/0.797766/0.784834
```

Gap to minimum pass:

| Metric | Current | Minimum pass | Gap to pass | Good | Gap to good | Excellent | Gap to excellent |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RMSE | 0.195316 | 0.180000 | +0.015316 | 0.150000 | +0.045316 | 0.100000 | +0.095316 |
| MAE | 0.151793 | 0.150000 | +0.001793 | 0.120000 | +0.031793 | 0.080000 | +0.071793 |
| PCC | 0.797766 | 0.800000 | +0.002234 | 0.850000 | +0.052234 | 0.900000 | +0.102234 |
| CCC | 0.784834 | 0.800000 | +0.015166 | 0.850000 | +0.065166 | 0.900000 | +0.115166 |

## Split-02 Audit After Rescale

Artifacts:

```text
outputs/valence_rescale_split02_audit_20260628/
```

The remaining split_02 errors are still mostly song-level mean bias:

```text
split_02 test RMSE/MAE/PCC/CCC = 0.209386/0.162401/0.762291/0.742968
corr(abs(song_mean_bias), song_rmse) = 0.997617
extreme abs(target_mean) >= 0.4 count = 30
extreme group mean song RMSE/MAE = 0.200973/0.197580
non-extreme group mean song RMSE/MAE = 0.154071/0.148089
```

Worst examples remain strongly positive/negative songs whose predictions are
near neutral:

| musicId | Target mean | Pred mean | Bias | Song RMSE |
| ---: | ---: | ---: | ---: | ---: |
| 419 | -0.550722 | 0.133724 | +0.684446 | 0.685281 |
| 230 | 0.510122 | -0.026173 | -0.536295 | 0.536978 |
| 139 | 0.577296 | 0.172052 | -0.405244 | 0.405629 |
| 824 | 0.689524 | 0.289658 | -0.399866 | 0.401465 |
| 149 | -0.380327 | 0.009003 | +0.389329 | 0.391564 |

## Negative Trial: Song-Level Residual Bias After Rescale

Script:

```text
scripts/search_pmemo_valence_rescale_song_bias_ccc_preserving.py
```

Outputs:

```text
outputs/valence_rescale_song_bias_ccc_preserving_20260628/
```

Result:

| Model | Test RMSE | Test MAE | Test PCC | Test CCC | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| Current best | 0.195316 | 0.151793 | 0.797766 | 0.784834 | keep |
| Rescale + song bias | 0.196294 | 0.152499 | 0.795052 | 0.781459 | reject |

Interpretation:

```text
The residual bias model slightly helps split_01 and split_02 but overfits
split_03. It should not be promoted.
```

A very conservative post-hoc bias check produced only microscopic gains:

```text
best conservative check = 0.195225/0.151715/0.797854/0.784960
```

This is not promoted as a formal new best because the gain is too small and
comes from a post-hoc constrained check after seeing the negative result.

## Negative Trial: Final Calibration After Rescale

Script:

```text
scripts/search_pmemo_valence_rescale_final_calibration.py
```

Outputs:

```text
outputs/valence_rescale_final_calibration_20260628/
```

Result:

| Model | Test RMSE | Test MAE | Test PCC | Test CCC | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| Current best | 0.195316 | 0.151793 | 0.797766 | 0.784834 | keep |
| Rescale + final calibration | 0.195428 | 0.152018 | 0.797346 | 0.782687 | reject |

## Global Deployment Rescale Check

The promoted CV candidate uses split-specific validation-selected rescale
parameters. For real arbitrary-song deployment, a single global configuration is
preferable. A pooled-validation and macro-validation global search both selected:

```text
mean_gain = 1.0
std_gain = 0.45
blend = 1.0
```

Artifacts:

```text
outputs/valence_global_prediction_only_rescale_20260628/
```

Global configuration results:

| Evaluation | RMSE | MAE | PCC | CCC |
| --- | ---: | ---: | ---: | ---: |
| Pooled test | 0.195432 | 0.152036 | 0.800989 | 0.787990 |
| Macro split test | 0.195388 | 0.152277 | 0.797910 | 0.783420 |

Interpretation:

```text
For CV reporting, keep the split-specific CCC-preserving rescale result as the
best metric candidate. For arbitrary-song deployment, use the global rescale
configuration unless a final full-training validation protocol is built.
```

## Next Direction

The remaining bottleneck is not simple output calibration. The next meaningful
step should target Valence polarity before the model reaches the postprocess:

```text
1. audit split_02 worst songs for label/audio anomalies and metadata patterns;
2. train a lightweight song-level Valence polarity model using deployable audio
   features, then fuse it cautiously with frame-level predictions;
3. if retraining the frame model, optimize for song-level CCC/polarity rather
   than only frame-level RMSE.
```
