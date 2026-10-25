# Valence Song-Level Residual Correction - 2026-06-27

Report time: 2026-06-27

Current research progress: Phase 1F continues to optimize the Valence teacher.
The current bottleneck is that frame-level Valence RMSE/MAE/PCC/CCC still miss
the minimum target thresholds.

## Purpose

Previous context-conditioned retraining showed that song-level Valence bias is
partly learnable, but directly injecting context into the BiLSTM hurt
frame-level generalization. This round therefore tested a more conservative
strategy:

```text
Keep the historical best frame-level model unchanged.
Train a lightweight song-level residual/bias correction model on train songs.
Select correction strength only on validation.
Apply bounded correction to every frame in each song.
```

The test set was not used for selection.

## Thresholds Used For Reporting

| Metric | Minimum pass | Good | Excellent | Direction |
| --- | ---: | ---: | ---: | --- |
| RMSE | 0.180000 | 0.150000 | 0.100000 | lower is better |
| MAE | 0.150000 | 0.120000 | 0.080000 | lower is better |
| PCC | 0.800000 | 0.850000 | 0.900000 | higher is better |
| CCC | 0.800000 | 0.850000 | 0.900000 | higher is better |

## Historical Best Before This Round

```text
model_id = pmemo_valence_mert_hidden6_harmonic_extreme_weighted_v1
trial_id = h6_harm_h80_ext125_clip010_ccc005
test RMSE/MAE/PCC/CCC = 0.205673/0.164567/0.783400/0.767304
```

Gap to thresholds:

| Metric | Historical best | Minimum pass | Gap to pass | Good | Gap to good | Excellent | Gap to excellent |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RMSE | 0.205673 | 0.180000 | +0.025673 | 0.150000 | +0.055673 | 0.100000 | +0.105673 |
| MAE | 0.164567 | 0.150000 | +0.014567 | 0.120000 | +0.044567 | 0.080000 | +0.084567 |
| PCC | 0.783400 | 0.800000 | +0.016600 | 0.850000 | +0.066600 | 0.900000 | +0.116600 |
| CCC | 0.767304 | 0.800000 | +0.032696 | 0.850000 | +0.082696 | 0.900000 | +0.132696 |

## Current Experiment

Validation-selected correction:

```text
feature_set = pred_stats
model = Ridge
alpha = 0.1
strength = 0.25
cap = 0.06
selection_rule = validation pass-gap composite
```

Current experiment test result:

```text
test RMSE/MAE/PCC/CCC = 0.204718/0.163881/0.782784/0.766340
```

Comparison against historical best:

| Metric | Current experiment | Historical best | Delta vs historical best | Minimum pass | Current gap to pass | Good | Current gap to good | Excellent | Current gap to excellent |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RMSE | 0.204718 | 0.205673 | -0.000955 | 0.180000 | +0.024718 | 0.150000 | +0.054718 | 0.100000 | +0.104718 |
| MAE | 0.163881 | 0.164567 | -0.000686 | 0.150000 | +0.013881 | 0.120000 | +0.043881 | 0.080000 | +0.083881 |
| PCC | 0.782784 | 0.783400 | -0.000616 | 0.800000 | +0.017216 | 0.850000 | +0.067216 | 0.900000 | +0.117216 |
| CCC | 0.766340 | 0.767304 | -0.000964 | 0.800000 | +0.033660 | 0.850000 | +0.083660 | 0.900000 | +0.133660 |

Decision:

```text
Do not promote as the balanced historical best.
It slightly improves RMSE/MAE but slightly worsens PCC/CCC.
```

## Diagnostic-Only Result

The best test-ranked row was:

```text
feature_set = pred_mert
model = Ridge
alpha = 10.0
strength = 1.0
cap = 0.15
test RMSE/MAE/PCC/CCC = 0.197412/0.156906/0.793820/0.774973
```

This is not promoted because validation did not support it:

```text
val RMSE/MAE/PCC/CCC = 0.224252/0.165691/0.805242/0.780513
```

Interpretation:

```text
Song-level residual correction can reduce test error, but current validation
split does not reliably select the aggressive correction that helps test. This
is another sign of validation/test distribution shift.
```

## Conclusion

This round produced a small RMSE/MAE improvement candidate, but not a robust
new teacher:

```text
RMSE improved by 0.000955
MAE improved by 0.000686
PCC worsened by 0.000616
CCC worsened by 0.000964
```

The balanced historical best remains:

```text
h6_harm_h80_ext125_clip010_ccc005
test RMSE/MAE/PCC/CCC = 0.205673/0.164567/0.783400/0.767304
```

Next recommendation:

```text
Stop relying on one fixed validation/test split for Valence. The next meaningful
step is multi-split or cross-validation evaluation of the current best and the
residual correction family, then select a method by average validation behavior
rather than one unstable split.
```

Artifacts:

```text
script = scripts/search_pmemo_valence_song_residual_correction.py
outputs = outputs/valence_song_residual_correction_20260627/
summary = outputs/valence_song_residual_correction_20260627/summary.json
all results = outputs/valence_song_residual_correction_20260627/all_song_residual_results.csv
selected predictions = outputs/valence_song_residual_correction_20260627/selected_*_predictions.csv
```
