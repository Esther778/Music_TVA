# Valence Shift and Song-Bias Postprocess Report

Report time: 2026-06-28 Asia/Shanghai

Current research progress: Phase 1 VA Teacher optimization. This round did not
train a new neural network. It diagnosed the current Valence model errors and
tested deployable post-processing methods.

## Starting Point

Previous best:

```text
model_id = pmemo_valence_regularized_stack_ensemble_20260628
test RMSE/MAE/PCC/CCC mean = 0.202132/0.156198/0.779322/0.764633
```

## Diagnostic Finding

The worst test songs showed a consistent pattern:

```text
extreme positive or negative song-level Valence target
-> model prediction is pulled toward neutral
```

Examples from split_02:

| musicId | Target mean | Pred mean | Song RMSE |
| ---: | ---: | ---: | ---: |
| 419 | -0.5507 | 0.1321 | 0.6841 |
| 230 | 0.5101 | -0.0144 | 0.5274 |
| 149 | -0.3803 | 0.0457 | 0.4272 |

This suggests that the remaining bottleneck is not only frame-level temporal
modeling. The model still underestimates song-level Valence polarity.

## Time-Shift Postprocess

A validation-selected edge-padded time shift was tested. Edge padding keeps the
same number of frames, so it is usable for full-song inference.

```text
script = scripts/search_pmemo_valence_shift_postprocess.py
outputs = outputs/valence_shift_postprocess_20260628/
```

Result:

| Model | Test RMSE | Test MAE | Test PCC | Test CCC |
| --- | ---: | ---: | ---: | ---: |
| Previous best | 0.202132 | 0.156198 | 0.779322 | 0.764633 |
| Shift postprocess | 0.200982 | 0.154802 | 0.782260 | 0.767101 |

Selected shifts:

| Split | Shift frames | Shift seconds |
| --- | ---: | ---: |
| split_01_seed_101 | -1 | -0.5 |
| split_02_seed_202 | -8 | -4.0 |
| split_03_seed_303 | 4 | 2.0 |

Interpretation:

```text
Time alignment contributes to the error, especially split_02. However, shift
direction is not fully stable across splits, so this should be treated as a
validation-selected postprocess rather than evidence of one universal delay.
```

## Song-Level Bias Correction

A deployable song-level residual correction was tested. It predicts a per-song
Valence bias using only statistics of the model's own predicted trajectory:

```text
pred_mean, pred_std, pred_min, pred_max, pred_range,
pred quantiles, head/tail means, tail-head difference, frame count
```

The correction model is Ridge regression trained on train songs and selected on
validation songs.

Two objectives were tested:

| Objective | Test RMSE | Test MAE | Test PCC | Test CCC | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| CCC-oriented bias | 0.201161 | 0.155201 | 0.784893 | 0.772986 | better CCC, worse RMSE/MAE |
| RMSE-aware bias | 0.200510 | 0.154051 | 0.784522 | 0.772036 | current balanced best |

Current best:

```text
model_id = pmemo_valence_song_bias_correction_rmse_aware_20260628
test RMSE/MAE/PCC/CCC mean = 0.200510/0.154051/0.784522/0.772036
```

## Gap To Thresholds

Positive gap means still short of the threshold.

| Metric | Current | Minimum pass | Gap to pass | Good | Gap to good | Excellent | Gap to excellent |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RMSE | 0.200510 | 0.180000 | +0.020510 | 0.150000 | +0.050510 | 0.100000 | +0.100510 |
| MAE | 0.154051 | 0.150000 | +0.004051 | 0.120000 | +0.034051 | 0.080000 | +0.074051 |
| PCC | 0.784522 | 0.800000 | +0.015478 | 0.850000 | +0.065478 | 0.900000 | +0.115478 |
| CCC | 0.772036 | 0.800000 | +0.027964 | 0.850000 | +0.077964 | 0.900000 | +0.127964 |

## Decision

Promote `pmemo_valence_song_bias_correction_rmse_aware_20260628` as the current
balanced Valence candidate.

It is still not a threshold-passing final teacher, but it improves all four
metrics over the previous best and directly addresses the neutral-prediction
problem for extreme-Valence songs.

## Next Step

The next high-value direction is not another small postprocess. The remaining
gap likely requires improving song-level Valence polarity inside training:

```text
1. add explicit song-level mean loss / polarity loss during Valence training;
2. keep frame-level loss and CCC loss;
3. compare against 0.200510/0.154051/0.784522/0.772036;
4. preserve full-song inference compatibility.
```

## Follow-Up Negative Trial

After this report's main postprocess result, a song-mean-enhanced BiLSTM trial
was tested:

```text
trial_id = bilstm_songmean_h80_do035_wd0005_ccc008_clip030_std002
change = clip_mse_weight increased to 0.30, clip_std_weight = 0.02
standalone test RMSE/MAE/PCC/CCC = 0.212325/0.167211/0.772542/0.764351
```

It did not improve over the current best:

```text
current best RMSE/MAE/PCC/CCC = 0.200510/0.154051/0.784522/0.772036
```

It was also tested as a weak ensemble member. The validation-selected ensemble
produced:

```text
test RMSE/MAE/PCC/CCC = 0.201026/0.155036/0.785756/0.752816
```

Decision:

```text
Do not promote the songmean trial. Increasing clip mean loss to 0.30 is too
unstable and damages CCC in ensemble use.
```
