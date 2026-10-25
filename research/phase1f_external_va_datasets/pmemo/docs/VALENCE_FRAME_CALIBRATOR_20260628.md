# Valence Frame Calibrator Report

Report time: 2026-06-28 Asia/Shanghai

Current research progress: Phase 1 VA Teacher optimization. Arousal has already
met the minimum target. Valence remains below the required threshold, so this
round tested deployable second-stage calibration on top of the current Valence
prediction pipeline.

## Starting Point

Previous balanced Valence candidate:

```text
model_id = pmemo_valence_song_bias_correction_rmse_aware_20260628
test RMSE/MAE/PCC/CCC mean = 0.200510/0.154051/0.784522/0.772036
```

## Method

A frame-level calibrator was trained using only features available at inference
time:

```text
frame prediction
local smoothed prediction
prediction delta
normalized time position
song-level predicted trajectory statistics
```

No ground-truth label is used as input during inference. The calibrator is
trained on train labels, selected on validation, and evaluated on held-out test
songs.

Candidate calibrators:

```text
Ridge
Ridge + polynomial features
ElasticNet
HistGradientBoostingRegressor
```

## Frame Calibrator Result

```text
script = scripts/search_pmemo_valence_frame_calibrator.py
outputs = outputs/valence_frame_calibrator_20260628/
```

| Model | Test RMSE | Test MAE | Test PCC | Test CCC |
| --- | ---: | ---: | ---: | ---: |
| Previous best | 0.200510 | 0.154051 | 0.784522 | 0.772036 |
| Frame calibrator | 0.197610 | 0.153505 | 0.791889 | 0.777734 |

Per-split selected models:

| Split | Selected model | Blend | Test RMSE | Test MAE | Test PCC | Test CCC |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| split_01_seed_101 | `hgb_lr002_leaf15_l2_01` | 1.00 | 0.181652 | 0.142437 | 0.844810 | 0.835152 |
| split_02_seed_202 | `hgb_lr003_leaf31_l2_001` | 1.00 | 0.213342 | 0.165516 | 0.754448 | 0.735606 |
| split_03_seed_303 | `hgb_lr002_leaf15_l2_01` | 0.50 | 0.197837 | 0.152563 | 0.776408 | 0.762444 |

## Frame Calibrator + Song Bias

After the frame calibrator, a small deployable song-level bias correction was
tested again. It uses only predicted trajectory statistics, then applies a
capped song-level offset.

```text
script = scripts/search_pmemo_valence_frame_calibrator_song_bias.py
outputs = outputs/valence_frame_calibrator_song_bias_20260628/
```

Result:

| Model | Test RMSE | Test MAE | Test PCC | Test CCC |
| --- | ---: | ---: | ---: | ---: |
| Frame calibrator | 0.197610 | 0.153505 | 0.791889 | 0.777734 |
| Frame calibrator + song bias | 0.197480 | 0.153529 | 0.793235 | 0.780052 |

This is the current best Valence candidate.

## Gap To Thresholds

Positive gap means still short of the threshold.

| Metric | Current | Minimum pass | Gap to pass | Good | Gap to good | Excellent | Gap to excellent |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RMSE | 0.197480 | 0.180000 | +0.017480 | 0.150000 | +0.047480 | 0.100000 | +0.097480 |
| MAE | 0.153529 | 0.150000 | +0.003529 | 0.120000 | +0.033529 | 0.080000 | +0.073529 |
| PCC | 0.793235 | 0.800000 | +0.006765 | 0.850000 | +0.056765 | 0.900000 | +0.106765 |
| CCC | 0.780052 | 0.800000 | +0.019948 | 0.850000 | +0.069948 | 0.900000 | +0.119948 |

## Extra Tree Check

Additional tree candidates were tested after the main frame calibrator:

```text
ExtraTreesRegressor
RandomForestRegressor
GradientBoostingRegressor
```

Best result from that check:

```text
test RMSE/MAE/PCC/CCC mean = 0.198113/0.153544/0.790003/0.777209
```

Decision:

```text
Do not promote. It is weaker than the current frame calibrator + song bias
candidate.
```

## Decision

Promote the following as the current Valence candidate:

```text
model_id = pmemo_valence_frame_calibrator_song_bias_20260628
test RMSE/MAE/PCC/CCC mean = 0.197480/0.153529/0.793235/0.780052
```

It is still not a threshold-passing final teacher, but it is materially closer:

```text
MAE gap to pass = +0.003529
PCC gap to pass = +0.006765
CCC gap to pass = +0.019948
RMSE gap to pass = +0.017480
```

## Next Step

The current postprocess has nearly exhausted simple prediction-only correction.
The next useful direction should target split_02 directly:

```text
1. inspect split_02 extreme-Valence songs and their feature/label distribution;
2. test robust training against extreme song-level Valence under fixed splits;
3. consider a two-head Valence model: frame curve + song-level polarity head;
4. keep 0.197480/0.153529/0.793235/0.780052 as the comparison baseline.
```
