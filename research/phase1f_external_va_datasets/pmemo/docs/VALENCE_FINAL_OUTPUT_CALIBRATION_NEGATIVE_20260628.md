# Valence Final Output Calibration Negative Trial

Report time: 2026-06-28 Asia/Shanghai

Current research progress: Phase 1 VA Teacher optimization. This experiment
tested whether the near-pass Valence model can be improved by a lightweight
final output calibration stage.

Current best Valence model remains:

```text
model_id = pmemo_valence_frame_calibrator_song_bias_20260628
test RMSE/MAE/PCC/CCC mean = 0.197480/0.153529/0.793235/0.780052
```

## Method

The experiment did not retrain the audio model. It used the current best
frame-level predictions and tested deployable output calibrators:

```text
fit set = train predictions
selection set = validation predictions
held-out report = test predictions
calibrators = identity, linear, ridge, polynomial ridge, Huber, isotonic,
              quantile mapping
extra search = blend weight and optional output clipping
```

Script:

```text
scripts/search_pmemo_valence_final_output_calibration.py
```

Outputs:

```text
outputs/valence_final_output_calibration_20260628/
```

## Main Result

| Model | Test RMSE | Test MAE | Test PCC | Test CCC | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| Current best | 0.197480 | 0.153529 | 0.793235 | 0.780052 | keep |
| Final output calibration | 0.197042 | 0.153111 | 0.792856 | 0.777329 | do not promote |

Delta versus current best:

```text
RMSE = -0.000439
MAE  = -0.000418
PCC  = -0.000380
CCC  = -0.002723
```

The calibration slightly reduces RMSE and MAE, but it also lowers PCC and CCC.
Because CCC is an active acceptance metric, this is not a valid improvement.

## CCC-Preserving Check

A second validation-only check constrained selection to avoid validation CCC
loss. It still failed to improve the official current best on test CCC.

| Selection rule | Test RMSE | Test MAE | Test PCC | Test CCC | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| CCC epsilon 0 | 0.197647 | 0.153615 | 0.792709 | 0.779977 | reject |
| CCC epsilon 0.001 | 0.197442 | 0.153655 | 0.792928 | 0.779294 | reject |
| CCC epsilon 0.002 | 0.196884 | 0.153002 | 0.793402 | 0.778051 | reject |

## Current Best Gap

Positive gap means still short of the threshold.

| Metric | Current | Minimum pass | Gap to pass | Good | Gap to good | Excellent | Gap to excellent |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RMSE | 0.197480 | 0.180000 | +0.017480 | 0.150000 | +0.047480 | 0.100000 | +0.097480 |
| MAE | 0.153529 | 0.150000 | +0.003529 | 0.120000 | +0.033529 | 0.080000 | +0.073529 |
| PCC | 0.793235 | 0.800000 | +0.006765 | 0.850000 | +0.056765 | 0.900000 | +0.106765 |
| CCC | 0.780052 | 0.800000 | +0.019948 | 0.850000 | +0.069948 | 0.900000 | +0.119948 |

## Conclusion

Do not promote this calibration. It is useful as an ablation showing that simple
post-hoc output reshaping can lower point error slightly, but it does not solve
the correlation/agreement bottleneck. The next Valence optimization should
focus on either:

```text
1. better audio representation for Valence polarity;
2. worst-song label/audio audit before more model tuning;
3. a separate deployable song-level Valence polarity module if it can improve
   CCC without using test labels or hidden annotation leakage.
```
