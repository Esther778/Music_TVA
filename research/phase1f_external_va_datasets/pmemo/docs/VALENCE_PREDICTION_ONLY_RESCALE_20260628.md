# Valence Prediction-Only Rescale Update

Report time: 2026-06-28 Asia/Shanghai

Current research progress: Phase 1 VA Teacher optimization. This experiment
improved the current Valence candidate using a deployable prediction-only
song-level rescaling step.

## Updated Current Best

Previous Valence best:

```text
model_id = pmemo_valence_frame_calibrator_song_bias_20260628
test RMSE/MAE/PCC/CCC mean = 0.197480/0.153529/0.793235/0.780052
```

New Valence best:

```text
model_id = pmemo_valence_prediction_only_rescale_ccc_preserving_20260628
test RMSE/MAE/PCC/CCC mean = 0.195316/0.151793/0.797766/0.784834
```

Delta versus previous best:

```text
RMSE = -0.002164
MAE  = -0.001736
PCC  = +0.004531
CCC  = +0.004782
```

## Method

The method does not retrain the audio model and does not require labels at
inference time. It uses only the predicted Valence curve of each song:

```text
new_pred = mean_gain * song_pred_mean + std_gain * (pred - song_pred_mean)
```

Interpretation:

```text
mean_gain adjusts whole-song Valence polarity
std_gain adjusts within-song curve amplitude
blend controls how strongly to apply the transform
```

Selection rule:

```text
fit requirement = none; prediction-only transform
parameter search = validation set only
test set = held out for reporting
CCC-preserving rule = select rescale candidate only if
                      val_ccc >= baseline_val_ccc - 0.001
```

Scripts:

```text
scripts/search_pmemo_valence_prediction_only_rescale.py
scripts/search_pmemo_valence_prediction_only_rescale_refine.py
scripts/materialize_pmemo_valence_rescale_ccc_preserving.py
```

Outputs:

```text
outputs/valence_prediction_only_rescale_20260628/
outputs/valence_prediction_only_rescale_refine_20260628/
outputs/valence_prediction_only_rescale_ccc_preserving_20260628/
```

## Result Comparison

| Model | Test RMSE | Test MAE | Test PCC | Test CCC | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| Previous best | 0.197480 | 0.153529 | 0.793235 | 0.780052 | replace |
| Coarse prediction-only rescale | 0.195836 | 0.152452 | 0.795591 | 0.780084 | useful |
| Refined rescale, lowest validation score | 0.194530 | 0.151292 | 0.797728 | 0.780528 | useful but CCC weaker |
| CCC-preserving refined rescale | 0.195316 | 0.151793 | 0.797766 | 0.784834 | promote |

The refined lowest-score variant has the best RMSE/MAE, but the CCC-preserving
variant is promoted because the project treats CCC as an active acceptance
metric.

## Selected Split Parameters

| Split | mean_gain | std_gain | blend | Test RMSE | Test MAE | Test PCC | Test CCC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| split_01_seed_101 | 1.000 | 0.500 | 0.900 | 0.181559 | 0.141920 | 0.844993 | 0.835335 |
| split_02_seed_202 | 0.975 | 0.500 | 1.000 | 0.209386 | 0.162401 | 0.762291 | 0.742968 |
| split_03_seed_303 | 1.050 | 0.500 | 1.000 | 0.195004 | 0.151058 | 0.786013 | 0.776199 |

## Gap To Thresholds

Positive gap means still short of the threshold.

| Metric | Current | Minimum pass | Gap to pass | Good | Gap to good | Excellent | Gap to excellent |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RMSE | 0.195316 | 0.180000 | +0.015316 | 0.150000 | +0.045316 | 0.100000 | +0.095316 |
| MAE | 0.151793 | 0.150000 | +0.001793 | 0.120000 | +0.031793 | 0.080000 | +0.071793 |
| PCC | 0.797766 | 0.800000 | +0.002234 | 0.850000 | +0.052234 | 0.900000 | +0.102234 |
| CCC | 0.784834 | 0.800000 | +0.015166 | 0.850000 | +0.065166 | 0.900000 | +0.115166 |

## Interpretation

This result is important because it improves all four metrics without changing
the model or using labels at inference. It suggests that part of the Valence
error comes from output-scale mismatch rather than feature extraction alone.

However, the model is still below the target line. MAE and PCC are now very
close to passing, while RMSE and especially CCC remain the main gaps.

Next steps:

```text
1. keep this rescale candidate as the current Valence best;
2. focus next on improving split_02, which still has the weakest test metrics;
3. avoid blind heavy retraining until the split_02 high-error songs are audited;
4. if training resumes, target song-level Valence polarity/CCC rather than only
   frame-level RMSE.
```
