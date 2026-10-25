# Valence Multi-Split Stability Check - 2026-06-27

Report time: 2026-06-27

Current research progress: Phase 1F is evaluating whether the current best
Valence teacher is stable enough for arbitrary-song prediction. Previous
single-split experiments produced a best Valence result around RMSE 0.2057,
but repeated tuning suggested validation/test instability.

## Purpose

This round stopped tuning on the original fixed split and tested the current
best Valence architecture under three new song-level 8:1:1 splits.

Model family:

```text
MERT hidden_6 768 + Librosa harmonic 36
Valence-only BiLSTM
extreme-weighted frame MSE + 0.05 frame CCC + 0.10 clip mean MSE
```

Each split:

```text
train songs = 613
validation songs = 77
test songs = 77
checkpoint selected by validation selection_score
test not used for selection
```

## Thresholds Used For Reporting

| Metric | Minimum pass | Good | Excellent | Direction |
| --- | ---: | ---: | ---: | --- |
| RMSE | 0.180000 | 0.150000 | 0.100000 | lower is better |
| MAE | 0.150000 | 0.120000 | 0.080000 | lower is better |
| PCC | 0.800000 | 0.850000 | 0.900000 | higher is better |
| CCC | 0.800000 | 0.850000 | 0.900000 | higher is better |

## Historical Single-Split Best

```text
model_id = pmemo_valence_mert_hidden6_harmonic_extreme_weighted_v1
trial_id = h6_harm_h80_ext125_clip010_ccc005
test RMSE/MAE/PCC/CCC = 0.205673/0.164567/0.783400/0.767304
```

Gap to thresholds:

| Metric | Historical single-split best | Minimum pass | Gap to pass | Good | Gap to good | Excellent | Gap to excellent |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RMSE | 0.205673 | 0.180000 | +0.025673 | 0.150000 | +0.055673 | 0.100000 | +0.105673 |
| MAE | 0.164567 | 0.150000 | +0.014567 | 0.120000 | +0.044567 | 0.080000 | +0.084567 |
| PCC | 0.783400 | 0.800000 | +0.016600 | 0.850000 | +0.066600 | 0.900000 | +0.116600 |
| CCC | 0.767304 | 0.800000 | +0.032696 | 0.850000 | +0.082696 | 0.900000 | +0.132696 |

## Multi-Split Base Model Results

| Split | Seed | Val RMSE | Val PCC | Val CCC | Test RMSE | Test MAE | Test PCC | Test CCC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| split_01 | 101 | 0.236698 | 0.662470 | 0.655687 | 0.194410 | 0.153705 | 0.819899 | 0.809436 |
| split_02 | 202 | 0.248610 | 0.722205 | 0.717709 | 0.232051 | 0.182200 | 0.721924 | 0.709718 |
| split_03 | 303 | 0.182693 | 0.851614 | 0.849593 | 0.214638 | 0.163528 | 0.749652 | 0.743822 |

Mean test result:

```text
RMSE/MAE/PCC/CCC mean = 0.213700/0.166477/0.763825/0.754325
RMSE/MAE/PCC/CCC std  = 0.015381/0.011819/0.041235/0.041382
```

Comparison to historical single-split best:

| Metric | Multi-split mean | Historical single-split best | Delta | Minimum pass | Mean gap to pass | Good | Mean gap to good | Excellent | Mean gap to excellent |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RMSE | 0.213700 | 0.205673 | +0.008027 | 0.180000 | +0.033700 | 0.150000 | +0.063700 | 0.100000 | +0.113700 |
| MAE | 0.166477 | 0.164567 | +0.001911 | 0.150000 | +0.016477 | 0.120000 | +0.046477 | 0.080000 | +0.086477 |
| PCC | 0.763825 | 0.783400 | -0.019576 | 0.800000 | +0.036175 | 0.850000 | +0.086175 | 0.900000 | +0.136175 |
| CCC | 0.754325 | 0.767304 | -0.012979 | 0.800000 | +0.045675 | 0.850000 | +0.095675 | 0.900000 | +0.145675 |

Decision:

```text
Do not treat the original single-split Valence result as stable final evidence.
The current architecture does not meet the minimum thresholds on multi-split
average.
```

## Multi-Split Residual Correction

The conservative song-level residual correction family was then evaluated
inside each split. Each correction was trained on train songs and selected by
validation only.

Mean test result after validation-selected residual correction:

```text
RMSE/MAE/PCC/CCC mean = 0.211757/0.163699/0.762573/0.753040
RMSE/MAE/PCC/CCC std  = 0.013142/0.009773/0.042646/0.045511
```

Comparison to multi-split base model:

| Metric | Residual-corrected mean | Base mean | Delta | Minimum pass | Corrected gap to pass |
| --- | ---: | ---: | ---: | ---: | ---: |
| RMSE | 0.211757 | 0.213700 | -0.001943 | 0.180000 | +0.031757 |
| MAE | 0.163699 | 0.166477 | -0.002778 | 0.150000 | +0.013699 |
| PCC | 0.762573 | 0.763825 | -0.001251 | 0.800000 | +0.037427 |
| CCC | 0.753040 | 0.754325 | -0.001285 | 0.800000 | +0.046960 |

Decision:

```text
Do not promote residual correction as the balanced Valence teacher.
It lowers RMSE/MAE slightly but does not improve PCC/CCC.
```

## Interpretation

This round changes the project conclusion:

```text
The Valence problem is not solved by the current MERT hidden_6 + harmonic
BiLSTM model. The original single-split result is useful as a reference point,
but multi-split average is weaker and still below the minimum target.
```

Important observed pattern:

```text
Split difficulty varies strongly.
One split reaches test PCC/CCC above 0.80, while other splits fall clearly
below. This means the model is sensitive to which songs appear in validation
and test.
```

## Next Recommendation

Stop small modifications to the current BiLSTM head. The next meaningful
model-improvement step should add either stronger data or a more robust
training objective:

```text
1. Keep multi-split as the official evaluation protocol.
2. Try a clip/song-level pretraining objective before frame-level fine-tuning.
3. Try a stronger but still lightweight temporal model, such as TCN or GRU-D,
   under the same multi-split protocol.
4. If possible, add DEAM/PMEmo joint training only after label-scale and
   time-axis compatibility are locked.
```

Artifacts:

```text
base script = scripts/train_pmemo_valence_multisplit_best.py
residual script = scripts/search_pmemo_valence_multisplit_residual_correction.py
base outputs = outputs/valence_multisplit_best_20260627/
residual outputs = outputs/valence_multisplit_residual_correction_20260627/
```
