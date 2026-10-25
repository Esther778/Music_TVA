# Arousal OpenSMILE Ensemble Teacher

Date: 2026-06-24

## Purpose

The immediate goal was to make Arousal reach the project minimum metric
standard while keeping the model usable for arbitrary complete-song prediction.

Earlier result:

```text
PMEmo official OpenSMILE + joint VA BiLSTM
Arousal RMSE/MAE/PCC/CCC = 0.186934 / 0.141942 / 0.837598 / 0.830559
```

This was close but still missed the Arousal RMSE target of `0.18`.

## What Changed

The model target was narrowed from joint VA prediction to Arousal-only
prediction. Selection also changed from mean VA validation RMSE to Arousal
validation RMSE.

Trials:

```text
scripts/train_pmemo_opensmile_arousal_trials.py
```

The strongest single model was:

```text
trial_id = a_only_h96_l1_do015_mse
model = 1-layer BiLSTM
hidden_dim = 96
dropout = 0.15
loss = MSE
best_epoch = 59
test RMSE/MAE/PCC/CCC = 0.180960 / 0.140092 / 0.852634 / 0.848122
```

This single model nearly passed, but RMSE still missed the minimum line by
`0.000960`.

## Final Arousal Ensemble

Validation-selected ensemble:

```text
model_id = pmemo_opensmile_arousal_ensemble_v1
selection rule = choose weights by validation RMSE only
test set = final reporting only
feature type = PMEmo-style OpenSMILE ComParE_2016 dynamic features
```

Members:

| Trial | Weight |
| --- | ---: |
| a_only_h128_l2_do025_mse | 0.15 |
| a_only_h160_l2_do030_mse_ccc | 0.30 |
| a_only_h192_l2_do035_mse_ccc_smooth | 0.20 |
| a_only_h96_l1_do015_mse | 0.35 |

Config:

```text
outputs/opensmile_arousal_trials/arousal_ensemble_v1.json
```

## Test Metrics

Canonical `[-1, 1]` scale:

| Metric | Current | Minimum pass | Gap to pass | Good | Gap to good | Excellent | Gap to excellent | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| RMSE | 0.176695 | 0.180000 | -0.003305 | 0.150000 | +0.026695 | 0.100000 | +0.076695 | Pass minimum |
| MAE | 0.135837 | 0.150000 | -0.014163 | 0.120000 | +0.015837 | 0.080000 | +0.055837 | Pass minimum |
| PCC | 0.859718 | 0.800000 | -0.059718 | 0.850000 | -0.009718 | 0.900000 | +0.040282 | Pass good |
| CCC | 0.848452 | 0.800000 | -0.048452 | 0.850000 | +0.001548 | 0.900000 | +0.051548 | Pass minimum, near good |

For RMSE/MAE lower is better. For PCC/CCC higher is better. Negative gap means
the threshold has been exceeded.

## Complete-Song Feature Extraction

The Python `opensmile` package was installed successfully.

PMEmo `dynamic_features.csv` was checked against OpenSMILE low-level
descriptors. The feature structure is:

```text
65 ComParE_2016 LLD mean/std
65 ComParE_2016 delta LLD mean/std
total feature_dim = 260
```

Best reproduction strategy:

```text
feature_set = ComParE_2016
feature_level = LowLevelDescriptors + LowLevelDescriptors_Deltas
window = trailing 1.0 s
hop = 0.5 s
```

On a PMEmo audio sanity check, this reproduced the official feature table with
flattened feature correlation about `0.996`. It is not bit-identical, so this
should be described as a PMEmo-style OpenSMILE reproduction.

## Complete-Song Prediction Scripts

Single song:

```bash
.venv/bin/python research/phase1f_external_va_datasets/pmemo/scripts/predict_full_song_arousal_opensmile_ensemble.py \
  --audio "songs/江南-林俊杰.mp3" \
  --output "research/phase1f_external_va_datasets/pmemo/outputs/opensmile_arousal_trials/fullsong_predictions/jiangnan_arousal_opensmile_ensemble.csv"
```

Batch:

```bash
.venv/bin/python research/phase1f_external_va_datasets/pmemo/scripts/predict_batch_full_song_arousal_opensmile_ensemble.py \
  --audio-dir songs \
  --output-dir research/phase1f_external_va_datasets/pmemo/outputs/opensmile_arousal_trials/fullsong_predictions_batch
```

Verified outputs:

```text
outputs/opensmile_arousal_trials/fullsong_predictions/jiangnan_arousal_opensmile_ensemble.csv
outputs/opensmile_arousal_trials/fullsong_predictions_batch_smoke/
batch smoke test = 2/2 songs succeeded
```

Output columns:

```text
time_sec
arousal_minus1_1
arousal_0_1
model_id
feature_type
```

## Decision

This is the current best Arousal Teacher candidate.

```text
Metric-qualified on PMEmo Arousal = yes
Complete-song inference path = yes
Use for batch Arousal pseudo-label generation = acceptable
Still needs multi-seed or external validation before final paper claim = yes
```

Next step should be Valence. Arousal can be frozen temporarily unless later
multi-seed validation shows instability.
