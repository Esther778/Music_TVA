# Valence Multi-Split CCC-Preserving Ensemble Report

Report time: 2026-06-27 11:05 Asia/Shanghai

Current research progress: Phase 1 VA Teacher optimization. Arousal has a
usable PMEmo OpenSMILE ensemble candidate. Valence is still below the minimum
teacher threshold, so this round focused on improving Valence under the same
song-level multi-split protocol.

## Purpose

The previous BiLSTM + TCN ensemble improved RMSE, MAE, and PCC over the
multi-split BiLSTM baseline, but slightly reduced CCC. This trial reused the
already trained BiLSTM and TCN predictions and changed only the validation
selection rule:

- stronger penalty for validation PCC/CCC gap;
- finer fusion weight search, 0.00 to 1.00 in steps of 0.01;
- validation-selected calibration: identity, bias, linear, mean/std;
- song-level moving-average smoothing windows: 1, 3, 5, 9, 15.

No new neural network was trained in this step.

## Current Model

```text
model_id = pmemo_valence_multisplit_ensemble_ccc_preserving_20260627
base_models =
  1. MERT hidden_6 + Librosa harmonic Valence-only BiLSTM
  2. MERT hidden_6 + Librosa harmonic lightweight TCN
selection = validation CCC/PCC-weighted pass-gap objective
postprocess = split-specific fusion weight + calibration + song-level smoothing
```

## Main Result

| Model | Test RMSE mean | Test MAE mean | Test PCC mean | Test CCC mean |
| --- | ---: | ---: | ---: | ---: |
| BiLSTM multi-split baseline | 0.213700 | 0.166477 | 0.763825 | 0.754325 |
| TCN multi-split baseline | 0.210383 | 0.165812 | 0.762037 | 0.728580 |
| Previous BiLSTM+TCN ensemble | 0.205385 | 0.158285 | 0.771277 | 0.752518 |
| Current CCC-preserving ensemble | 0.203066 | 0.156576 | 0.777286 | 0.760357 |

Compared with the previous ensemble:

| Metric | Previous ensemble | Current ensemble | Delta |
| --- | ---: | ---: | ---: |
| RMSE | 0.205385 | 0.203066 | -0.002320 |
| MAE | 0.158285 | 0.156576 | -0.001710 |
| PCC | 0.771277 | 0.777286 | +0.006010 |
| CCC | 0.752518 | 0.760357 | +0.007840 |

Compared with the BiLSTM multi-split baseline:

| Metric | BiLSTM baseline | Current ensemble | Delta |
| --- | ---: | ---: | ---: |
| RMSE | 0.213700 | 0.203066 | -0.010634 |
| MAE | 0.166477 | 0.156576 | -0.009902 |
| PCC | 0.763825 | 0.777286 | +0.013462 |
| CCC | 0.754325 | 0.760357 | +0.006032 |

## Per-Split Result

| Split | Selected fusion | Calibration | Smooth window | Test RMSE | Test MAE | Test PCC | Test CCC |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| split_01_seed_101 | 0.46 BiLSTM + 0.54 TCN | bias | 15 | 0.190091 | 0.149989 | 0.828259 | 0.809312 |
| split_02_seed_202 | 0.50 BiLSTM + 0.50 TCN | bias | 15 | 0.218571 | 0.166973 | 0.731995 | 0.714218 |
| split_03_seed_303 | 0.53 BiLSTM + 0.47 TCN | linear | 15 | 0.200535 | 0.152765 | 0.771605 | 0.757542 |

## Gap To Thresholds

Positive gap means the model is still short of that threshold.

| Metric | Current | Minimum pass | Gap to pass | Good | Gap to good | Excellent | Gap to excellent |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RMSE | 0.203066 | 0.180000 | +0.023066 | 0.150000 | +0.053066 | 0.100000 | +0.103066 |
| MAE | 0.156576 | 0.150000 | +0.006576 | 0.120000 | +0.036576 | 0.080000 | +0.076576 |
| PCC | 0.777286 | 0.800000 | +0.022714 | 0.850000 | +0.072714 | 0.900000 | +0.122714 |
| CCC | 0.760357 | 0.800000 | +0.039643 | 0.850000 | +0.089643 | 0.900000 | +0.139643 |

## Historical Context

Best single-split Valence reference:

```text
pmemo_valence_mert_hidden6_harmonic_extreme_weighted_v1
test RMSE/MAE/PCC/CCC = 0.205673/0.164567/0.783400/0.767304
```

The current CCC-preserving ensemble has better multi-split RMSE and MAE than
that single-split reference, but its mean PCC and CCC are still lower. Therefore
it should be treated as the current best multi-split candidate, not as a final
threshold-passing Valence teacher.

## Decision

Promote this run as the current best multi-split Valence candidate:

```text
pmemo_valence_multisplit_ensemble_ccc_preserving_20260627
test RMSE/MAE/PCC/CCC mean = 0.203066/0.156576/0.777286/0.760357
```

It is still not acceptable as the final Valence teacher under the user's minimum
target because all four metrics remain slightly below threshold. The most urgent
gaps are RMSE and CCC.

## Next Step

The next experiment should target the split_02 weakness, because split_02 is
dragging down all average metrics. Recommended next low-cost direction:

1. train a Valence-specific model with stronger regularization and less
   overfitting on the same three splits;
2. keep the current CCC-preserving ensemble as the comparison baseline;
3. select checkpoints by validation CCC/PCC plus RMSE, not by RMSE alone;
4. report current, historical best, and threshold gaps every time.

Artifacts:

```text
script = scripts/search_pmemo_valence_multisplit_ensemble_ccc_preserving.py
outputs = outputs/valence_multisplit_ensemble_ccc_preserving_20260627/
summary = outputs/valence_multisplit_ensemble_ccc_preserving_20260627/summary.json
all candidates = outputs/valence_multisplit_ensemble_ccc_preserving_20260627/all_ensemble_results.csv
```
