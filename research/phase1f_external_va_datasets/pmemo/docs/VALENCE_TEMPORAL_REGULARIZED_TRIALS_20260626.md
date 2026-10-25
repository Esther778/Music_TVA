# Valence Temporal-Regularized Trials - 2026-06-26

Report time: 2026-06-26

Current research progress: Phase 1F is still optimizing a Valence-specific
teacher model for arbitrary-song VA prediction. Arousal already has a
threshold-passing candidate, while Valence remains the main bottleneck.

## Purpose

This round continued from the current best Valence model:

```text
MERT hidden_6 + Librosa harmonic + Valence-only BiLSTM
```

The experiment tested whether the current plateau could be improved by:

- adding weak temporal smoothness regularization
- matching predicted frame-to-frame delta with target delta
- adding clip-level standard-deviation matching
- increasing regularization by reducing hidden size or increasing dropout and weight decay

## Thresholds Used For Reporting

| Metric | Minimum pass | Good | Excellent | Direction |
| --- | ---: | ---: | ---: | --- |
| RMSE | 0.180000 | 0.150000 | 0.100000 | lower is better |
| MAE | 0.150000 | 0.120000 | 0.080000 | lower is better |
| PCC | 0.800000 | 0.850000 | 0.900000 | higher is better |
| CCC | 0.800000 | 0.850000 | 0.900000 | higher is better |

## Historical Best Before This Round

| Item | Value |
| --- | --- |
| model_id | `pmemo_valence_mert_hidden6_harmonic_extreme_weighted_v1` |
| trial_id | `h6_harm_h80_ext125_clip010_ccc005` |
| feature | MERT hidden_6 768 + Librosa harmonic 36 |
| model | Valence-only BiLSTM, projection 320, hidden_dim 80, dropout 0.30 |
| loss | extreme-weighted frame MSE + 0.05 frame CCC + 0.10 clip mean MSE |
| extreme_weight | 1.25 |

Historical best test metrics:

| Metric | Historical best | Minimum pass | Gap to pass | Good | Gap to good | Excellent | Gap to excellent |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RMSE | 0.205673 | 0.180000 | +0.025673 | 0.150000 | +0.055673 | 0.100000 | +0.105673 |
| MAE | 0.164567 | 0.150000 | +0.014567 | 0.120000 | +0.044567 | 0.080000 | +0.084567 |
| PCC | 0.783400 | 0.800000 | +0.016600 | 0.850000 | +0.066600 | 0.900000 | +0.116600 |
| CCC | 0.767304 | 0.800000 | +0.032696 | 0.850000 | +0.082696 | 0.900000 | +0.132696 |

## Current Experiment Results

| Trial | Main change | Best epoch | Val RMSE | Val PCC | Val CCC | Test RMSE | Test MAE | Test PCC | Test CCC | Decision |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `h6_harm_h80_ext125_clip010_ccc005_smooth002` | weak prediction smoothness | 19 | 0.213992 | 0.822451 | 0.812614 | 0.205690 | 0.164579 | 0.783381 | 0.767284 | not promoted |
| `h6_harm_h80_ext125_clip010_std003_delta020` | std match + delta match | 19 | 0.213812 | 0.822795 | 0.812932 | 0.205730 | 0.164539 | 0.783150 | 0.767066 | not promoted |
| `h6_harm_h80_ext125_clip010_ccc005_delta020` | delta match | 19 | 0.213900 | 0.822627 | 0.812817 | 0.205768 | 0.164634 | 0.783257 | 0.767163 | not promoted |
| `h6_harm_h64_ext125_clip010_ccc005_do035_wd0005` | smaller hidden + stronger regularization | 8 | 0.224703 | 0.811121 | 0.794849 | 0.208508 | 0.169193 | 0.767218 | 0.760375 | not promoted |
| `h6_harm_h80_ext125_clip010_ccc005_do040_wd0005` | higher dropout + stronger weight decay | 19 | 0.217321 | 0.817569 | 0.810913 | 0.219912 | 0.174870 | 0.768314 | 0.748349 | not promoted |

Best current-experiment test RMSE:

```text
trial = h6_harm_h80_ext125_clip010_ccc005_smooth002
test RMSE/MAE/PCC/CCC = 0.205690/0.164579/0.783381/0.767284
```

Comparison against historical best:

| Metric | Current experiment best | Historical best | Delta vs historical best | Minimum pass | Current gap to pass | Good | Current gap to good | Excellent | Current gap to excellent |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RMSE | 0.205690 | 0.205673 | +0.000017 | 0.180000 | +0.025690 | 0.150000 | +0.055690 | 0.100000 | +0.105690 |
| MAE | 0.164579 | 0.164567 | +0.000012 | 0.150000 | +0.014579 | 0.120000 | +0.044579 | 0.080000 | +0.084579 |
| PCC | 0.783381 | 0.783400 | -0.000019 | 0.800000 | +0.016619 | 0.850000 | +0.066619 | 0.900000 | +0.116619 |
| CCC | 0.767284 | 0.767304 | -0.000020 | 0.800000 | +0.032716 | 0.850000 | +0.082716 | 0.900000 | +0.132716 |

Decision:

```text
Do not promote any temporal-regularized or stronger-regularized trial.
The historical best remains h6_harm_h80_ext125_clip010_ccc005.
```

## Interpretation

The first three trials are numerically almost identical to the historical best.
This means weak temporal smoothness, delta matching, and clip standard-deviation
matching do not materially change the learned solution.

The two stronger regularization trials are worse, so the current bottleneck is
not simply overfitting from too-large hidden size or too-low dropout. The model
appears to have reached a local plateau for the current input family:

```text
MERT hidden_6 + harmonic features + single BiLSTM head
```

## Next Recommended Experiment

The next useful direction should change the information pathway, not just add
small loss weights:

```text
1. Build a Valence-specific two-branch model:
   local frame branch = MERT hidden_6 + harmonic BiLSTM
   song-context branch = pooled song-level MERT/harmonic statistics
   fusion = context-conditioned frame prediction

2. Keep the same song-level split and validation-based selection.

3. Compare every trial against the historical best and threshold table.
```

Reason:

```text
Valence errors are dominated by song-level affective bias and several shared
hard songs. A context-conditioned model is more directly aligned with that
failure mode than further scalar loss-weight tuning.
```

Artifacts:

```text
script = scripts/train_pmemo_mert_hidden6_harmonic_temporal_regularized.py
outputs = outputs/mert_hidden6_harmonic_temporal_regularized/
comparison = outputs/mert_hidden6_harmonic_temporal_regularized/comparison.csv
```
