# Shadow Validation Trend-Residual Experiment

## Protocol

- Original train songs: 1,441.
- Train-sub: 1,261 songs.
- Shadow validation: 180 songs.
- Split seed: 2026.
- Stratification: song-level Valence and Arousal quantiles.
- Existing validation frozen during model and post-processing selection.
- Test not used.

## Models

Both candidates used:

```text
MERT layer 6 + MERT final layer + 118-d handcrafted features
-> feature-wise gate
-> BiLSTM
```

The trend-residual model added:

- a 5-second trend head;
- a 10-second trend head;
- a local residual head;
- frame CCC and Pearson losses;
- 5-second and 10-second window CCC losses;
- low-weight ranking and variance losses.

## Shadow Results

| Model | RMSE | MAE | PCC | CCC |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 0.185229 | 0.144518 | 0.671930 | 0.663702 |
| Trend-residual | 0.185096 | 0.142694 | 0.671715 | 0.663185 |
| Fixed 50/50 ensemble + smoothing | 0.180815 | 0.139727 | 0.683143 | 0.671636 |

The model-level gain did not meet the predefined 0.003 RMSE threshold.

## Frozen Validation Confirmation

After model and post-processing selection, both models were refit on all
1,441 training songs using shadow-selected fixed epoch counts:

- baseline: 21 epochs;
- trend-residual: 36 epochs.

The fixed 50/50 ensemble used zero shift, Gaussian sigma 8 frames, and no
calibration.

| Metric | Frozen validation |
| --- | ---: |
| Valence RMSE | 0.224760 |
| Valence MAE | 0.169021 |
| Valence PCC | 0.669015 |
| Valence CCC | 0.606852 |

The candidate is rejected. The existing Valence ensemble remains selected:

```text
RMSE=0.202493
MAE=0.153771
PCC=0.724139
CCC=0.692638
```

## Main Finding

Frame-weighted Valence distributions differ materially:

```text
full train mean/std=0.0845/0.2472
shadow validation mean/std=0.0980/0.2426
frozen validation mean/std=0.0252/0.2901
```

The shadow split accurately represents the training distribution but not
the frozen validation distribution. Continued single-split tuning is
unlikely to provide reliable gains. The next defensible model-development
protocol is song-level cross-validation or OOF training across the original
training set.
