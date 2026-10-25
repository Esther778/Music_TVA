# Valence Split-02 Audit and Negative Trials

Report time: 2026-06-28 Asia/Shanghai

Current research progress: Phase 1 VA Teacher optimization. Current Valence
candidate remains:

```text
model_id = pmemo_valence_frame_calibrator_song_bias_20260628
test RMSE/MAE/PCC/CCC mean = 0.197480/0.153529/0.793235/0.780052
```

## Split-02 Difficulty Audit

Artifacts:

```text
outputs/valence_split02_difficulty_audit_20260628/song_level_error_audit.csv
outputs/valence_split02_difficulty_audit_20260628/split02_test_worst15.csv
outputs/valence_split02_difficulty_audit_20260628/split02_error_correlations.csv
outputs/valence_split02_difficulty_audit_20260628/summary.json
```

The worst split_02 test songs are dominated by extreme song-level Valence labels
whose predictions are pulled toward neutral:

| musicId | Frames | Target mean | Pred mean | Bias | MAE | RMSE | Target std | Pred std |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 419 | 30 | -0.550722 | 0.137153 | 0.687875 | 0.687875 | 0.688701 | 0.033913 | 0.001288 |
| 230 | 26 | 0.510122 | -0.026844 | -0.536966 | 0.536966 | 0.537659 | 0.027729 | 0.009815 |
| 139 | 34 | 0.577296 | 0.176464 | -0.400833 | 0.400833 | 0.401297 | 0.016154 | 0.004313 |
| 824 | 40 | 0.689524 | 0.297085 | -0.392439 | 0.392439 | 0.393913 | 0.037701 | 0.006137 |
| 149 | 100 | -0.380326 | 0.009234 | 0.389560 | 0.389560 | 0.391637 | 0.043714 | 0.009378 |

Extreme vs non-extreme split_02 test songs:

| Group | Count | Mean song RMSE | Mean song MAE |
| --- | ---: | ---: | ---: |
| `abs(target_mean) >= 0.4` | 30 | 0.198678 | 0.195366 |
| `abs(target_mean) < 0.4` | 47 | 0.159661 | 0.152041 |

Interpretation:

```text
The bottleneck is song-level Valence polarity. The model often predicts a very
flat near-neutral curve for songs whose labels are strongly positive or
strongly negative.
```

## Negative Trial 1: Auxiliary Polarity Head

Detailed report:

```text
docs/VALENCE_POLARITY_HEAD_NEGATIVE_20260628.md
```

Result:

| Model | Test RMSE | Test MAE | Test PCC | Test CCC |
| --- | ---: | ---: | ---: | ---: |
| Current best | 0.197480 | 0.153529 | 0.793235 | 0.780052 |
| Polarity-head BiLSTM | 0.216594 | 0.169938 | 0.760860 | 0.753016 |

Fusion with current best:

```text
test RMSE/MAE/PCC/CCC = 0.200445/0.154616/0.785439/0.763568
```

Decision:

```text
Do not promote. The auxiliary polarity head is worse standalone and hurts
ensemble CCC.
```

## Negative Trial 2: Audio-Feature Song Bias

An audio-driven residual correction was tested after the current best. It used
whole-song MERT and harmonic Librosa statistics to predict per-song residual
bias.

```text
script = scripts/search_pmemo_valence_audio_song_bias.py
outputs = outputs/valence_audio_song_bias_20260628/
```

Result:

| Model | Test RMSE | Test MAE | Test PCC | Test CCC |
| --- | ---: | ---: | ---: | ---: |
| Current best | 0.197480 | 0.153529 | 0.793235 | 0.780052 |
| Audio song-bias correction | 0.198976 | 0.155140 | 0.789913 | 0.773182 |

Decision:

```text
Do not promote. It overfits song-level residuals and worsens all four mean test
metrics.
```

## Current Best Gap

Positive gap means still short of the threshold.

| Metric | Current | Minimum pass | Gap to pass | Good | Gap to good | Excellent | Gap to excellent |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RMSE | 0.197480 | 0.180000 | +0.017480 | 0.150000 | +0.047480 | 0.100000 | +0.097480 |
| MAE | 0.153529 | 0.150000 | +0.003529 | 0.120000 | +0.033529 | 0.080000 | +0.073529 |
| PCC | 0.793235 | 0.800000 | +0.006765 | 0.850000 | +0.056765 | 0.900000 | +0.106765 |
| CCC | 0.780052 | 0.800000 | +0.019948 | 0.850000 | +0.069948 | 0.900000 | +0.119948 |

## Next Direction

Do not keep adding simple song-level heads or high-dimensional audio residual
Ridge corrections. Better next options:

```text
1. inspect whether worst songs are label outliers or genre/style outliers;
2. train with explicit extreme-song sampling, but evaluate without changing test;
3. train separate Valence calibrators for low-confidence flat-prediction songs;
4. consider collecting/adding more Valence-extreme training examples if available.
```
