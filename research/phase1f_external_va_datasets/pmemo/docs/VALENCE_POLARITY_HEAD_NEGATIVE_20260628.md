# Valence Polarity Head Negative Trial

Report time: 2026-06-28 Asia/Shanghai

Current research progress: Phase 1 VA Teacher optimization. The current Valence
candidate remains:

```text
model_id = pmemo_valence_frame_calibrator_song_bias_20260628
test RMSE/MAE/PCC/CCC mean = 0.197480/0.153529/0.793235/0.780052
```

## Purpose

The previous diagnostics showed that high-error songs often have extreme
song-level Valence targets but predictions are pulled toward neutral. This trial
tested whether an auxiliary song-level polarity head inside the BiLSTM training
could reduce that problem.

## Model

```text
script = scripts/train_pmemo_valence_multisplit_polarity_head.py
trial_id = bilstm_polarity_h80_do035_song020_clip015
feature = MERT hidden_6 + Librosa harmonic
frame model = BiLSTM
auxiliary head = song-level pooled hidden state -> song Valence mean
loss = frame weighted MSE + frame CCC + clip mean/std + song-head MSE + frame/song consistency
```

## Standalone Result

| Model | Test RMSE | Test MAE | Test PCC | Test CCC |
| --- | ---: | ---: | ---: | ---: |
| Current best | 0.197480 | 0.153529 | 0.793235 | 0.780052 |
| Polarity-head BiLSTM | 0.216594 | 0.169938 | 0.760860 | 0.753016 |

Per split:

| Split | Test RMSE | Test MAE | Test PCC | Test CCC |
| --- | ---: | ---: | ---: | ---: |
| split_01_seed_101 | 0.201640 | 0.159796 | 0.808363 | 0.796421 |
| split_02_seed_202 | 0.221276 | 0.176174 | 0.752504 | 0.745146 |
| split_03_seed_303 | 0.226866 | 0.173844 | 0.721712 | 0.717482 |

## Fusion Check

The polarity-head outputs were also tested as weak ensemble members with the
current best. Validation-selected fusion produced:

```text
test RMSE/MAE/PCC/CCC mean = 0.200445/0.154616/0.785439/0.763568
```

This is worse than the current best, especially CCC.

## Decision

Do not promote.

```text
The simple auxiliary song-level polarity head does not solve the Valence
polarity problem. It improves neither standalone performance nor ensemble
performance against the current best.
```

## Interpretation

The failure suggests that the issue is not simply lack of a song-level loss.
The model may need one of the following instead:

```text
1. better feature representation for Valence polarity;
2. stronger or more reliable song-level supervision;
3. explicit extreme-song sampling/weighting;
4. split_02 difficult-song audit to identify label or feature mismatch.
```

For future comparisons, keep the current baseline:

```text
0.197480/0.153529/0.793235/0.780052
```
