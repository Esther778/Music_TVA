# Valence Extreme-Song Sampler Negative Trials

Report time: 2026-06-28 Asia/Shanghai

Current research progress: Phase 1 VA Teacher optimization. This experiment
tested whether the split_02 extreme-Valence failure mode can be fixed by
sampling strongly positive/negative training songs more often.

Current best Valence model remains:

```text
model_id = pmemo_valence_frame_calibrator_song_bias_20260628
test RMSE/MAE/PCC/CCC mean = 0.197480/0.153529/0.793235/0.780052
```

## Rationale

The split_02 audit showed that many high-error songs have extreme song-level
Valence means, while model predictions are flat and near-neutral. The sampler
therefore increased the chance that extreme-Valence training songs appear in
each epoch, without changing validation or test data.

The model family was intentionally kept stable:

```text
feature = MERT hidden_6 768 + Librosa harmonic 36
model = Valence-only BiLSTM
evaluation = fixed three song-level splits from valence_multisplit_best_20260627
validation/test leakage = no
```

Script:

```text
scripts/train_pmemo_valence_multisplit_extreme_sampler.py
```

Outputs:

```text
outputs/valence_multisplit_extreme_sampler_20260628/
```

## Results

| Trial | Song sampler | Test RMSE | Test MAE | Test PCC | Test CCC | Decision |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Current best | none, frame calibrator + song bias | 0.197480 | 0.153529 | 0.793235 | 0.780052 | keep |
| `bilstm_extsong_sampler_h80_s05_clip010_ccc005` | gentle, strength 0.5 | 0.210717 | 0.163264 | 0.765355 | 0.750240 | reject |
| `bilstm_extsong_sampler_h80_s2_clip010_ccc005` | stronger, strength 2.0 | 0.219585 | 0.170319 | 0.758261 | 0.746395 | reject |

Delta versus current best:

| Trial | RMSE delta | MAE delta | PCC delta | CCC delta |
| --- | ---: | ---: | ---: | ---: |
| gentle sampler | +0.013237 | +0.009735 | -0.027881 | -0.029812 |
| stronger sampler | +0.022105 | +0.016791 | -0.034974 | -0.033657 |

Positive delta is worse for RMSE/MAE. Negative delta is worse for PCC/CCC.

## Fusion Check

The rejected sampler outputs were also checked for possible complementary
information. We searched the sampler blend weight on the validation split only,
then reported the held-out test split.

Artifacts:

```text
outputs/valence_extreme_sampler_fusion_check_20260628/
```

| Fusion | Test RMSE | Test MAE | Test PCC | Test CCC | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| Current best | 0.197480 | 0.153529 | 0.793235 | 0.780052 | keep |
| Current best + gentle sampler | 0.197322 | 0.153187 | 0.793468 | 0.779815 | do not promote |
| Current best + stronger sampler | 0.198176 | 0.153951 | 0.791930 | 0.779484 | reject |

The gentle sampler fusion slightly improves RMSE, MAE, and PCC, but its CCC is
lower than the current best. Since CCC is one of the active acceptance metrics,
this fusion is recorded as a useful diagnostic signal rather than a promoted
model.

## Gap Of Current Best

The current best model still has the following gaps to the active thresholds:

| Metric | Current | Minimum pass | Gap to pass | Good | Gap to good | Excellent | Gap to excellent |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RMSE | 0.197480 | 0.180000 | +0.017480 | 0.150000 | +0.047480 | 0.100000 | +0.097480 |
| MAE | 0.153529 | 0.150000 | +0.003529 | 0.120000 | +0.033529 | 0.080000 | +0.073529 |
| PCC | 0.793235 | 0.800000 | +0.006765 | 0.850000 | +0.056765 | 0.900000 | +0.106765 |
| CCC | 0.780052 | 0.800000 | +0.019948 | 0.850000 | +0.069948 | 0.900000 | +0.119948 |

## Conclusion

Extreme-song sampling is not the next main path. Both gentle and stronger
samplers degraded all mean test metrics. This suggests that the remaining
Valence problem is not simply caused by insufficient exposure to extreme songs.

The likely bottleneck is one or more of:

```text
1. audio features still do not encode Valence polarity strongly enough;
2. some extreme-Valence labels are hard or inconsistent with audio-only cues;
3. the model can learn local Valence variation but struggles with whole-song
   polarity calibration under cross-song testing.
```

Next useful directions:

```text
1. inspect worst-song labels/audio metadata before more blind training;
2. test robust outlier-aware calibration using only validation-selected rules;
3. consider a lightweight song-level polarity classifier only if it is trained
   and evaluated as a separate deployable module, not as a hidden label leak;
4. preserve pmemo_valence_frame_calibrator_song_bias_20260628 as the current
   best until a new model beats it on all or most metrics.
```
