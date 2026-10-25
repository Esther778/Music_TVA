# PMEmo Valence Song-Balanced Loss Trials

Report time: 2026-06-26 Asia/Shanghai

Current research progress: Valence remains below the current minimum Teacher
threshold. The active best before this round is:

```text
model_id = pmemo_valence_mert_hidden6_harmonic_extreme_weighted_v1
trial_id = h6_harm_h80_ext125_clip010_ccc005
test RMSE/MAE/PCC/CCC = 0.205673/0.164567/0.783400/0.767304
```

## Purpose

The previous global-head experiment showed that song-level Valence is partly
learnable, but direct auxiliary sharing harmed frame-level prediction. This
round tests a simpler idea: make the frame-level loss song-balanced.

Standard frame MSE averages all valid frames together, so songs with more
frames receive more training weight. Song-balanced loss first averages frame
loss within each song and then averages across songs in the batch.

## Script

```text
research/phase1f_external_va_datasets/pmemo/scripts/train_pmemo_mert_hidden6_harmonic_song_balanced.py
```

## Results

| Trial | Best Epoch | Test RMSE | Test MAE | Test PCC | Test CCC | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| h6_harm_h80_songbal_ext100_clip010 | 10 | 0.213541 | 0.168214 | 0.766066 | 0.752800 | weaker |
| h6_harm_h80_songbal_ext125_clip010 | 10 | 0.213521 | 0.168113 | 0.766276 | 0.752934 | weaker |
| h6_harm_h80_songbal_ext150_clip010 | 10 | 0.213518 | 0.168045 | 0.766556 | 0.753133 | weaker |
| h6_harm_h80_songbal_ext125_clip008 | 10 | 0.213400 | 0.168022 | 0.766357 | 0.753054 | weaker |
| h6_harm_h80_songbal_ext125_clip010_ccc008 | 32 | 0.210740 | 0.170878 | 0.779175 | 0.767904 | test CCC slightly higher, but RMSE/MAE/PCC weaker |

## Important Observation

The last trial achieved strong validation correlation:

```text
trial_id = h6_harm_h80_songbal_ext125_clip010_ccc008
best validation epoch = 32
validation RMSE/PCC/CCC = 0.21140/0.831/0.827
test RMSE/MAE/PCC/CCC = 0.210740/0.170878/0.779175/0.767904
```

This did not generalize to the test split. It confirms that validation-only
improvement is not enough for Teacher promotion.

## Decision

Do not promote any song-balanced trial. The current best remains:

```text
model_id = pmemo_valence_mert_hidden6_harmonic_extreme_weighted_v1
trial_id = h6_harm_h80_ext125_clip010_ccc005
test RMSE/MAE/PCC/CCC = 0.205673/0.164567/0.783400/0.767304
```

## Interpretation

Song-balanced loss does not solve the Valence bottleneck. The biggest remaining
issue is likely not only loss weighting; it is a combination of:

```text
1. Valence distribution shift across train/validation/test songs.
2. Limited feature separability for whole-song Valence direction.
3. Validation-selected improvements failing to generalize to test.
```

## Next Step

The next useful step should be diagnostic rather than another blind loss
variant:

1. Compare train/validation/test song-level Valence mean distributions.
2. Identify test songs where all current models fail.
3. Consider split-robust evaluation or cross-validation before further model
   architecture changes.
