# Valence Song-Mean Fusion Negative Result

Report time: 2026-06-28

Current research progress: Phase 1 VA Teacher optimization. Arousal has a
usable OpenSMILE ensemble candidate for arbitrary-song prediction. Valence is
still below the minimum target and remains the current bottleneck.

## Purpose

The previous split_02 audit showed that remaining Valence errors are strongly
related to song-level mean bias: many difficult songs have a clearly positive
or negative target mean, while the prediction remains too neutral.

This experiment tested whether a lightweight song-level Valence mean module can
correct that bias on top of the current best frame-level Valence prediction.

## Input Model

```text
model_id = pmemo_valence_prediction_only_rescale_ccc_preserving_20260628
test RMSE/MAE/PCC/CCC = 0.195316/0.151793/0.797766/0.784834
```

Current gap to minimum pass:

| Metric | Current | Minimum pass | Gap |
| --- | ---: | ---: | ---: |
| RMSE | 0.195316 | 0.180000 | +0.015316 |
| MAE | 0.151793 | 0.150000 | +0.001793 |
| PCC | 0.797766 | 0.800000 | +0.002234 |
| CCC | 0.784834 | 0.800000 | +0.015166 |

## Method

New script:

```text
scripts/search_pmemo_valence_song_mean_fusion.py
```

Outputs:

```text
outputs/valence_song_mean_fusion_20260628/
```

The script keeps the current frame-level curve and trains a song-level regressor
to predict each song's Valence mean. The final frame prediction is:

```text
corrected_valence = current_valence + strength * clip(predicted_song_mean - current_song_mean)
```

Feature sets tested:

```text
prediction summary only
MERT hidden_6 song statistics + Librosa harmonic statistics
MERT hidden_6 song statistics + all Librosa statistics
```

Regressors tested:

```text
Ridge
Huber
ElasticNet
PCA + Ridge
PCA + HistGradientBoosting
PCA + RandomForest
```

Selection was validation-only. The no-change baseline was included as a
candidate so the search could reject the correction.

## Result 1: Unconstrained Per-Split Selection

```text
baseline test RMSE/MAE/PCC/CCC = 0.195316/0.151793/0.797766/0.784834
selected test RMSE/MAE/PCC/CCC = 0.194170/0.152897/0.798307/0.773211
```

Delta:

```text
RMSE = -0.001147
MAE  = +0.001104
PCC  = +0.000542
CCC  = -0.011623
```

Decision: reject. RMSE improves slightly, but MAE and CCC get worse.

## Result 2: CCC-Preserving Reanalysis

After adding a validation-side CCC-preserving constraint:

```text
baseline test RMSE/MAE/PCC/CCC = 0.195316/0.151793/0.797766/0.784834
selected test RMSE/MAE/PCC/CCC = 0.193478/0.151876/0.800305/0.779717
```

Delta:

```text
RMSE = -0.001838
MAE  = +0.000083
PCC  = +0.002539
CCC  = -0.005117
```

Decision: reject as new best. PCC crosses 0.80, but CCC still falls and MAE
does not improve.

Gap to minimum pass for this constrained result:

| Metric | Current | Minimum pass | Gap | Status |
| --- | ---: | ---: | ---: | --- |
| RMSE | 0.193478 | 0.180000 | +0.013478 | Not pass |
| MAE | 0.151876 | 0.150000 | +0.001876 | Not pass |
| PCC | 0.800305 | 0.800000 | -0.000305 | Pass |
| CCC | 0.779717 | 0.800000 | +0.020283 | Not pass |

## Result 3: Single Global Configuration

Using one global configuration for all three splits, selected only by average
validation performance:

```text
selected config = mert_all_librosa + hgb_pca32 + strength 0.9 + cap010
baseline test RMSE/MAE/PCC/CCC = 0.195316/0.151793/0.797766/0.784834
selected test RMSE/MAE/PCC/CCC = 0.195064/0.151832/0.797500/0.782109
```

Decision: reject. The global configuration is more deployment-like, but the
gain is too small and CCC/PCC decline.

## Conclusion

Song-level Valence mean correction confirms the diagnosis that global Valence
polarity matters, but it is not sufficient as a standalone fix. It can slightly
reduce RMSE or push PCC over 0.80 in one constrained analysis, but it hurts CCC,
which is central for continuous VA curve agreement.

Current Valence best remains:

```text
pmemo_valence_prediction_only_rescale_ccc_preserving_20260628
test RMSE/MAE/PCC/CCC = 0.195316/0.151793/0.797766/0.784834
```

## Next Technical Direction

Do not keep tuning song-mean postprocess alone. The next useful experiments are:

1. Build a deployable Valence prediction script from MP3 to MERT hidden_6 +
   Librosa harmonic features, so the current best can actually run on arbitrary
   full songs.
2. For metric improvement, train a Valence model with a stronger global
   emotion branch inside the network rather than after prediction.
3. If using Arousal as auxiliary information, use only out-of-fold or separately
   deployable predicted Arousal, not true Arousal labels from the test split.
4. Consider a two-head system for final use: keep the Arousal-only model for
   Arousal and continue Valence-specific modeling separately.
