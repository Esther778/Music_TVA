# External VA Improvement Advice Audit

Last updated: 2026-06-21

This document maps the externally supplied recommendations to the current
project state. It prevents repeating completed work and identifies the
remaining high-value experiments.

## Confirmed Evaluation Facts

### Label scale

DEAM averaged dynamic Valence and Arousal annotations use the theoretical
range:

```text
[-1, +1]
```

No 0-1 normalization is applied before metric calculation.

The current observed values are:

| Split | Valence range | Arousal range |
| --- | --- | --- |
| train | -0.832 to 0.716 | -0.772 to 0.898 |
| validation | -0.768 to 0.632 | -0.713 to 0.785 |
| test | -0.566 to 0.580 | -0.531 to 0.692 |

Dynamic labels start at `sample_15000ms`. The first 15 seconds are excluded
because beginning-of-excerpt annotations are considered unstable.

The source tables contain 1223 Valence columns and 1224 Arousal columns. The
current paired VA preparation uses their common length.

### Metrics

The project already reports:

```text
overall RMSE and MAE
Valence/Arousal RMSE
Valence/Arousal MAE
Valence/Arousal Pearson r
Valence/Arousal CCC
mean CCC
```

### Mean baseline

On the current song-level test set:

```text
training-mean baseline RMSE=0.249551
OpenL3+MLP aligned single-seed RMSE=0.208569
MERT+BiLSTM current RMSE=0.202452
```

Both learned models beat the mean baseline, but the margin is moderate.

### Split protocol

The current scientific experiments use the same song-level split:

```text
train=1441 songs
validation=180 songs
test=181 songs
song overlap=0
```

The earlier frame-level split remains only as a published-protocol
reproduction and must not be used for unseen-song claims.

## Time Alignment Audit

MERT alignment is explicit:

```text
time_sec=15.0, 15.5, 16.0, ...
labels=sample_15000ms, sample_15500ms, sample_16000ms, ...
```

An OpenL3 issue was found:

- old code used `features[-T:]`,
- a typical 45-second song therefore started at 15.5 seconds,
- labels started at 15.0 seconds,
- the feature sequence was shifted 0.5 seconds later than the labels.

All 1802 songs support exact OpenL3 alignment from timestamp index 30,
corresponding to 15.0 seconds.

Single-seed correction result:

| Metric | Old tail alignment | Exact 15 s alignment | Change |
| --- | ---: | ---: | ---: |
| validation RMSE | 0.222834 | 0.220933 | -0.001902 |
| test RMSE | 0.210084 | 0.208569 | -0.001515 |
| test Valence RMSE | 0.221901 | 0.220053 | -0.001848 |
| test Arousal RMSE | 0.197562 | 0.196414 | -0.001147 |
| test Pearson V | 0.414900 | 0.422190 | +0.007291 |
| test Pearson A | 0.654082 | 0.655921 | +0.001839 |
| test mean CCC | 0.500731 | 0.501581 | +0.000850 |

The correction is valid but small. Additional OpenL3 seeds are paused by user
decision until the single-run value is assessed.

## Advice Already Completed

- MERT features have already replaced OpenL3 as the main modern backbone
  experiment.
- MERT+MLP, MERT+BiLSTM, and MERT+Transformer have already been trained.
- Separate VA metrics, CCC, MAE, and Pearson have already been implemented.
- Song-level split and leakage audit are complete.
- MSE weighting and temporal smoothness trials have already been attempted in
  limited form.

Therefore, repeating these items unchanged would not be useful.

## Advice To Execute Next

1. Three-seed MERT+BiLSTM baseline verification.
2. Validation-only annotation lag search.
3. Multi-scale temporal context features.
4. Song-balanced loss.
5. MERT intermediate-layer or learned layer-mixture experiment.
6. Combined MSE, CCC, and first-difference trajectory loss.
7. Traditional music-feature fusion, starting with RMS/onset/chroma/key-mode.
8. Partial MERT fine-tuning only after frozen-feature experiments plateau.

Jukebox is not a current priority because it is substantially heavier and less
practical than improving the existing MERT pipeline.

## Current Decision

Do not continue widening OpenL3 MLP heads. The exact time correction reduced
RMSE by less than 1%, which confirms alignment mattered but was not the main
bottleneck.

The next highest-information experiment is MERT annotation-lag calibration,
after establishing the three-seed MERT+BiLSTM baseline.

## Implemented Follow-Up

The requested pure-audio comparison was implemented in:

```text
research/phase1c_spectral_feature_fusion_va
```

Completed:

- log-Mel multi-scale CNN + channel/spatial attention + BiLSTM,
- Librosa 118-dimensional handcrafted features + BiLSTM,
- log-Mel and handcrafted fusion,
- unified song-level split,
- separate Valence/Arousal RMSE, MAE, PCC, and CCC.

Main finding: MERT remains strongest by validation RMSE and Valence metrics.
Handcrafted/fusion features are strongest for Arousal. The next fusion should
combine MERT with handcrafted features.
