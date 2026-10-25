# Phase 1C Run Log

## 2026-06-21

### Project Created

Purpose:

- add a pure log-Mel spectral baseline,
- add a Librosa handcrafted music-feature baseline,
- add a spectral/handcrafted fusion baseline,
- compare all methods on the existing DEAM song-level split.

Current OpenL3 and MERT pipelines are pure audio:

```text
OpenL3 input=audio embedding only
MERT input=audio embedding only
lyrics/metadata/history VA input=none
```

Research boundaries:

- DAMFF values supplied by the user are retained as:
  DEAM2015 Arousal RMSE `0.2401`, Valence RMSE `0.3403`;
  MER1101 Arousal RMSE `0.1494`, Valence RMSE `0.2595`.
- Do not attribute `0.112/0.109` to DAMFF on DEAM.
- ENMI values `0.0440/0.0352` are not used as a pure-audio target because the
  rolling forecast may include historical music-emotion features.
- Exact DAMFF/ENMI source papers were not independently located in the
  accessible search results during this run; do not invent bibliographic
  details. Add the original PDFs when available.

Implementation strategy:

```text
log-Mel: continuous spectrogram from 15 s onward
handcrafted: MFCC, spectral, rhythmic, chroma and tonal descriptors
spectral model: multi-scale CNN + CBAM + BiLSTM
fusion model: spectral temporal features + handcrafted projection + BiLSTM
```

### Feature Extraction Completed

```text
songs=1802/1802
train=1441
validation=180
test=181
log-Mel bins=64
handcrafted dimension=118
target frames min/max=60/1223
invalid files=0
generated size=334 MB
```

The final handcrafted vector includes pooled mean/std descriptors from:

```text
MFCC
spectral centroid/bandwidth/flatness/contrast
zero-crossing rate
chroma
tonnetz
RMS
onset strength
selected tempogram bands
```

### Single-Seed Baselines Completed

All new models used seed `2026`.

| Model | Best epoch | Val RMSE | Test RMSE |
| --- | ---: | ---: | ---: |
| Handcrafted + BiLSTM | 44 | 0.246835 | 0.204089 |
| Log-Mel DAMFF-inspired | 25 | 0.235476 | 0.213163 |
| Log-Mel + handcrafted fusion | 12 | 0.239780 | 0.201222 |

Full per-dimension RMSE, MAE, PCC, and CCC:

```text
docs/ABLATION_RESULTS.md
```

Decision:

- MERT+BiLSTM remains the selected scientific candidate because it has the
  lowest validation RMSE.
- Do not select the fusion model from its lower test RMSE.
- Handcrafted features are promising specifically for Arousal.
- Next experiment should fuse MERT with handcrafted features and retain
  separate Valence/Arousal output heads.
