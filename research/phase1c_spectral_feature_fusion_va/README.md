# Phase 1C: Spectral And Music-Feature VA Modeling

Status: first single-seed ablation completed.

## Goal

Build pure-audio, song-level VA baselines inspired by two research directions:

1. DAMFF-style spectral modeling:
   log-Mel spectrogram, multi-scale CNN, channel/spatial attention, and BiLSTM.
2. ENMI-style feature fusion:
   Librosa music descriptors plus deep log-Mel features and a temporal model.

This project does not use previous VA labels as input. DEAM Valence/Arousal is
used only as the supervised target.

## Unified Comparison

All experiments must use:

```text
DEAM dynamic VA range=[-1,+1]
label start=15.0 seconds
frame hop=0.5 seconds
song-level split=1441/180/181
selection metric=validation RMSE
```

Planned comparison:

```text
OpenL3 + MLP
MERT + MLP/BiLSTM
log-Mel multi-scale CNN + CBAM + BiLSTM
Librosa handcrafted features + BiLSTM
log-Mel + handcrafted feature fusion + BiLSTM
```

Required reporting:

```text
Valence/Arousal RMSE
Valence/Arousal MAE
Valence/Arousal PCC
Valence/Arousal CCC
overall RMSE/MAE and mean CCC
```

## Layout

```text
configs/experiment.json
docs/RUN_LOG.md
scripts/extract_audio_features.py
scripts/train_audio_va.py
data/features/                 generated and gitignored
outputs/                       generated and gitignored
```

## Method Boundary

- The architecture is inspired by the stated DAMFF direction, not claimed as
  an exact reproduction without the original implementation details.
- The feature fusion is inspired by ENMI, but excludes historical emotion
  features and rolling-forecast targets.
- Reported DAMFF and ENMI paper numbers are contextual references only and are
  not treated as directly comparable results.

Current results:

```text
docs/ABLATION_RESULTS.md
```

Current decision: MERT+BiLSTM remains the validation-selected model.
Handcrafted features improve Arousal and should next be fused with MERT rather
than used as a standalone replacement.
