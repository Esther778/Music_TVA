# Research Workspace

This directory separates completed baseline work from the current section-aware
music emotion modeling work.

## Phase 1: Completed VA Prediction Baseline

Path:

```text
research/phase1_va_prediction/openl3_va_baseline
```

This is the completed OpenL3-based continuous Valence-Arousal prediction
project copied from the previous paper. It includes DEAM preprocessing,
OpenL3 feature extraction, MLP/BiLSTM/Transformer training, saved models,
training histories, and example prediction outputs.

Treat this folder as a reproducible baseline. Do not mix new section-aware
training code into it unless intentionally backporting utilities.

## Phase 2: In Progress Section-Aware VA Modeling

Path:

```text
research/phase2_section_aware_va
```

This is the current research task. It will use reviewed song sections from the
QC database plus frame-level VA predictions to build a two-layer music emotion
model:

1. Frame-level model: predict continuous Valence-Arousal from 0.5 second audio
   features.
2. Section-aware model: combine section boundaries and section types with the
   frame-level VA curve to model full-song emotional structure.

The expected outputs are frame-level VA predictions and section-level emotional
progression features.
