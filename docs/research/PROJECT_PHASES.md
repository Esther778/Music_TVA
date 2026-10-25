# Research Project Phases

## Phase 1 Completed: OpenL3 VA Prediction

Folder:

```text
research/phase1_va_prediction/openl3_va_baseline
```

Scope:

- DEAM dynamic Valence-Arousal label preparation.
- OpenL3 512-dimensional audio embedding extraction.
- MLP, BiLSTM, and Transformer VA regression experiments.
- Saved baseline models and training histories.
- MLP-based prediction interface for generating frame-level VA curves.

Current interpretation:

- This phase is complete and should be treated as the baseline emotion
  prediction capability from the previous paper.
- It should not be mixed with Phase 2 section-aware code.

## Phase 2 In Progress: Section-Aware Complete-Song Emotion Modeling

Folder:

```text
research/phase2_section_aware_va
```

Scope:

- Generate frame-level VA predictions for reviewed songs.
- Join 0.5 second VA frames with reviewed section boundaries and section types.
- Train a two-layer music emotion model:
  - layer 1: frame-level continuous VA prediction,
  - layer 2: section-aware emotional structure modeling.
- Preserve per-10-song batch data and metrics.
- Produce frame-level VA predictions and section-level emotional progression
  features.

Initial data rule:

- 75 reviewed songs are available in the database.
- Use the first 70 reviewed songs for the first training pass.
- Split train / validation / test as 8:1:1.

Research use:

- Characterize emotional progression, climax organization, section contrast,
  and related section-aware emotion structure measures in human-created Chinese
  popular music.
- Provide the audio-side baseline for later lyric-audio VA alignment in the
  graduation thesis.
- Do not perform human/AIGC migration comparison; that historical direction was
  removed from the graduation-thesis scope on 2026-10-25.
