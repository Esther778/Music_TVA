# Phase 2 Section-Aware VA Modeling

Status: in progress.

## Goal

Build a section-aware complete-song emotion structure modeling pipeline on top
of the completed Phase 1 VA prediction baseline.

The model design is two-layer:

1. Frame-level VA prediction:
   - input: 0.5 second audio features,
   - output: continuous Valence-Arousal curve.
2. Section-aware emotion structure modeling:
   - input: frame-level VA curve plus reviewed section boundaries and section
     types,
   - output: section-level emotional progression features for complete songs.

The research target is to compare human music and AIGC music in emotional
progression, climax organization, and section contrast.

## Current Data Rule

- The database currently has 75 reviewed songs.
- The first training pass should use the first 70 reviewed songs.
- Train / validation / test split ratio: 8:1:1.
- Training should be tracked in batches of 10 songs.
- Accuracy metrics for every batch must be preserved.
- Intermediate data must be preserved, including manifests, frame-level VA
  predictions, section-aligned VA tables, split files, model checkpoints,
  metrics, and plots.

## Directory Layout

```text
configs/                  experiment configs
docs/                     design notes and run logs
scripts/                  Phase 2 scripts, not yet implemented
data/manifests/           reviewed-song lists and train/val/test splits
data/frame_va/            0.5 second frame-level VA predictions
data/section_va/          frame VA joined with section boundaries/types
data/training_batches/    batch-by-batch training data snapshots
outputs/models/           model checkpoints
outputs/metrics/          per-batch and final metrics
outputs/predictions/      final frame-level and section-level predictions
outputs/plots/            diagnostic plots
```

## Do Not Start Yet

Code implementation and training should start only after the project folder
structure has been reviewed and accepted.
