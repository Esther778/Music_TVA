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
scripts/                  manifest, VA prediction, alignment, training scripts
data/manifests/           reviewed-song lists and train/val/test splits
data/openl3_features/     expected per-song 512-d OpenL3 features for MLP input
data/frame_va/            0.5 second frame-level VA predictions
data/section_va/          frame VA joined with section boundaries/types
data/training_batches/    batch-by-batch training data snapshots
outputs/models/           model checkpoints
outputs/metrics/          per-batch and final metrics
outputs/predictions/      final frame-level and section-level predictions
outputs/plots/            diagnostic plots
```

## Current Run State

- Supabase input snapshots have been saved in `data/manifests/raw/`.
- `phase2_initial_70_manifest.csv` and `phase2_initial_70_splits.csv` have
  been generated.
- Split counts are train 56, validation 7, test 7.
- All selected songs have local audio files.
- The trained Phase 1 MLP model is available and is the required VA predictor.
- Current blocker: the selected songs do not yet have matching precomputed
  OpenL3 feature files in `data/openl3_features/`, so MLP inference cannot
  produce frame-level VA curves yet.

## Script Entry Points

```bash
.venv/bin/python research/phase2_section_aware_va/scripts/build_manifest.py
.venv/bin/python research/phase2_section_aware_va/scripts/predict_frame_va.py --features-dir research/phase2_section_aware_va/data/openl3_features
.venv/bin/python research/phase2_section_aware_va/scripts/align_sections.py
.venv/bin/python research/phase2_section_aware_va/scripts/train_section_model.py
```

`predict_frame_va.py` intentionally does not load or install OpenL3 unless
`--extract-audio` is passed. The normal Phase 2 path is to use existing
per-song feature files and the trained Phase 1 MLP weights.
