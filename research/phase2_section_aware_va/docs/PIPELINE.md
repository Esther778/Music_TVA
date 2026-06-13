# Phase 2 Section-Aware VA Pipeline

Last updated: 2026-06-13

## Current Objective

Build the second-stage section-aware music emotion modeling pipeline.

Phase 1 is complete and lives at:

```text
research/phase1_va_prediction/openl3_va_baseline
```

Phase 2 uses Phase 1's frame-level VA prediction capability and combines it
with reviewed song sections from the QC database.

## User-Specified Initial Run

- Database currently has 75 reviewed songs.
- Use the first 70 reviewed songs for the first training pass.
- Split train / validation / test as 8:1:1.
- Record training metrics every 10 songs.
- Preserve all intermediate data and training outputs.
- Do not discard batch outputs; future reviewed songs should be able to extend
  this run.

## Pipeline Stages

### 1. Reviewed-Song Manifest

Query Supabase for reviewed songs from `song_review_summary`, then keep the
first 70 songs using a stable ordering. Save:

```text
data/manifests/reviewed_songs_YYYYMMDD.csv
data/manifests/phase2_initial_70_manifest.csv
data/manifests/phase2_initial_70_splits.csv
```

### 2. Frame-Level VA Prediction

Use the Phase 1 OpenL3 + MLP predictor:

```text
research/phase1_va_prediction/openl3_va_baseline/outputs/models/mlp/mlp_best.pt
research/phase1_va_prediction/openl3_va_baseline/outputs/models/mlp/mlp_input_norm.npz
```

The MLP does not consume MP3 audio directly. Its required input is a sequence
of 512-dimensional OpenL3 features, one row per 0.5 second frame.

Normal Phase 2 inference path:

```text
data/openl3_features/<song_id>.npy
or
data/openl3_features/<song_id>.npz
```

The `.npy` format is expected to be shaped `(frames, 512)`. The `.npz` format
may contain `embedding` or `embedding_norm`; if `time_sec` is absent,
timestamps are reconstructed at 0.5 second intervals.

The prediction script uses the trained Phase 1 MLP weights. It does not load
OpenL3 unless `--extract-audio` is explicitly requested.

Save one CSV per song:

```text
data/frame_va/<song_id>_frame_va.csv
```

Expected columns:

```text
song_id,title,artist,time_sec,valence,arousal
```

### 3. Section Alignment

Query reviewed human annotations from Supabase and align each 0.5 second VA
frame to the corresponding reviewed section. Save:

```text
data/section_va/<song_id>_frame_section_va.csv
data/section_va/phase2_initial_70_frame_section_va.csv
```

Expected columns include:

```text
song_id,title,artist,time_sec,valence,arousal,section_index,section_type,
section_start_sec,section_end_sec
```

### 4. Section-Level Feature Extraction

Aggregate frame-level VA inside each section and compute transition features.
Save:

```text
data/section_va/phase2_initial_70_section_features.csv
```

Feature families:

- section duration,
- mean/std/min/max valence and arousal,
- entry/exit/delta valence and arousal,
- linear slope over time,
- peak timing,
- contrast against previous and next section,
- song-normalized section position.

### 5. Two-Layer Training

Layer 1:

- frame-level VA predictor from audio features.
- initial implementation may reuse Phase 1 MLP as the frozen or warm-start
  baseline.

Layer 2:

- section-aware emotion structure model trained on section-level features.
- train incrementally in batches of 10 songs.
- log each batch's train/validation/test metrics.

Save:

```text
data/training_batches/
outputs/models/
outputs/metrics/
outputs/predictions/
outputs/plots/
```

## Metrics To Preserve

Frame-level:

- MSE,
- RMSE,
- Pearson r for valence,
- Pearson r for arousal.

Section-level:

- section-level VA reconstruction RMSE,
- Pearson r for section mean valence,
- Pearson r for section mean arousal,
- transition delta error,
- per-section-type summary metrics.

## Current Status

- Pipeline documentation created and updated.
- Phase 2 script entry points created:
  - `scripts/build_manifest.py`
  - `scripts/predict_frame_va.py`
  - `scripts/align_sections.py`
  - `scripts/train_section_model.py`
- Supabase snapshots saved:
  - `data/manifests/raw/reviewed_songs_20260613.json`
  - `data/manifests/raw/human_annotations_20260613.json`
- Initial 70-song manifest and split files generated:
  - `data/manifests/phase2_initial_70_manifest.csv`
  - `data/manifests/phase2_initial_70_splits.csv`
- Current split counts: train 56, validation 7, test 7.
- Local audio exists for all 70 selected songs.
- Current blocker: no matching precomputed OpenL3 feature files were found for
  the first 70 selected songs. `data/manifests/phase2_initial_70_feature_audit.csv`
  records the missing feature inputs.
