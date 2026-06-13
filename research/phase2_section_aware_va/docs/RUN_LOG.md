# Phase 2 Run Log

This log is updated as the section-aware VA pipeline progresses. It should be
read together with `PIPELINE.md`.

## 2026-06-13

### Starting Point

- Phase 1 OpenL3 VA baseline has been moved to
  `research/phase1_va_prediction/openl3_va_baseline`.
- Phase 2 workspace has been created at
  `research/phase2_section_aware_va`.
- The user confirmed the Phase 2 task:
  - use reviewed songs from Supabase,
  - generate frame-level VA,
  - join VA with reviewed sections,
  - train a two-layer section-aware emotion model,
  - preserve batch data and metrics,
  - first pass uses the first 70 reviewed songs out of 75.

### Runtime Estimate Before Implementation

- Manifest and split generation: 5-15 minutes.
- Frame-level VA generation for 70 songs: approximately 1-3 hours, depending
  on OpenL3 extraction speed.
- Section alignment and feature extraction: 10-30 minutes.
- First two-layer training pass: approximately 1-3 hours for a lightweight
  reproducible baseline, longer if larger models are introduced.

### Actions Completed

- Created `docs/PIPELINE.md`.
- Created this run log.
- Created Phase 2 scripts:
  - `scripts/common.py`
  - `scripts/fetch_supabase_inputs.sh`
  - `scripts/build_manifest.py`
  - `scripts/predict_frame_va.py`
  - `scripts/align_sections.py`
  - `scripts/train_section_model.py`
- Saved Supabase snapshots:
  - `data/manifests/raw/reviewed_songs_20260613.json`
  - `data/manifests/raw/human_annotations_20260613.json`
- Generated first-run manifest files:
  - `data/manifests/reviewed_songs_20260613.csv`
  - `data/manifests/phase2_initial_70_manifest.csv`
  - `data/manifests/phase2_initial_70_splits.csv`
- Current Supabase snapshot contains 81 reviewed songs. The first run uses
  the first 70 according to `updated_at.asc,id.asc`.
- Split counts are train 56, validation 7, test 7.
- All 70 selected songs have local MP3 files.
- Verified the "trained MLP + precomputed feature" path. The selected songs
  currently have no matching feature files in
  `data/openl3_features/`, so frame-level VA generation produced:

```text
completed=0 skipped=70
```

- Wrote the missing-feature audit:
  `data/manifests/phase2_initial_70_feature_audit.csv`.

### Current Technical Constraint

The trained Phase 1 MLP cannot take MP3 files directly. It takes 0.5-second,
512-dimensional OpenL3 feature rows as input. Because the user asked not to
download or install OpenL3, the current Phase 2 run must wait for precomputed
OpenL3 features for the 70 selected songs, or for explicit approval to extract
those features locally with an existing/installed extractor.

### Next Actions

1. Provide or generate `data/openl3_features/<song_id>.npy` or `.npz` for the
   70 selected songs.
2. Run `predict_frame_va.py --features-dir data/openl3_features` to generate
   `data/frame_va/<song_id>_frame_va.csv`.
3. Run `align_sections.py --annotations-json data/manifests/raw/human_annotations_20260613.json`.
4. Run `train_section_model.py` to train cumulative 10-song section-aware
   batches and preserve metrics.
