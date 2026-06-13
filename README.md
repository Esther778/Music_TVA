# Music VA Section-Aware Research Project

Last updated: 2026-06-13

This project builds a research and data-production pipeline for complete-song
music emotion modeling. It combines automatic pop-song section segmentation,
human QC, Supabase storage, and continuous Valence-Arousal prediction.

The current research direction is section-aware music emotion structure
modeling: use frame-level VA curves together with reviewed song sections to
analyze emotional progression, climax organization, and section contrast in
human music and AIGC music.

## Current Status

Phase 1 is complete:

```text
research/phase1_va_prediction/openl3_va_baseline
```

This is the copied baseline from the previous paper. It uses DEAM annotations
and OpenL3 audio embeddings to train MLP, BiLSTM, and Transformer continuous
VA regression models. Treat it as the completed baseline VA prediction
capability.

Phase 2 is in progress:

```text
research/phase2_section_aware_va
```

This phase will generate section-labeled VA data and train a two-layer music
emotion model:

1. Frame-level layer: predict continuous Valence-Arousal from 0.5 second audio
   features.
2. Section-aware layer: combine frame-level VA predictions with section
   boundaries and section types to model full-song emotional structure.

Current Phase 2 run state:

- Supabase reviewed-song and human-annotation snapshots have been saved under
  `research/phase2_section_aware_va/data/manifests/raw/`.
- The first 70 reviewed songs have been selected and split 56 / 7 / 7 for
  train / validation / test.
- The Phase 2 scripts now use the completed Phase 1 MLP weights for VA
  prediction.
- Frame-level VA generation is waiting for per-song 512-dimensional OpenL3
  feature files. The first 70 reviewed songs currently have no matching
  precomputed OpenL3 feature files in the Phase 2 feature directory.

Detailed phase notes:

```text
docs/research/PROJECT_PHASES.md
research/README.md
research/phase2_section_aware_va/README.md
```

## Project Layout

```text
annotation_tool/                  Supabase-backed human QC console
docs/research/                    research phase documentation
hybrid_structure/                 current functional section pipeline
research/                         research workspace
  phase1_va_prediction/           completed OpenL3 VA baseline
  phase2_section_aware_va/        current section-aware VA modeling work
songs/                            local mp3 library, gitignored
outputs/                          generated transcripts, sections, features, gitignored
netlify_publish/                  built static QC package, gitignored
```

Important workflow documents:

```text
PROJECT_HANDOFF.md
NEW_SONG_PIPELINE_WORKFLOW.md
LYRIC_CALIBRATION_WORKFLOW.md
NETLIFY_DEPLOY_WORKFLOW.md
annotation_tool/WORKFLOW_STATUS.md
annotation_tool/README.md
```

## Data And QC Platform

The QC platform is hosted at:

```text
https://musicva.netlify.app
```

The annotation tool uses Supabase tables and views for:

```text
songs
model_runs
auto_sections
lyric_lines
human_annotations
section_reviews
song_review_summary
section_review_queue
human_annotations_readable
```

The current section label taxonomy is:

```text
intro, verse, pre_chorus, chorus, bridge, interlude, outro
```

## New Song Pipeline

Newly collected audio files go into:

```text
songs/songs2
```

Files in `songs/songs2` may have inconsistent names as long as title and artist
are present. After validation and dedupe, valid songs are normalized to:

```text
歌名-歌手.mp3
```

The pipeline order is:

```text
inspect and dedupe songs/songs2
normalize filenames and move valid mp3 files into songs/
write annotation_tool/today_new_songs_YYYYMMDD.txt
run annotation_tool/run_model_outputs_for_songs.py
import generated manifest with annotation_tool/import_local_results.py
verify Supabase and QC pending_review visibility
```

Do not insert new songs into Supabase before model output exists.

## Local QC Development

Run the static QC console locally from the repository root:

```bash
python3 -m http.server 5177
```

Open:

```text
http://127.0.0.1:5177/annotation_tool/
```

Avoid using `file://` for normal QC testing because local audio path behavior
differs from the HTTP server.

## Netlify Deployment

The user normally performs production deployment from macOS Terminal.

Build and deploy command:

```bash
cd "/Users/esther/Desktop/New project"
python3 annotation_tool/build_netlify_publish.py
npx netlify deploy --prod --dir=netlify_publish
```

Codex should not run production deployment unless explicitly asked. If a
frontend/platform change is made, local source files and `netlify_publish/`
must stay synchronized before deployment.

## Research Phase 2 Initial Rules

Current user-specified rules for the next training task:

- The database currently has 75 reviewed songs.
- Use the first 70 reviewed songs for the first training pass.
- Split train / validation / test as 8:1:1.
- Train and record metrics in batches of 10 songs.
- Preserve training data, frame-level VA predictions, section-labeled VA
  tables, split manifests, model checkpoints, metrics, and plots.
- Use the trained Phase 1 MLP model for VA prediction. The MLP requires
  0.5-second, 512-dimensional OpenL3 features as input; if those features are
  not already available, frame-level VA cannot be generated from the MLP alone.

## Documentation Rule

This README is the top-level project map and must be kept current. During
future tasks, update this file when project phase boundaries, directory layout,
major data flow, training workflow, deployment workflow, or database/QC behavior
changes.

Also update the more specific workflow documents when a change belongs there:

```text
NEW_SONG_PIPELINE_WORKFLOW.md
LYRIC_CALIBRATION_WORKFLOW.md
PROJECT_HANDOFF.md
NETLIFY_DEPLOY_WORKFLOW.md
annotation_tool/WORKFLOW_STATUS.md
research/phase2_section_aware_va/README.md
```

## Git And Generated Data

Large local data and generated outputs are intentionally gitignored, including:

```text
songs/
outputs/
netlify_publish/
research/phase1_va_prediction/openl3_va_baseline/data/
research/phase1_va_prediction/openl3_va_baseline/artifacts/
research/phase1_va_prediction/openl3_va_baseline/outputs/
research/phase2_section_aware_va/data/
research/phase2_section_aware_va/outputs/
```

Keep commits scoped to the task. Do not mix unrelated dirty files into research
or QC updates.
