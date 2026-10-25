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

## 2026-06-16

### EI Paper Method Review

Created:

```text
docs/EI_PAPER_METHOD_REVIEW.md
```

The document records:

- whether the two-stage design can support an EI-level paper,
- the core methodological weaknesses,
- required fixes before writing,
- recommended paper contribution claims,
- suggested paper structure,
- section-level features,
- Stage 2 evaluation design,
- immediate next actions.

Key conclusion: the design is viable if the paper is framed as
section-aware complete-song emotion structure modeling. The largest weakness is
that Stage 2 VA labels are model-derived pseudo labels rather than independent
human VA ground truth.

### Reviewed Song Count Update

User reported that the current Phase 2 reviewed/annotated complete-song count
has increased to:

```text
95 songs
```

Current local snapshots are stale and still show older counts:

```text
reviewed_songs_20260613.csv=70 songs
phase2_initial_70_manifest.csv=70 songs
raw/reviewed_songs_20260613.json=81 songs
```

Before generating complete-song pseudo-VA, refresh the reviewed-song manifest
and split files from the current source of truth.

## 2026-06-21

### Phase 1 VA Teacher Frozen (Historical, Later Superseded)

The operational Teacher for Phase 2 pseudo-VA generation is now fixed:

```text
teacher_model_id=openl3_mlp_w512_v1
feature_backbone=OpenL3 mel256 music 512-d
frame_hop=0.5 seconds
head=MLP [512,256], dropout 0.10
deployment_mode=mean ensemble
seeds=2026,3407,614
```

Selection rule:

- architecture and deployment mode were selected by validation RMSE,
- test metrics were used only for the final report,
- the ensemble was selected because validation RMSE improved from `0.109146`
  for the best single model to `0.099575`.

Frozen Teacher final test metrics:

```text
RMSE=0.0984247178
MAE=0.0734129474
Pearson r(V)=0.9226814508
Pearson r(A)=0.9409965277
CCC(V)=0.9081979990
CCC(A)=0.9329394102
mean CCC=0.9205687046
```

Reproducibility files:

```text
research/phase1_va_prediction/openl3_va_baseline/configs/frozen_teacher_v1.json
research/phase1_va_prediction/openl3_va_baseline/docs/TEACHER_MODEL_CARD.md
research/phase1_va_prediction/openl3_va_baseline/scripts/freeze_va_teacher.py
research/phase1_va_prediction/openl3_va_baseline/scripts/predict_frozen_teacher.py
```

Verification completed:

- all three checkpoint metrics reproduced from saved weights,
- normalization statistics exactly match the MLP training split,
- checkpoint and normalization SHA-256 hashes were recorded,
- a 13,000-frame inference smoke test reproduced the frozen test RMSE with
  absolute difference below `1e-8`.

Critical limitation:

- the OpenL3 result uses the published frame-level random split,
- it is not a song-level generalization estimate,
- it cannot be used to claim OpenL3 superiority over song-level MERT,
- generated Phase 2 VA values are pseudo-VA rather than human ground truth.

Next Phase 2 actions:

1. Refresh the reviewed-song and annotation manifests to the current 95-song
   source of truth.
2. Adapt `predict_frame_va.py` to load the frozen Teacher configuration.
3. Audit or extract matching 0.5-second, 512-dimensional OpenL3 features.
4. Generate frame-level pseudo-VA while preserving per-song logs and failures.

This provisional decision was superseded by the song-level validation in the
next section. Do not follow these generation steps until final Teacher
selection is completed.

### Song-Level Validation Changed The Teacher Decision

OpenL3+MLP was retrained and evaluated on the exact MERT song split.

```text
OpenL3 three-seed ensemble:
val_rmse=0.2222753316
test_rmse=0.2083092183
test_mean_ccc=0.4941812754

MERT+BiLSTM current baseline:
val_rmse=0.2199336708
test_rmse=0.2024518376
test_mean_ccc=0.5382629439
```

Decision update:

- the OpenL3 engineering configuration remains available,
- it is no longer considered the final scientific Teacher,
- do not generate the final 95-song pseudo-VA corpus yet,
- first run MERT+BiLSTM with the same three seeds and evaluate a seed ensemble,
- choose the final Teacher using validation RMSE, with CCC and per-dimension
  metrics as supporting evidence.

## 2026-07-20

### Blind Validation Stage 1 Refresh And Non-Chinese Song Removal

The paused Supabase project was resumed and the blind-safe Stage 1 input audit
was rerun. Only song identity/audio metadata and section boundaries were read;
section types, lyrics, review history, and any VA/climax judgments remained
excluded.

At the user's request, `Love Story - Taylor Swift`
(`local_64ca8af42c6b`) was permanently removed because the study corpus now
excludes non-Chinese songs. The deletion covered:

- Supabase `songs`, `model_runs`, `auto_sections`, `lyric_lines`,
  `human_annotations`, `section_reviews`, and `audio_features_2hz` records;
- the local and Netlify-publish MP3 copies;
- dedicated section/lyric outputs and matching rows in local manifests and
  snapshots.

Post-deletion read-only verification:

```text
reviewed songs=99
section boundary rows=1096
songs with local audio=99
songs missing boundaries=0
invalid boundary rows=0
nonpositive song durations=0
```

One section-continuity issue remains for manual QC:

```text
song=倒数 - G.E.M.邓紫棋
song_id=local_105ab4011af8
section 9 end=162.0 s
section 10 start=173.0 s
unassigned gap=11.0 s
```

Do not start the 15-song blind sample until this gap is corrected and the
Stage 1 audit is rerun.

### Blind Validation Stage 1 Passed

The user corrected the `倒数` section gap. A fresh read-only Supabase snapshot
and audit produced:

```text
reviewed songs=99
unique section boundaries=1097
songs with continuous boundaries=99
local audio files=99
invalid/gap/overlap rows=0
```

Stage 1 is complete. Full details and field-use restrictions are recorded in
`docs/BLIND_VALIDATION_STAGE1_AUDIT_20260720.md`. Stage 2 sampling has not yet
started.

### Blind Validation Stage 2 Sample Frozen

A deterministic 5-by-3 stratified sample was selected from the 99-song corpus
with seed `20260720`. One song was selected per duration-by-section-count cell,
with no repeated artist or title.

```text
sample songs=15
unique artists=15
covered strata=15/15
frozen section boundaries=179
manifest_sha256=1d6d624c27c6c2fd6317b6b146b97a935846ade4467553cbfb3ef9551d96aff5
boundaries_sha256=4327ad8bef8d4a3c0f9bcba71c4a5fd3ab9743a77c29c2b03ebac8af6d5ca7f1
```

The selected list and complete protocol are in
`docs/BLIND_VALIDATION_STAGE2_SAMPLE_20260720.md`. No VA inference has started.

### Blind Validation Stage 3 Frame-Level Pseudo-VA Complete

The frozen final clean Phase 1 Teachers were applied to the 15-song audio-only
sample:

```text
Valence=pmemo_final_clean_valence_teacher_v1
Arousal=pmemo_final_clean_arousal_teacher_v1
features=MERT hidden-6 + PMEmo-style OpenSMILE dynamic 260d
deployment=30 s windows, 15 s stride, Hann overlap-add
frame hop=0.5 s
```

The 30-second deployment window was fixed before accessing human labels. A
direct 204.6-second forward pass had collapsed to V/A standard deviations of
`0.001199/0.004199`; the windowed pass produced `0.086861/0.200925` while
remaining audio-only and using the same frozen weights.

Final batch and independent audit:

```text
songs=15/15
frames=7375
failed songs=0
collapse warnings=0
combined-table consistency=pass
combined SHA-256=9fd7d7b758ba707c8e1c8fdde57548eb0eb8225e87b40ae987c1ed39f1582357
```

Compatibility findings retained for the paper:

- MERT hidden-6 re-extraction correlation with a saved PMEmo feature was
  `0.999683`.
- Raw-MP3 OpenSMILE reproduction is not bit-equivalent to PMEmo's official
  features; the checked song correlation was `0.881348`.
- Outputs are pseudo-VA and require the planned section-level human validation.

Full provenance and limitations:
`docs/BLIND_VALIDATION_STAGE3_INFERENCE_20260720.md`.

Stage 3 is complete. Stage 4 section alignment has not started and requires
explicit user confirmation.

## 2026-07-21

### Blind Validation Stage 4 Section Alignment Complete

The 7375 frozen Stage 3 pseudo-VA frames were aligned to the 179 frozen Stage 2
section boundaries. No Supabase query was made and section types, lyrics,
review confidence, human emotion levels, change labels, and climax answers
remained withheld.

Alignment rule:

```text
section_start_sec <= frame_center_time < section_end_sec
```

Audit result:

```text
songs=15
sections=179
frames=7375
assigned=7375
unassigned=0
overlap=0
sections without frames=0
independent consistency checks=pass
```

Outputs include frame-section alignment, section-level emotion features, and
song-level structure summaries. `climax_section_order_by_mean_arousal` is only
a model-derived candidate definition and is not a validated human climax.

Full report:
`docs/BLIND_VALIDATION_STAGE4_ALIGNMENT_20260721.md`.

Stage 4 is complete. Stage 5 human/section-type comparison has not started and
requires explicit user confirmation.

### Blind Validation Stage 5 Structural Analysis Complete, Direct Human Ratings Pending

After Stage 3/4 outputs were frozen, 179 held-out section types were fetched by
their frozen annotation IDs. All IDs and boundaries matched; all 179 sections
were approved and had a structural type. The database has no human VA,
change-direction, or climax fields.

Song-level paired exploratory result:

```text
Chorus minus Verse Valence=+0.093922, 13/15 positive, p=0.000305
Chorus minus Verse Arousal=+0.235362, 15/15 positive, p=0.000061
```

Direct Verse-to-Chorus and Pre-chorus-to-Chorus transitions also showed
positive mean Arousal changes. Repeated Chorus sections were not more
consistent than repeated Verse sections (`p=0.599487` for pairwise VA
distance), so that negative result is retained.

Maximum-mean-Arousal candidates were Chorus or Bridge in 12/15 songs and at a
normalized midpoint of at least 0.70 in 13/15 songs. These are not human climax
hit rates.

A 179-row model-blind annotation sheet and guide were created for the missing
direct human validation. Full report:
`docs/BLIND_VALIDATION_STAGE5_STRUCTURAL_ANALYSIS_20260721.md`.

Stage 5 structural analysis is complete. Stage 5 direct emotion validation
remains pending the human annotation sheet and the Teacher must remain frozen.

### Paper Scope Updated And Human Annotation Workbook Created

- The user removed AIGC comparison from the current EI paper scope.
- The active scope is section-aware emotional progression, section contrast,
  and climax organization in human-created Chinese popular music.
- Historical AIGC design notes remain in place only for traceability and are
  marked as superseded by this scope decision.
- Created the model-blind Excel workbook:
  `outputs/human_validation/blind_sample_15_human_section_emotion_annotation.xlsx`.
- Workbook contents:
  - Chinese rating protocol and field definitions;
  - 15 songs and 179 frozen section rows;
  - dropdown validation for Valence, Arousal, change direction, and climax;
  - first-section `not_applicable` handling;
  - formula-driven per-song completion and exactly-one-climax checks.
- The workbook contains no model VA values, climax candidates, or structural
  analysis results. The frozen Teachers remain untouched.

### Human Validation Workbook Audio And Strict Input Update

- Added reproducible section-audio builder:
  `scripts/build_blind_human_validation_audio_clips.py`.
- Cropped all 179 frozen section intervals from the 15 local source MP3 files.
- Audio format: MP3, 128 kbps, 44.1 kHz, stereo.
- Duration audit: 179/179 clips passed with absolute error at most 0.08 s.
- `New Boy` ends at 252.839184 s although the frozen rounded boundary is
  253.0 s. Its final 238.0-253.0 s clip therefore contains approximately
  0.16 s of trailing silence so the frozen interval remains unchanged.
- Added `播放本段` to every workbook section row using relative links into the
  adjacent `audio_clips/` directory.
- Added strict Excel list validation with `errorStyle=stop` for:
  - rater ID (`rater_01`, `rater_02`);
  - Valence (`low`, `mid`, `high`);
  - Arousal (`low`, `mid`, `high`);
  - change direction (`up`, `down`, `stable`);
  - climax (`0`, `1`);
  - first-section change (`not_applicable`).
- Workbook XML audit confirmed all list rules and 179 hyperlink formulas.
- LibreOffice independent-open audit recognized 179/179 audio links as link
  annotations and displayed the friendly label `▶ 播放`.

### Human Validation Browser Console Added

- Added the standalone builder:
  `scripts/build_blind_human_validation_console.py`.
- Added the versioned HTML template:
  `web/human_validation_console_template.html`.
- Generated the primary manual interface:
  `outputs/human_validation/blind_sample_15_human_validation_console.html`.
- The console loads all 15 songs and 179 audited section clips without model
  predictions. Every row provides play/pause, a seekable progress bar, and
  current/total playback time.
- Required ratings remain restricted to the protocol dropdown values. Selecting
  a climax automatically clears any previous climax in the same song.
- Browser-local autosave and UTF-8 CSV export were added. The Excel workbook is
  retained as a fallback because spreadsheet hyperlinks do not provide a
  reliable in-cell player on macOS.
- Final static and local-serving checks:
  - JavaScript syntax passed;
  - embedded rows=179, songs=15, unique audio paths=179;
  - required play, pause, seek, time-update, autosave, and CSV-export logic
    present;
  - HTML returned HTTP 200 as `text/html`;
  - sample section audio returned HTTP 200 as `audio/mpeg`.

### Human Validation Console Narrow-Screen Layout Update

- Reworked the protocol explanation into clearly separated, bold-labeled
  Valence, Arousal, change, climax, and judgment-principle rows.
- Added a no-horizontal-scroll layout below 760 px:
  - song and artist are combined;
  - section identity and interval are stacked;
  - audio uses the full row width;
  - Valence/Arousal and change/climax controls use two-column rows;
  - notes use the full row width.
- Below 520 px, toolbar commands and the audio time/progress controls reflow to
  prevent text or controls from exceeding the viewport.
- Removed the desktop table's former 1540 px minimum width. All columns now use
  a fixed 100% proportional width allocation, with wrapping and shrinkable
  controls, and horizontal scrolling is disabled for the page and table shell.

### Human Climax Protocol Corrected

- Removed the assumption that every song has exactly one principal climax.
- `is_climax` now denotes a perceptible local emotional peak and allows zero,
  one, or multiple positive sections per song.
- Removed automatic clearing of other climax selections and removed the
  exactly-one-climax requirement from song completion.
- Withheld `section_type` from both the rendered page and embedded browser
  data to avoid Chorus-related expectation bias; results will be joined to
  frozen structure data later by `human_annotation_id`.
- Changed the annotation standard from a collapsible disclosure to an
  always-visible panel and added the requirement to rate each song in order.
- Bumped browser autosave storage from `v3` to `v4` so annotations created
  under the superseded rule cannot silently carry into the corrected protocol.
- Established the canonical exported-response directory:
  `data/blind_validation/stage5/human_annotations/`, with one CSV per rater.

## 2026-07-22

### Two-Song Human Validation Preview

- Received `data/blind_validation/stage5/human_annotations/吕靖怡.csv`.
- Audit passed: 179 unique frozen IDs, zero boundary drift, 21 complete rows
  covering exactly two complete songs, and no partially filled other songs.
- Exploratory mean within-song Spearman was `0.569` for Valence and `0.793`
  for Arousal.
- Pooled Spearman was `0.076` for Valence and `0.780` for Arousal, suggesting
  a possible cross-song Valence calibration issue that cannot be resolved at
  n=2.
- All seven human emotional peaks mapped to Chorus only after blind ratings
  were complete. The frozen top-Arousal candidate was an exact human-peak hit
  for one song and an adjacent hit for the other.
- Full provisional interpretation and guardrails:
  `docs/BLIND_VALIDATION_PARTIAL_HUMAN_RESULTS_20260722.md`.

### Human Change Definition And Column Width Update

- Reduced proportional widths for song title from 11% to 8%, artist from 12%
  to 11%, and audio from 24% to 18%.
- Reallocated the space to Valence, Arousal, intensity change, climax, and
  notes so dropdown contents remain visible without horizontal scrolling.
- Defined `previous section` as `section_order - 1` within the same song,
  never the previous occurrence of the same structural type.
- Defined `human_change_from_prev` specifically as perceived Arousal/energy
  change: up, down, or stable. It does not represent Valence direction.
- Existing Valence, Arousal, and climax values remain intact. The first two
  songs' change labels are provisional until the rater reviews them under the
  clarified definition.

### Stage 5B Rater 01 Complete

- Received a completed `吕靖怡.csv`: 15/15 songs, 179/179 sections.
- Audit passed with 179 unique frozen IDs, zero boundary drift, no missing or
  invalid values, and correct first-section `not_applicable` handling.
- Mean within-song Spearman: Valence `0.345`, Arousal `0.503`.
- Pooled section Spearman: Valence `0.419`, Arousal `0.589`.
- Between-song mean Spearman: Valence `0.400`, Arousal `0.732`.
- Non-stable change direction matched model Arousal delta in `84/117=71.8%`.
- Frozen maximum-mean-Arousal candidate: exact human-peak hit `9/15`; adjacent
  hit `13/15`.
- Human Chorus lift was positive for Valence and Arousal in `14/15` songs.
- Added the reproducible evaluator `scripts/analyze_blind_human_validation.py`
  and saved machine-readable results under
  `outputs/human_validation/rater_01_evaluation/`.
- Full report:
  `docs/BLIND_VALIDATION_HUMAN_RESULTS_RATER01_20260722.md`.
- This remains one-rater ordinal validation. Do not report RMSE/CCC against
  these labels or tune the frozen Teachers from these results.

### Cross-Song VA Anchor Candidate Preparation

- Preserved the completed `吕靖怡.csv` as the original pilot annotation; no
  response values or official annotation protocol were overwritten.
- Built 12 model-free calibration candidates: two examples for each of
  Valence low/mid/high and Arousal low/mid/high.
- All 12 source songs are outside the frozen 15-song blind-validation sample;
  blind-test overlap is zero.
- Candidate intervals are contained within reviewed human section boundaries.
- Teacher predictions were not read or used for candidate selection. Basic
  acoustic descriptors were used only to reject an unsuitable Arousal-mid
  interval and are retained in the candidate audit table.
- Added the reproducible builder
  `scripts/build_va_anchor_candidates.py`, source template
  `web/va_anchor_candidate_review_template.html`, and candidate outputs under
  `outputs/human_validation/calibration_candidates/`.
- Candidate audit passed: 12/12 audio files exist, all duration errors are
  within 0.08 seconds, embedded page rows=12, unique songs=12, and JavaScript
  syntax passed.
- Candidate status remains `candidate_only_not_frozen`. The official guide and
  reannotation version must not be generated until the user confirms which
  clips are sufficiently typical cross-song scale anchors.
- Automated browser visual QA was unavailable because the browser security
  policy blocks local `file://` navigation; static HTML/data/audio checks
  passed, and the user will perform the required listening and visual review.

### VA Anchors Frozen And Anchored Reannotation Console Built

- Received and archived the complete human candidate review at
  `outputs/human_validation/calibration_candidates/human_reviews/`
  with 12/12 judgments resolved.
- Human review accepted 10 proposed labels and corrected two Valence labels:
  - `阴天`: low -> mid;
  - `同桌的你`: mid -> low.
- The corrections preserve exactly two anchors for every combination of
  Valence/Arousal and low/mid/high; no replacement candidate was required.
- Frozen anchor tables:
  - `outputs/human_validation/calibration_candidates/frozen_va_anchors_v1.csv`;
  - `outputs/human_validation/calibration_candidates/frozen_va_anchors_v1.json`.
- Added reproducible validation/freezing script
  `scripts/freeze_va_anchor_review.py`.
- Built the fresh reannotation console
  `outputs/human_validation/blind_sample_15_human_validation_console_anchored_v1.html`
  using `scripts/build_blind_human_validation_anchored_console.py`.
- The anchored console uses a new local-storage key and export filename. It
  does not import or overwrite the original pilot annotations.
- Final static audit passed: 179 embedded section rows, 179 section audio
  files, 12 frozen anchors, 12 anchor audio files, and valid JavaScript syntax.
- The Teacher remains frozen; neither candidate confirmation nor anchored
  reannotation modifies model weights or uses model outputs as human prompts.

## 2026-07-23

### Anchored Reannotation Segmented Controls

- Replaced the four per-section rating dropdowns in the anchored reannotation
  console with in-cell segmented button groups:
  - Valence: low / mid / high;
  - Arousal: low / mid / high;
  - change: up / down / stable;
  - climax: no / yes.
- The selected button is highlighted and saved immediately through the same
  anchored-v1 browser storage key, so existing anchored-page progress remains
  readable.
- First-section change remains locked to `not_applicable` and is displayed as
  a disabled `不适用` button.
- The original pilot page remains unchanged.
- Static audit passed: 179 embedded rows, segmented-control renderer and click
  handler present, old rating-select renderer absent, and JavaScript syntax
  valid.
- Renamed the visible `高潮` control to `局部情绪峰值` and clarified that it is
  neither Chorus recognition nor a requirement to mark every Chorus. The
  machine-readable field remains `is_climax` for analysis compatibility.
- Corrected a stale pilot-page message: the anchored version now explicitly
  states that old answers are not loaded and all 179 sections must be rated
  again.

### Anchored V1 Human Validation Complete

- Received the complete anchored-v1 response and archived it as
  `data/blind_validation/stage5/human_annotations/blind_sample_15_human_annotations_anchored_v1_rater_01.csv`.
- SHA-256:
  `4ef639e1ae3170c9263653fbc3f19c972ad9f736a3fa6605a29e5dbcb6715e02`.
- Audit passed: 179 rows, 179 unique frozen IDs, 15 songs, no missing values,
  and zero start/end boundary drift.
- Main anchored-v1 results:
  - pooled section Spearman: Valence `0.583773`, Arousal `0.533223`;
  - within-song macro Spearman: Valence `0.398162`, Arousal `0.503876`;
  - between-song mean Spearman: Valence `0.697051`, Arousal `0.577301`;
  - non-stable Arousal-change sign accuracy: `87/129=67.4%`;
  - frozen top-Arousal exact/adjacent peak hits: `8/15` and `13/15`.
- Human Chorus-minus-Verse lift remained positive and statistically nonzero:
  Valence `+0.578` (`p=0.00358`) and Arousal `+0.759` (`p=0.00130`).
- Of 57 human local peaks, 54 were Chorus; this is reported as blind local
  peak/Chorus overlap, not as independent Chorus ground truth.
- Added reproducible Pilot-versus-anchored comparison script
  `scripts/compare_human_validation_rounds.py` and machine-readable comparison
  JSON under `outputs/human_validation/rater_01_anchored_v1_evaluation/`.
- Full interpretation and paper-writing guardrails:
  `docs/BLIND_VALIDATION_HUMAN_RESULTS_ANCHORED_V1_20260723.md`.
- The unified-anchor result is the formal one-rater ordinal validation. The
  Teacher remains frozen, and no human result was used for model tuning.

## 2026-07-24

### Same-Rater Delayed Review Complete

- Before recomputing model agreement, the user reported that continuous
  listening fatigue affected confidence and that delayed relistening felt more
  accurate. The delayed review was therefore designated as the primary result
  before metrics were inspected; the initial anchored-v1 result remains a
  sensitivity analysis.
- Archived the delayed review as
  `blind_sample_15_human_annotations_anchored_v1_delayed_review_rater_01.csv`.
- SHA-256:
  `25b1cf7d53e78bf0bbd2fc5550b870e65d99aa643e74dc144e8fbf4507377f43`.
- Audit passed: 179 rows, 179 unique IDs, no missing values, and zero boundary
  drift.
- Compared with the initial anchored review, 24 sections changed at least one
  field: Valence 11, Arousal 18, change direction 1, local peak 0.
- Same-rater delayed stability:
  - Valence exact agreement `93.9%`, quadratic-weighted Kappa `0.930`;
  - Arousal exact agreement `89.9%`, quadratic-weighted Kappa `0.882`;
  - change exact agreement `99.4%`, Kappa `0.992`;
  - local peak exact agreement and Kappa `1.000`.
- Updated primary model-agreement results:
  - pooled Valence/Arousal Spearman `0.570/0.556`;
  - within-song macro Valence/Arousal `0.385/0.519`;
  - between-song Valence/Arousal `0.640/0.532`;
  - nonstable change sign accuracy `88/129=68.2%`;
  - exact/adjacent top-Arousal peak hits unchanged at `8/15` and `13/15`.
- Chorus-minus-Verse human lift: Valence `+0.678` (`p=0.00132`) and
  Arousal `+0.914` (`p=0.000948`). Human/model lift correlation was `0.182`
  for Valence and `0.592` for Arousal.
- Added `scripts/compare_anchored_delayed_review.py` and the machine-readable
  `initial_vs_delayed_review.json`.
- Full report:
  `docs/BLIND_VALIDATION_HUMAN_RESULTS_DELAYED_REVIEW_20260724.md`.
- The Kappa values measure intra-rater test-retest stability, not inter-rater
  reliability. The Teacher remains frozen.

## 2026-07-25

### VA Anchor Expansion Supplement Prepared

- The user requested expansion from two to five reference examples for every
  Valence/Arousal × low/mid/high combination.
- Prepared 18 supplemental candidates: three new songs for each of the six
  dimension-level combinations. If all proposed/final labels remain balanced,
  the frozen anchor set will grow from 12 to 30 clips, five per combination.
- Safeguards:
  - 18 unique new songs;
  - zero overlap with the frozen 15-song validation sample;
  - zero overlap with the 12 existing anchor songs;
  - all intervals contained in one reviewed human section boundary;
  - no Teacher prediction used;
  - basic acoustic descriptors retained only as candidate screening metadata.
- Added reproducible builder
  `scripts/build_va_anchor_supplement_candidates.py`.
- Generated candidate assets under
  `outputs/human_validation/calibration_candidates_v2_supplement/`, including
  18 audio clips, CSV/JSON audit manifests, and
  `va_anchor_supplement_candidate_review.html`.
- Static audit passed: 18 embedded records, 18 unique songs, 18 available audio
  files, separate browser storage key and export filename, and valid
  JavaScript syntax.
- Status remains `supplement_candidate_only_not_frozen`. Do not update the
  formal anchor guide or annotation console until the human review is exported
  and all six final groups contain exactly five anchors.

### VA Anchor V2 Frozen From Chat Confirmation

- The user listened to all 18 supplemental candidates and stated in chat that
  every candidate fit its proposed level. No review CSV was exported.
- Preserved that provenance explicitly as `confirmation_method =
  user_chat_statement`; no exported-review claim was created.
- Archived the resulting confirmation table at
  `outputs/human_validation/calibration_candidates_v2_supplement/human_reviews/va_anchor_supplement_chat_confirmation_20260725.csv`.
- Froze the combined reference library as
  `frozen_va_anchors_v2_30clips.csv` and
  `frozen_va_anchors_v2_30clips.json` under
  `outputs/human_validation/calibration_candidates_v2_supplement/`.
- Freeze audit passed: 30 clips, 30 unique songs, exactly five references for
  every Valence/Arousal x low/mid/high group, zero overlap with the frozen
  15-song blind sample, and no Teacher predictions used.
- Added reproducible freezer
  `scripts/freeze_va_anchor_supplement_v2.py`.

### Independent 30-Reference Annotation Console Built

- Added `scripts/build_blind_human_validation_anchored_v2_console.py` and built
  `outputs/human_validation/blind_sample_15_human_validation_console_anchored_v2_30refs.html`.
- The page contains five references per VA level, uses a new local-storage key
  and a version-specific CSV filename, and does not import prior answers.
- Static audit passed: 179 section rows across 15 songs, 179 available section
  clips, 30 available reference clips, and valid JavaScript syntax.
- The existing anchored-v1 delayed-review result remains the current formal
  result. It was collected with 12 references and is not retroactively relabeled
  as a 30-reference result. A v2 result requires a fresh 179-section annotation.

## 2026-07-26

### Same-Rater 30-Reference Review Analyzed

- The user reported relistening after a break and revising ratings with the
  expanded 30-clip reference library before model-agreement metrics were
  recomputed. Model predictions remained hidden and the Teachers remained
  frozen.
- Archived the reviewed CSV as
  `data/blind_validation/stage5/human_annotations/blind_sample_15_human_annotations_anchored_v2_30refs_review_rater_01.csv`.
- SHA-256:
  `26f7a20a2a11ff044e7e6188ec914c388a1da01d9baf117ce15ac1299b54eb09`.
- The existing Python/Pandas/SciPy evaluation pipeline was reused. Audit
  passed: 179 rows, 179 unique IDs, 15 songs, no missing categorical fields,
  and zero frozen-boundary drift.
- Relative to the 2026-07-24 delayed review, 38 sections changed at least one
  field: Valence 23, Arousal 20, change direction 0, local peak 1.
- Same-rater agreement versus the delayed review:
  - Valence exact `87.15%`, quadratic-weighted Kappa `0.858`;
  - Arousal exact `88.83%`, quadratic-weighted Kappa `0.877`;
  - change exact/Kappa `1.000`;
  - local peak exact `99.44%`, Kappa `0.987`.
- Updated model-agreement results:
  - pooled Valence/Arousal Spearman `0.614/0.576`;
  - within-song macro Valence/Arousal `0.437/0.525`;
  - between-song Valence/Arousal `0.742/0.614`;
  - nonstable change sign accuracy `88/129=68.2%`;
  - exact/adjacent top-Arousal peak hits `8/15` and `12/15`.
- Chorus-minus-Verse human lift remained positive: Valence `+0.667` in 13/15
  songs; Arousal `+0.802` in 13/15 songs. Human/model lift Spearman was `0.228`
  for Valence and `0.586` for Arousal.
- Added reproducible comparison script
  `scripts/compare_delayed_and_30ref_review.py`.
- Full report:
  `docs/BLIND_VALIDATION_HUMAN_RESULTS_30REF_REVIEW_20260726.md`.
- This is a calibrated same-rater review, not a second independent rater. The
  previous delayed-review result is retained as sensitivity evidence.

### Human Validation Result Frozen

- Added `scripts/freeze_human_validation_30refs.py` and froze the current
  calibrated same-rater result as
  `data/blind_validation/stage5/human_annotations/frozen_human_validation_30refs_rater01_20260726.json`.
- The freeze script re-audited 179 rows/15 songs, all categorical values,
  first-section change rules, the 30 model-free anchors, formal evaluation
  outputs, Stage 3 Teacher IDs/hashes, Stage 4 alignment, and Stage 5 structure
  manifests.
- Frozen paper role: primary ordinal human-validation result; the anchored-v1
  delayed review remains protocol-sensitivity evidence.
- Guardrails frozen in the manifest: Teacher unchanged, predictions hidden
  during annotation, anchors selected without Teacher predictions, no
  held-out tuning, ordinal labels are not continuous VA ground truth, and
  same-rater stability is not inter-rater reliability.
- Updated the project, research, Phase 2, pipeline, and annotation-response
  README records. Next research stage is full-corpus inference for the 99
  reviewed Chinese songs.

### Safe Storage Cleanup

- Removed only regenerable Python/test/type-check caches, macOS `.DS_Store`
  files, and the empty temporary lightweight cross-check directory.
- Approximately 208 MB of cache data was removed.
- No audio, model, environment, source dataset, annotation, experiment output,
  report, or historical result was deleted.

### Full-Corpus Frozen Teacher Inference And Structure Audit

- Froze 99 songs and 1,097 continuous section boundaries in
  `data/full_corpus/stage1/`; the stale denormalized section-count field was
  not used.
- Hash-audited and hard-linked 15 existing blind outputs; 84 songs required
  new inference.
- Frozen Teacher inference completed 99/99 songs and 49,763 0.5-second frames
  in 3,043.35 seconds. Teacher and input hashes remained fixed.
- Independent output audit passed 99/99 songs, 0 failures, 0 curve-collapse
  warnings, and exact manifest-order combined-table reconstruction within the
  `5e-8` numeric serialization tolerance.
- Local MP3 decoder warnings were retained in the run trace. The affected
  songs passed decoded-duration and later section-coverage checks.
- First alignment attempt correctly stopped on four frames equal to final
  section endpoints. The full-corpus protocol now states internal `[start,end)`
  and final `[start,end]`; rerun passed with 49,763/49,763 assigned frames,
  1,097/1,097 represented sections, 0 overlaps, and 0 empty sections.
- Supabase structural-label reveal passed for 1,097/1,097 frozen IDs with zero
  boundary mismatch and no human VA/change/climax fields read.
- All 99 prediction artifacts are retained. Chinese-language primary analysis
  excludes `ANGEL` and `Easy On Me`, yielding 97 songs, 75 artists, and 1,076
  sections.
- Core Chorus-minus-Verse analysis has 96 paired songs. Mean pseudo-VA lift is
  `+0.101` Valence and `+0.266` Arousal; all six exploratory paired tests remain
  significant after Holm correction.
- Sample-sufficiency audit: current corpus is sufficient for exploratory
  analysis; optional external-validity extension is 30 songs from 15 randomly
  selected QQ-ranking artists across three rank strata. Do not automatically
  expand to 170 songs.
- A second independent rater on the frozen 15-song validation set remains more
  important for reliability than additional one-rater VA judgments.
- Full report:
  `docs/FULL_CORPUS_INFERENCE_AND_SAMPLE_AUDIT_20260726.md`.

### Reproducible 97-Song Exploratory Analysis

- Added `scripts/run_full_corpus_exploratory_analysis.py` and the isolated
  output package `analysis/full_corpus_97/`; frozen upstream data were not
  modified.
- Audit passed for 97 songs, 75 artists, 1,076 sections, and 48,804 frames:
  zero duplicate IDs/orders, zero continuity errors, zero unrecognized types,
  and all frames assigned.
- Generated cleaned analysis copies, section mapping, 21 machine-readable
  result tables, 11 publication-oriented figures in PNG/SVG, a data audit,
  findings summary, discovery log, limitations note, and full Chinese report.
- Primary Chorus-minus-Verse song-level lift: Valence `+0.100881` (90.6% of
  songs positive) and Arousal `+0.265919` (100% positive); duration weighting
  produced essentially the same result.
- Direct Verse-to-Chorus lift: Valence `+0.053937`, Arousal `+0.138921`.
- First-to-last Chorus change: Valence `+0.087258`, Arousal `+0.208959`.
  However, in 89 songs with repeated Verse and Chorus, the Chorus-minus-Verse
  repeat-change difference was Valence `-0.025173` (raw/Holm p=`0.109/0.437`)
  and Arousal `+0.014652` (raw/Holm p=`0.596/0.934`). Interpretation changed to general late-song
  accumulation, not Chorus-specific amplification.
- Model-derived max-Arousal section was Chorus in 53/97 songs; 54.6% occurred
  in the final quarter. Largest transition landed on Outro in 36/97 songs.
- Exact full-form sequences were highly heterogeneous: 78 patterns in 97
  songs; the most frequent appeared in four songs.
- Trajectory clustering was retained as a negative result: best silhouette
  `0.151755`, subsample ARI `0.423793`, and KMeans-Ward ARI `0.334682`; no
  classification was accepted. This remains a method-audit record rather than
  a musicological conclusion about emotional-arc types.
- Visual QA completed on section distributions, normalized mean trajectory,
  cluster diagnostics, and effect-size forest plot.

### Popular-Music Interpretation Extension

- Added `scripts/analyze_musicological_implications.py`; frozen upstream data
  and prior statistical outputs were not changed.
- Generated 8 musicological tables, 3 figure sets in PNG/SVG, an auditable
  evidence matrix, and 18 case-study candidates. All CSV files are readable,
  contain no missing values, and the extension manifest status is `pass`.
- Collapsing adjacent identical labels yielded 53 full-form sequences across
  97 songs, but stable local grammar remained visible. The most frequent
  three-section motif was Chorus-Interlude-Verse in 71 songs.
- Verse-Pre-chorus-Chorus showed staged Arousal buildup; Chorus function is
  therefore framed as energetic arrival rather than unconditional happiness.
- The corpus is interpreted as cyclic late-song accumulation and partial
  release. Interlude is a rotation hinge, Bridge is contextual contrast, and
  Outro locally releases energy without fully restoring the opening state.
- The report explicitly separates data-supported formal functions from
  mechanisms requiring listening, harmony, timbre, rhythm, or production
  analysis.
- Report:
  `analysis/full_corpus_97/reports/MUSICOLOGICAL_INTERPRETATION_REPORT_ZH.md`.

### Emotional Organization And Section Functions

- Added `scripts/analyze_emotional_organization_strategies.py` and generated 9
  tables plus 3 figure sets in PNG/SVG. No frozen upstream data or prior
  results were modified.
- The descriptive threshold is `±0.05`; it was declared in code and the
  manifest and was not tuned to maximize any pattern count.
- Of 72 strict Verse-Pre-chorus-Chorus events, 47 showed staged Arousal buildup,
  15 reserved the main rise for the Chorus entry, 4 reached a Pre-chorus high
  then continued approximately level, and 1 rose then fell at Chorus.
- Across 368 Chorus entries: 111 positive releases, 89 Arousal-led intensity
  releases, 71 continuations, 50 mixed/Valence-led entries, 30 reverse entries,
  14 high-Arousal negative-Valence releases, and 3 soothing releases.
- Bridge return uses 31/35 valid events with both a prior Chorus center and a
  later Chorus; 16/31 (`51.6%`) moved closer to the prior Chorus region.
- Among comparable Intro/Outro songs, `21.1%` ended within VA distance `0.10`
  of the Intro. This is a descriptive emotional-return rule, not a form label.
- `23.6%` of 979 adjacent transitions crossed an absolute VA quadrant, but the
  negative-Valence/high-Arousal region contains only 15/1,076 sections. The
  quadrant result is calibration-sensitive and cannot support absence claims.
- `74.2%` of songs have an Arousal range at least 1.25 times their Valence
  range. `11.3%` concentrate at least half of their section-level VA path in
  the two largest transitions.
- Generated 97 continuous strategy profiles and 97 rule-based Chinese song
  portraits. Each portrait stores its numeric basis and explicitly avoids
  lyrics and causal claims about harmony, rhythm, timbre, or production.
- Manifest status: `pass`; transition count equals 1,076 sections minus 97
  songs.
- Report:
  `analysis/full_corpus_97/reports/EMOTIONAL_ORGANIZATION_AND_SECTION_FUNCTIONS_ZH.md`.

### Consolidated Full Research Report

- Added
  `reports/ei_section_aware_full_report_20260726/FULL_RESEARCH_REPORT_ZH.md`
  and its report-folder README.
- The report covers the full chain from the published OpenL3/DEAM starting
  point, strict song-level correction, MERT/DEAM experiments, PMEmo migration,
  separate final Valence/Arousal Teachers, the frozen 115-clip test,
  complete-song windowed inference, 15-song same-rater validation, and the
  97-song section-aware analysis.
- Every major interpretation now follows a fixed four-part structure: numeric
  data, emotional process, popular-music formal function, and evidence limit.
- The report explicitly identifies the 97-song corpus as researcher-collected
  convenience sampling. A scientific sampling frame was not reconstructed
  retrospectively.
- Priority open items are frozen in the report: second independent rater,
  ranking-stratified external sample or explicit convenience-sample scope,
  complete-song continuous VA validation, raw-MP3 feature-domain audit, and
  blind listening case studies with measurable musical features.
- Removed trajectory-cluster stability from substantive conclusions; the
  historical numbers remain solely as method-audit evidence.

### Chinese Paper Draft

- Added
  `reports/ei_section_aware_full_report_20260726/PAPER_DRAFT_ZH.md`.
- Reorganized the full research record into a paper structure: abstract,
  introduction, related work, data and research design, method, Teacher
  evaluation, section-aware results, discussion, validity threats, conclusion,
  declarations, references, and reproducibility appendix.
- The paper's contribution is section-aware emotional organization rather than
  a second pseudo-label prediction model. Teacher outputs are consistently
  described as pseudo-VA.
- Preserved four P0 editorial gaps: convenience sampling, second independent
  rater, complete-song continuous-VA ground truth, and raw-MP3 OpenSMILE
  feature-domain equivalence.
- Did not treat failed trajectory clustering as evidence that emotional arcs do
  not exist; clustering remains method-audit evidence only.

## 2026-08-02

### Data-Only External Analysis Package

- Created the standalone data directory:
  `exports/gpt_section_va_data_package_20260802/`.
- Created the validated ZIP archive:
  `exports/gpt_section_va_data_package_20260802.zip`.
- The package contains frozen Teacher evaluation data, corpus inventory,
  frame-level pseudo-VA, frame-section alignment, section/song features,
  human-validation data, analysis-ready tables, derived CSV/JSON tables, and
  machine-readable protocol/audit manifests.
- The package excludes audio, model weights, NPZ feature caches, figures,
  reports, paper prose, source code, and newly written analysis.
- ZIP integrity check passed with no errors.
