# Phase 2 Section-Aware VA Pipeline

Last updated: 2026-07-26

## Objective

Use frozen audio-only VA Teachers to generate complete-song pseudo-VA curves,
align them with human-reviewed musical sections, and quantify emotional
progression, section contrast, and climax organization. The scientific main
line is interpretable structure analysis, not pseudo-label retraining.

Current paper scope: human-created Chinese popular music only. The former AIGC
comparison plan was removed on 2026-07-21.

## Frozen Teacher

```text
Valence=pmemo_final_clean_valence_teacher_v1
Arousal=pmemo_final_clean_arousal_teacher_v1
features=MERT-v1-95M hidden-6 + PMEmo-style OpenSMILE dynamic 260d
architecture=separate Global-Local BiGRU
frame hop=0.5 s
complete-song deployment=30 s window, 15 s stride, Hann overlap-add
```

Outputs are pseudo-VA, not human ground truth. Raw-MP3 OpenSMILE extraction is
not bit-equivalent to PMEmo's released feature CSV, so all complete-song
conclusions require external validation.

## Current Corpus

```text
reviewed Chinese songs=99
reviewed continuous section boundaries=1097
non-Chinese songs excluded
```

Older `phase2_initial_70_*` files are historical and are not the current blind
validation path.

## Blind Validation Stages

### Stage 1: Input Audit

- Refresh only song identity/audio metadata and section boundaries.
- Exclude section types, lyrics, reviews, VA answers, and climax answers.
- Verify local audio and continuous, non-overlapping boundaries.

Result: 99/99 songs passed after correcting one boundary gap.

### Stage 2: Sample Freeze

- Deterministic seed `20260720`.
- Five duration strata by three section-count strata.
- One song per cell, with unique title and artist.
- Freeze audio and boundary hashes.

Result: 15 songs and 179 boundaries frozen.

### Stage 3: Audio-Only Pseudo-VA

- Verify frozen audio and Teacher hashes.
- Extract MERT/OpenSMILE features from MP3.
- Predict 0.5-second VA using the frozen 30/15/Hann deployment rule.
- Do not read section boundaries, types, or human answers.

Result: 7375 frames, 15/15 songs passed, 0 collapse warnings.

### Stage 4: Boundary Alignment

- Reveal only frozen boundary intervals.
- Assign frame center `t` using `start <= t < end`.
- Extract section and song structure features.
- Keep section types and human emotion answers hidden.

Result: 7375/7375 frames assigned to 179/179 sections, no overlap or missing
section.

### Stage 5A: Held-Out Structural Analysis

- Reveal section types only after Stage 3/4 outputs are frozen.
- Join by frozen `human_annotation_id` and reject boundary drift.
- Use song-level paired comparisons, not section rows as independent samples.
- Report positive and negative results without Teacher tuning.

Result: strong within-song Chorus Arousal lift; no evidence that repeated
Chorus is more VA-consistent than repeated Verse. This is indirect structural
evidence, not direct human VA accuracy.

### Stage 5B: Direct Human Emotion Validation

Current status: the 30-reference calibrated rater-01 review is complete and
frozen for 15/15 songs and 179/179 sections. A second independent rater and
inter-rater reliability remain unavailable.

Final same-rater annotation and freeze manifest:

```text
data/blind_validation/stage5/human_annotations/blind_sample_15_human_annotations_anchored_v2_30refs_review_rater_01.csv
data/blind_validation/stage5/human_annotations/frozen_human_validation_30refs_rater01_20260726.json
```

Primary model-agreement results are pooled Valence/Arousal Spearman
`0.614/0.576`, within-song macro `0.437/0.525`, and nonstable Arousal-change
sign accuracy `88/129=68.2%`. These are one-rater ordinal validation results,
not continuous-VA RMSE/CCC and not inter-rater reliability. The frozen Teacher
must not be tuned using these results.

Fill the model-blind 179-row sheet with:

```text
human_valence_level=low|mid|high
human_arousal_level=low|mid|high
human_change_from_prev=up|down|stable
is_climax=0|1 (zero, one, or multiple positive sections per song)
```

The 12-reference delayed review is retained as protocol-sensitivity evidence.
The next pipeline step is full-corpus inference and section-structure feature
extraction for all 99 reviewed Chinese songs.

## Full-Corpus Stages - Completed 2026-07-26

1. `build_full_corpus_manifest.py` froze 99 songs and 1,097 boundaries, then
   hash-verified and hard-linked the 15 existing blind outputs.
2. `predict_full_corpus_final_teacher.py` produced 49,763 complete-song frames
   with the frozen Teachers. Runtime was 3,043.35 seconds; 99/99 songs passed.
3. `audit_full_corpus_outputs.py` verified hashes, model IDs, 0.5-second time
   axes, decoded duration, finite range, curve variance, and combined-table
   reconstruction. Result: pass, 0 collapse warnings.
4. `align_full_corpus_sections.py` aligned all 49,763 frames to all 1,097
   sections. Internal intervals are `[start,end)` and final intervals are
   `[start,end]`; four exact endpoint frames were included in final sections.
5. `fetch_full_corpus_section_types.py` revealed 1,097/1,097 types only after
   alignment passed, with zero ID or boundary mismatch and no human VA fields.
6. `analyze_full_corpus_structure.py` retained all predictions but excluded
   `ANGEL` and `Easy On Me` from the 97-song Chinese-language primary analysis.
7. `assess_full_corpus_sample_sufficiency.py` added Holm correction and froze
   the sample-expansion assessment. Current exploratory analysis is sufficient;
   optional stronger external validity uses a 30-song QQ-ranking stratified
   extension, not the full 170-song 3/2/1 expansion.

Primary full-corpus report:

```text
docs/FULL_CORPUS_INFERENCE_AND_SAMPLE_AUDIT_20260726.md
```

## Stage 6: Reproducible 97-Song Exploratory Analysis - Completed 2026-07-26

1. Audited 97 songs, 1,076 sections, and 48,804 in-scope frames; found no
   duplicate IDs, order errors, discontinuities, unapproved rows, or unknown
   section types.
2. Preserved source files and generated analysis copies with raw/normalized
   section labels, occurrence indices, previous/next types, within-song
   centered/z-scored VA, and duration weights.
3. Used song-level paired tests, 5,000-draw song bootstrap intervals, effect
   sizes, and Holm correction. Section-row plots are descriptive only.
4. Added transition direction sensitivity at thresholds 0.03/0.05/0.07,
   repeated-section difference-in-differences, three separate peak definitions,
   song-level path/entropy features, counterexamples, and exact-form sequences.
5. Compared KMeans and Ward trajectory clustering for k=2-5 with 100 80%
   subsamples. The best silhouette `0.152` and stability ARI `0.424` fail the
   frozen threshold. This records that the preset classification attempt was
   not validated; it is not treated as a substantive finding about emotional
   arc types.
6. Main interpretation: Chorus has a strong Arousal-led lift over Verse, but
   later Chorus growth is not greater than later Verse growth. The evidence
   supports general late-song accumulation rather than Chorus-specific repeat
   amplification.

Outputs and full Chinese report:

```text
analysis/full_corpus_97/
analysis/full_corpus_97/reports/EXPLORATORY_ANALYSIS_REPORT_ZH.md
```

## Stage 7: Popular-Music Interpretation Layer - Completed 2026-07-26

1. Added `scripts/analyze_musicological_implications.py` without modifying the
   frozen full-corpus data or Stage 6 statistical outputs.
2. Reconstructed adjacent-same-label-collapsed forms and local three-section
   motifs. The corpus contains 53 collapsed full forms, while recurring local
   patterns such as Chorus-Interlude-Verse occur across many songs.
3. Reframed Verse-Pre-chorus-Chorus as an Arousal-led
   initiation-buildup-arrival trajectory, not a universal increase in
   positive Valence.
4. Interpreted repeated Chorus growth as part of general late-song
   accumulation because its increase does not exceed repeated Verse growth.
5. Treated Interlude as a frequent rotation hinge, Bridge as contextual
   contrast without a fixed VA direction, and Outro as local release with
   incomplete return to the opening state.
6. Added an evidence matrix separating direct support, indirect support, and
   listening-analysis candidates. Harmony, timbre, rhythm, and production
   mechanisms remain outside the present evidence.
7. Generated 8 tables, 3 PNG/SVG figure sets, 18 auditable case-study
   candidates, and a Chinese musicological interpretation report. Manifest
   status: pass.

Primary report:

```text
analysis/full_corpus_97/reports/MUSICOLOGICAL_INTERPRETATION_REPORT_ZH.md
```

## Stage 8: Emotional Organization And Section Functions - Completed 2026-07-26

1. Added `scripts/analyze_emotional_organization_strategies.py` as a
   descriptive extension. It adds no inferential tests and does not assign
   songs to fixed classes.
2. Froze `±0.05` as a transparent descriptive threshold for local rise,
   decline, or approximate stability. The threshold is not a psychological
   boundary and was not optimized against the observed category counts.
3. Split 72 strict Verse-Pre-chorus-Chorus events into buildup processes and
   368 Chorus entries into two-dimensional release functions.
4. Added 35 Bridge deviation/return events, 94 Outro closure events, and 979
   adjacent VA-region transitions. Bridge-return denominators exclude events
   without both a prior Chorus center and a later Chorus.
5. Added 34 continuous song-level strategy fields covering baseline position,
   V/A range, contrast, accumulation, repetition consistency, return,
   transition concentration, separate peak positions, closure, and V/A
   coupling.
6. Generated 97 rule-based Chinese song portraits with a numeric evidence
   field and an explicit pseudo-VA interpretation boundary.
7. Absolute VA quadrants are retained only as a calibration-sensitive aid:
   negative-Valence/high-Arousal contains 15/1,076 sections and is not used for
   corpus-wide absence claims.
8. Audit passed: 97 unique strategy profiles, 97 unique portraits, and 979
   transitions equal the expected section-minus-song count.

Primary report:

```text
analysis/full_corpus_97/reports/EMOTIONAL_ORGANIZATION_AND_SECTION_FUNCTIONS_ZH.md
```

## Stage 9: Consolidated Research Report - Completed 2026-07-26

1. Added the full Chinese research report at
   `reports/ei_section_aware_full_report_20260726/FULL_RESEARCH_REPORT_ZH.md`.
2. Integrated the experiment history from OpenL3/DEAM through MERT, PMEmo,
   separate Valence/Arousal global-local BiGRU Teachers, final frozen test,
   complete-song inference, blind validation, and 97-song analysis.
3. Standardized all substantive findings as
   data evidence -> emotional process -> popular-music function -> evidence
   boundary.
4. Marked the 97-song corpus as a convenience sample; no retrospective random
   sampling claim or invented sampling protocol was added.
5. Registered priority open items: scientific sampling, a second independent
   rater, complete-song continuous VA ground truth, raw-MP3 OpenSMILE domain
   equivalence, and lower-level musical-mechanism validation.

## Stage 10: Chinese Paper Draft - Completed 2026-07-26

1. Added the journal-style Chinese manuscript at
   `reports/ei_section_aware_full_report_20260726/PAPER_DRAFT_ZH.md`.
2. Positioned the frozen Teachers as enabling measurement models and the
   section-aware emotional-organization analysis as the paper's main
   contribution.
3. Preserved explicit editorial markers for convenience sampling, second-rater
   validation, complete-song continuous VA ground truth, and raw-MP3 feature
   equivalence.
4. Kept trajectory clustering as method-audit evidence only; the paper does
   not claim that fixed emotional-arc types exist or do not exist.
6. Trajectory clustering remains only a method-audit record and is excluded
   from substantive musicological conclusions.

## Reproducible Entry Points

```bash
.venv/bin/python research/phase2_section_aware_va/scripts/predict_blind_sample_final_teacher.py
.venv/bin/python research/phase2_section_aware_va/scripts/audit_blind_stage3_outputs.py
.venv/bin/python research/phase2_section_aware_va/scripts/align_blind_stage4_sections.py
.venv/bin/python research/phase2_section_aware_va/scripts/fetch_blind_stage5_section_types.py
.venv/bin/python research/phase2_section_aware_va/scripts/analyze_blind_stage5_section_types.py
.venv/bin/python research/phase2_section_aware_va/scripts/build_blind_stage5_human_validation_template.py
.venv/bin/python research/phase2_section_aware_va/scripts/compare_delayed_and_30ref_review.py
.venv/bin/python research/phase2_section_aware_va/scripts/freeze_human_validation_30refs.py
MPLBACKEND=Agg .venv/bin/python research/phase2_section_aware_va/scripts/run_full_corpus_exploratory_analysis.py
MPLBACKEND=Agg .venv/bin/python research/phase2_section_aware_va/scripts/analyze_musicological_implications.py
MPLBACKEND=Agg .venv/bin/python research/phase2_section_aware_va/scripts/analyze_emotional_organization_strategies.py
```

For figures, use `MPLBACKEND=Agg` with
`scripts/plot_blind_stage5_results.py`.

## Scientific Guardrails

- Never describe pseudo-VA as ground truth.
- Never tune the Teacher using the 15 held-out songs.
- Keep the song as the statistical unit for section-type comparisons.
- Report sample count, confidence interval, effect size, and negative results.
- Do not call maximum predicted Arousal a correct climax without human labels.
- Do not reintroduce AIGC comparisons into the current paper without a separate
  preregistered corpus design and an explicit scope revision.
