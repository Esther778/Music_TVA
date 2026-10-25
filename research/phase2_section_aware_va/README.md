# Phase 2 Section-Aware VA Modeling

Status: blind human validation frozen; full-corpus inference, section
alignment, structure extraction, and sample-sufficiency audit complete.

## Goal

Build a section-aware complete-song emotion structure modeling pipeline on top
of the completed Phase 1 VA prediction baseline.

The model design is two-layer:

1. Frame-level VA prediction:
   - input: 0.5 second audio features,
   - output: continuous Valence-Arousal curve.
2. Section-aware emotion structure analysis:
   - input: frame-level VA curve plus reviewed section boundaries and section
     types,
   - output: section-level emotional progression features for complete songs.

The research target is to characterize emotional progression, climax
organization, and section contrast in human-created Chinese popular music.
AIGC comparison was removed from the paper scope on 2026-07-21 and from the
graduation-thesis scope on 2026-10-25.

## Current Data Rule

- The frozen reviewed corpus contains 99 songs after removing `Love Story`.
  All 99 retain prediction artifacts; `ANGEL` and `Easy On Me` are excluded
  from the Chinese-language primary analysis, leaving 97 in-scope songs.
- Existing local `phase2_initial_70_*` manifests are stale and should be
  refreshed before pseudo-VA generation.
- Intermediate data must be preserved, including manifests, frame-level VA
  predictions, section-aligned VA tables, model checkpoints, metrics, and
  plots.
- Phase 2 outputs are pseudo-VA. They must not be described as independent
  human VA ground truth.

## Directory Layout

```text
configs/                  experiment configs
docs/                     design notes and run logs
scripts/                  manifest, VA prediction, alignment, training scripts
data/manifests/           reviewed-song lists and train/val/test splits
data/blind_validation/    frozen 15-song validation protocol and outputs
data/full_corpus/         frozen 99-song inference, alignment, and analysis
data/frame_va/            later full-corpus 0.5-second pseudo-VA predictions
data/section_va/          frame VA joined with section boundaries/types
data/training_batches/    batch-by-batch training data snapshots
outputs/models/           model checkpoints
outputs/metrics/          per-batch and final metrics
outputs/predictions/      final frame-level and section-level predictions
outputs/plots/            diagnostic plots
```

## Current Run State

- Stage 1 passed on the 99-song reviewed Chinese corpus: 99 local MP3 files,
  1097 continuous section boundaries, and no invalid/gap/overlap rows.
- Stage 2 froze a deterministic 15-song duration-by-section-count sample with
  15 unique artists and 179 boundaries.
- Stage 3 used the frozen final clean Valence and Arousal Teachers with MERT
  hidden-6 plus PMEmo-style OpenSMILE features.
- Complete-song deployment uses fixed 30-second windows, 15-second stride, and
  Hann overlap-add because direct multi-minute inference collapses outside the
  Teacher's short-clip training distribution.
- Stage 3 produced 7375 pseudo-VA frames for 15/15 songs. Independent audit:
  15 passed, 0 failed, 0 collapse warnings.
- Stage 4 aligned all 7375 frames to all 179 frozen sections: 0 unassigned
  frames, 0 overlaps, and 0 sections without frames.
- Stage 4 produced frame-section, section-feature, and song-structure tables
  without reading section types or human emotion/climax answers.
- Stage 5 revealed 179/179 approved section types after outputs were frozen.
  Chorus predicted Arousal exceeded Verse in 15/15 songs; predicted Valence did
  so in 13/15 songs.
- Stage 5B direct validation is complete for one rater after a model-blind,
  30-reference calibrated review of all 179 sections.
- Pooled human/Teacher Spearman is `0.614` for Valence and `0.576` for Arousal;
  within-song macro Spearman is `0.437/0.525`.
- The frozen primary annotation manifest is
  `data/blind_validation/stage5/human_annotations/frozen_human_validation_30refs_rater01_20260726.json`.
- The 12-reference delayed review remains sensitivity evidence. Same-rater
  stability is not inter-rater reliability, and ordinal ratings are not
  continuous VA ground truth.
- The frozen Teachers must not be tuned using these held-out results.
- Full-corpus inference is complete: 99/99 songs, 49,763 frames, 0 failures,
  and 0 curve-collapse warnings. Fifteen songs were hash-verified and reused.
- Full-corpus alignment covers 49,763/49,763 frames and 1,097/1,097 sections,
  with 0 unassigned frames, 0 overlaps, and 0 empty sections.
- The Chinese-language primary set contains 97 songs, 75 artists, and 1,076
  sections. Chorus-minus-Verse paired coverage is 96 songs.
- The current set is sufficient for exploratory analysis. A 30-song frozen
  QQ-ranking stratified extension is recommended only to strengthen external
  validity; a second independent rater remains the higher validation priority.
- Full report: `docs/FULL_CORPUS_INFERENCE_AND_SAMPLE_AUDIT_20260726.md`.
- The complete 97-song exploratory package is frozen under
  `analysis/full_corpus_97/`: cleaned analysis copies, 37 CSV result tables,
  17 figures in PNG/SVG, audit, discovery log, limitations, statistical report,
  and a separate Chinese musicological interpretation report.
- Chorus-minus-Verse lift is `+0.101` Valence and `+0.266` Arousal. The later
  Chorus increase is not greater than the later Verse increase, so the current
  evidence supports general late-song accumulation rather than a unique
  repeated-Chorus effect.
- Trajectory clustering remains a method-audit result only: the tested
  preset classification did not pass its stability threshold and is not used
  as a substantive musicological conclusion.
- The musicological extension interprets the corpus as shared local formal
  grammar with whole-song diversity, Arousal-led Chorus arrival, cyclic
  late-song accumulation, contextual Bridge contrast, and partial Outro
  release. These are bounded interpretations of pseudo-VA evidence, not direct
  claims about harmony, timbre, rhythm, or production mechanisms.
- The deeper emotional-organization layer separates Pre-chorus buildup,
  Chorus release, Bridge deviation/return, Outro closure, and VA-region
  movement; it also provides continuous strategy profiles and traceable
  Chinese portraits for all 97 songs. It introduces no new inferential tests
  and no fixed song classes.
- A consolidated research report from Phase 1 model development through Phase
  2 popular-music interpretation is available at
  `../../reports/ei_section_aware_full_report_20260726/FULL_RESEARCH_REPORT_ZH.md`.
  It marks convenience sampling, second-rater validation, complete-song
  ground truth, raw-MP3 feature equivalence, and musical-mechanism analysis as
  explicit open items rather than completed evidence.
- The corresponding journal-style Chinese manuscript is available at
  `../../reports/ei_section_aware_full_report_20260726/PAPER_DRAFT_ZH.md`.

## Research Writing Notes

The EI paper feasibility review, core methodological risks, and recommended
writing structure are recorded in:

```text
docs/EI_PAPER_METHOD_REVIEW.md
```

Important current note: the Phase 2 scientific framing remains section-aware
structure analysis. The executable Teachers are frozen, but raw-MP3 OpenSMILE
features have a measured domain gap from PMEmo's released feature CSV. Treat
their complete-song outputs as pseudo-VA and validate section-level trends
against human judgments before drawing structural conclusions.

## Script Entry Points

```bash
.venv/bin/python research/phase2_section_aware_va/scripts/predict_blind_sample_final_teacher.py
.venv/bin/python research/phase2_section_aware_va/scripts/audit_blind_stage3_outputs.py
.venv/bin/python research/phase2_section_aware_va/scripts/align_blind_stage4_sections.py
.venv/bin/python research/phase2_section_aware_va/scripts/analyze_blind_stage5_section_types.py
.venv/bin/python research/phase2_section_aware_va/scripts/analyze_blind_human_validation.py --human-csv research/phase2_section_aware_va/data/blind_validation/stage5/human_annotations/blind_sample_15_human_annotations_anchored_v2_30refs_review_rater_01.csv --output-dir research/phase2_section_aware_va/outputs/human_validation/rater_01_anchored_v2_30refs_review_evaluation
.venv/bin/python research/phase2_section_aware_va/scripts/compare_delayed_and_30ref_review.py
.venv/bin/python research/phase2_section_aware_va/scripts/freeze_human_validation_30refs.py
.venv/bin/python research/phase2_section_aware_va/scripts/build_full_corpus_manifest.py
.venv/bin/python research/phase2_section_aware_va/scripts/predict_full_corpus_final_teacher.py
.venv/bin/python research/phase2_section_aware_va/scripts/audit_full_corpus_outputs.py
.venv/bin/python research/phase2_section_aware_va/scripts/align_full_corpus_sections.py
.venv/bin/python research/phase2_section_aware_va/scripts/fetch_full_corpus_section_types.py
.venv/bin/python research/phase2_section_aware_va/scripts/analyze_full_corpus_structure.py
.venv/bin/python research/phase2_section_aware_va/scripts/assess_full_corpus_sample_sufficiency.py
MPLBACKEND=Agg .venv/bin/python research/phase2_section_aware_va/scripts/run_full_corpus_exploratory_analysis.py
MPLBACKEND=Agg .venv/bin/python research/phase2_section_aware_va/scripts/analyze_musicological_implications.py
MPLBACKEND=Agg .venv/bin/python research/phase2_section_aware_va/scripts/analyze_emotional_organization_strategies.py
```

Detailed provenance is in:

```text
docs/BLIND_VALIDATION_STAGE3_INFERENCE_20260720.md
docs/BLIND_VALIDATION_STAGE4_ALIGNMENT_20260721.md
docs/BLIND_VALIDATION_STAGE5_STRUCTURAL_ANALYSIS_20260721.md
docs/FULL_CORPUS_INFERENCE_AND_SAMPLE_AUDIT_20260726.md
```
