# Research Workspace

This directory separates completed baseline work from the current section-aware
music emotion modeling work.

## Phase 1: Published OpenL3 Baseline

Path:

```text
research/phase1_va_prediction/openl3_va_baseline
```

This contains the published OpenL3-based continuous Valence-Arousal prediction
project plus the later PyTorch MLP tuning and engineering Teacher freeze:

```text
openl3_mlp_w512_v1
three-seed mean ensemble
validation RMSE=0.099575
test RMSE=0.098425
```

See `docs/TEACHER_MODEL_CARD.md` in that folder. Its metrics use the published
frame-level random split and are not a song-level generalization estimate.
Keep this result as a published baseline and engineering reference, not as the
final complete-song Teacher.

## Phase 1C: Spectral And Feature-Fusion Ablation

Path:

```text
research/phase1c_spectral_feature_fusion_va
```

Completed pure-audio single-seed baselines:

```text
log-Mel multi-scale CNN + channel/spatial attention + BiLSTM
Librosa handcrafted features + BiLSTM
log-Mel + handcrafted fusion
```

Handcrafted features and fusion show their strongest gains on Arousal, but
DEAM-only complete-song Valence remains the main bottleneck.

## Phase 1E: Strict DEAM Teacher Optimization

Path:

```text
research/phase1e_teacher_optimization
```

This stage tests whether DEAM alone can produce a complete-song VA Teacher
under leakage-free song-level and complete-song protocols.

Current preserved conclusion:

```text
DEAM complete-song subset = 58 songs
five-fold Valence Global-Local outer-test RMSE/MAE/PCC/CCC =
0.267009/0.205546/0.463024/0.430917
best complete-song global Ridge outer-test Mean RMSE/MAE/PCC/CCC =
0.242049/0.193788/0.513339/0.393422
decision = keep as ablations, not final Teacher
```

Snapshot:

```text
research/phase1e_teacher_optimization/docs/TEACHER_PROGRESS_SNAPSHOT_20260624.md
```

## Phase 1F: External VA Dataset Integration

Path:

```text
research/phase1f_external_va_datasets
```

PMEmo integration has started because the DEAM-only complete-song subset is too
small and unstable for the target Teacher standard, especially for Valence.

Current PMEmo entry point:

```text
research/phase1f_external_va_datasets/pmemo
```

Place PMEmo raw files under `pmemo/data/raw/`, then audit before training:

```bash
.venv/bin/python research/phase1f_external_va_datasets/pmemo/scripts/audit_pmemo_dataset.py
```

Current PMEmo status:

```text
official 2019 update downloaded
dynamic_songs = 767
dynamic_frame_rows = 36434
frame_step = 0.5 s
label_range = 0 to 1
initial manifests completed under pmemo/data/processed/
```

First PMEmo-only OpenSMILE baselines are complete. The best current PMEmo-only
model is OpenSMILE dynamic features + BiLSTM:

```text
canonical scale = [-1, 1]
test RMSE/MAE/Mean PCC/Mean CCC = 0.232909/0.176988/0.704445/0.688687
Arousal test PCC/CCC = 0.837598/0.830559
Valence test PCC/CCC = 0.571292/0.546814
```

This confirms the same pattern seen in DEAM: Arousal is learnable, Valence is
the main bottleneck. The next Teacher-oriented step is MERT feature extraction
for PMEmo, then PMEmo pretraining followed by DEAM fine-tuning.

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
progression features. The frozen reviewed corpus contains 99 songs.
Blind-safe Stages 1-4 are complete: a stratified 15-song sample was frozen and
the final clean separate Valence/Arousal Teachers generated 7375 frame-level
pseudo-VA rows. All frames were then aligned to all 179 frozen sections with 0
unassigned frames, 0 overlaps, and 0 sections without frames. Stage 5 then
revealed 179/179 held-out section types: Chorus predicted Arousal exceeded
Verse in 15/15 songs. Model-blind direct validation is now frozen for one rater
after a 30-reference calibrated review of all 179 sections. Pooled
human/Teacher Spearman is `0.614/0.576` for Valence/Arousal; within-song macro
is `0.437/0.525`. The result is ordinal same-rater evidence, not continuous VA
ground truth or inter-rater reliability. Full-corpus inference is now complete:
99/99 songs, 49,763 frames, and 1,097/1,097 aligned sections passed audit. The
Chinese-language primary analysis contains 97 songs, 75 artists, and 1,076
sections after excluding two foreign-language songs from statistics while
retaining their artifacts. The current corpus is sufficient for exploratory
analysis; a 30-song ranking-stratified extension is optional for stronger
external validity. See the Stage 3-5, human-validation, and full-corpus reports
under `phase2_section_aware_va/docs/`.

The reproducible 97-song exploratory analysis is complete under
`phase2_section_aware_va/analysis/full_corpus_97/`. Its strongest result is a
Chorus-minus-Verse lift of `+0.101` Valence and `+0.266` Arousal. Repeated
Chorus increases do not exceed repeated Verse increases after a paired
difference-in-differences check, so the current interpretation is general
late-song accumulation rather than Chorus-specific amplification. Trajectory
clustering did not meet the preregistered descriptive stability threshold and
is retained only as a method-audit result, not a musicological conclusion.
The follow-up musicological layer is recorded in
`phase2_section_aware_va/analysis/full_corpus_97/reports/MUSICOLOGICAL_INTERPRETATION_REPORT_ZH.md`.
It reframes the same frozen evidence in terms of formal function, local
Verse/Pre-chorus/Chorus teleology, rotation, repetition with variation,
late-song accumulation, Bridge contrast, and Outro release while preserving
explicit evidence boundaries.
The next descriptive layer separates buildup, release, deviation/return,
closure, two-dimensional VA movement, repeated-section reinterpretation, and
continuous song-level organization strategies. It also generates a traceable
rule-based portrait for every song; see
`phase2_section_aware_va/analysis/full_corpus_97/reports/EMOTIONAL_ORGANIZATION_AND_SECTION_FUNCTIONS_ZH.md`.
The complete cross-phase Chinese research report, including model history,
final Teacher rigor, full-song deployment, human validation, layered
data-emotion-music-function conclusions, and explicit open items, is stored at
`../reports/ei_section_aware_full_report_20260726/FULL_RESEARCH_REPORT_ZH.md`.
The paper-form Chinese manuscript is stored at
`../reports/ei_section_aware_full_report_20260726/PAPER_DRAFT_ZH.md`.
