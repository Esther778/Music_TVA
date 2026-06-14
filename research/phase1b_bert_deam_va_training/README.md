# Phase 1B BERT-Style DEAM VA Training

Status: stage 3 complete; best current VA head is `bilstm_baseline`.

This folder is an independent retraining workspace. It does not modify the
completed OpenL3 baseline in `research/phase1_va_prediction/openl3_va_baseline`.

## Goal

Retrain the first-stage continuous Valence-Arousal predictor on DEAM using a
BERT-style music/audio representation, then compare MLP, BiLSTM, and
Transformer VA heads.

The default encoder for the next implementation stage is:

```text
m-a-p/MERT-v1-95M
```

This is treated as the BERT-style audio backbone for the project. Text BERT is
not used because the DEAM supervision target is aligned to audio, not lyrics.

## Five Stages

1. Data audit and independent workspace setup.
2. MERT feature extraction from DEAM audio at 0.5s frame resolution.
3. MLP, BiLSTM, and Transformer training with controlled hyperparameter trials.
4. Model comparison and best-model selection.
5. Export the selected model for Phase 2 section-aware VA generation.

## Stage 2 MERT Feature Extraction

Stage 2 files:

```text
configs/stage2_mert_features.json
scripts/extract_mert_features.py
scripts/audit_mert_features.py
requirements-mert.txt
```

The feature extractor saves one file per song:

```text
data/features_mert/<song_id>.npz
```

Each file is aligned to the DEAM dynamic VA target timeline:

```text
0.5s hop, starting at 15.0s, ending at the available audio duration
```

The DEAM dynamic-label table has 1223 possible columns, but many DEAM audio
files are 45-second excerpts. The extractor therefore keeps only label frames
covered by the actual audio. Most excerpt files produce 60 or 61 frames; longer
files produce more frames.

## Current State

Stage 1 created:

```text
data/manifests/data_manifest.csv
data/manifests/split_manifest.csv
data/manifests/audit_summary.json
docs/PIPELINE.md
docs/RUN_LOG.md
```

Current audit result:

- DEAM audio files: 1802
- Valence label songs: 1802
- Arousal label songs: 1802
- Common usable songs: 1802
- Missing audio or labels: 0
- Dynamic labels start at 15000 ms and use 500 ms hop.
- Song-level split: train 1441, validation 180, test 181.

Stage 2 extraction is now complete:

- MERT feature files: 1802 / 1802
- Feature dimension: 768
- Frame count range: 60 to 1223
- Split coverage: train 1441, validation 180, test 181

Stage 3 training is now complete:

- Dataset after invalid-label filtering: train 101248 frames, validation 13427
  frames, test 15320 frames.
- Trained trial count: 9, covering MLP, BiLSTM, and Transformer heads.
- Current best model by test RMSE: `bilstm_baseline`.
- Best validation RMSE: 0.219934.
- Best test RMSE: 0.202452.
- Test Pearson r: Valence 0.511522, Arousal 0.641069.
- Best checkpoint is local-only under
  `outputs/models/bilstm/bilstm_baseline_best.pt`.

Committed result summary:

```text
docs/STAGE3_RESULTS.md
```

Local generated outputs:

```text
data/datasets/mert_va/
outputs/metrics/
outputs/models/
outputs/predictions/
outputs/results/
```

These generated outputs are intentionally ignored by git, except for the
tracked narrative summaries in `docs/`.

## Important Constraint

The previous OpenL3 MLP/BiLSTM/Transformer checkpoints cannot be directly
reused after switching to MERT/BERT-style features. The input representation is
different, so the VA heads must be retrained and compared again.
