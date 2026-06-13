# Phase 1B Pipeline

Last updated: 2026-06-13

## Objective

Train a new first-stage VA predictor using DEAM and a BERT-style music/audio
encoder. The trained model will replace the old OpenL3-based VA output layer
for the later section-aware Phase 2 pipeline.

## Stage 1: Data Audit

Inputs are read from the completed Phase 1 DEAM copy:

```text
research/phase1_va_prediction/openl3_va_baseline/data/DEAM/MEMD_audio
research/phase1_va_prediction/openl3_va_baseline/data/DEAM/annotations/annotations averaged per song/dynamic (per second annotations)/valence.csv
research/phase1_va_prediction/openl3_va_baseline/data/DEAM/annotations/annotations averaged per song/dynamic (per second annotations)/arousal.csv
```

DEAM dynamic annotations are wide CSV tables. Columns are named like
`sample_15000ms`, `sample_15500ms`, and continue at 0.5 second intervals. The
training target is therefore already compatible with the requested 0.5s frame
resolution.

Stage 1 produces:

```text
data/manifests/data_manifest.csv
data/manifests/split_manifest.csv
data/manifests/audit_summary.json
```

The split is song-level, not frame-level, to avoid leakage across train,
validation, and test.

## Stage 2: BERT-Style Audio Feature Extraction

Default encoder:

```text
m-a-p/MERT-v1-95M
```

Planned Hugging Face loading pattern:

```python
from transformers import AutoModel, Wav2Vec2FeatureExtractor

model = AutoModel.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True)
processor = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True)
```

Expected feature output:

```text
data/features_mert/<song_id>.npz
```

Each feature file should contain:

```text
song_id
time_sec
embedding
```

Current extraction implementation:

```text
scripts/extract_mert_features.py
configs/stage2_mert_features.json
```

The script uses chunked inference to reduce memory pressure. For each DEAM
song, it pools MERT hidden states into the DEAM dynamic-label timeline covered
by the actual audio:

```text
start=15.0s
hop=0.5s
max_label_frames=1223
```

The dynamic-label CSVs contain 1223 possible sample columns, but many DEAM
audio files are 45-second excerpts. The feature file therefore stores a
variable-length sequence:

```text
embedding: (covered_label_frames, hidden_dim)
```

Most excerpt songs produce 60 or 61 frames. Longer files produce more frames.
The matching labels must be selected by each file's `time_sec` values during
dataset construction.

Small test command:

```bash
.venv/bin/python research/phase1b_bert_deam_va_training/scripts/extract_mert_features.py --split train --limit 1
```

Feature inventory command:

```bash
.venv/bin/python research/phase1b_bert_deam_va_training/scripts/audit_mert_features.py
```

The extraction step must record:

- model name,
- transformers version,
- sampling rate,
- layer selection,
- pooling strategy,
- frame hop,
- extraction failures.

Dependency policy:

- do not install or use OpenL3,
- use PyTorch, librosa, soundfile, and transformers,
- if MERT loading fails with a newer transformers version, use
  `transformers<=4.44.0`.

Stage 2 final inventory:

```text
feature_files=1802/1802
feature_dim=768
frame_count_min=60
frame_count_max=1223
train_features=1441
val_features=180
test_features=181
```

## Stage 3: Model Training And Tuning

Models to train:

- MLP,
- BiLSTM,
- Transformer.

Each model family starts with a baseline config, then receives additional
hyperparameter trials if validation metrics are unstable or clearly improvable.
Trials are decided pragmatically from validation RMSE and Pearson r.

Primary metrics:

- RMSE,
- MSE,
- Pearson r for Valence,
- Pearson r for Arousal.

## Stage 4: Best Model Selection

Selection priority:

1. lowest test RMSE,
2. stable Valence and Arousal Pearson r,
3. smooth and musically plausible VA curves if numeric metrics are close.

Outputs:

```text
outputs/results/model_comparison.csv
outputs/results/best_model_report.md
outputs/models/best_va_model.pt
```

## Stage 5: Phase 2 Export

The selected first-stage model generates frame-level VA for reviewed full
songs. Phase 2 then aligns those frame-level predictions to human-reviewed
section boundaries and section types.
