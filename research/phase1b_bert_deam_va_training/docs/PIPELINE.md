# Phase 1B Pipeline

Last updated: 2026-06-21

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
- MAE,
- CCC,
- Valence RMSE / Arousal RMSE,
- Valence MAE / Arousal MAE,
- Pearson r for Valence,
- Pearson r for Arousal.

Current selection rule:

```text
train split: model fitting only
validation split: model and hyperparameter selection
test split: final reporting only, never used for tuning
```

Primary selection metric is validation RMSE. If validation RMSE is close, use
validation CCC, Valence/Arousal balance, train-validation gap, and model
complexity as secondary criteria.

Stage 3 implementation:

```text
configs/stage3_training_plan.json
scripts/build_mert_va_datasets.py
scripts/train_mert_mlp.py
scripts/train_mert_sequence.py
scripts/aggregate_training_results.py
scripts/training_utils.py
```

Dataset construction aligns each feature file's `time_sec` values back to the
DEAM dynamic label columns. Frames with non-finite features or labels are
dropped before training and counted in `dataset_summary.json`.

Final dataset:

```text
train_frames=101248
val_frames=13427
test_frames=15320
train_dropped_invalid_frames=1451
val_dropped_invalid_frames=181
test_dropped_invalid_frames=182
feature_dim=768
```

Training used CPU in this environment. A direct Torch check reported MPS as
available, but the long-running training script process did not reliably expose
MPS, so Stage 3 was completed on CPU to preserve a stable training run.

Completed trials:

```text
MLP: mlp_baseline, mlp_wide, mlp_light_dropout
BiLSTM: bilstm_baseline, bilstm_deep, bilstm_wide
Transformer: transformer_baseline, transformer_deep, transformer_wide
```

RMSE reduction trials are planned for all three families, not only BiLSTM:

```text
MLP: stronger dropout, smaller/wider heads, loss weighting, gradient clipping
BiLSTM: hidden size 64/96/128, dropout 0.35, weight decay, lower LR, smoothing loss
Transformer: stronger dropout/weight decay, smaller model, smoothing loss
```

Result aggregation command:

```bash
.venv/bin/python research/phase1b_bert_deam_va_training/scripts/aggregate_training_results.py
```

Tracked summary:

```text
docs/STAGE3_RESULTS.md
```

## Stage 4: Best Model Selection

Selection priority:

1. lowest validation RMSE,
2. higher validation CCC,
3. stable Valence and Arousal metrics,
4. smaller train-validation gap,
5. lower model complexity if metrics are close.

Outputs:

```text
outputs/results/model_comparison.csv
outputs/results/best_model_report.md
outputs/models/best_va_model.pt
```

Current Stage 4 candidate from Stage 3:

```text
model_family=bilstm
trial_id=bilstm_baseline
best_epoch=42
val_rmse=0.219934
test_rmse=0.202452
test_pearson_valence=0.511522
test_pearson_arousal=0.641069
checkpoint=outputs/models/bilstm/bilstm_baseline_best.pt
```

This candidate remains provisional until the RMSE-reduction trials and multi-seed
stability checks are complete.

The next RMSE improvement work is specified in:

```text
docs/RMSE_IMPROVEMENT_PLAN.md
```

The plan prioritizes annotation-lag calibration, multi-scale context,
song-balanced loss, MERT layer selection, and limited encoder fine-tuning over
another large round of head-only hyperparameter changes.

## Stage 5: Phase 2 Export

The selected first-stage model generates frame-level VA for reviewed full
songs. Phase 2 then aligns those frame-level predictions to human-reviewed
section boundaries and section types.
