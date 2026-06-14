# Phase 1B Run Log

This log records the BERT-style/MERT + DEAM retraining workflow.

## 2026-06-13

### User Decision

- Rebuild the first-stage VA predictor in a new folder.
- Do not modify the earlier OpenL3 baseline.
- Replace the old OpenL3 feature extractor with a BERT-style music/audio
  representation, using MERT as the default backbone.
- Retrain MLP, BiLSTM, and Transformer on DEAM.
- Compare tuned versions and choose the best model for Phase 2 VA output.

### Stage 1 Plan

1. Create independent workspace.
2. Audit DEAM audio and dynamic VA labels.
3. Generate song-level manifest.
4. Generate song-level train/validation/test split manifest.
5. Record the exact data interpretation before feature extraction or training.

### Stage 1 Completed

Created independent workspace:

```text
research/phase1b_bert_deam_va_training
```

Generated:

```text
data/manifests/data_manifest.csv
data/manifests/split_manifest.csv
data/manifests/audit_summary.json
```

Audit summary:

```text
audio_count=1802
valence_song_count=1802
arousal_song_count=1802
common_song_count=1802
missing_audio_for_labels=0
audio_without_labels=0
label_start_ms=15000
label_end_ms=626000
label_hop_ms=500
label_frames=1223
split_seed=42
train=1441
val=180
test=181
```

Decision recorded: the split is song-level rather than frame-level, because
frame-level random splitting would leak the same song into train, validation,
and test.

### Next Stage

Stage 2 will add MERT feature extraction code. It will require `transformers`
and Hugging Face model loading for `m-a-p/MERT-v1-95M`. No OpenL3 installation
should be attempted.

### Stage 2 Started

Added:

```text
configs/stage2_mert_features.json
scripts/extract_mert_features.py
scripts/audit_mert_features.py
requirements-mert.txt
```

Current dependency check:

```text
torch=2.8.0
librosa=0.11.0
soundfile=0.13.1
pandas=2.3.3
transformers=missing
```

The extraction script is designed to:

- load `m-a-p/MERT-v1-95M` through Hugging Face,
- process long audio in chunks,
- pool hidden states into the DEAM 0.5s label timeline,
- save one `.npz` file per song in `data/features_mert/`,
- append extraction results to `outputs/feature_audit.csv`.

Next action is to install `transformers<=4.44.0`, then run a one-song smoke
test before launching full feature extraction.

### Stage 2 Completed

Installed:

```text
transformers==4.44.0
```

MERT loading and extraction:

```text
model_name=m-a-p/MERT-v1-95M
sample_rate=24000
feature_dim=768
```

Important correction during extraction:

- The first implementation assumed every song should output all 1223 label
  frames from the wide DEAM annotation table.
- A smoke/full-run audit showed most DEAM audio files are 45-second excerpts,
  so forcing 1223 frames would fill hundreds of frames outside the actual
  audio.
- The extraction script was corrected to keep only label times covered by the
  actual audio duration.
- Erroneous intermediate feature files from that first attempt were removed
  before the final full run.

Final extraction result:

```text
features=1802/1802
train=1441
val=180
test=181
feature_dim=768
frame_count_min=60
frame_count_mean=73.145949
frame_count_median=61
frame_count_max=1223
full_extraction_elapsed_sec=3463.124
```

Generated:

```text
data/features_mert/*.npz
outputs/feature_audit.csv
outputs/mert_feature_inventory.csv
```

Stage 3 can now build MERT+DEAM datasets and train/tune MLP, BiLSTM, and
Transformer heads.

## 2026-06-14

### Stage 3 Completed

Added:

```text
configs/stage3_training_plan.json
scripts/training_utils.py
scripts/build_mert_va_datasets.py
scripts/train_mert_mlp.py
scripts/train_mert_sequence.py
scripts/aggregate_training_results.py
docs/STAGE3_RESULTS.md
```

Dataset build:

```text
train_frames=101248
val_frames=13427
test_frames=15320
feature_dim=768
train_dropped_invalid_frames=1451
val_dropped_invalid_frames=181
test_dropped_invalid_frames=182
```

Important correction during Stage 3:

- The first MLP run produced NaN metrics.
- Investigation showed the MERT feature arrays were finite, but DEAM dynamic
  labels contained some NaN values.
- Dataset construction was updated to drop frames where either features or VA
  labels are non-finite.
- Training scripts were updated to assert finite arrays and fail clearly if a
  non-finite metric appears.
- A later device test with a fake `--trial __none__` truncated the MLP aggregate
  CSV, but each MLP trial JSON remained intact. `aggregate_training_results.py`
  was added to rebuild family and global comparison tables from trial JSON.

Training environment:

```text
torch=2.8.0
transformers=4.44.0
training_device=cpu
```

MPS note: a direct Torch check reported MPS availability, but the training
script process did not reliably expose MPS. Stage 3 was completed on CPU for a
stable run.

Completed trials:

```text
mlp_baseline
mlp_wide
mlp_light_dropout
bilstm_baseline
bilstm_deep
bilstm_wide
transformer_baseline
transformer_deep
transformer_wide
```

Best current model by test RMSE:

```text
model_family=bilstm
trial_id=bilstm_baseline
best_epoch=42
parameters=920066
train_rmse=0.051500
val_rmse=0.219934
test_rmse=0.202452
test_pearson_valence=0.511522
test_pearson_arousal=0.641069
checkpoint=outputs/models/bilstm/bilstm_baseline_best.pt
```

Local generated outputs:

```text
data/datasets/mert_va/
outputs/metrics/
outputs/models/
outputs/predictions/
outputs/results/
```

These outputs are ignored by git because they include generated arrays and
checkpoints. The human-readable Stage 3 result summary is tracked at:

```text
docs/STAGE3_RESULTS.md
```
