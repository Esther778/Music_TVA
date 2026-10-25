# Arousal Complete-Song Prediction Model

Date: 2026-06-24

## Purpose

The PMEmo OpenSMILE + BiLSTM Arousal baseline has stronger metrics, but it uses
the released PMEmo `dynamic_features.csv`. That is not enough for arbitrary
complete-song prediction because the project does not yet have a matching
OpenSMILE extraction pipeline.

This run trains an Arousal-only model using features extracted directly from
audio with librosa. It is weaker than the official OpenSMILE baseline, but it
can be applied to any local MP3/WAV/FLAC/M4A file.

## Model

```text
model_id = pmemo_librosa_arousal_bilstm_v1
training data = PMEmo 2019 dynamic annotations
label scale = [-1, 1]
target = Arousal only
feature type = librosa handcrafted features extractable from arbitrary audio
feature dim = 118
model = 2-layer BiLSTM
hidden_dim = 128
dropout = 0.25
optimizer = AdamW
learning_rate = 8e-4
weight_decay = 1e-4
best_epoch = 3
```

Checkpoint:

```text
outputs/arousal_fullsong_librosa/arousal_bilstm_best.pt
```

## PMEmo Song-Level Split

| Split | Songs | Frames |
| --- | ---: | ---: |
| Train | 614 | 29184 |
| Validation | 77 | 3912 |
| Test | 76 | 3338 |

## Test Metrics

Canonical `[-1, 1]` scale:

| Metric | Current |
| --- | ---: |
| RMSE | 0.216204 |
| MAE | 0.169500 |
| PCC | 0.781381 |
| CCC | 0.771253 |

## Gap To Project Targets

| Metric | Current | Minimum pass | Gap to pass | Good | Gap to good | Excellent | Gap to excellent | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| RMSE | 0.216204 | 0.180000 | +0.036204 | 0.150000 | +0.066204 | 0.100000 | +0.116204 | Not pass |
| MAE | 0.169500 | 0.150000 | +0.019500 | 0.120000 | +0.049500 | 0.080000 | +0.089500 | Not pass |
| PCC | 0.781381 | 0.800000 | +0.018619 | 0.850000 | +0.068619 | 0.900000 | +0.118619 | Not pass |
| CCC | 0.771253 | 0.800000 | +0.028747 | 0.850000 | +0.078747 | 0.900000 | +0.128747 | Not pass |

For RMSE/MAE lower is better. For PCC/CCC higher is better. Positive gap means
the model has not reached that threshold.

## Complete-Song Inference Test

Test audio:

```text
songs/江南-林俊杰.mp3
```

Output:

```text
outputs/arousal_fullsong_librosa/fullsong_predictions/jiangnan_arousal.csv
frames = 535
duration_sec = 267.944671
hop = 0.5 s
```

Output columns:

```text
time_sec
arousal_minus1_1
arousal_0_1
model_id
feature_type
```

## Prediction Scripts

Single song:

```bash
.venv/bin/python research/phase1f_external_va_datasets/pmemo/scripts/predict_full_song_arousal.py \
  --audio "songs/江南-林俊杰.mp3" \
  --output "research/phase1f_external_va_datasets/pmemo/outputs/arousal_fullsong_librosa/fullsong_predictions/jiangnan_arousal.csv"
```

Batch directory:

```bash
.venv/bin/python research/phase1f_external_va_datasets/pmemo/scripts/predict_batch_full_song_arousal.py \
  --audio-dir songs \
  --output-dir research/phase1f_external_va_datasets/pmemo/outputs/arousal_fullsong_librosa/fullsong_predictions_batch
```

## Decision

This model is usable as an engineering complete-song Arousal predictor, because
it can process arbitrary MP3 files end to end.

It is not yet a scientifically final Arousal Teacher, because it misses all
project minimum thresholds:

```text
RMSE gap = +0.036204
MAE gap = +0.019500
PCC gap = +0.018619
CCC gap = +0.028747
```

The stronger PMEmo OpenSMILE + BiLSTM baseline remains the best metric baseline,
but it needs an OpenSMILE extraction pipeline before it can be used for
arbitrary complete songs.
