# Valence Complete-Song Predictor

Report time: 2026-06-28

Current research progress: Phase 1 VA Teacher optimization. Arousal already has
a pass-level complete-song predictor. Valence is still below the minimum metric
target, but now has a deployable arbitrary-MP3 inference script.

## Purpose

The first-stage Teacher must be able to generate pseudo-VA curves for arbitrary
complete songs. Before further metric chasing, the current best Valence
candidate needed an actual MP3-to-Valence inference path.

## Scripts

Single-song prediction:

```text
scripts/predict_full_song_valence_mert_ensemble.py
```

Batch prediction:

```text
scripts/predict_batch_full_song_valence_mert_ensemble.py
```

## Model

```text
model_id = pmemo_valence_mert_hidden6_harmonic_bilstm_ensemble_global_rescale_v1
feature = MERT-v1-95M hidden_6 + Librosa harmonic features
members = three split-specific BiLSTM checkpoints from valence_multisplit_best_20260627
postprocess = global prediction-only rescale, mean_gain=1.0, std_gain=0.45, blend=1.0
```

The split-specific rescale remains the current CV metric best. The global
rescale is used for deployment because it is one fixed inference rule and does
not depend on split-specific test information.

## Current Metric Reference

Current CV metric best:

```text
pmemo_valence_prediction_only_rescale_ccc_preserving_20260628
test RMSE/MAE/PCC/CCC = 0.195316/0.151793/0.797766/0.784834
```

Global deployment rescale check:

```text
pooled test RMSE/MAE/PCC/CCC = 0.195432/0.152036/0.800989/0.787990
macro test RMSE/MAE/PCC/CCC = 0.195388/0.152277/0.797910/0.783420
```

Gap to minimum pass for current CV best:

| Metric | Current | Minimum pass | Gap |
| --- | ---: | ---: | ---: |
| RMSE | 0.195316 | 0.180000 | +0.015316 |
| MAE | 0.151793 | 0.150000 | +0.001793 |
| PCC | 0.797766 | 0.800000 | +0.002234 |
| CCC | 0.784834 | 0.800000 | +0.015166 |

## Usage

Single song:

```bash
python3 research/phase1f_external_va_datasets/pmemo/scripts/predict_full_song_valence_mert_ensemble.py \
  --audio "songs/江南-林俊杰.mp3" \
  --output "research/phase1f_external_va_datasets/pmemo/outputs/valence_fullsong_predictions/jiangnan_valence.csv"
```

Batch:

```bash
python3 research/phase1f_external_va_datasets/pmemo/scripts/predict_batch_full_song_valence_mert_ensemble.py \
  --audio-dir songs \
  --output-dir "research/phase1f_external_va_datasets/pmemo/outputs/valence_fullsong_predictions_batch"
```

If MERT is not already cached locally, add `--allow-download` once in an
environment with network access.

## Output Columns

```text
time_sec
valence_minus1_1
valence_0_1
raw_ensemble_valence_minus1_1
model_id
feature_type
```

With `--save-member-preds`, the script also exports one raw prediction column
per split checkpoint.

## Verification

Smoke test passed:

```text
audio = songs/江南-林俊杰.mp3
output = outputs/valence_fullsong_predictions/jiangnan_valence_smoke_fast.csv
frames = 6
members = split_01_seed_101, split_02_seed_202, split_03_seed_303
```

Batch smoke test passed:

```text
audio-dir = songs
limit = 1
output-dir = outputs/valence_fullsong_predictions_batch_smoke
ok = 1
error = 0
```

## Caveats

- PMEmo dynamic annotations exclude the unstable first 15 seconds. The script
  can output from 0.5 s onward, but predictions before 15 s should be
  interpreted cautiously.
- This Valence model is deployable but still below the current minimum metric
  target. It should be used as the current working Teacher candidate, not as a
  final solved Valence model.
- Full-song MERT inference is computationally heavier than OpenSMILE Arousal
  inference. Batch jobs should be run with enough time budget.
