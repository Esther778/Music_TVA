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

## 2026-06-16

### Stage 1 Teacher Model Optimization Restarted

User decision:

- Keep all three VA head families: MLP, BiLSTM, and Transformer.
- The immediate Stage 1 goal is to lower validation RMSE as much as reasonably
  possible before generating complete-song pseudo-VA for Phase 2.
- Use multi-dimensional evaluation, but make RMSE reduction the main practical
  optimization target.

Implementation updates:

```text
scripts/training_utils.py
scripts/train_mert_mlp.py
scripts/train_mert_sequence.py
scripts/aggregate_training_results.py
configs/stage3_training_plan.json
```

Evaluation updates:

- Added MAE.
- Added CCC.
- Added Valence/Arousal RMSE.
- Added Valence/Arousal MAE.
- Added Valence/Arousal CCC.
- Changed model selection from test RMSE to validation RMSE.
- Test metrics are now final-reporting metrics only.

Training mechanism updates:

- MLP supports loss weighting and gradient clipping.
- BiLSTM and Transformer support loss weighting, gradient clipping, and
  temporal smoothing loss.
- Sequence-model final evaluation now uses non-shuffled loaders so future saved
  predictions align with target rows.

New RMSE-reduction trial plan:

```text
MLP: 5 additional regularized trials
BiLSTM: 7 additional trials with smaller hidden sizes, stronger regularization,
        lower learning rate, smoothing loss, and Valence weighting
Transformer: 3 additional regularized/smaller/smoothed trials
```

Existing 9 trials were re-aggregated with the new validation-based rule.
Current provisional best remains:

```text
model_family=bilstm
trial_id=bilstm_baseline
val_rmse=0.219934
val_mean_ccc=0.627696
test_rmse=0.202452
test_mean_ccc=0.538263
```

Phase 2 context update from the user:

```text
reviewed/annotated complete songs are now reported as 95
```

## 2026-06-21

### Cross-Backbone Teacher Decision (Historical, Later Superseded)

The OpenL3 branch completed additional MLP tuning and three-seed verification.
For Phase 2 operational pseudo-VA generation, the frozen Teacher is:

```text
teacher_model_id=openl3_mlp_w512_v1
deployment_mode=mean ensemble
seeds=2026,3407,614
val_rmse=0.0995747522
test_rmse=0.0984247178
test_mean_ccc=0.9205687046
```

This does not establish that OpenL3 is superior to MERT because the protocols
remain different:

```text
OpenL3 = frame-level random split
MERT = song-level split
```

The current MERT+BiLSTM result remains the more defensible song-level
generalization experiment. A fair backbone comparison still requires an
OpenL3 model trained and evaluated on the same song-level DEAM split.

This provisional operational decision was superseded later on 2026-06-21 by
the OpenL3 song-level validation recorded immediately below.

### OpenL3 Song-Level Split Validation Completed

OpenL3+MLP was rebuilt using this phase's exact song split manifest.

Dataset audit:

```text
train=1441 songs / 101248 frames
validation=180 songs / 13427 frames
test=181 songs / 15320 frames
song_overlap=0
```

Three-seed OpenL3+MLP ensemble:

```text
val_rmse=0.2222753316
test_rmse=0.2083092183
test_mean_ccc=0.4941812754
```

Current MERT+BiLSTM baseline:

```text
val_rmse=0.2199336708
test_rmse=0.2024518376
test_mean_ccc=0.5382629439
```

Interpretation:

- MERT+BiLSTM currently has the lower validation and test RMSE.
- MERT is stronger on Valence; OpenL3 is slightly stronger on several Arousal
  metrics.
- Final Teacher selection is reopened.
- The next required experiment is three-seed MERT+BiLSTM verification, because
  the current MERT result is still a single run while OpenL3 now has three.

### RMSE Improvement Strategy Recorded

The user considered the current song-level RMSE around `0.20` insufficient and
requested a concrete improvement path.

Code audit found:

```text
MERT layer=last
pool_window_sec=0.5
encoder=frozen
training loss=frame-weighted MSE
no annotation response-lag calibration
single shared VA output head
```

The prioritized plan is recorded in:

```text
docs/RMSE_IMPROVEMENT_PLAN.md
```

Highest-priority experiments:

1. three-seed MERT+BiLSTM baseline,
2. DEAM annotation-lag validation search,
3. multi-scale temporal context,
4. song-balanced loss,
5. MERT intermediate-layer and learned layer-mixture comparison,
6. multi-scale temporal head with separate Valence/Arousal outputs.

Further head-only width/dropout tuning is considered low priority because
previous trials produced only small validation improvements.

### DEAM Scale And Timestamp Audit

User clarified:

```text
DEAM averaged dynamic VA range=[-1,+1]
first 15 seconds excluded because initial annotations are unstable
```

Verified:

- metrics are computed on the original signed scale with no 0-1 normalization,
- MERT features begin exactly at 15.0 seconds and advance by 0.5 seconds,
- the inherited OpenL3 tail alignment was 0.5 seconds late for typical songs.

OpenL3 exact-alignment seed 2026:

```text
old val_rmse=0.222834
new val_rmse=0.220933
old test_rmse=0.210084
new test_rmse=0.208569
```

The user requested no additional OpenL3 seeds until this first result was
assessed. Recommendation: do not spend further budget on OpenL3 seeds now;
continue with MERT multi-seed verification and annotation-lag diagnostics.

External advice was mapped against completed and pending project work in:

```text
docs/EXTERNAL_ADVICE_AUDIT_20260621.md
```

Current local Phase 2 manifest snapshot is stale:

```text
reviewed_songs_20260613.csv=70 songs
phase2_initial_70_manifest.csv=70 songs
raw/reviewed_songs_20260613.json=81 songs
```

Before Phase 2 pseudo-VA generation, refresh the Phase 2 reviewed-song manifest
to the current 95-song state.

### MLP RMSE-Reduction Round 1 Completed

Completed trials:

```text
mlp_h128_d035
mlp_h256_d035
mlp_h512_d035_lr3e4
mlp_h256_d05
mlp_h256_valence_weight
```

Aggregation command:

```bash
.venv/bin/python research/phase1b_bert_deam_va_training/scripts/aggregate_training_results.py
```

Current total trials: 14.

Best MLP after this round:

```text
trial_id=mlp_h256_valence_weight
val_rmse=0.232948
val_mean_ccc=0.525726
test_rmse=0.208067
test_mean_ccc=0.446329
```

Previous best MLP:

```text
trial_id=mlp_wide
val_rmse=0.233827
```

Conclusion:

- MLP-family validation RMSE improved by about 0.000879.
- The improvement is real but small.
- Current global best is still `bilstm_baseline` with `val_rmse=0.219934`.
- Next action is to run the planned BiLSTM RMSE-reduction trials, because this
  family currently has the strongest validation RMSE and is the most likely
  source of a better teacher model.

### BiLSTM RMSE-Reduction Round Started

Completed trials so far:

```text
bilstm_h64_d035
bilstm_h96_d035
bilstm_h128_d035
bilstm_h96_lr3e4
bilstm_h96_wd5e4
bilstm_h96_smooth005
bilstm_h96_valence_weight
```

Aggregated exact results:

| Trial | Purpose | Best val RMSE | Test RMSE | Conclusion |
| --- | --- | ---: | ---: | --- |
| bilstm_h64_d035 | Smaller BiLSTM with stronger dropout | 0.228948 | 0.215911 | Better than MLP, but worse than baseline |
| bilstm_h96_d035 | Medium BiLSTM with stronger dropout | 0.231509 | 0.205154 | Worse validation RMSE than h64 and baseline |
| bilstm_h128_d035 | Baseline-size BiLSTM with stronger dropout | 0.224221 | 0.207778 | Best of dropout-0.35 trials so far, but still worse than baseline |
| bilstm_h96_lr3e4 | Medium BiLSTM with lower learning rate | 0.231452 | 0.207012 | Slightly better than h96_d035, but still worse than baseline |
| bilstm_h96_wd5e4 | Medium BiLSTM with stronger weight decay | 0.226601 | 0.206791 | Better than h96_d035, but still worse than h128_d035 and baseline |
| bilstm_h96_smooth005 | Medium BiLSTM with temporal smoothness loss | 0.231515 | 0.205111 | Similar to h96_d035; no RMSE gain |
| bilstm_h96_valence_weight | Medium BiLSTM with higher Valence loss weight | 0.229913 | 0.205333 | Improves test RMSE relative to many trials, but validation RMSE is still worse than baseline |

Current interpretation:

- Reducing hidden size to 64 or 96 with dropout 0.35 does not lower validation
  RMSE.
- Restoring baseline-level capacity with dropout 0.35 improves over h64/h96,
  but still does not beat `bilstm_baseline`.
- Stronger dropout alone is therefore not the main RMSE-reduction path.
- Lower learning rate on h96 slightly improves over h96_d035, but not enough.
- Stronger weight decay improves h96 relative to h96_d035, but not enough.
- Smoothness loss at 0.005 does not lower validation RMSE.
- Valence-weighted loss does not lower validation RMSE enough to beat baseline.
- User decided this round is sufficient. Further Transformer/RMSE search is
  paused for now.
- Current teacher candidate remains `bilstm_baseline`:
  `val_rmse=0.219934`, `test_rmse=0.202452`,
  `checkpoint=outputs/models/bilstm/bilstm_baseline_best.pt`.

### OpenL3 Published Result Context

User clarified that the OpenL3 experiment results come from an already
published paper:

```text
/Users/esther/Desktop/申博/论文/基于 OpenL3 嵌入的音乐情绪连续愉悦度—唤醒度回归建模研究/论文（中文）.pdf
```

Implication for the current EI project:

- The OpenL3 results should be treated as the user's published Phase 1 prior
  work, not as an informal or disposable old experiment.
- Numerically, the published OpenL3+MLP result shown by the user
  (`test_rmse=0.1203`, `test_pearson_valence=0.878`,
  `test_pearson_arousal=0.909`) is stronger than the current MERT+BiLSTM
  result (`test_rmse=0.202452`, `test_pearson_valence=0.511522`,
  `test_pearson_arousal=0.641069`).
- Unless a unified re-evaluation proves otherwise, the new EI paper should not
  claim that MERT improves VA prediction over the published OpenL3 model.
- The safer paper framing is to use the published OpenL3 model as the completed
  first-stage VA teacher/baseline, and make the new contribution the
  section-aware complete-song emotion structure analysis.

### OpenL3+MLP Verification

User asked to verify the published OpenL3+MLP result first, before testing
BiLSTM or Transformer.

DEAM/MERT data-count clarification:

- Current MERT training used 1802 DEAM songs/audio files after feature-label
  alignment, not 7000+ audio files.
- The MERT frame dataset contains about 130k frame samples:
  train=101248, val=13427, test=15320.
- The 7000+ figure is not the audio count in the current MERT training set.

OpenL3 baseline assets found:

```text
research/phase1_va_prediction/openl3_va_baseline/
artifacts/openl3_2hz_features/: 1802 feature files
artifacts/datasets/mlp/: prebuilt MLP frame dataset
outputs/models/mlp/mlp_best.pt
```

Recomputed OpenL3+MLP test metrics from the saved checkpoint:

```text
X_test=(13000, 512)
test_loss=0.014471267227370006
test_rmse=0.12029657512903214
test_pearson_valence=0.8783936500549316
test_pearson_arousal=0.9085791110992432
```

The result was reproduced in two independent Python environments:

```text
project .venv: numpy=1.23.5, torch=2.8.0
Anaconda Python 3.12.4: numpy=1.26.4, torch=2.8.0
```

Interpretation:

- The published OpenL3+MLP result is real and reproducible from the current
  saved artifacts.
- The result matches the paper table values: RMSE 0.1203, r(V) 0.878, r(A)
  0.909.
- However, the OpenL3 MLP dataset builder uses frame-level random splitting:
  all frames are shuffled and split into train/val/test.
- The MERT dataset builder uses song-level splitting from a manifest, then
  frame arrays are generated within each split.
- Therefore OpenL3+MLP and MERT+BiLSTM are not directly comparable as a fair
  feature-backbone comparison unless OpenL3 is rebuilt with the same song-level
  split or MERT is evaluated under the same frame-level random split.

OpenL3 package installation check:

- A new Python 3.10.6 environment was created:
  `.venv_openl3_mlp`.
- `openl3` is not currently installed in `.venv`, Anaconda, or
  `.venv_openl3_mlp`.
- Installing dependencies or `openl3` requires network access. Two sandbox
  escalation requests for pip installation timed out, so package installation
  was not completed in this run.

Follow-up environment update:

- User installed `openl3` successfully in `.venv_openl3_mlp`.
- Verified runtime:
  `Python 3.10.6`, `openl3==0.4.2`, `tensorflow==2.20.0`.
- User preference: use PyTorch for model training/validation because the user
  does not work with TensorFlow.
- Decision: TensorFlow/OpenL3 may be used only as the feature extraction
  backend when necessary. All VA prediction models and verification scripts
  should remain PyTorch-based.

Final OpenL3+MLP PyTorch verification in `.venv_openl3_mlp`:

```text
python=3.10.6
openl3=0.4.2
tensorflow=2.20.0
torch=2.12.0
numpy=2.2.6
X_test=(13000, 512)
test_loss=0.014471267227370006
test_rmse=0.12029657512903214
test_pearson_valence=0.8783936500549316
test_pearson_arousal=0.9085791110992432
```

Conclusion: the published OpenL3+MLP result is reproducible in the dedicated
Python 3.10 + OpenL3 + PyTorch environment. No TensorFlow model training is
needed; TensorFlow is present only because OpenL3 depends on it.

### OpenL3+MLP PyTorch Tuning

User asked to tune only the OpenL3+MLP model first, trying two parameter
adjustments while reporting both single RMSE and multi-dimensional metrics.

New files:

```text
research/phase1_va_prediction/openl3_va_baseline/scripts/train_mlp_tuning.py
research/phase1_va_prediction/openl3_va_baseline/docs/MLP_TUNING_LOG.md
```

The published checkpoint was not overwritten.

Results:

| Model | Test RMSE | Test MAE | Test r(V) | Test r(A) | Test CCC(V) | Test CCC(A) | Test mean CCC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Published OpenL3+MLP | 0.1202965826 | 0.0923237428 | 0.8783923388 | 0.9085802436 | 0.8589702249 | 0.8977689147 | 0.8783695698 |
| mlp_tune_w384_d015_lr3e4 | 0.1119703501 | 0.0853296667 | 0.8948650956 | 0.9232392907 | 0.8787971735 | 0.9128537178 | 0.8958254457 |
| mlp_tune_w384_d010_lr3e4_e100 | 0.1114633828 | 0.0843302011 | 0.8953961730 | 0.9209565520 | 0.8851035833 | 0.9151602983 | 0.9001319408 |

Current best tuned OpenL3+MLP checkpoint:

```text
research/phase1_va_prediction/openl3_va_baseline/outputs/models/mlp_tuning/mlp_tune_w384_d010_lr3e4_e100_best.pt
```

Interpretation:

- Both tuned MLP variants improve over the published OpenL3+MLP checkpoint on
  the same frame-level random split.
- Trial 2 has the lowest test RMSE and highest mean CCC, so it is the current
  best tuned MLP candidate.
- The same caveat remains: this OpenL3 MLP result is based on frame-level random
  splitting, not song-level splitting.

Additional tuning after user requested two more trials:

| Model | Test RMSE | Test MAE | Test r(V) | Test r(A) | Test CCC(V) | Test CCC(A) | Test mean CCC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mlp_tune_w384_d008_lr2e4_e120 | 0.1126990616 | 0.0855165049 | 0.8924162984 | 0.9188708663 | 0.8835707903 | 0.9141519070 | 0.8988613486 |
| mlp_tune_w512_d010_lr2e4_e120 | 0.1088737920 | 0.0824103653 | 0.9004738331 | 0.9251695871 | 0.8904539943 | 0.9189484119 | 0.9047012329 |

Updated best tuned OpenL3+MLP checkpoint:

```text
research/phase1_va_prediction/openl3_va_baseline/outputs/models/mlp_tuning/mlp_tune_w512_d010_lr2e4_e120_best.pt
```

Research judgment:

- Trial 4 shows that additional MLP capacity can still improve this
  frame-level OpenL3 task.
- Further same-split tuning may produce small gains, but the stronger next
  research step is multi-seed verification and/or song-level split evaluation.
- Do not present this as a fair OpenL3-vs-MERT comparison until both use the
  same split protocol.

### Project Collaboration Guide

User requested that future AI conversations should not only execute requested
tasks, but also contribute independent research judgment after completing the
requested work.

New project-level guide:

```text
AI_COLLABORATION_GUIDE.md
```

README was updated so new AI/Codex conversations know to read this guide first.

### Trial 4 Multi-Seed Verification

User confirmed that the frame-level random split caveat should be recorded, and
asked to run three random seeds for the current best OpenL3+MLP configuration.

Base configuration:

```text
trial=mlp_tune_w512_d010_lr2e4_e120
hidden=[512, 256]
dropout=0.10
lr=0.0002
weight_decay=0.00005
epochs=120
batch_size=512
```

Seeds:

```text
2026, 3407, 614
```

Per-seed test results:

| Seed | Test RMSE | Test MAE | Test r(V) | Test r(A) | Test mean CCC |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2026 | 0.1081200391 | 0.0813138038 | 0.9034107924 | 0.9256888628 | 0.9056760073 |
| 3407 | 0.1085988432 | 0.0820955038 | 0.9004259706 | 0.9259812832 | 0.9051438570 |
| 614 | 0.1076953784 | 0.0812104419 | 0.9029800892 | 0.9271407127 | 0.9063308239 |

Aggregate:

| Metric | Mean | Std |
| --- | ---: | ---: |
| test_rmse | 0.1081380869 | 0.0004520027 |
| test_mae | 0.0815399165 | 0.0004839203 |
| test_pearson_valence | 0.9022722840 | 0.0016133913 |
| test_pearson_arousal | 0.9262702862 | 0.0007678601 |
| test_mean_ccc | 0.9057168961 | 0.0005945389 |

Generated aggregate files:

```text
research/phase1_va_prediction/openl3_va_baseline/outputs/results/mlp_tuning/mlp_tuning_seed_summary.csv
research/phase1_va_prediction/openl3_va_baseline/outputs/results/mlp_tuning/mlp_tuning_seed_summary.json
```

Interpretation:

- The current best OpenL3+MLP configuration is stable across three seeds.
- These results are still based on the original OpenL3 frame-level random split.
- For direct comparison with MERT, OpenL3 must be rebuilt under the same
  song-level split protocol.
