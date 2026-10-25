# OpenL3+MLP Tuning Log

## 2026-06-17

Purpose:

- Keep the published OpenL3+MLP model unchanged.
- Try two small PyTorch MLP parameter adjustments to see whether the published
  result can be improved.
- Report both single RMSE and multi-dimensional evaluation metrics.

Published baseline reproduced in `.venv_openl3_mlp`:

```text
python=3.10.6
openl3=0.4.2
torch=2.12.0
numpy=2.2.6
X_test=(13000, 512)
test_rmse=0.12029657512903214
test_pearson_valence=0.8783936500549316
test_pearson_arousal=0.9085791110992432
```

Important comparison caveat:

- This OpenL3 MLP dataset uses frame-level random splitting.
- The current MERT dataset uses song-level splitting.
- Therefore this tuning log evaluates whether OpenL3+MLP can improve over its
  own published baseline, not whether OpenL3 is fairly better than MERT.

Metrics reported:

```text
RMSE
MAE
Valence RMSE
Arousal RMSE
Valence MAE
Arousal MAE
Valence Pearson r
Arousal Pearson r
Valence CCC
Arousal CCC
Mean CCC
```

### Trial 1: `mlp_tune_w384_d015_lr3e4`

Parameter change relative to the published MLP:

| Field | Published MLP | Trial 1 |
| --- | ---: | ---: |
| hidden | [256, 128] | [384, 192] |
| dropout | 0.20 | 0.15 |
| learning rate | 0.0005 | 0.0003 |
| weight decay | 0.0001 | 0.00005 |
| batch size | 256 | 512 |
| max epochs | 40 | 70 |
| early stopping | none | patience 12 |

Rationale:

- Slightly increase capacity.
- Slightly reduce dropout because the frame-level task already has many samples.
- Reduce learning rate to make the wider model converge more smoothly.
- Keep regularization present but weaker than the published baseline.

Command:

```bash
.venv_openl3_mlp/bin/python research/phase1_va_prediction/openl3_va_baseline/scripts/train_mlp_tuning.py \
  --trial mlp_tune_w384_d015_lr3e4 \
  --device cpu
```

Result:

| Metric | Value |
| --- | ---: |
| best_epoch | 70 |
| parameters | 271298 |
| val_rmse | 0.1139739305 |
| test_rmse | 0.1119703501 |
| test_mae | 0.0853296667 |
| test_rmse_valence | 0.1136946604 |
| test_rmse_arousal | 0.1102190837 |
| test_mae_valence | 0.0865758136 |
| test_mae_arousal | 0.0840835124 |
| test_pearson_valence | 0.8948650956 |
| test_pearson_arousal | 0.9232392907 |
| test_ccc_valence | 0.8787971735 |
| test_ccc_arousal | 0.9128537178 |
| test_mean_ccc | 0.8958254457 |

Interpretation:

- Trial 1 improves over the published MLP baseline:
  `test_rmse` changes from `0.1202965751` to `0.1119703501`.
- Pearson and CCC also improve.
- Best epoch is the final epoch 70, so the model may not have fully converged.

### Trial 2: `mlp_tune_w384_d010_lr3e4_e100`

Parameter change based on Trial 1:

| Field | Trial 1 | Trial 2 |
| --- | ---: | ---: |
| hidden | [384, 192] | [384, 192] |
| dropout | 0.15 | 0.10 |
| learning rate | 0.0003 | 0.0003 |
| weight decay | 0.00005 | 0.00005 |
| batch size | 512 | 512 |
| max epochs | 70 | 100 |
| early stopping | patience 12 | patience 15 |

Rationale:

- Trial 1 was still improving at epoch 70.
- Lower dropout tests whether the model was still slightly underfitting.
- Longer training gives the same direction more room to converge.

Command:

```bash
.venv_openl3_mlp/bin/python research/phase1_va_prediction/openl3_va_baseline/scripts/train_mlp_tuning.py \
  --trial mlp_tune_w384_d010_lr3e4_e100 \
  --device cpu
```

Status: pending.

Result:

| Metric | Value |
| --- | ---: |
| best_epoch | 82 |
| parameters | 271298 |
| val_rmse | 0.1122192740 |
| test_rmse | 0.1114633828 |
| test_mae | 0.0843302011 |
| test_rmse_valence | 0.1125023514 |
| test_rmse_arousal | 0.1104146391 |
| test_mae_valence | 0.0848213807 |
| test_mae_arousal | 0.0838390067 |
| test_pearson_valence | 0.8953961730 |
| test_pearson_arousal | 0.9209565520 |
| test_ccc_valence | 0.8851035833 |
| test_ccc_arousal | 0.9151602983 |
| test_mean_ccc | 0.9001319408 |

Interpretation:

- Trial 2 improves over Trial 1 in test RMSE and mean CCC.
- Trial 2 has slightly lower test Pearson Arousal than Trial 1, but better
  RMSE, Valence Pearson, Valence/Arousal CCC, and mean CCC.
- Current best tuned MLP is `mlp_tune_w384_d010_lr3e4_e100`.

## Current Comparison

| Model | Test RMSE | Test MAE | Test r(V) | Test r(A) | Test CCC(V) | Test CCC(A) | Test mean CCC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Published OpenL3+MLP | 0.1202965826 | 0.0923237428 | 0.8783923388 | 0.9085802436 | 0.8589702249 | 0.8977689147 | 0.8783695698 |
| Trial 1: w384 d0.15 lr3e-4 | 0.1119703501 | 0.0853296667 | 0.8948650956 | 0.9232392907 | 0.8787971735 | 0.9128537178 | 0.8958254457 |
| Trial 2: w384 d0.10 lr3e-4 e100 | 0.1114633828 | 0.0843302011 | 0.8953961730 | 0.9209565520 | 0.8851035833 | 0.9151602983 | 0.9001319408 |

Best checkpoint:

```text
outputs/models/mlp_tuning/mlp_tune_w384_d010_lr3e4_e100_best.pt
```

Generated result files:

```text
outputs/results/mlp_tuning/mlp_tuning_summary.csv
outputs/results/mlp_tuning/mlp_tune_w384_d015_lr3e4_summary.json
outputs/results/mlp_tuning/mlp_tune_w384_d010_lr3e4_e100_summary.json
```

## Additional User Request

The user asked the AI to contribute independent research judgment beyond simply
executing requested tasks. Based on the first two MLP tuning trials:

- More capacity helped.
- Lower dropout helped.
- Longer training helped.
- Further improvement is possible, but expected gains are likely marginal.

Two additional trials are planned:

### Trial 3: `mlp_tune_w384_d008_lr2e4_e120`

Purpose:

- Keep the best width from Trial 2.
- Lower dropout from 0.10 to 0.08.
- Lower learning rate from 0.0003 to 0.0002.
- Extend training to 120 epochs.

Hypothesis:

- Trial 2 still showed useful validation improvement at later epochs.
- A lower learning rate may stabilize late-stage optimization.
- Slightly lower dropout tests whether the model remains under-regularized or
  still benefits from less regularization.

### Trial 4: `mlp_tune_w512_d010_lr2e4_e120`

Purpose:

- Increase capacity from [384, 192] to [512, 256].
- Keep dropout at 0.10.
- Use a lower learning rate to reduce instability from the larger model.

Hypothesis:

- If the frame-level OpenL3 task is still capacity-limited, the wider model may
  reduce RMSE further.
- If validation metrics stop improving or test metrics degrade, then the current
  best model is probably near the useful capacity limit for this split.

### Trial 3 Result

| Metric | Value |
| --- | ---: |
| best_epoch | 112 |
| parameters | 271298 |
| val_rmse | 0.1141125932 |
| test_rmse | 0.1126990616 |
| test_mae | 0.0855165049 |
| test_rmse_valence | 0.1137603447 |
| test_rmse_arousal | 0.1116276979 |
| test_mae_valence | 0.0861369520 |
| test_mae_arousal | 0.0848960727 |
| test_pearson_valence | 0.8924162984 |
| test_pearson_arousal | 0.9188708663 |
| test_ccc_valence | 0.8835707903 |
| test_ccc_arousal | 0.9141519070 |
| test_mean_ccc | 0.8988613486 |

Interpretation:

- Trial 3 did not beat Trial 2.
- Lowering dropout to 0.08 and learning rate to 0.0002 made training more
  conservative, but validation and test metrics were weaker.
- The result suggests the best direction is not simply "less dropout and lower
  learning rate".

### Trial 4 Result

| Metric | Value |
| --- | ---: |
| best_epoch | 83 |
| parameters | 394498 |
| val_rmse | 0.1097721979 |
| test_rmse | 0.1088737920 |
| test_mae | 0.0824103653 |
| test_rmse_valence | 0.1099759564 |
| test_rmse_arousal | 0.1077603549 |
| test_mae_valence | 0.0830277652 |
| test_mae_arousal | 0.0817929655 |
| test_pearson_valence | 0.9004738331 |
| test_pearson_arousal | 0.9251695871 |
| test_ccc_valence | 0.8904539943 |
| test_ccc_arousal | 0.9189484119 |
| test_mean_ccc | 0.9047012329 |

Interpretation:

- Trial 4 is the best OpenL3+MLP tuning result so far.
- Increasing capacity to [512, 256] improved RMSE, Pearson, and CCC on the same
  frame-level random split.
- The original published MLP was likely under-capacity for this frame-level
  split.

## Updated Current Comparison

| Model | Test RMSE | Test MAE | Test r(V) | Test r(A) | Test CCC(V) | Test CCC(A) | Test mean CCC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Published OpenL3+MLP | 0.1202965826 | 0.0923237428 | 0.8783923388 | 0.9085802436 | 0.8589702249 | 0.8977689147 | 0.8783695698 |
| Trial 1: w384 d0.15 lr3e-4 | 0.1119703501 | 0.0853296667 | 0.8948650956 | 0.9232392907 | 0.8787971735 | 0.9128537178 | 0.8958254457 |
| Trial 2: w384 d0.10 lr3e-4 e100 | 0.1114633828 | 0.0843302011 | 0.8953961730 | 0.9209565520 | 0.8851035833 | 0.9151602983 | 0.9001319408 |
| Trial 3: w384 d0.08 lr2e-4 e120 | 0.1126990616 | 0.0855165049 | 0.8924162984 | 0.9188708663 | 0.8835707903 | 0.9141519070 | 0.8988613486 |
| Trial 4: w512 d0.10 lr2e-4 e120 | 0.1088737920 | 0.0824103653 | 0.9004738331 | 0.9251695871 | 0.8904539943 | 0.9189484119 | 0.9047012329 |

Updated best checkpoint:

```text
outputs/models/mlp_tuning/mlp_tune_w512_d010_lr2e4_e120_best.pt
```

Updated recommendation:

- More same-split tuning may still reduce RMSE slightly, but the scientific
  value of continuing to tune this exact split is now limited.
- Recommended next checks before using this as a headline result:
  - run the best Trial 4 configuration with 3 random seeds,
  - report mean and standard deviation,
  - rebuild OpenL3+MLP under song-level split if directly comparing with MERT,
  - keep the frame-level random split caveat explicit in paper writing.

## Trial 4 Multi-Seed Verification

Purpose:

- Verify that the best Trial 4 configuration is stable across random seeds.
- This is still evaluated on the original OpenL3 frame-level random split.

Command pattern:

```bash
.venv_openl3_mlp/bin/python research/phase1_va_prediction/openl3_va_baseline/scripts/train_mlp_tuning.py \
  --trial mlp_tune_w512_d010_lr2e4_e120 \
  --seed <SEED> \
  --device cpu
```

Seeds:

```text
2026
3407
614
```

Per-seed results:

| Seed | Best epoch | Test RMSE | Test MAE | Test r(V) | Test r(A) | Test CCC(V) | Test CCC(A) | Test mean CCC |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026 | 119 | 0.1081200391 | 0.0813138038 | 0.9034107924 | 0.9256888628 | 0.8920747638 | 0.9192773104 | 0.9056760073 |
| 3407 | 80 | 0.1085988432 | 0.0820955038 | 0.9004259706 | 0.9259812832 | 0.8900264502 | 0.9202612042 | 0.9051438570 |
| 614 | 119 | 0.1076953784 | 0.0812104419 | 0.9029800892 | 0.9271407127 | 0.8920245767 | 0.9206370115 | 0.9063308239 |

Mean and standard deviation:

| Metric | Mean | Std |
| --- | ---: | ---: |
| val_rmse | 0.1093183334 | 0.0001704494 |
| val_mean_ccc | 0.9012564619 | 0.0001793489 |
| test_rmse | 0.1081380869 | 0.0004520027 |
| test_mae | 0.0815399165 | 0.0004839203 |
| test_rmse_valence | 0.1092293561 | 0.0007089066 |
| test_rmse_arousal | 0.1070347155 | 0.0004696569 |
| test_pearson_valence | 0.9022722840 | 0.0016133913 |
| test_pearson_arousal | 0.9262702862 | 0.0007678601 |
| test_ccc_valence | 0.8913752635 | 0.0011683761 |
| test_ccc_arousal | 0.9200585087 | 0.0007021474 |
| test_mean_ccc | 0.9057168961 | 0.0005945389 |

Generated aggregate files:

```text
outputs/results/mlp_tuning/mlp_tuning_seed_summary.csv
outputs/results/mlp_tuning/mlp_tuning_seed_summary.json
```

Conclusion:

- Trial 4 is stable across the three tested seeds.
- The 3-seed average `test_rmse=0.1081380869` is better than the single
  seed-42 Trial 4 result (`test_rmse=0.1088737920`) and the published baseline
  (`test_rmse=0.1202965826`).
- This remains a frame-level random split result, so it should not be described
  as a fair song-level generalization result.

## 2026-06-21: Phase 1 Teacher Frozen

The three Trial 4 seed checkpoints were evaluated as a prediction ensemble.
The ensemble candidate was compared with the best single checkpoint using
validation RMSE only.

Selection result:

```text
best_single_by_validation=seed 2026
best_single_val_rmse=0.1091462299
ensemble_val_rmse=0.0995747522
selected_mode=mean_ensemble
teacher_model_id=openl3_mlp_w512_v1
```

Final frozen Teacher test metrics:

| Metric | Value |
| --- | ---: |
| test RMSE | 0.0984247178 |
| test MAE | 0.0734129474 |
| test Valence RMSE | 0.0994812995 |
| test Arousal RMSE | 0.0973566547 |
| test Pearson r(V) | 0.9226814508 |
| test Pearson r(A) | 0.9409965277 |
| test CCC(V) | 0.9081979990 |
| test CCC(A) | 0.9329394102 |
| test mean CCC | 0.9205687046 |

New reproducibility files:

```text
configs/frozen_teacher_v1.json
docs/TEACHER_MODEL_CARD.md
scripts/freeze_va_teacher.py
scripts/predict_frozen_teacher.py
outputs/results/mlp_tuning/frozen_teacher_v1_audit.json
```

The old published checkpoint and inference script remain unchanged. The frozen
Teacher is a separate Phase 2 operational artifact.

## 2026-06-21: Song-Level Validation Completed

The tuned MLP was retrained using the exact MERT song-level split and the same
three random seeds.

```text
train=1441 songs / 101248 frames
validation=180 songs / 13427 frames
test=181 songs / 15320 frames
song_overlap=0
```

Three-seed OpenL3+MLP ensemble:

```text
validation_rmse=0.2222753316
test_rmse=0.2083092183
test_mean_ccc=0.4941812754
```

Current same-split MERT+BiLSTM:

```text
validation_rmse=0.2199336708
test_rmse=0.2024518376
test_mean_ccc=0.5382629439
```

Conclusion:

- the frame-level random split result substantially overestimates unseen-song
  performance,
- MERT+BiLSTM currently leads on validation RMSE under the shared song split,
- final Teacher selection is reopened until MERT+BiLSTM receives matching
  three-seed verification,
- full details are in `docs/SONG_LEVEL_VALIDATION.md`.
