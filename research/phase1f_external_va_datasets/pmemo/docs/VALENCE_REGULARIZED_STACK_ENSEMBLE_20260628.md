# Valence Regularized Stack Ensemble Report

Report time: 2026-06-28 Asia/Shanghai

Current research progress: Phase 1 VA Teacher optimization. Arousal already has
a usable candidate. Valence remains below the minimum target, so this round
continued targeted optimization around the current multi-split Valence best.

## Purpose

The previous current best was:

```text
pmemo_valence_multisplit_ensemble_ccc_preserving_20260627
test RMSE/MAE/PCC/CCC mean = 0.203066/0.156576/0.777286/0.760357
```

The main weakness was still `split_02`, and the earlier BiLSTM/TCN models showed
overfitting. This round tested whether smaller, more regularized BiLSTM models
could provide complementary predictions for ensemble use.

## Regularized BiLSTM Trials

| Trial | Hidden | Dropout | Weight decay | CCC loss | Clip std | Test RMSE | Test MAE | Test PCC | Test CCC | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `bilstm_reg_h64_do045_wd001_ccc010_clip010_ext115` | 64 | 0.45 | 0.0010 | 0.10 | 0.02 | 0.214701 | 0.169402 | 0.777821 | 0.770887 | useful as ensemble member |
| `bilstm_reg_h56_do050_wd0015_ccc012_clip012_ext110` | 56 | 0.50 | 0.0015 | 0.12 | 0.03 | 0.216119 | 0.168262 | 0.766801 | 0.761663 | negative standalone result |

Interpretation:

```text
Stronger regularization improved or preserved CCC in some cases, but it did not
solve RMSE/MAE. The regularized models are not good standalone teachers.
However, the first regularized model is useful as a weak ensemble member.
```

## Ensemble Results

Step 1 fused the previous best with the first regularized model:

```text
pmemo_valence_current_regularized_ensemble_20260628
test RMSE/MAE/PCC/CCC mean = 0.202390/0.156369/0.778800/0.763963
```

Step 2 stacked the second regularized model on top of that ensemble:

```text
pmemo_valence_regularized_stack_ensemble_20260628
test RMSE/MAE/PCC/CCC mean = 0.202132/0.156198/0.779322/0.764633
```

## Current Best Comparison

| Model | Test RMSE | Test MAE | Test PCC | Test CCC |
| --- | ---: | ---: | ---: | ---: |
| BiLSTM multi-split baseline | 0.213700 | 0.166477 | 0.763825 | 0.754325 |
| BiLSTM + TCN CCC-preserving ensemble | 0.203066 | 0.156576 | 0.777286 | 0.760357 |
| Current + regularized ensemble | 0.202390 | 0.156369 | 0.778800 | 0.763963 |
| Current regularized stack ensemble | 0.202132 | 0.156198 | 0.779322 | 0.764633 |

Change vs previous best:

| Metric | Previous best | Current stack | Delta |
| --- | ---: | ---: | ---: |
| RMSE | 0.202390 | 0.202132 | -0.000258 |
| MAE | 0.156369 | 0.156198 | -0.000171 |
| PCC | 0.778800 | 0.779322 | +0.000522 |
| CCC | 0.763963 | 0.764633 | +0.000670 |

## Gap To Thresholds

Positive gap means still short of the threshold.

| Metric | Current | Minimum pass | Gap to pass | Good | Gap to good | Excellent | Gap to excellent |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RMSE | 0.202132 | 0.180000 | +0.022132 | 0.150000 | +0.052132 | 0.100000 | +0.102132 |
| MAE | 0.156198 | 0.150000 | +0.006198 | 0.120000 | +0.036198 | 0.080000 | +0.076198 |
| PCC | 0.779322 | 0.800000 | +0.020678 | 0.850000 | +0.070678 | 0.900000 | +0.120678 |
| CCC | 0.764633 | 0.800000 | +0.035367 | 0.850000 | +0.085367 | 0.900000 | +0.135367 |

## Per-Split Selection

| Split | Selected extra weight | Calibration | Test RMSE | Test MAE | Test PCC | Test CCC |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| split_01_seed_101 | 0.07 reg2 | bias | 0.187739 | 0.149102 | 0.832780 | 0.820213 |
| split_02_seed_202 | 0.00 reg2 | identity | 0.218121 | 0.166725 | 0.733581 | 0.716143 |
| split_03_seed_303 | 0.00 reg2 | identity | 0.200535 | 0.152765 | 0.771605 | 0.757542 |

## Decision

Promote `pmemo_valence_regularized_stack_ensemble_20260628` as the current best
multi-split Valence candidate, but still not as a final threshold-passing
teacher.

The improvement is small. It confirms that regularized weak models can slightly
improve the ensemble, but the current bottleneck is no longer simple dropout or
weight decay. The next useful direction should focus on split robustness and
data/label alignment rather than just making the same BiLSTM smaller.

Artifacts:

```text
regularized training script = scripts/train_pmemo_valence_multisplit_regularized_bilstm.py
current + reg ensemble script = scripts/search_pmemo_valence_current_regularized_ensemble.py
stack ensemble script = scripts/search_pmemo_valence_regularized_stack_ensemble.py
regularized outputs = outputs/valence_multisplit_regularized_bilstm_20260628/
current + reg outputs = outputs/valence_current_regularized_ensemble_20260628/
stack outputs = outputs/valence_regularized_stack_ensemble_20260628/
```

## Next Step

Recommended next experiment:

```text
Stop increasing regularization on the same architecture. Diagnose split_02 and
test data/label-level improvements:
1. split_02 distribution and difficult-song audit;
2. label shift or target smoothing specifically under multi-split protocol;
3. train with validation objective that explicitly preserves CCC while reducing
   RMSE;
4. compare every result against 0.202132/0.156198/0.779322/0.764633.
```
