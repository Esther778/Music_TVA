# Valence Multi-Split BiLSTM + TCN Ensemble - 2026-06-27

Report time: 2026-06-27

Current research progress: Phase 1F is improving the Valence teacher under the
multi-split gate. BiLSTM has better CCC than TCN, while TCN has slightly better
RMSE/MAE. This round tested validation-selected prediction fusion.

## Purpose

Combine complementary predictions:

```text
BiLSTM = stronger curve concordance
TCN = lower absolute error
Ensemble = validation-selected weighted average, optional calibration
```

The test set was not used for selecting weights or calibration.

## Evaluation Protocol

```text
same three song-level splits as prior multi-split experiments
for each split:
  use validation set to choose weight and calibration
  evaluate once on held-out test set
```

Candidate search:

```text
weight_tcn = 0.00 to 1.00 step 0.05
calibration = identity / bias / linear / mean_std
selection = validation pass-gap composite
```

## Thresholds Used For Reporting

| Metric | Minimum pass | Good | Excellent | Direction |
| --- | ---: | ---: | ---: | --- |
| RMSE | 0.180000 | 0.150000 | 0.100000 | lower is better |
| MAE | 0.150000 | 0.120000 | 0.080000 | lower is better |
| PCC | 0.800000 | 0.850000 | 0.900000 | higher is better |
| CCC | 0.800000 | 0.850000 | 0.900000 | higher is better |

## Reference Results

| Model | Test RMSE | Test MAE | Test PCC | Test CCC |
| --- | ---: | ---: | ---: | ---: |
| BiLSTM multi-split mean | 0.213700 | 0.166477 | 0.763825 | 0.754325 |
| TCN multi-split mean | 0.210383 | 0.165812 | 0.762037 | 0.728580 |

## Current Ensemble Results

Selected per split:

| Split | Selected weight | Calibration | Test RMSE | Test MAE | Test PCC | Test CCC |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| split_01 | 0.30 BiLSTM + 0.70 TCN | bias | 0.193069 | 0.152511 | 0.822262 | 0.802050 |
| split_02 | 0.35 BiLSTM + 0.65 TCN | bias | 0.220894 | 0.168585 | 0.724255 | 0.702779 |
| split_03 | 0.50 BiLSTM + 0.50 TCN | linear | 0.202194 | 0.153761 | 0.767313 | 0.752723 |

Ensemble multi-split mean:

```text
test RMSE/MAE/PCC/CCC mean = 0.205385/0.158285/0.771277/0.752518
test RMSE/MAE/PCC/CCC std  = 0.011582/0.007301/0.040109/0.040528
```

## Comparison Against BiLSTM Multi-Split Mean

| Metric | Ensemble mean | BiLSTM mean | Delta | Minimum pass | Ensemble gap to pass | Good | Ensemble gap to good | Excellent | Ensemble gap to excellent |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RMSE | 0.205385 | 0.213700 | -0.008314 | 0.180000 | +0.025385 | 0.150000 | +0.055385 | 0.100000 | +0.105385 |
| MAE | 0.158285 | 0.166477 | -0.008192 | 0.150000 | +0.008285 | 0.120000 | +0.038285 | 0.080000 | +0.078285 |
| PCC | 0.771277 | 0.763825 | +0.007452 | 0.800000 | +0.028723 | 0.850000 | +0.078723 | 0.900000 | +0.128723 |
| CCC | 0.752518 | 0.754325 | -0.001808 | 0.800000 | +0.047482 | 0.850000 | +0.097482 | 0.900000 | +0.147482 |

## Decision

```text
Promote as the current best multi-split Valence candidate, but not as a
threshold-passing final teacher.
```

Reason:

```text
Compared with BiLSTM multi-split mean, the ensemble improves RMSE, MAE, and
PCC. CCC is slightly lower by 0.001808, but much better than standalone TCN.
Overall it is the strongest multi-split compromise tested so far.
```

## Interpretation

The fusion result confirms that BiLSTM and TCN carry complementary information.
The model still misses the minimum pass thresholds, but the gap is smaller:

```text
RMSE gap to pass = +0.025385
MAE gap to pass = +0.008285
PCC gap to pass = +0.028723
CCC gap to pass = +0.047482
```

The major unresolved issue remains CCC. For section-aware downstream use,
curve shape consistency is still not strong enough.

## Next Recommendation

The next step should target CCC directly without giving back the RMSE/MAE gain:

```text
1. Try validation-selected CCC-preserving ensemble objective:
   require CCC not lower than BiLSTM by more than a small tolerance.
2. Try post-hoc smooth calibration on ensemble predictions.
3. If that fails, move to song-level pretraining + frame-level fine-tuning.
```

Artifacts:

```text
script = scripts/search_pmemo_valence_multisplit_bilstm_tcn_ensemble.py
outputs = outputs/valence_multisplit_bilstm_tcn_ensemble_20260627/
summary = outputs/valence_multisplit_bilstm_tcn_ensemble_20260627/summary.json
all results = outputs/valence_multisplit_bilstm_tcn_ensemble_20260627/all_ensemble_results.csv
```
