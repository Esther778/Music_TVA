# Valence Multi-Split TCN Trials - 2026-06-27

Report time: 2026-06-27

Current research progress: Phase 1F is trying to improve the Valence teacher
under the new multi-split gate. The previous BiLSTM architecture did not meet
minimum thresholds on multi-split average, so this round tested a lightweight
temporal convolutional network.

## Purpose

This round tested whether replacing the BiLSTM head with a lightweight TCN
could improve multi-split Valence robustness.

Model family:

```text
Input = MERT hidden_6 768 + Librosa harmonic 36
Model = projection + dilated residual TCN
dilations = 1, 2, 4, 8
channels = 128
loss = extreme-weighted frame MSE + 0.05 frame CCC + 0.10 clip mean MSE
```

Evaluation protocol:

```text
same three song-level splits as VALENCE_MULTISPLIT_STABILITY_20260627
each split selects checkpoint by validation selection_score
test is held out per split
```

## Thresholds Used For Reporting

| Metric | Minimum pass | Good | Excellent | Direction |
| --- | ---: | ---: | ---: | --- |
| RMSE | 0.180000 | 0.150000 | 0.100000 | lower is better |
| MAE | 0.150000 | 0.120000 | 0.080000 | lower is better |
| PCC | 0.800000 | 0.850000 | 0.900000 | higher is better |
| CCC | 0.800000 | 0.850000 | 0.900000 | higher is better |

## Reference Results

Historical single-split best:

```text
h6_harm_h80_ext125_clip010_ccc005
test RMSE/MAE/PCC/CCC = 0.205673/0.164567/0.783400/0.767304
```

BiLSTM multi-split base mean:

```text
test RMSE/MAE/PCC/CCC = 0.213700/0.166477/0.763825/0.754325
```

## Current TCN Results

Per-split test results:

| Split | Seed | Test RMSE | Test MAE | Test PCC | Test CCC |
| --- | ---: | ---: | ---: | ---: | ---: |
| split_01 | 101 | 0.200427 | 0.160077 | 0.808743 | 0.785331 |
| split_02 | 202 | 0.228470 | 0.176249 | 0.710328 | 0.671305 |
| split_03 | 303 | 0.202251 | 0.161110 | 0.767038 | 0.729104 |

TCN multi-split mean:

```text
test RMSE/MAE/PCC/CCC mean = 0.210383/0.165812/0.762037/0.728580
test RMSE/MAE/PCC/CCC std  = 0.012811/0.007392/0.040333/0.046552
```

## Comparison Against BiLSTM Multi-Split Mean

| Metric | TCN mean | BiLSTM mean | Delta | Minimum pass | TCN gap to pass | Good | TCN gap to good | Excellent | TCN gap to excellent |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RMSE | 0.210383 | 0.213700 | -0.003317 | 0.180000 | +0.030383 | 0.150000 | +0.060383 | 0.100000 | +0.110383 |
| MAE | 0.165812 | 0.166477 | -0.000665 | 0.150000 | +0.015812 | 0.120000 | +0.045812 | 0.080000 | +0.085812 |
| PCC | 0.762037 | 0.763825 | -0.001788 | 0.800000 | +0.037963 | 0.850000 | +0.087963 | 0.900000 | +0.137963 |
| CCC | 0.728580 | 0.754325 | -0.025745 | 0.800000 | +0.071420 | 0.850000 | +0.121420 | 0.900000 | +0.171420 |

## Decision

```text
Do not promote TCN as the balanced Valence teacher.
It improves RMSE and MAE slightly, but PCC and especially CCC decrease.
```

## Interpretation

The TCN is better at reducing absolute error on average, but it produces curves
with weaker concordance. Since the downstream project cares about emotional
trajectory shape and section-level emotion progression, CCC/PCC cannot be
ignored.

Practical reading:

```text
TCN is a useful ablation and may be part of an ensemble or auxiliary branch,
but it is not a replacement for the current BiLSTM teacher family.
```

## Next Recommendation

The next model-improvement step should not simply swap temporal heads one by
one. The most useful next attempt is a multi-objective model that explicitly
separates:

```text
1. frame-level local trajectory prediction
2. song-level Valence mean/range prediction
3. validation-selected fusion that preserves PCC/CCC
```

This can be implemented as an ensemble/fusion between BiLSTM and TCN
predictions under the same multi-split protocol. TCN may contribute lower
RMSE/MAE while BiLSTM preserves better CCC.

Artifacts:

```text
script = scripts/train_pmemo_valence_multisplit_tcn.py
outputs = outputs/valence_multisplit_tcn_20260627/
summary = outputs/valence_multisplit_tcn_20260627/summary.json
```
