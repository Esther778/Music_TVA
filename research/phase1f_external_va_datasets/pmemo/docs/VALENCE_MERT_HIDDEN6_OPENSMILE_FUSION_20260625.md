# PMEmo Valence MERT Hidden_6 + OpenSMILE Fusion Trials

Report time: 2026-06-25

Current research progress: Valence remains the active bottleneck. The current
best model before this round is MERT hidden_6 + Librosa harmonic features. This
round tests whether adding full PMEmo OpenSMILE dynamic features improves
Valence.

## Historical Best Before This Round

```text
model_id = pmemo_valence_mert_hidden6_harmonic_focus_v1
trial_id = hidden6_harm_h80_proj320_do030_ccc005
feature = MERT hidden_6 768 + Librosa harmonic 36
model = Valence-only BiLSTM, projection 320, hidden_dim 80, dropout 0.30
loss = MSE + 0.05 * CCC loss
test RMSE/MAE/PCC/CCC = 0.206191/0.165833/0.778422/0.757760
```

## Input Features

```text
MERT layer = hidden_6
MERT dim = 768
OpenSMILE dim = 260
fusion dim = 1028
OpenSMILE source = outputs/opensmile_baseline/dataset/sequence/*.npz
OpenSMILE state = already train-normalized sequence X
```

Important implementation note:

```text
The first attempted script version incorrectly applied feature_norm.npz again
to OpenSMILE sequence X. That double normalization created extremely large
values and was interrupted. Those interrupted results are invalid and must not
be reported as experiments.
```

The fixed script uses OpenSMILE sequence `X` directly and only applies MERT's
own training-set normalization to the MERT branch.

## Fixed Fusion Results

| Trial | Best Epoch | Test RMSE | Test MAE | Test PCC | Test CCC |
| --- | ---: | ---: | ---: | ---: | ---: |
| h6_smile_h64_proj256_do030_mse | 21 | 0.218166 | 0.177587 | 0.745083 | 0.734991 |
| h6_smile_h80_proj320_do030_ccc005 | 18 | 0.210139 | 0.170445 | 0.761105 | 0.753956 |
| h6_smile_h96_proj384_do035_ccc005 | 13 | 0.211589 | 0.172449 | 0.758355 | 0.746935 |

Best model from this round:

```text
trial_id = h6_smile_h80_proj320_do030_ccc005
test RMSE/MAE/PCC/CCC = 0.210139/0.170445/0.761105/0.753956
```

## Comparison With Historical Best

| Metric | Historical Best | This Round Best | Change |
| --- | ---: | ---: | ---: |
| RMSE | 0.206191 | 0.210139 | +0.003947 |
| MAE | 0.165833 | 0.170445 | +0.004611 |
| PCC | 0.778422 | 0.761105 | -0.017317 |
| CCC | 0.757760 | 0.753956 | -0.003804 |

Decision:

```text
Do not promote MERT hidden_6 + OpenSMILE fusion.
It is a useful ablation, but it is weaker than the current hidden_6 + harmonic
best on all test metrics.
```

## Interpretation

The validation curves for the fixed fusion looked strong, with some epochs
reaching validation PCC/CCC above 0.80. However, the test set did not improve.
This suggests that full OpenSMILE features add capacity and validation-set
fit, but do not improve held-out generalization under the current split.

OpenSMILE still remains useful as:

1. A strong Arousal feature family.
2. A possible carefully selected or regularized Valence feature source.
3. A candidate for late fusion or validation-weighted ensemble, not naive full
   concatenation.

## Current Best After This Round

The current Valence best remains unchanged:

```text
model_id = pmemo_valence_mert_hidden6_harmonic_focus_v1
trial_id = hidden6_harm_h80_proj320_do030_ccc005
test RMSE/MAE/PCC/CCC = 0.206191/0.165833/0.778422/0.757760
status = current best Valence, still not pass
```

Gap to minimum target:

| Metric | Current Best | Minimum Pass | Gap To Pass |
| --- | ---: | ---: | ---: |
| RMSE | 0.206191 | 0.180000 | +0.026191 |
| MAE | 0.165833 | 0.150000 | +0.015833 |
| PCC | 0.778422 | 0.800000 | +0.021578 |
| CCC | 0.757760 | 0.800000 | +0.042240 |

## Next Step

The next useful direction should not be direct full-feature concatenation.
Prefer:

1. Validation-selected late fusion between current hidden_6 + harmonic and
   OpenSMILE-family predictions.
2. Clip-level auxiliary loss on the hidden_6 + harmonic model.
3. Validation-selected calibration/postprocessing with strict test reporting.
