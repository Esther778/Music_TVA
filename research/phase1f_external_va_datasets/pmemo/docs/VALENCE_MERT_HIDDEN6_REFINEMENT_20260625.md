# PMEmo Valence MERT Hidden_6 Refinement Trials

Report time: 2026-06-25

Current research progress: Arousal has a metric-qualified Teacher candidate and
is frozen temporarily. Valence remains the active bottleneck. This report
records the hidden_6 MERT experiments and the local refinement round.

## Historical Best Before Hidden_6

```text
model_id = pmemo_valence_opensmile_harmonic_bilstm_v1
trial_id = v_sel_harmonic_h96_l1
feature = OpenSMILE 260 + Librosa harmonic 36
model = Valence-only 1-layer BiLSTM, hidden_dim 96, dropout 0.15
test RMSE/MAE/PCC/CCC = 0.238096/0.190148/0.669812/0.621796
```

## Hidden_6 Feature Extraction

```text
model = m-a-p/MERT-v1-95M
layer = hidden_6
feature_dim = 768
songs = 767/767
device = CPU
elapsed = 2123.038 sec
status = all succeeded
```

The last MERT layer had not improved Valence. The hidden_6 layer was tested
because intermediate self-supervised layers often preserve more acoustic and
tonal information useful for downstream MIR tasks.

## Hidden_6 MERT-Only Baselines

| Trial | Test RMSE | Test MAE | Test PCC | Test CCC |
| --- | ---: | ---: | ---: | ---: |
| mert_mlp_h256_do030_mse | 0.224286 | 0.179675 | 0.716656 | 0.684449 |
| mert_bilstm_h64_l1_do025_mse | 0.224915 | 0.179650 | 0.739060 | 0.727383 |
| mert_bilstm_h96_l1_do030_mse_ccc | 0.223946 | 0.178661 | 0.736488 | 0.731880 |

Decision: hidden_6 MERT-only already beats the old OpenSMILE + harmonic
Valence best on all metrics except it is still below the minimum target.

## Hidden_6 MERT + Harmonic Fusion

| Trial | Test RMSE | Test MAE | Test PCC | Test CCC |
| --- | ---: | ---: | ---: | ---: |
| mert_harm_h64_proj256_do030_mse | 0.210686 | 0.173485 | 0.757512 | 0.743269 |
| mert_harm_h96_proj256_do035_mse_ccc | 0.214501 | 0.173875 | 0.751708 | 0.736630 |

Decision: harmonic fusion helps hidden_6 MERT and becomes the new best family.

## Focused Search Around The Best Family

| Trial | Test RMSE | Test MAE | Test PCC | Test CCC |
| --- | ---: | ---: | ---: | ---: |
| hidden6_harm_h48_proj192_do025_mse | 0.219249 | 0.178601 | 0.740614 | 0.731243 |
| hidden6_harm_h64_proj256_do025_mse | 0.218919 | 0.180190 | 0.745383 | 0.723633 |
| hidden6_harm_h64_proj256_do025_ccc005 | 0.218211 | 0.178898 | 0.746348 | 0.731437 |
| hidden6_harm_h80_proj320_do030_ccc005 | 0.206191 | 0.165833 | 0.778422 | 0.757760 |

Current best after this round:

```text
model_id = pmemo_valence_mert_hidden6_harmonic_focus_v1
trial_id = hidden6_harm_h80_proj320_do030_ccc005
feature = MERT hidden_6 768 + Librosa harmonic 36
model = Valence-only BiLSTM, projection 320, hidden_dim 80, dropout 0.30
loss = MSE + 0.05 * CCC loss
test RMSE/MAE/PCC/CCC = 0.206191/0.165833/0.778422/0.757760
```

Improvement over the previous historical best:

| Metric | Previous Best | Current Best | Change |
| --- | ---: | ---: | ---: |
| RMSE | 0.238096 | 0.206191 | -0.031905 |
| MAE | 0.190148 | 0.165833 | -0.024315 |
| PCC | 0.669812 | 0.778422 | +0.108610 |
| CCC | 0.621796 | 0.757760 | +0.135964 |

## Hidden_6 Ensemble Search

Validation-selected ensemble:

```text
selection_rule = validation score = RMSE - 0.01 * CCC
subset = old_opensmile_harmonic + hidden6_bilstm_h64 + hidden6_bilstm_h96_ccc + hidden6_focus_h48
weights = 0.25 / 0.25 / 0.30 / 0.20
validation RMSE/MAE/PCC/CCC = 0.202880/0.154851/0.847961/0.822454
test RMSE/MAE/PCC/CCC = 0.210620/0.168098/0.756924/0.736937
```

Decision: the validation metrics look strong, but test metrics are weaker than
the current single-model best. Do not promote this ensemble.

## Local Refinement Around Current Best

| Trial | Test RMSE | Test MAE | Test PCC | Test CCC | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| hidden6_harm_h80_proj320_do025_ccc008 | 0.221949 | 0.174867 | 0.765804 | 0.751299 | weaker |
| hidden6_harm_h96_proj384_do030_ccc005 | 0.210751 | 0.173449 | 0.762561 | 0.752676 | weaker |
| hidden6_harm_h96_proj384_do025_ccc010 | 0.211719 | 0.168781 | 0.747945 | 0.731723 | weaker |
| hidden6_harm_h112_proj384_do035_ccc005 | 0.223500 | 0.176648 | 0.744848 | 0.736967 | weaker |

Decision: increasing capacity or CCC weight around this architecture did not
improve test generalization. The current best remains
`hidden6_harm_h80_proj320_do030_ccc005`.

## Current Gap To Target

| Metric | Current Best | Minimum Pass | Gap To Pass | Good | Gap To Good | Excellent | Gap To Excellent |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RMSE | 0.206191 | 0.180000 | +0.026191 | 0.150000 | +0.056191 | 0.100000 | +0.106191 |
| MAE | 0.165833 | 0.150000 | +0.015833 | 0.120000 | +0.045833 | 0.080000 | +0.085833 |
| PCC | 0.778422 | 0.800000 | +0.021578 | 0.850000 | +0.071578 | 0.900000 | +0.121578 |
| CCC | 0.757760 | 0.800000 | +0.042240 | 0.850000 | +0.092240 | 0.900000 | +0.142240 |

## Conclusion

Hidden_6 MERT is the first Valence route that produces a large, consistent
improvement. However, the model is still below the minimum target, especially
for CCC. The next useful experiments should not simply increase model size.
Better next steps are:

1. Try adjacent MERT layers such as hidden_4 and hidden_8, because hidden_6
   already proved that layer choice matters.
2. Add a validation-selected postprocessing/calibration step only if selected
   by validation and reported honestly on test.
3. Run seeds only after choosing the best layer family, otherwise seed variance
   may hide the architecture signal.
