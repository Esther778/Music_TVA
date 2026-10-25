# PMEmo Valence MERT Adjacent Layer Trials

Report time: 2026-06-25

Current research progress: Arousal is temporarily frozen with a
metric-qualified Teacher candidate. Valence remains the bottleneck. The
previous best Valence model used MERT hidden_6 plus Librosa harmonic features.
This report tests whether adjacent MERT layers hidden_4 and hidden_8 can improve
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

Current gap to minimum Valence thresholds:

| Metric | Current Best | Minimum Pass | Gap To Pass |
| --- | ---: | ---: | ---: |
| RMSE | 0.206191 | 0.180000 | +0.026191 |
| MAE | 0.165833 | 0.150000 | +0.015833 |
| PCC | 0.778422 | 0.800000 | +0.021578 |
| CCC | 0.757760 | 0.800000 | +0.042240 |

## Feature Extraction

| Layer | Songs | Feature Dim | Elapsed Sec | Status |
| --- | ---: | ---: | ---: | --- |
| hidden_4 | 767/767 | 768 | 1653.597 | all succeeded |
| hidden_8 | 767/767 | 768 | 1615.072 | all succeeded |

## Hidden_4 Results

### MERT-Only

| Trial | Test RMSE | Test MAE | Test PCC | Test CCC |
| --- | ---: | ---: | ---: | ---: |
| mert_mlp_h256_do030_mse | 0.230233 | 0.181124 | 0.697688 | 0.656575 |
| mert_bilstm_h64_l1_do025_mse | 0.215706 | 0.171362 | 0.737009 | 0.714613 |
| mert_bilstm_h96_l1_do030_mse_ccc | 0.236438 | 0.185889 | 0.713455 | 0.687377 |

### MERT + Harmonic

| Trial | Test RMSE | Test MAE | Test PCC | Test CCC |
| --- | ---: | ---: | ---: | ---: |
| mert_harm_h64_proj256_do030_mse | 0.217875 | 0.172992 | 0.729576 | 0.708676 |
| mert_harm_h96_proj256_do035_mse_ccc | 0.231852 | 0.183173 | 0.717054 | 0.702894 |

### Focus Search

| Trial | Test RMSE | Test MAE | Test PCC | Test CCC |
| --- | ---: | ---: | ---: | ---: |
| hidden4_harm_h48_proj192_do025_mse | 0.229112 | 0.182817 | 0.707483 | 0.693677 |
| hidden4_harm_h64_proj256_do025_mse | 0.222663 | 0.180941 | 0.721750 | 0.691679 |
| hidden4_harm_h64_proj256_do025_ccc005 | 0.219282 | 0.178151 | 0.735324 | 0.716099 |
| hidden4_harm_h80_proj320_do030_ccc005 | 0.230895 | 0.183518 | 0.726086 | 0.688056 |

Best hidden_4 result:

```text
trial_id = mert_bilstm_h64_l1_do025_mse
test RMSE/MAE/PCC/CCC = 0.215706/0.171362/0.737009/0.714613
```

Decision: hidden_4 is weaker than hidden_6 and should not be continued as the
main Valence path.

## Hidden_8 Results

### MERT-Only

| Trial | Test RMSE | Test MAE | Test PCC | Test CCC |
| --- | ---: | ---: | ---: | ---: |
| mert_mlp_h256_do030_mse | 0.221806 | 0.175742 | 0.729772 | 0.683783 |
| mert_bilstm_h64_l1_do025_mse | 0.227734 | 0.175392 | 0.722936 | 0.712882 |
| mert_bilstm_h96_l1_do030_mse_ccc | 0.220952 | 0.176094 | 0.761698 | 0.742613 |

### MERT + Harmonic

| Trial | Test RMSE | Test MAE | Test PCC | Test CCC |
| --- | ---: | ---: | ---: | ---: |
| mert_harm_h64_proj256_do030_mse | 0.214950 | 0.166041 | 0.750379 | 0.743133 |
| mert_harm_h96_proj256_do035_mse_ccc | 0.229768 | 0.180240 | 0.736712 | 0.727461 |

### Focus Search

| Trial | Test RMSE | Test MAE | Test PCC | Test CCC |
| --- | ---: | ---: | ---: | ---: |
| hidden8_harm_h48_proj192_do025_mse | 0.223811 | 0.180002 | 0.722295 | 0.673881 |
| hidden8_harm_h64_proj256_do025_mse | 0.215377 | 0.165770 | 0.748016 | 0.739406 |
| hidden8_harm_h64_proj256_do025_ccc005 | 0.217074 | 0.167003 | 0.747737 | 0.740371 |
| hidden8_harm_h80_proj320_do030_ccc005 | 0.228289 | 0.176893 | 0.725303 | 0.711950 |

Best hidden_8 result:

```text
trial_id = mert_harm_h64_proj256_do030_mse
test RMSE/MAE/PCC/CCC = 0.214950/0.166041/0.750379/0.743133
```

Decision: hidden_8 has a competitive MAE but worse RMSE, PCC, and CCC than
hidden_6. It should not replace the current best model.

## Current Best After This Round

No adjacent-layer model beat hidden_6.

```text
current_best_model_id = pmemo_valence_mert_hidden6_harmonic_focus_v1
current_best_trial = hidden6_harm_h80_proj320_do030_ccc005
test RMSE/MAE/PCC/CCC = 0.206191/0.165833/0.778422/0.757760
status = still best, still not pass
```

Updated gap to target is unchanged:

| Metric | Current Best | Minimum Pass | Gap To Pass | Good | Gap To Good | Excellent | Gap To Excellent |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RMSE | 0.206191 | 0.180000 | +0.026191 | 0.150000 | +0.056191 | 0.100000 | +0.106191 |
| MAE | 0.165833 | 0.150000 | +0.015833 | 0.120000 | +0.045833 | 0.080000 | +0.085833 |
| PCC | 0.778422 | 0.800000 | +0.021578 | 0.850000 | +0.071578 | 0.900000 | +0.121578 |
| CCC | 0.757760 | 0.800000 | +0.042240 | 0.850000 | +0.092240 | 0.900000 | +0.142240 |

## Interpretation

Layer choice matters. The earlier improvement was not simply caused by using
MERT instead of OpenSMILE; it specifically depended on hidden_6. Lower hidden_4
appears to be too acoustic/local for this Valence target, while higher hidden_8
loses some of the useful signal needed for stable numeric prediction.

Next useful steps:

1. Test hidden_5 and hidden_7 only if time allows, because hidden_6 is bracketed
   by weaker hidden_4 and hidden_8.
2. Prefer improving the hidden_6 family with better validation-selected
   calibration, clip-level auxiliary loss, or MERT hidden_6 plus OpenSMILE
   fusion rather than increasing BiLSTM size.
3. Do not run random seeds until the architecture family is stable enough; seed
   runs should confirm a selected model, not search blindly.
