# PMEmo Valence Harmonic Objective Trials

Report time: 2026-06-25

Current research progress: Arousal has a metric-qualified Teacher candidate.
Valence remains the bottleneck. This report records a low-cost training
objective search based on the current best Valence feature family:
OpenSMILE 260 + Librosa harmonic 36.

## Historical Best Before This Round

```text
model_id = pmemo_valence_opensmile_harmonic_bilstm_v1
trial_id = v_sel_harmonic_h96_l1
feature = OpenSMILE 260 + Librosa harmonic 36
model = 1-layer BiLSTM
test RMSE/MAE/PCC/CCC = 0.238096/0.190148/0.669812/0.621796
```

## Objective Variants

Script:

```text
scripts/train_pmemo_valence_harmonic_objectives.py
```

Test results:

| Trial | Test RMSE | Test MAE | Test PCC | Test CCC | RMSE delta vs historical | MAE delta vs historical | PCC delta vs historical | CCC delta vs historical |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| v_harm_smoothmix025_mse_delta002 | 0.241111 | 0.194222 | 0.667377 | 0.595257 | +0.003014 | +0.004074 | -0.002435 | -0.026540 |
| v_harm_smoothmix050_mse_clip020 | 0.259033 | 0.206768 | 0.614323 | 0.569021 | +0.020937 | +0.016621 | -0.055489 | -0.052775 |
| v_harm_huber_ccc005_clip010 | 0.254994 | 0.201753 | 0.629095 | 0.592792 | +0.016898 | +0.011605 | -0.040717 | -0.029004 |
| v_harm_mse_ccc010_clip020_delta002 | 0.260053 | 0.210878 | 0.636779 | 0.576495 | +0.021957 | +0.020730 | -0.033033 | -0.045301 |

## Threshold Gap For Best Trial In This Round

Best trial in this round by test RMSE:

```text
trial_id = v_harm_smoothmix025_mse_delta002
test RMSE/MAE/PCC/CCC = 0.241111/0.194222/0.667377/0.595257
```

Gap to minimum pass:

| Metric | Current | Minimum pass | Gap to pass | Status |
| --- | ---: | ---: | ---: | --- |
| RMSE | 0.241111 | 0.180000 | +0.061111 | Not pass |
| MAE | 0.194222 | 0.150000 | +0.044222 | Not pass |
| PCC | 0.667377 | 0.800000 | +0.132623 | Not pass |
| CCC | 0.595257 | 0.800000 | +0.204743 | Not pass |

## Decision

```text
No objective variant beat the historical best.
The historical best remains pmemo_valence_opensmile_harmonic_bilstm_v1.
Training-objective tweaks alone are not enough.
```

Next direction:

```text
Test MERT intermediate-layer embeddings. The previous MERT experiment used
last-layer embeddings only; MIR representations often differ by layer, and
middle layers may preserve more local acoustic/tonal information.
```
