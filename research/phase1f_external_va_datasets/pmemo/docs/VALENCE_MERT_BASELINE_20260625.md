# PMEmo Valence MERT Baseline Trials

Report time: 2026-06-25

Current research progress: Arousal has a metric-qualified PMEmo Teacher
candidate. Valence remains below target. Before this report, the current best
Valence model was `pmemo_valence_opensmile_harmonic_bilstm_v1`.

## Current Best Before MERT

```text
model_id = pmemo_valence_opensmile_harmonic_bilstm_v1
trial_id = v_sel_harmonic_h96_l1
feature = OpenSMILE 260 + Librosa harmonic 36
model = 1-layer BiLSTM
test RMSE/MAE/PCC/CCC = 0.238096/0.190148/0.669812/0.621796
status = current best Valence, not pass
```

## PMEmo-Aligned MERT Feature Extraction

Reason:

Old local MERT features were not reused because their time axes did not match
the current PMEmo chorus-label time ranges. New features were extracted from
the actual PMEmo chorus audio and aligned to the current PMEmo `time_sec`
labels.

Script:

```text
scripts/extract_pmemo_mert_features.py
```

Extraction result:

```text
model = m-a-p/MERT-v1-95M
transformers = 4.44.0
device = CPU
songs = 767/767
feature_dim = 768
elapsed = 1484.002 sec
output = outputs/mert_pmemo/features/
audit = outputs/mert_pmemo/feature_audit.csv
```

## MERT-Only Valence Baselines

Script:

```text
scripts/train_pmemo_mert_valence_baseline.py
```

Results:

| Trial | Best epoch | Test RMSE | Test MAE | Test PCC | Test CCC |
| --- | ---: | ---: | ---: | ---: | ---: |
| mert_mlp_h256_do030_mse | 18 | 0.263013 | 0.212605 | 0.588462 | 0.530630 |
| mert_bilstm_h64_l1_do025_mse | 18 | 0.260726 | 0.202863 | 0.614882 | 0.595830 |
| mert_bilstm_h96_l1_do030_mse_ccc | 21 | 0.259879 | 0.207687 | 0.631615 | 0.600791 |

Decision:

```text
MERT-only did not beat OpenSMILE + harmonic.
MERT BiLSTM is better than MERT MLP, but not enough.
```

## MERT + Harmonic Fusion

Script:

```text
scripts/train_pmemo_mert_harmonic_valence_fusion.py
```

Results:

| Trial | Best epoch | Test RMSE | Test MAE | Test PCC | Test CCC |
| --- | ---: | ---: | ---: | ---: | ---: |
| mert_harm_h64_proj256_do030_mse | 8 | 0.253964 | 0.198529 | 0.643646 | 0.624557 |
| mert_harm_h96_proj256_do035_mse_ccc | 10 | 0.257721 | 0.200896 | 0.620578 | 0.603758 |

Decision:

```text
MERT + harmonic improves over MERT-only, but still does not beat the current
OpenSMILE + harmonic best.
Simple feature concatenation is not enough for Valence.
```

## Current Valence Best After MERT Trials

The current best Valence model remains unchanged:

```text
model_id = pmemo_valence_opensmile_harmonic_bilstm_v1
test RMSE/MAE/PCC/CCC = 0.238096/0.190148/0.669812/0.621796
```

Gap to thresholds:

| Metric | Current | Minimum pass | Gap to pass | Good | Gap to good | Excellent | Gap to excellent |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RMSE | 0.238096 | 0.180000 | +0.058096 | 0.150000 | +0.088096 | 0.100000 | +0.138096 |
| MAE | 0.190148 | 0.150000 | +0.040148 | 0.120000 | +0.070148 | 0.080000 | +0.110148 |
| PCC | 0.669812 | 0.800000 | +0.130188 | 0.850000 | +0.180188 | 0.900000 | +0.230188 |
| CCC | 0.621796 | 0.800000 | +0.178204 | 0.850000 | +0.228204 | 0.900000 | +0.278204 |

## Technical Interpretation

The result is not that MERT is useless. The more precise conclusion is:

```text
PMEmo-aligned MERT last-layer embeddings with simple MLP/BiLSTM heads do not
improve Valence under the current split and training setup.
```

Likely next steps:

1. Test earlier/intermediate MERT layers instead of only the last layer.
2. Use gated fusion or cross-attention rather than raw concatenation.
3. Diagnose PMEmo Valence split distribution and per-song failure cases.
4. Consider Valence-specific label smoothing or clip-level auxiliary loss.

Do not keep blindly increasing BiLSTM size on the current last-layer MERT
features.
