# PMEmo Valence Selected Fusion And Ensemble Trials

Report time: 2026-06-25 09:28 CST

Current research progress: Arousal Teacher has reached the current minimum
metric target on PMEmo and has a complete-song inference path. The active
problem is Valence. This report records the first low-cost Valence improvement
round after OpenSMILE-only Valence failed to pass the target.

## Baseline Before This Round

Previous best Valence model:

```text
trial_id = v_only_h96_l1_do015_mse
feature = PMEmo official OpenSMILE dynamic features, 260 dim
model = Valence-only 1-layer BiLSTM
hidden_dim = 96
dropout = 0.15
loss = MSE
best_epoch = 34
test RMSE/MAE/PCC/CCC = 0.251189/0.201264/0.640771/0.612084
```

Gap to minimum pass:

| Metric | Current | Minimum pass | Gap to pass | Status |
| --- | ---: | ---: | ---: | --- |
| RMSE | 0.251189 | 0.180000 | +0.071189 | Not pass |
| MAE | 0.201264 | 0.150000 | +0.051264 | Not pass |
| PCC | 0.640771 | 0.800000 | +0.159229 | Not pass |
| CCC | 0.612084 | 0.800000 | +0.187916 | Not pass |

## Round 1: Full OpenSMILE + Librosa Fusion

Setup:

```text
OpenSMILE dim = 260
Librosa dim = 118
Fusion dim = 378
split = PMEmo song-level split, seed 42
target = Valence only, [-1, 1] scale
```

Result:

| Trial | Test RMSE | Test MAE | Test PCC | Test CCC |
| --- | ---: | ---: | ---: | ---: |
| v_fusion_h96_l1_do015_mse | 0.256539 | 0.205612 | 0.633366 | 0.581590 |
| v_fusion_h112_l1_do020_mse | 0.257421 | 0.205100 | 0.624862 | 0.579176 |
| v_fusion_h128_l2_do030_mse_ccc | 0.254445 | 0.198235 | 0.616781 | 0.576494 |
| v_fusion_h160_l2_do035_mse_ccc_smooth | 0.274710 | 0.214841 | 0.574219 | 0.555636 |

Decision:

```text
Full 118-dim Librosa fusion did not beat OpenSMILE-only.
Likely reason: full handcrafted feature fusion introduces noisy or redundant
dimensions for Valence.
```

## Round 2: Selected Librosa Feature Fusion

Motivation:

Valence is usually more sensitive to timbre, harmony, and tonal information
than to raw energy alone. Instead of appending all Librosa features, this round
tested smaller groups:

```text
harmonic = chroma + tonnetz mean/std, 36 dim
mfcc = MFCC mean/std, 40 dim
harmonic+mfcc = 76 dim
harmonic+spectral = 58 dim
```

Results:

| Trial | Feature group | Test RMSE | Test MAE | Test PCC | Test CCC |
| --- | --- | ---: | ---: | ---: | ---: |
| v_sel_harmonic_h96_l1 | harmonic | 0.238096 | 0.190148 | 0.669812 | 0.621796 |
| v_sel_mfcc_h96_l1 | mfcc | 0.271094 | 0.210262 | 0.615902 | 0.583832 |
| v_sel_harmonic_mfcc_h96_l1 | harmonic+mfcc | 0.259930 | 0.207392 | 0.613822 | 0.578730 |
| v_sel_harmonic_spectral_h128_l2_ccc | harmonic+spectral | 0.262084 | 0.205781 | 0.654324 | 0.621902 |

Current best Valence model after this round:

```text
trial_id = v_sel_harmonic_h96_l1
feature = OpenSMILE 260 + selected Librosa harmonic 36
model = Valence-only 1-layer BiLSTM
hidden_dim = 96
dropout = 0.15
loss = MSE
best_epoch = 22
test RMSE/MAE/PCC/CCC = 0.238096/0.190148/0.669812/0.621796
```

Improvement over previous OpenSMILE-only best:

| Metric | Previous best | New best | Change |
| --- | ---: | ---: | ---: |
| RMSE | 0.251189 | 0.238096 | -0.013093 |
| MAE | 0.201264 | 0.190148 | -0.011116 |
| PCC | 0.640771 | 0.669812 | +0.029041 |
| CCC | 0.612084 | 0.621796 | +0.009712 |

Gap to thresholds for current best:

| Metric | Current | Minimum pass | Gap to pass | Good | Gap to good | Excellent | Gap to excellent |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RMSE | 0.238096 | 0.180000 | +0.058096 | 0.150000 | +0.088096 | 0.100000 | +0.138096 |
| MAE | 0.190148 | 0.150000 | +0.040148 | 0.120000 | +0.070148 | 0.080000 | +0.110148 |
| PCC | 0.669812 | 0.800000 | +0.130188 | 0.850000 | +0.180188 | 0.900000 | +0.230188 |
| CCC | 0.621796 | 0.800000 | +0.178204 | 0.850000 | +0.228204 | 0.900000 | +0.278204 |

Decision:

```text
This is the new Valence best, but it is still far below the target.
Selected harmonic features help. Full Librosa fusion and MFCC fusion do not.
```

## Round 3: Harmonic-Focused Parameter Search

Setup:

The selected harmonic result became the new best, so a narrow parameter search
was run around it.

Results:

| Trial | Test RMSE | Test MAE | Test PCC | Test CCC |
| --- | ---: | ---: | ---: | ---: |
| v_harm_h80_l1_do020_lr7e4 | 0.257778 | 0.202610 | 0.600980 | 0.568818 |
| v_harm_h112_l1_do020_lr7e4 | 0.274076 | 0.220532 | 0.574080 | 0.525411 |
| v_harm_h96_l1_do025_lr6e4_ccc005 | 0.256182 | 0.202359 | 0.623795 | 0.581931 |
| v_harm_h96_l2_do025_lr6e4_ccc010 | 0.272686 | 0.222216 | 0.553950 | 0.501256 |

Decision:

```text
Focused parameter tuning did not beat v_sel_harmonic_h96_l1.
Do not continue blind BiLSTM hyperparameter tuning in this feature space.
```

## Round 4: Validation-Selected Ensemble

Candidate models:

```text
opensmile_v_only_h96
selected_harmonic_h96
selected_harmonic_spectral
selected_harmonic_mfcc
fusion_full_h128
harmonic_focus_ccc005
```

Selection rule:

```text
convex weights, step 0.05
subset size up to 4
selected by validation RMSE only
```

Best validation ensemble:

```text
subset = opensmile_v_only_h96 + selected_harmonic_spectral + harmonic_focus_ccc005
weights = 0.50 / 0.30 / 0.20
validation RMSE/MAE/PCC/CCC = 0.230119/0.177624/0.804424/0.753117
test RMSE/MAE/PCC/CCC = 0.245146/0.194812/0.667431/0.629125
```

Decision:

```text
The ensemble improves validation and slightly improves test CCC over the
current single-model best, but it worsens test RMSE and MAE.
It is useful as an ablation, not as the current frozen Valence Teacher.
```

## Current Valence Best After This Report

```text
model_id = pmemo_valence_opensmile_harmonic_bilstm_v1
trial_id = v_sel_harmonic_h96_l1
feature = OpenSMILE 260 + Librosa harmonic 36
model = 1-layer BiLSTM
hidden_dim = 96
dropout = 0.15
loss = MSE
selection = validation RMSE
test RMSE/MAE/PCC/CCC = 0.238096/0.190148/0.669812/0.621796
status = current best Valence, not pass
```

## Next Technical Direction

The best result improved all four test metrics, but the remaining gap is too
large for more small BiLSTM tuning to be the main route.

Recommended next steps:

1. Build PMEmo-aligned MERT features from the actual PMEmo chorus audio and
   current PMEmo timestamps. Do not reuse old MERT features whose time axis
   does not match PMEmo.
2. Train MERT-only Valence MLP/BiLSTM as the next baseline.
3. If MERT-only improves Valence, test MERT + harmonic fusion.
4. Keep OpenSMILE + harmonic as the current low-cost baseline and ablation.
