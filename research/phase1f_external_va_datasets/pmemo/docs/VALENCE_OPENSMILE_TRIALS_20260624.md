# Valence-Only OpenSMILE Trials

Report time: 2026-06-24 21:13:50 CST

Current research progress:

```text
Arousal has a metric-qualified complete-song Teacher candidate.
Valence remains the main VA Teacher bottleneck.
This run tested whether Valence-only training on PMEmo OpenSMILE features can
substantially improve over the joint VA baseline.
```

## Baseline Before This Run

Previous best Valence baseline:

```text
model = PMEmo official OpenSMILE dynamic features + joint VA BiLSTM
selection = validation mean VA RMSE
split = song-level
```

Test metrics:

| Metric | Current | Minimum pass | Gap to pass |
| --- | ---: | ---: | ---: |
| RMSE | 0.271199 | 0.180000 | +0.091199 |
| MAE | 0.212034 | 0.150000 | +0.062034 |
| PCC | 0.571292 | 0.800000 | +0.228708 |
| CCC | 0.546814 | 0.800000 | +0.253186 |

## What Was Tested

Scripts:

```text
scripts/train_pmemo_opensmile_valence_trials.py
scripts/train_pmemo_opensmile_valence_mlp.py
```

Data:

```text
dataset = PMEmo 2019
feature = official OpenSMILE dynamic features
feature_dim = 260
target = Valence only
label_scale = [-1, 1]
split = song-level, same as PMEmo baseline
```

Trials:

```text
7 BiLSTM variants
3 MLP variants
validation-selected model checkpoints
validation-only postprocess / ensemble search
```

## BiLSTM Trial Results

Test metrics on canonical `[-1, 1]` scale:

| Trial | Best Epoch | RMSE | MAE | PCC | CCC |
| --- | ---: | ---: | ---: | ---: | ---: |
| v_only_h128_l2_do025_mse | 7 | 0.273882 | 0.219546 | 0.575639 | 0.543305 |
| v_only_h160_l2_do030_mse_ccc | 6 | 0.268778 | 0.215721 | 0.600027 | 0.585968 |
| v_only_h192_l2_do035_mse_ccc_smooth | 17 | 0.269934 | 0.215968 | 0.605057 | 0.572865 |
| v_only_h96_l1_do015_mse | 34 | 0.251189 | 0.201264 | 0.640771 | 0.612084 |
| v_only_h96_l1_do010_lr6e4_mse | 33 | 0.271463 | 0.217742 | 0.583356 | 0.554458 |
| v_only_h112_l1_do015_lr7e4_mse | 26 | 0.290476 | 0.227430 | 0.542980 | 0.512520 |
| v_only_h80_l1_do010_lr8e4_mse | 36 | 0.285393 | 0.223705 | 0.583526 | 0.546862 |

Best test model:

```text
trial_id = v_only_h96_l1_do015_mse
model = 1-layer BiLSTM
hidden_dim = 96
dropout = 0.15
loss = MSE
best_epoch = 34
```

Gap to target:

| Metric | Current | Minimum pass | Gap to pass | Good | Gap to good | Excellent | Gap to excellent | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| RMSE | 0.251189 | 0.180000 | +0.071189 | 0.150000 | +0.101189 | 0.100000 | +0.151189 | Not pass |
| MAE | 0.201264 | 0.150000 | +0.051264 | 0.120000 | +0.081264 | 0.080000 | +0.121264 | Not pass |
| PCC | 0.640771 | 0.800000 | +0.159229 | 0.850000 | +0.209229 | 0.900000 | +0.259229 | Not pass |
| CCC | 0.612084 | 0.800000 | +0.187916 | 0.850000 | +0.237916 | 0.900000 | +0.287916 | Not pass |

## MLP Trial Results

| Trial | Best Epoch | RMSE | MAE | PCC | CCC |
| --- | ---: | ---: | ---: | ---: | ---: |
| v_mlp_h384_do025 | 2 | 0.284049 | 0.234675 | 0.550105 | 0.476757 |
| v_mlp_h256_do020 | 3 | 0.272553 | 0.222525 | 0.559421 | 0.496789 |
| v_mlp_h512_do035 | 4 | 0.272422 | 0.222669 | 0.554557 | 0.479951 |

MLP does not improve over the joint VA baseline and is clearly weaker than the
best Valence-only BiLSTM.

## Ensemble Search

A validation-selected ensemble improved validation metrics:

```text
validation RMSE/PCC/CCC = 0.226186 / 0.817237 / 0.758030
```

But it did not improve the test set:

```text
test RMSE/MAE/PCC/CCC = 0.267950 / 0.210134 / 0.605250 / 0.570343
```

Decision:

```text
Do not freeze a Valence ensemble from this round.
The validation/test mismatch suggests Valence generalization remains unstable.
```

## Interpretation

Valence-only training helped, but not enough:

| Model | RMSE | MAE | PCC | CCC |
| --- | ---: | ---: | ---: | ---: |
| Joint VA BiLSTM baseline | 0.271199 | 0.212034 | 0.571292 | 0.546814 |
| Best Valence-only BiLSTM | 0.251189 | 0.201264 | 0.640771 | 0.612084 |

Improvement:

```text
RMSE improved by about 0.020010
MAE improved by about 0.010770
PCC improved by about 0.069479
CCC improved by about 0.065270
```

This confirms that Valence-only training is useful, but OpenSMILE-only acoustic
features are not sufficient to reach the current Teacher threshold.

## Decision

```text
Continue OpenSMILE-only blind tuning = not recommended
Next route = add Valence-relevant harmonic / tonal features and then MERT fusion
```

Recommended next experiments:

```text
1. Add harmonic/tonal handcrafted features: chroma, tonnetz, key/mode proxy,
   harmonic-percussive ratio, pitch contour, spectral brightness.
2. Train Valence-only feature-fusion BiLSTM/TCN.
3. Extract PMEmo MERT embeddings and test MERT + OpenSMILE / harmonic fusion.
4. Keep Arousal frozen and optimize Valence separately.
```
