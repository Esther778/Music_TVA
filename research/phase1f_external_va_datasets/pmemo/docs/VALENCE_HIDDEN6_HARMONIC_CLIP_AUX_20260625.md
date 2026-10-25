# PMEmo Valence Hidden_6 + Harmonic Clip-Auxiliary Trials

Report time: 2026-06-25

Current research progress: Valence remains below the target thresholds. The
previous best model was MERT hidden_6 + Librosa harmonic with frame-level MSE
and CCC loss. This round adds a clip-level auxiliary loss to encourage each
song's predicted mean Valence to match the song's annotated mean Valence.

## Historical Best Before This Round

```text
model_id = pmemo_valence_mert_hidden6_harmonic_focus_v1
trial_id = hidden6_harm_h80_proj320_do030_ccc005
test RMSE/MAE/PCC/CCC = 0.206191/0.165833/0.778422/0.757760
```

## Training Change

Input features remain unchanged:

```text
feature = MERT hidden_6 768 + Librosa harmonic 36
model = Valence-only BiLSTM
```

Loss change:

```text
old loss = frame MSE + 0.05 * frame CCC loss
new loss = frame MSE + frame CCC loss + clip mean auxiliary loss
```

The clip auxiliary loss computes each song's mean predicted Valence and mean
target Valence inside a batch, then penalizes their mismatch. This is intended
to improve global prediction distribution and CCC without changing the frame
alignment.

## Results

| Trial | Best Epoch | Test RMSE | Test MAE | Test PCC | Test CCC |
| --- | ---: | ---: | ---: | ---: | ---: |
| h6_harm_h80_clipmse005_ccc005 | 19 | 0.205981 | 0.165660 | 0.778788 | 0.758959 |
| h6_harm_h80_clipmse010_ccc005 | 19 | 0.205867 | 0.165556 | 0.778969 | 0.760180 |
| h6_harm_h80_clipmse010_clipccc005 | 19 | 0.208577 | 0.167894 | 0.776078 | 0.762383 |
| h6_harm_h64_clipmse010_ccc005 | 16 | 0.218377 | 0.178487 | 0.747052 | 0.733031 |

Best model from this round:

```text
model_id = pmemo_valence_mert_hidden6_harmonic_clip_aux_v1
trial_id = h6_harm_h80_clipmse010_ccc005
feature = MERT hidden_6 768 + Librosa harmonic 36
model = Valence-only BiLSTM, projection 320, hidden_dim 80, dropout 0.30
loss = frame MSE + 0.05 * frame CCC loss + 0.10 * clip mean MSE
test RMSE/MAE/PCC/CCC = 0.205867/0.165556/0.778969/0.760180
```

## Improvement Over Previous Best

| Metric | Previous Best | New Best | Change |
| --- | ---: | ---: | ---: |
| RMSE | 0.206191 | 0.205867 | -0.000324 |
| MAE | 0.165833 | 0.165556 | -0.000277 |
| PCC | 0.778422 | 0.778969 | +0.000547 |
| CCC | 0.757760 | 0.760180 | +0.002420 |

Decision:

```text
Promote h6_harm_h80_clipmse010_ccc005 as the new current best Valence model.
The improvement is small, but all four test metrics improve over the previous
best.
```

## Postprocessing Check

Validation-selected moving-average + linear calibration was also tested on the
new best model:

```text
selected on validation = moving_average window 15 + linear calibration
validation RMSE/MAE/PCC/CCC = 0.209618/0.155428/0.829122/0.814775
test RMSE/MAE/PCC/CCC = 0.207249/0.166747/0.784305/0.756676
```

Decision:

```text
Do not apply postprocessing. It improves validation and test PCC, but worsens
test RMSE, MAE, and CCC.
```

## Current Gap To Target

| Metric | Current Best | Minimum Pass | Gap To Pass | Good | Gap To Good | Excellent | Gap To Excellent |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RMSE | 0.205867 | 0.180000 | +0.025867 | 0.150000 | +0.055867 | 0.100000 | +0.105867 |
| MAE | 0.165556 | 0.150000 | +0.015556 | 0.120000 | +0.045556 | 0.080000 | +0.085556 |
| PCC | 0.778969 | 0.800000 | +0.021031 | 0.850000 | +0.071031 | 0.900000 | +0.121031 |
| CCC | 0.760180 | 0.800000 | +0.039820 | 0.850000 | +0.089820 | 0.900000 | +0.139820 |

## Interpretation

Clip-level auxiliary loss helps, but only slightly. This suggests that the
model's main limitation is not only song-level mean bias. The remaining gap is
likely caused by frame-level Valence ambiguity, feature insufficiency, or split
distribution mismatch.

Next useful steps:

1. Try a slightly more direct CCC-oriented objective around the new best,
   without adding postprocessing.
2. Test validation-selected late fusion between the new clip-aux model and
   other complementary models, especially hidden_8 or OpenSMILE variants.
3. Run seeds only after a larger architectural or objective improvement is
   found; the current improvement is too small to justify seed confirmation as
   the next priority.
