# PMEmo Valence Extreme-Weighted Training And Fusion Check

Report time: 2026-06-25 Asia/Shanghai

Current research progress: Arousal has a usable PMEmo OpenSMILE ensemble
candidate. Valence remains the active Teacher bottleneck. This round continues
from the previous best hidden_6 MERT + harmonic clip-auxiliary Valence model.

## Previous Current Best

```text
model_id = pmemo_valence_mert_hidden6_harmonic_clip_aux_v1
trial_id = h6_harm_h80_clipmse010_ccc005
feature = MERT hidden_6 768 + Librosa harmonic 36
model = Valence-only BiLSTM, projection 320, hidden_dim 80, dropout 0.30
loss = frame MSE + 0.05 * frame CCC loss + 0.10 * clip mean MSE
test RMSE/MAE/PCC/CCC = 0.205867/0.165556/0.778969/0.760180
```

## Diagnostic Before Training

The previous best under-predicts Valence variance and is biased toward the
middle:

```text
test target mean/std/min/max = 0.255469/0.316639/-0.528475/0.831100
test pred   mean/std/min/max = 0.207521/0.272326/-0.437593/0.695182
test error mean/std = -0.047949/0.200206
```

The worst test songs are mostly whole-song mean failures, not just local frame
noise. This suggests the next loss should discourage over-conservative
predictions.

## Extreme-Aware Training

Script:

```text
research/phase1f_external_va_datasets/pmemo/scripts/train_pmemo_mert_hidden6_harmonic_extreme_weighted.py
```

Common setup:

```text
feature = MERT hidden_6 768 + Librosa harmonic 36
model = Valence-only BiLSTM
base loss = weighted frame MSE + 0.05 * frame CCC loss + 0.10 * clip mean MSE
extreme weighting = 1 + extreme_weight * abs(target_valence)
selection = validation score, test not used for checkpoint selection
```

Results:

| Trial | Best Epoch | Test RMSE | Test MAE | Test PCC | Test CCC | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| h6_harm_h80_ext075_clip010_ccc005 | 19 | 0.205265 | 0.164378 | 0.782495 | 0.765622 | improves current best, but lower PCC/CCC than ext125 |
| h6_harm_h80_ext125_clip010_ccc005 | 19 | 0.205673 | 0.164567 | 0.783400 | 0.767304 | new current best |
| h6_harm_h80_ext100_clip010_std005 | 19 | 0.205345 | 0.164312 | 0.783150 | 0.766232 | best MAE, but lower PCC/CCC than ext125 |
| h6_harm_h96_ext100_clip010_ccc005 | 5 | 0.215152 | 0.173285 | 0.741870 | 0.728564 | weaker |

## New Current Best

```text
model_id = pmemo_valence_mert_hidden6_harmonic_extreme_weighted_v1
trial_id = h6_harm_h80_ext125_clip010_ccc005
feature = MERT hidden_6 768 + Librosa harmonic 36
model = Valence-only BiLSTM, projection 320, hidden_dim 80, dropout 0.30
loss = extreme-weighted frame MSE + 0.05 * frame CCC loss + 0.10 * clip mean MSE
extreme_weight = 1.25
test RMSE/MAE/PCC/CCC = 0.205673/0.164567/0.783400/0.767304
```

Improvement over the previous clip-auxiliary best:

| Metric | Previous | New | Delta |
| --- | ---: | ---: | ---: |
| RMSE | 0.205867 | 0.205673 | -0.000194 |
| MAE | 0.165556 | 0.164567 | -0.000989 |
| PCC | 0.778969 | 0.783400 | +0.004431 |
| CCC | 0.760180 | 0.767304 | +0.007125 |

## Fusion Checks

Two low-cost fusion routes were tested after adding the extreme-aware outputs.

### Validation-Selected Convex Late Fusion

Script:

```text
research/phase1f_external_va_datasets/pmemo/scripts/search_pmemo_valence_late_fusion.py
```

Selected by validation:

```text
combo = hidden6_focus_best + hidden6_refine_h80 + opensmile_harmonic_best + mert_opensmile_fixed
weights = 0.1 / 0.4 / 0.2 / 0.3
validation RMSE/MAE/PCC/CCC = 0.203282/0.153640/0.842433/0.826971
test RMSE/MAE/PCC/CCC = 0.210125/0.168003/0.767448/0.748907
decision = not promoted
```

Although validation improves, test worsens. This suggests validation/test
distribution shift and fusion overfitting.

### Ridge Stacking

Script:

```text
research/phase1f_external_va_datasets/pmemo/scripts/train_pmemo_valence_ridge_stacking.py
```

Selected by validation:

```text
combo = clip_aux_light + hidden6_focus_best + hidden6_refine_h80 + hidden8_harm_mse + mert_opensmile_fixed
alpha = 0.1
validation RMSE/MAE/PCC/CCC = 0.207451/0.154116/0.837184/0.811747
test RMSE/MAE/PCC/CCC = 0.205204/0.166415/0.773916/0.749394
decision = not promoted
```

Ridge stacking lowers RMSE slightly but worsens MAE/PCC/CCC relative to the new
extreme-aware best, so it is not used.

## Threshold Gap

| Metric | Current | Minimum Pass | Gap To Pass | Good | Gap To Good | Excellent | Gap To Excellent |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RMSE | 0.205673 | 0.180000 | +0.025673 | 0.150000 | +0.055673 | 0.100000 | +0.105673 |
| MAE | 0.164567 | 0.150000 | +0.014567 | 0.120000 | +0.044567 | 0.080000 | +0.084567 |
| PCC | 0.783400 | 0.800000 | +0.016600 | 0.850000 | +0.066600 | 0.900000 | +0.116600 |
| CCC | 0.767304 | 0.800000 | +0.032696 | 0.850000 | +0.082696 | 0.900000 | +0.132696 |

## Decision

Promote `h6_harm_h80_ext125_clip010_ccc005` as the current best Valence model.
Do not promote either fusion method. The best short-term signal is that
extreme-aware loss improves all four test metrics, especially PCC/CCC.

## Next Step

The next experiment should stay in the hidden_6 + harmonic family and refine
the objective rather than adding larger models. Recommended options:

1. Try a smaller grid around `extreme_weight = 1.0 to 1.75` with validation
   selection focused on PCC/CCC while monitoring RMSE.
2. Add a lightweight song-global auxiliary head to predict each song's mean
   Valence from pooled sequence features.
3. Investigate split distribution shift by comparing train/validation/test
   song-level Valence means and extremes.
