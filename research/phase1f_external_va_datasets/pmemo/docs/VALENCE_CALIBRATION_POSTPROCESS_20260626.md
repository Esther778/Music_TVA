# Valence Calibration/Postprocess Trials - 2026-06-26

Report time: 2026-06-26

Current research progress: Phase 1F is optimizing a Valence-specific teacher
model on PMEmo before using the teacher for complete-song VA prediction and
section-aware emotion structure analysis.

## Purpose

The previous split diagnostics showed that the current best Valence model
underpredicts the PMEmo test-set mean by about `-0.0498` and slightly
underpredicts variance. This round tested validation-selected postprocessing
instead of changing the network:

- moving average smoothing by song
- bias correction
- linear calibration
- mean/std calibration
- scale-about-mean calibration

Selection rule:

```text
Choose postprocess configuration by validation pass-gap composite.
The test set is not used for model or postprocess selection.
```

## Thresholds Used For Reporting

| Metric | Minimum pass | Good | Excellent | Direction |
| --- | ---: | ---: | ---: | --- |
| RMSE | 0.180000 | 0.150000 | 0.100000 | lower is better |
| MAE | 0.150000 | 0.120000 | 0.080000 | lower is better |
| PCC | 0.800000 | 0.850000 | 0.900000 | higher is better |
| CCC | 0.800000 | 0.850000 | 0.900000 | higher is better |

## Historical Best Before This Round

| Item | Value |
| --- | --- |
| model_id | `pmemo_valence_mert_hidden6_harmonic_extreme_weighted_v1` |
| trial_id | `h6_harm_h80_ext125_clip010_ccc005` |
| feature | MERT hidden_6 768 + Librosa harmonic 36 |
| model | Valence-only BiLSTM, projection 320, hidden_dim 80, dropout 0.30 |
| loss | extreme-weighted frame MSE + 0.05 frame CCC + 0.10 clip mean MSE |
| extreme_weight | 1.25 |

Historical best test metrics:

| Metric | Historical best | Minimum pass | Gap to pass | Good | Gap to good | Excellent | Gap to excellent |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RMSE | 0.205673 | 0.180000 | +0.025673 | 0.150000 | +0.055673 | 0.100000 | +0.105673 |
| MAE | 0.164567 | 0.150000 | +0.014567 | 0.120000 | +0.044567 | 0.080000 | +0.084567 |
| PCC | 0.783400 | 0.800000 | +0.016600 | 0.850000 | +0.066600 | 0.900000 | +0.116600 |
| CCC | 0.767304 | 0.800000 | +0.032696 | 0.850000 | +0.082696 | 0.900000 | +0.132696 |

## Current Experiment Result

Validation-selected postprocess:

```text
base model = mert_opensmile_fusion
postprocess = linear calibration
moving_average_window = 15 frames
linear a = 0.9693826822188533
linear b = -0.0281918609115746
```

Validation metrics:

```text
RMSE/MAE/PCC/CCC = 0.205076/0.156142/0.837162/0.824111
```

Test metrics:

| Metric | Current experiment | Historical best | Delta vs historical best | Minimum pass | Gap to pass | Good | Gap to good | Excellent | Gap to excellent |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RMSE | 0.212722 | 0.205673 | +0.007049 | 0.180000 | +0.032722 | 0.150000 | +0.062722 | 0.100000 | +0.112722 |
| MAE | 0.174221 | 0.164567 | +0.009654 | 0.150000 | +0.024221 | 0.120000 | +0.054221 | 0.080000 | +0.094221 |
| PCC | 0.766435 | 0.783400 | -0.016965 | 0.800000 | +0.033565 | 0.850000 | +0.083565 | 0.900000 | +0.133565 |
| CCC | 0.742755 | 0.767304 | -0.024550 | 0.800000 | +0.057245 | 0.850000 | +0.107245 | 0.900000 | +0.157245 |

Decision:

```text
Do not promote. The validation-selected postprocess worsened all four test
metrics relative to the historical best.
```

## Diagnostic-Only Observation

The best test RMSE found inside the search was:

```text
base model = best_refine_rmse
postprocess = identity + moving_average_window 15
test RMSE/MAE/PCC/CCC = 0.202868/0.162256/0.788413/0.769605
```

This row is not promoted because it is identified by test-set ranking. It is
only a diagnostic clue that longer smoothing may be useful, but the smoothing
choice must be fixed by validation or by a pre-registered rule before it can be
used as a formal teacher result.

## Conclusion

This round did not improve the official Valence teacher. The current best
remains:

```text
pmemo_valence_mert_hidden6_harmonic_extreme_weighted_v1
h6_harm_h80_ext125_clip010_ccc005
test RMSE/MAE/PCC/CCC = 0.205673/0.164567/0.783400/0.767304
```

The result also confirms that validation/test distribution shift is still the
main blocker. The next useful step should not be another broad calibration
search. A better next experiment is to train a small set of validation-stable
models that directly target the current failure mode:

```text
MERT hidden_6 + harmonic BiLSTM
+ prediction smoothing loss or low-pass temporal head
+ stronger Valence mean/variance regularization
+ strict validation-selected smoothing rule
```

Artifacts:

```text
script = scripts/search_pmemo_valence_calibration.py
outputs = outputs/valence_calibration_20260626/
summary = outputs/valence_calibration_20260626/calibration_summary.json
all rows = outputs/valence_calibration_20260626/all_calibration_results.csv
top rows = outputs/valence_calibration_20260626/top_calibration_results.csv
```
