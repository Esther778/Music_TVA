# Valence Static Auxiliary Diagnostic Run

Report time: 2026-06-28

Current research progress: Phase 1 VA Teacher optimization. Arousal has reached
the current minimum target with the OpenSMILE ensemble. Valence is still below
minimum pass and remains the bottleneck.

## Purpose

After song-level mean fusion failed to improve CCC, this diagnostic tested a
stronger idea: add PMEmo `static_annotations.csv` Valence as an auxiliary
song-level target during dynamic Valence model training.

The motivation was reasonable: static Valence may help the model learn the
global positive/negative polarity of a song, which is currently the main source
of Valence error.

## Script

```text
scripts/train_pmemo_valence_multisplit_static_aux.py
```

Output directory:

```text
outputs/valence_multisplit_static_aux_20260628/
```

The script was intentionally added as a separate experiment and does not
overwrite the current best model.

## Current Best Reference

```text
model_id = pmemo_valence_prediction_only_rescale_ccc_preserving_20260628
test RMSE/MAE/PCC/CCC = 0.195316/0.151793/0.797766/0.784834
```

Gap to minimum pass:

| Metric | Current | Minimum pass | Gap |
| --- | ---: | ---: | ---: |
| RMSE | 0.195316 | 0.180000 | +0.015316 |
| MAE | 0.151793 | 0.150000 | +0.001793 |
| PCC | 0.797766 | 0.800000 | +0.002234 |
| CCC | 0.784834 | 0.800000 | +0.015166 |

## Diagnostic Runs

Two settings were tried and stopped early because validation performance was
clearly worse than the current best path.

### Static Aux, Stronger Weight

```text
trial_id = bilstm_static_aux_h80_do030_sw006_cons002
static_head_weight = 0.06
static_frame_consistency_weight = 0.02
```

Observed validation trend before abort:

```text
split_01 best observed val RMSE approximately 0.236, PCC approximately 0.66, CCC approximately 0.66
split_02 early val RMSE approximately 0.25-0.30, PCC/CCC approximately 0.59-0.70
```

### Static Aux, Very Light Weight

```text
trial_id = bilstm_static_aux_h80_do030_sw001_cons000
static_head_weight = 0.01
static_frame_consistency_weight = 0.00
checkpoint selection = frame-level validation metrics only
```

Observed validation trend before abort:

```text
split_01 best observed val RMSE approximately 0.236, PCC approximately 0.66, CCC approximately 0.66
split_02 early val RMSE approximately 0.255-0.308, PCC/CCC approximately 0.59-0.70
```

## Decision

Do not continue this exact static auxiliary direction.

The static PMEmo annotation is useful metadata, but using it directly as a
training auxiliary target appears to conflict with the frame-level dynamic
Valence objective under the current architecture. It pulls the model toward a
coarse global target and does not solve dynamic frame-level CCC.

Current Valence best remains:

```text
pmemo_valence_prediction_only_rescale_ccc_preserving_20260628
test RMSE/MAE/PCC/CCC = 0.195316/0.151793/0.797766/0.784834
```

## Next Direction

The next useful direction should not be another static auxiliary head. Better
options:

1. Package the current best Valence model into an arbitrary-MP3 inference
   script, because the first-stage Teacher must be usable on complete songs.
2. For metric improvement, build out-of-fold predicted Arousal features and use
   them as deployable auxiliary context for Valence. Do not use true Arousal
   labels from the Valence test split.
3. Try a lighter post-training ensemble only if it is selected by validation and
   compared against the current best on the same three splits.
