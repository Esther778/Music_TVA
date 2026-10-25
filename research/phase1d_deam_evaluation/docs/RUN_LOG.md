# Phase 1D Run Log

## 2026-06-21

Created a dedicated DEAM evaluation phase.

Confirmed before implementation:

```text
label scale=[-1,+1]
label start=15.0 seconds
label hop=0.5 seconds
split=song-level 1441/180/181
MERT prediction ordering reconstructs exactly from sequence_index.csv
```

The evaluation pipeline will report frame, fixed-window, and clip metrics,
plus mean baseline, label-shift diagnostics, validation-selected smoothing,
and validation-fitted affine calibration.

### Evaluation Completed

Generated:

```text
outputs/multi_level_evaluation.csv
outputs/label_shift_scan.csv
outputs/smoothing_validation_scan.csv
outputs/diagnostic_summary.json
docs/EVALUATION_RESULTS.md
```

Main conclusions:

- `0s` is the validation-RMSE-optimal integer shift.
- Gaussian smoothing with sigma 2 frames reduces test RMSE from `0.202452` to
  `0.200521`.
- Validation-fitted affine calibration does not transfer and is rejected.
- Clip-level aggregation reaches `RMSE=0.187807` and `MAE=0.149477`, but mean
  PCC/CCC remain `0.643315/0.601326`.
- Representation learning, especially Valence, is the next bottleneck.
