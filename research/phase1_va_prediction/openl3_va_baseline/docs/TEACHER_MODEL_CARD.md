# Frozen VA Teacher Model Card

Last updated: 2026-06-21

## Engineering Freeze Decision

The Phase 1 operational VA Teacher is frozen as:

```text
teacher_model_id=openl3_mlp_w512_v1
feature_backbone=OpenL3 mel256 music 512-d
frame_hop=0.5 seconds
head=MLP [512, 256], dropout 0.10
deployment_mode=mean ensemble of seeds 2026, 3407, and 614
output_order=Valence, Arousal
```

The ensemble averages the three models' frame-level VA predictions. The
deployment mode was selected using validation RMSE only. Test metrics were not
used to choose the model, seed, or deployment mode.

Scientific status update on 2026-06-21:

- this configuration remains reproducible and available for engineering,
- song-level validation produced `val_rmse=0.222275` and
  `test_rmse=0.208309` for the corresponding three-seed ensemble,
- MERT+BiLSTM is currently better on the same song split,
- final Teacher selection is reopened pending three-seed MERT+BiLSTM
  verification,
- do not generate the final 95-song pseudo-VA corpus from this configuration
  until that comparison is complete.

See:

```text
docs/SONG_LEVEL_VALIDATION.md
```

Machine-readable configuration:

```text
configs/frozen_teacher_v1.json
```

## Selection Evidence

Best single checkpoint by validation RMSE:

| Item | Value |
| --- | ---: |
| seed | 2026 |
| validation RMSE | 0.109146 |
| test RMSE | 0.108120 |
| test mean CCC | 0.905676 |

Frozen three-seed ensemble:

| Metric | Validation | Test |
| --- | ---: | ---: |
| RMSE | 0.099575 | 0.098425 |
| MAE | 0.074208 | 0.073413 |
| Valence RMSE | 0.099825 | 0.099481 |
| Arousal RMSE | 0.099324 | 0.097357 |
| Valence Pearson r | 0.919577 | 0.922681 |
| Arousal Pearson r | 0.937730 | 0.940997 |
| Valence CCC | 0.904427 | 0.908198 |
| Arousal CCC | 0.928864 | 0.932939 |
| Mean CCC | 0.916645 | 0.920569 |

Compared with the published OpenL3+MLP test RMSE of `0.120297`, the frozen
ensemble reduces RMSE by approximately `18.18%` on the same dataset split and
evaluation protocol. It also reduces test RMSE by approximately `8.97%`
relative to the best single checkpoint selected by validation RMSE.

## Reproducibility

The configuration records:

- architecture and feature parameters,
- all three checkpoint paths and SHA-256 hashes,
- normalization path and SHA-256 hash,
- per-member validation/test metrics,
- ensemble validation/test metrics,
- the selection rule and evaluation limitation.

Rebuild and audit the frozen configuration:

```bash
.venv_openl3_mlp/bin/python \
  research/phase1_va_prediction/openl3_va_baseline/scripts/freeze_va_teacher.py
```

Run inference on raw, precomputed 512-dimensional OpenL3 features:

```bash
.venv_openl3_mlp/bin/python \
  research/phase1_va_prediction/openl3_va_baseline/scripts/predict_frozen_teacher.py \
  INPUT_FEATURES.npy OUTPUT_FRAME_VA.csv
```

The tuned model uses the same training-split normalization statistics as the
published MLP. The audit script verifies this before freezing the Teacher.

## Claim Boundary

This result uses the original OpenL3 frame-level random 80/10/10 split. Frames
from one song may occur in more than one split. Therefore:

- the metrics are valid for reproducing and improving the published protocol,
- they are not a strict song-level generalization estimate,
- they must not be directly compared with the current song-level MERT metrics,
- pseudo-VA generated for Phase 2 must be described as model-derived estimates,
  not human ground truth.

The scientific robustness task that remains is a song-level OpenL3 evaluation
using the same DEAM song split as MERT. That control experiment was completed
on 2026-06-21 and is documented in `docs/SONG_LEVEL_VALIDATION.md`. Its result
reopened final Teacher selection, so Phase 2 final pseudo-VA generation is
blocked until MERT+BiLSTM receives matching three-seed verification.
