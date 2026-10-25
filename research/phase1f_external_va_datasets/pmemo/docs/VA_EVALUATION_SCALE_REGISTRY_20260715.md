# VA Evaluation Scale Registry

Recorded at: 2026-07-15 CST

Current stage: Phase 1 VA Teacher development. Results below use the 652-song
locked development pool with three disjoint song folds. The final locked
115-song test remains excluded.

## Unified Current Best VA Summary

Common comparison protocol:
- Dataset: PMEmo 2019.
- Label scale: continuous Valence/Arousal in `[-1, 1]`.
- Data: 652 development clips; final 115-song test excluded.
- Split: three disjoint song-level OOF folds.
- Evaluation: aggregate every clip's frame prediction and target to one mean,
  compute metrics within each fold, then report the three-fold macro mean.

| Dimension | Current model | RMSE | MAE | PCC | CCC | Minimum status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Valence | `pmemo_valence_locked_global_shift_amp_guarded_20260715` | 0.178526 | 0.140722 | 0.821365 | 0.805417 | all pass |
| Arousal | `pmemo_arousal_locked_opensmile_static_songmean_fine_20260715` | 0.174339 | 0.133155 | 0.870038 | 0.860214 | all pass |
| Minimum line | - | 0.180000 | 0.150000 | 0.800000 | 0.800000 | - |

This is the canonical table for comparing the current best V and A teachers
under one qualified evaluation scale. It does not replace frame-level metrics
and must not be described as full-length commercial-song or final-test
performance.

## Mandatory Reporting Rule

The statement `Valence RMSE < 0.18` is valid only at the complete-song
clip-level aggregation scale:

```text
aggregate each song's frame predictions to one mean Valence
aggregate each song's frame targets to one mean Valence
compute metrics over songs within each fold
report the macro mean across the three folds
```

It must not be reported as a 0.5-second frame-level result.

## Current Valence Best by Scale

Model:
`pmemo_valence_locked_global_shift_amp_guarded_20260715`

| Scale | RMSE | MAE | PCC | CCC | Minimum status |
| --- | ---: | ---: | ---: | ---: | --- |
| 0.5 s frame | 0.181284 | 0.142308 | 0.824000 | 0.807112 | RMSE not pass; other three pass |
| Complete song/clip | 0.178526 | 0.140722 | 0.821365 | 0.805417 | all four pass |

Interpretation:
- Valence passes the minimum line at complete-song scale.
- Valence does not pass the minimum RMSE line at 0.5-second frame scale.
- Neither result is a final locked-test result.

## Current Arousal Best at Complete-Song Scale

Model:
`pmemo_arousal_locked_opensmile_static_songmean_fine_20260715`

Aggregation: mean frame prediction and mean frame target within each song,
followed by metrics over songs.

| Fold | Songs | RMSE | MAE | PCC | CCC |
| --- | ---: | ---: | ---: | ---: | ---: |
| fold_01 | 225 | 0.183445 | 0.142214 | 0.853187 | 0.841001 |
| fold_02 | 220 | 0.163131 | 0.123611 | 0.881114 | 0.874501 |
| fold_03 | 207 | 0.176442 | 0.133641 | 0.875811 | 0.865140 |
| Three-fold macro | 652 | 0.174339 | 0.133155 | 0.870038 | 0.860214 |

Pooled 652-song secondary check:

```text
RMSE / MAE / PCC / CCC =
0.174574 / 0.133215 / 0.869489 / 0.860215
```

Interpretation:
- Arousal complete-song macro metrics pass all four minimum criteria.
- PCC and CCC reach the good line; RMSE and MAE do not.
- Fold_01 RMSE is above `0.18`, so the correct claim is that the three-fold
  macro result passes, not that every individual fold passes.
- This is development-fold evidence, not final locked-test performance.

## Thresholds

| Metric | Minimum pass | Good | Excellent |
| --- | ---: | ---: | ---: |
| RMSE | <= 0.180 | <= 0.150 | <= 0.100 |
| MAE | <= 0.150 | <= 0.120 | <= 0.080 |
| PCC | >= 0.800 | >= 0.850 | >= 0.900 |
| CCC | >= 0.800 | >= 0.850 | >= 0.900 |

## Paper-Writing Boundary

Allowed:
- `The Valence teacher reached an RMSE of 0.1785 at the complete-song
  aggregation level on locked development folds.`
- `The Arousal teacher reached a complete-song macro RMSE of 0.1743, with PCC
  and CCC above 0.86.`

Not allowed:
- `Valence frame-level RMSE is below 0.18.`
- `All Arousal folds have RMSE below 0.18.`
- `These are final-test results.`
