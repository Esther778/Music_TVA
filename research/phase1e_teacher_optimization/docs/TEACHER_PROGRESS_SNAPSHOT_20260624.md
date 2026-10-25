# Phase 1E Teacher Progress Snapshot

Date: 2026-06-24

## Status

The current DEAM-only complete-song Teacher attempt is preserved as an
ablation baseline, not a final scientific Teacher.

The important result is negative but useful:

```text
DEAM complete-song subset = 58 songs
Primary issue = complete-song Valence baseline generalization
Single-fold validation success did not generalize to locked test
Five-fold complete-song outer-test confirms instability
```

## Preserved Candidate Artifacts

Machine-readable registry:

```text
research/phase1e_teacher_optimization/docs/TEACHER_CANDIDATE_REGISTRY_20260624.json
```

### Fold-0 Validation Candidate

```text
Valence checkpoint:
research/phase1e_teacher_optimization/outputs/valence_global_local_full_song_fold0/finetune_best.pt

Arousal checkpoint:
research/phase1e_teacher_optimization/outputs/bilstm_full_song_transfer_fold0/finetune_best.pt

Fold-0 validation postprocess search:
research/phase1e_teacher_optimization/outputs/valence_global_local_full_song_fold0/validation_postprocess_v2_search.json
research/phase1e_teacher_optimization/outputs/valence_global_local_full_song_fold0/validation_postprocess_v2_no_isotonic_wide_search.json
```

Best conservative validation result before locked test:

```text
Valence smoothing = moving average 321 frames = 160.5 s
Arousal smoothing = gaussian sigma 14 frames = 7.0 s
Calibration = quadratic, no isotonic
Validation Mean RMSE/MAE/PCC/CCC = 0.169705/0.132280/0.837199/0.824150
Validation Valence RMSE/MAE/PCC/CCC = 0.181051/0.136789/0.834619/0.821164
Validation Arousal RMSE/MAE/PCC/CCC = 0.157545/0.127771/0.839779/0.827136
```

Locked Fold-0 test result:

```text
Conservative V2 postprocess test Mean RMSE/MAE/PCC/CCC = 0.236773/0.174319/0.400369/0.376076
Conservative V2 postprocess test Valence RMSE/MAE/PCC/CCC = 0.285205/0.203472/0.206967/0.198378
Conservative V2 postprocess test Arousal RMSE/MAE/PCC/CCC = 0.175445/0.145165/0.593771/0.553774
```

Decision: not promoted.

### Five-Fold Valence Global-Local Checkpoints

```text
research/phase1e_teacher_optimization/outputs/valence_global_local_full_song_fold0/finetune_best.pt
research/phase1e_teacher_optimization/outputs/valence_global_local_full_song_fold1/finetune_best.pt
research/phase1e_teacher_optimization/outputs/valence_global_local_full_song_fold2/finetune_best.pt
research/phase1e_teacher_optimization/outputs/valence_global_local_full_song_fold3/finetune_best.pt
research/phase1e_teacher_optimization/outputs/valence_global_local_full_song_fold4/finetune_best.pt
```

Five-fold outer-test Valence result:

```text
RMSE=0.267009
MAE=0.205546
PCC=0.463024
CCC=0.430917
```

Decision: retain as ablation only.

### Complete-Song Global Ridge Baseline

```text
research/phase1e_teacher_optimization/outputs/full_song_global_ridge_cv/summary.json
```

Five-fold outer-test result:

```text
Mean RMSE/MAE/PCC/CCC=0.242049/0.193788/0.513339/0.393422
Valence RMSE/MAE/PCC/CCC=0.253628/0.200424/0.490952/0.407597
Arousal RMSE/MAE/PCC/CCC=0.229888/0.187151/0.535725/0.379247
```

Decision: useful diagnostic baseline, not a dynamic Teacher.

## Current Technical Conclusion

DEAM alone is not enough for the current complete-song Teacher target if the
requirement is strict complete-song generalization. The 58 complete songs are
too few for reliable validation selection, especially for Valence.

The next stage should introduce an external VA dataset, starting with PMEmo,
to strengthen cross-song and cross-dataset emotion baseline learning.

## Do Not Lose

The following facts should be preserved in future reports:

- The OpenL3 frame-random result is an engineering baseline only, because it is
  not a song-level generalization estimate.
- The Fold-0 validation improvement is not final proof because it failed the
  locked outer test.
- The Global-Local Valence model is not the final Teacher.
- PMEmo integration is a data-strategy change, not a small architecture tweak.
