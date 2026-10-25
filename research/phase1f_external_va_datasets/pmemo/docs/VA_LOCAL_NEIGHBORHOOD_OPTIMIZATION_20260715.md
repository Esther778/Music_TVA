# VA Local-Neighborhood Optimization Report

Report time: 2026-07-15 10:36 CST

Current research stage: Phase 1 VA Teacher development. All model selection in
this report used the 652-song PMEmo development pool and its three song-level
folds. The prospective 115-song final locked test was not evaluated or used for
selection.

## Purpose

Perform a small, low-risk optimization around the registered Arousal and
Valence development bests without changing the backbone family or introducing
non-audio inputs.

## Arousal

Reference:

- model: `pmemo_arousal_locked_mert_smile_refine_complete_20260715`
- RMSE / MAE / PCC / CCC:
  `0.179684 / 0.139846 / 0.858232 / 0.848188`

Change:

- Retained the existing nine-branch prediction.
- Added a bounded residual correction from the already extracted 56-dimensional
  Discogs-EffNet MTG-Jamendo mood/theme audio-tag sequence.
- For every held fold, StandardScaler + Ridge was fitted on the other two folds.
- Selected setting: current-time tags, Ridge alpha `30`, residual strength
  `0.09`, correction cap `0.04`.

Candidate:

- model: `pmemo_arousal_locked_effnet_moodtheme_fine_complete_20260715`
- output: `outputs/arousal_locked_effnet_moodtheme_fine_complete_20260715`
- RMSE / MAE / PCC / CCC:
  `0.179600 / 0.139630 / 0.858358 / 0.848826`
- delta:
  `-0.000084 / -0.000216 / +0.000127 / +0.000639`

Song-stratified paired bootstrap, 10,000 resamples:

- RMSE delta 95% CI: `[-0.000375, +0.000194]`
- MAE delta 95% CI: `[-0.000503, +0.000059]`
- PCC delta 95% CI: `[-0.000328, +0.000602]`
- CCC delta 95% CI: `[+0.000136, +0.001156]`

Decision:

- Register as the new point-estimate development best because all four metrics
  improve and all four minimum criteria pass.
- Treat the gain as small. Only the CCC improvement is clearly separated from
  zero in this bootstrap; final-test superiority is not established.

## Valence

Reference:

- model: `pmemo_valence_locked_effnet_moodtheme_tags_balanced_final_complete_20260715`
- RMSE / MAE / PCC / CCC:
  `0.181537 / 0.142310 / 0.823441 / 0.806919`

Checks:

1. A narrow Ridge/correction-cap search found only negligible changes. The
   cap-`0.15` candidate was not promoted because all bootstrap intervals crossed
   zero.
2. A single global, deployable calibration was tested: preserve each song's
   predicted mean and multiply frame deviations around that mean by a common
   amplitude. No labels or metadata are required at inference.
3. The coarse and fine development searches selected amplitude `0.82`; no
   smoothing was selected.

Candidate:

- model: `pmemo_valence_locked_global_amp_final_complete_20260715`
- output: `outputs/valence_locked_global_amp_final_complete_20260715`
- RMSE / MAE / PCC / CCC:
  `0.181372 / 0.142309 / 0.823809 / 0.807109`
- delta:
  `-0.000164 / -0.0000005 / +0.000367 / +0.000189`

Song-stratified paired bootstrap, 10,000 resamples:

- RMSE delta 95% CI: `[-0.000269, -0.000076]`
- MAE delta 95% CI: `[-0.000098, +0.000095]`
- PCC delta 95% CI: `[+0.000168, +0.000620]`
- CCC delta 95% CI: `[+0.000020, +0.000390]`

Decision:

- Register as the new point-estimate development best. RMSE, PCC, and CCC
  improvements are directionally stable under song bootstrap; MAE is
  effectively unchanged.
- Minimum status: MAE, PCC, and CCC pass; RMSE remains above `0.18` by
  `0.001372`.
- The local post-processing path is now closed. Further tuning of amplitude,
  cap, smoothing, or fold-specific calibration is unlikely to provide a
  scientifically meaningful improvement.

## Threshold Comparison

| Dimension | Metric | Current | Minimum | Gap | Good | Excellent |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Arousal | RMSE | 0.179600 | 0.180 | -0.000400 | 0.150 | 0.100 |
| Arousal | MAE | 0.139630 | 0.150 | -0.010370 | 0.120 | 0.080 |
| Arousal | PCC | 0.858358 | 0.800 | +0.058358 | 0.850 | 0.900 |
| Arousal | CCC | 0.848826 | 0.800 | +0.048826 | 0.850 | 0.900 |
| Valence | RMSE | 0.181372 | 0.180 | +0.001372 | 0.150 | 0.100 |
| Valence | MAE | 0.142309 | 0.150 | -0.007691 | 0.120 | 0.080 |
| Valence | PCC | 0.823809 | 0.800 | +0.023809 | 0.850 | 0.900 |
| Valence | CCC | 0.807109 | 0.800 | +0.007109 | 0.850 | 0.900 |

For error metrics, a negative gap means the threshold is passed. For
correlation metrics, a positive gap means the threshold is passed.

## Reproducibility Artifacts

- `scripts/search_pmemo_arousal_locked_effnet_moodtheme_correction.py`
- `scripts/search_pmemo_valence_locked_global_smooth_amp.py`
- `scripts/compare_pmemo_va_paired_song_bootstrap.py`
- `scripts/repair_pmemo_locked_dev_prediction_coverage.py`
- Candidate directories contain frame predictions, candidate tables,
  summaries, and paired-bootstrap JSON files.

Coverage audit:
- The MERT+OpenSMILE inner join had removed five final edge frames across five
  songs. Both reference and candidate tables were restored with within-song
  edge hold.
- Registered complete artifacts contain `31,044/31,044` expected development
  frames, all finite, with no song overlap across folds.

The 115-song final locked test remains untouched. These are development
selection results, not final test results.
