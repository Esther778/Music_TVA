# Phase 1 VA Teacher: Global Review And Next Optimization Decision

Written at: 2026-07-14 CST

Current stage: Phase 1 VA Teacher development. Phase 2 section-aware analysis
has not started. This document records a planning decision only; no new model
was trained and the final locked 115-song test was not accessed.

## Current Reliable Development References

### Valence

- Model: `pmemo_valence_locked_effnet_moodtheme_tags_balanced_final_20260714`
- Development protocol: 652 songs, three locked song-level out-of-fold splits.
- RMSE / MAE / PCC / CCC:
  `0.181546 / 0.142319 / 0.823404 / 0.806874`.
- Minimum-line status: MAE, PCC, and CCC pass; RMSE misses by `0.001546`.
- Status: provisional development best, not a final-test result and not yet a
  single clean arbitrary-MP3 inference package.

### Arousal

- Development candidate:
  `pmemo_arousal_rmse_primary_postprocess_20260629`.
- Development RMSE / MAE / PCC / CCC:
  `0.179278 / 0.138949 / 0.855885 / 0.848522`.
- A later inspected blind audit produced
  `0.201161 / 0.161095 / 0.838482 / 0.831313` before post-processing.
- Interpretation: Arousal correlation is stable, but blind RMSE and MAE did not
  pass. The inspected blind split cannot be reused as a new blind test.

## Experiment Families Already Covered

1. OpenL3, MERT, MuQ, MusiCNN, Discogs-EffNet, and MAEST audio representations.
2. MLP, BiLSTM, TCN, Transformer, Transformer-BiLSTM, tree, Ridge, and ensemble
   prediction heads.
3. Log-Mel, Librosa harmonic/tonal features, OpenSMILE, and deep-feature fusion.
4. DEAM-only, PMEmo-only, DEAM-PMEmo transfer/fusion, and external-dataset
   transfer gates.
5. Song-balanced, extreme-weighted, CCC, clip-mean, clip-variance, temporal
   smoothness, polarity, ranking, and auxiliary-task objectives.
6. Label shifts, smoothing, affine calibration, rescaling, song-mean/bias
   correction, residual correction, stacking, and seed bagging.
7. Strict song-level evaluation, nested correction audits, deployability checks,
   and blind/final-test protocol revisions.

## Global Diagnosis

- The main Valence error is song-level mean/polarity rather than local curve
  smoothness. Local temporal and output-calibration families are saturated.
- Recent gains are at the `0.0001-0.001` scale and bootstrap intervals often
  cross zero. Continuing broad post-hoc search on the same folds risks
  development-set overfitting.
- Fold 3 remains the main distribution-shift bottleneck. Aggregate tuning that
  helps folds 1 and 2 but hurts fold 3 should not be promoted.
- The current Valence best is a multi-stage artifact chain. A clean raw-MP3,
  arbitrary-length Teacher is more important than forcing a development RMSE
  below `0.18` with another fragile correction.
- The final locked 115-song test must remain untouched until one architecture
  and inference policy are frozen.

## Next Optimization Direction

Primary direction: one preregistered, pure-audio, dual-timescale Valence model.

1. Run one music-specific CLAP representation gate on the locked development
   folds. CLAP is used only as an audio feature extractor at inference; no
   lyrics, chord recognizer, or user-provided auxiliary annotation is required.
2. If the gate gives a guarded cross-fold gain, train a clean factorized model:
   a global semantic branch predicts song-level Valence mean and a local
   MERT/EffNet branch predicts a zero-mean dynamic residual.
3. Use one fixed loss family: song-balanced frame MSE, song-mean MSE, mild CCC,
   residual zero-mean constraint, and temporal smoothness. Hyperparameters are
   selected inside the development protocol only.
4. Require improvement or non-degradation on RMSE, MAE, PCC, and CCC, inspect
   all three folds, and reject gains that depend only on aggregate reuse.
5. After freezing the architecture, package raw-MP3 full-song inference, retrain
   on all 652 development songs, and evaluate the final locked 115 songs once.

Fallback: if the CLAP gate is neutral or infeasible, do not resume broad
post-processing. Build the same clean global-local factorization using the
already materialized EffNet mood/theme semantic branch, then freeze and test.

## Explicitly Closed Directions

- Auxiliary chord recognition or a required chord-extraction dependency.
- More MAEST, MusiCNN, EffNet, or MuQ post-hoc correction searches.
- More smoothing, affine calibration, per-fold delay, or song-mean correction
  sweeps on the same predictions.
- Test-driven optimization or repeated use of the final locked test.
