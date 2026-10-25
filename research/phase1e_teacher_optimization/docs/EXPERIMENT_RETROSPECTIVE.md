# Phase 1 Teacher Experiment Retrospective

Date: 2026-06-22

## What Worked

- Replacing frame-random splitting with strict song-level splitting removed
  the largest source of leakage.
- Exact 15-second label alignment and 0.5-second timestamp matching made the
  evaluation reproducible.
- MERT layer 6 was consistently more useful than the final layer alone.
- Handcrafted features and model ensembling produced strong Arousal results.
- Dimension-specific modeling was justified: Arousal reached the requested
  thresholds while Valence remained the bottleneck.
- Reporting RMSE, MAE, PCC, and CCC separately exposed failures that a single
  average RMSE would hide.

## What Did Not Work

- Repeated tuning on one validation split increased validation-overfitting
  risk. The best validation ensemble is a development result, not an
  unbiased performance estimate.
- Increasing model complexity did not solve Valence. Train RMSE frequently
  fell below 0.09 while validation Valence RMSE remained near 0.20.
- Harmonic-context TCN-BiLSTM, target smoothing, and trend-residual heads did
  not provide transferable improvements.
- Affine calibration reduced RMSE in some development experiments but could
  reduce CCC. It cannot improve PCC under a positive linear transform.
- Strong smoothing slightly reduced RMSE but suppressed dynamic variation
  and could lower CCC.
- A single train-derived shadow validation split represented the training
  distribution but not the existing validation distribution.

## Main Diagnosis

The remaining limitation is not primarily BiLSTM depth or loss weighting.
It is a combination of:

1. Valence labels are harder and less directly determined by local acoustic
   energy than Arousal.
2. The current frozen MERT representations are not sufficiently invariant
   across song distributions for Valence.
3. Train and validation Valence distributions differ:

```text
full train mean/std=0.0845/0.2472
frozen validation mean/std=0.0252/0.2901
```

4. The current validation and historical test have already influenced model
   development, so neither should be treated as a pristine final estimate.
5. Frame-weighted metrics allow long songs to contribute more observations
   than short songs. Both micro frame-level and macro song-level metrics are
   needed.

## Experiments To Stop

- More BiLSTM width/depth grids.
- More small loss-weight adjustments on the same validation split.
- More validation-fitted linear ensembles.
- More target smoothing grids.
- Additional local handcrafted feature engineering without a new validation
  protocol.
- MERT-330M or LoRA before establishing a robust cross-validation baseline.

## Next Experimental Program

### Stage A: Evaluation Reset

Use stratified song-level five-fold cross-validation. Stratification should
cover song-level mean Valence, mean Arousal, duration, and clip type.

For every fold report:

- frame-level micro metrics;
- song-macro metrics;
- 5-second and 10-second fixed-window metrics;
- Valence and Arousal separately;
- mean and standard deviation across folds.

Freeze all model and post-processing choices before evaluating the five
folds. Use out-of-fold predictions for calibration or stacking.

### Stage B: Minimal Representation Ablation

Compare only three predeclared systems:

1. MERT final-layer BiLSTM baseline;
2. MERT layer-6 gated BiLSTM;
3. learned scalar mix of MERT layers 3, 6, 9, and final.

Use the same small prediction head and training protocol. This isolates
whether Valence gains come from representation rather than head complexity.

The scalar-mix experiment proceeds only if it improves mean cross-validation
Valence RMSE by at least 0.005 and improves PCC or CCC.

Status update:

- Five-fold Ridge screening confirmed layer 6 beats the final layer in 5/5
  folds.
- Fixed layer6/final mixtures improve monotonically with more layer 6, and
  none beat pure layer 6.
- Frame-micro gains are dominated by between-song mean prediction.
- Layer-6 within-song centered PCC is only 0.0411.

The immediate priority is therefore global-local temporal modeling, not
additional frozen-layer mixing.

Status update:

- The global-local temporal model lowers five-fold Valence RMSE by 0.005490
  relative to global-local Ridge and improves all five folds.
- Cross-fitted OOF stacking reaches RMSE 0.183046 and MAE 0.139923.
- The minimum Valence RMSE/PCC/CCC criteria are still not reached.
- The next justified step is representation adaptation with the same
  global-local head, not additional prediction-head tuning.

### Stage C: Representation Adaptation

Only after Stage B:

- first try a small adapter on the top MERT layers;
- freeze most of MERT;
- train Valence only;
- compare against the frozen scalar-mix model in the same folds.

MERT-330M frozen features are a secondary alternative if adapter training is
not feasible.

### Stage D: Final Evidence

Use OOF predictions for a nonnegative linear ensemble or calibration. Do not
fit ensemble weights on the reporting split.

For final paper evidence use:

- nested/grouped cross-validation, or
- a newly acquired blind external dataset/holdout.

The existing validation and historical test can be reported as secondary
legacy splits, not the only final evidence.

## Decision Rule

The next model is accepted only if it produces a cross-fold improvement that
is both:

- practically meaningful: Valence RMSE improves by at least 0.005;
- consistent: at least four of five folds improve, with no material PCC/CCC
  degradation.

Random seeds are run only after a model passes this fold-level criterion.
