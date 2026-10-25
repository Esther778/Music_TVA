# Song-Level VA RMSE Improvement Plan

Last updated: 2026-06-21

## Current Starting Point

All future optimization must keep the fixed DEAM song-level split:

```text
train=1441 songs / 101248 frames
validation=180 songs / 13427 frames
test=181 songs / 15320 frames
```

Current best:

```text
MERT-v1-95M last hidden layer
0.5-second mean pooling
BiLSTM baseline
validation RMSE=0.219934
test RMSE=0.202452
test mean CCC=0.538263
```

The frame-level random split must not be used to lower the headline RMSE.

Confirmed label definition:

```text
theoretical range=[-1,+1]
dynamic supervision begins at 15.0 seconds
no 0-1 normalization
```

## Diagnosed Bottlenecks

1. MERT extraction uses only the last hidden layer. Music emotion information
   may be stronger in an intermediate layer or a learned layer mixture.
2. Each target frame receives only a 0.5-second pooled representation. Valence
   often depends on longer harmonic and melodic context.
3. DEAM annotations may contain listener response delay. No label-lag
   calibration has been performed.
4. The current loss weights every valid frame equally. Long songs therefore
   contribute more optimization weight than short songs.
5. The model uses MSE as its main objective. It does not directly optimize CCC
   or trajectory direction.
6. The frozen MERT encoder cannot adapt its representation to VA regression.
7. Valence remains substantially harder than Arousal. Purely local acoustic
   features may be insufficient for harmony, mode, and semantic content.

An OpenL3 timestamp audit also found and corrected a 0.5-second inherited
alignment error. The correction lowered single-seed test RMSE from `0.210084`
to `0.208569`, so it was real but not the main source of the remaining error.

Detailed external-advice mapping and evaluation audit:

```text
docs/EXTERNAL_ADVICE_AUDIT_20260621.md
```

## Experiment Block A: Low-Cost High-Priority Checks

Run these before designing a larger model.

### A1. Annotation Lag Search

Test label shifts on the validation set:

```text
0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0 seconds
```

The shift must be applied consistently within every song, with edge frames
removed rather than padded. Select the shift using validation RMSE only.

Purpose: compensate for the delay between hearing music and moving the
continuous annotation control.

### A2. Multi-Scale Context

Build features centered at each 0.5-second target using:

```text
0.5 s, 1 s, 2 s, 4 s, and 8 s context
```

Start with concatenated mean and standard deviation pooling around each target.
This is cheaper than fine-tuning MERT and tests whether missing temporal context
is the main limitation.

### A3. Song-Balanced Training

Compute the loss as:

```text
mean(frame loss within each song), then mean across songs
```

This prevents long songs from dominating the gradient.

### A4. Better Loss

Compare:

```text
MSE
Huber
MSE + 0.1 * (1 - mean CCC)
MSE + CCC + first-difference trajectory loss
```

Use validation RMSE as the selection metric. CCC and trajectory losses are
supporting regularizers, not replacement test-selection metrics.

## Experiment Block B: Representation Improvement

### B1. MERT Layer Search

Re-extract multiple MERT layers in one forward pass and test:

```text
intermediate layers
last layer
mean of selected layers
learned weighted layer sum
```

Do not rerun MERT separately for each layer. Store all selected layers once.

### B2. Stronger Temporal Head

Test a restrained multi-scale temporal model:

```text
projection -> dilated temporal convolution -> BiLSTM -> residual VA head
```

This should model local changes and longer progression without using an
oversized Transformer on only 1802 songs.

### B3. Separate VA Heads

Use one shared temporal encoder followed by separate Valence and Arousal
regression heads. Their optimal context and regularization may differ.

## Experiment Block C: Higher-Cost Improvements

### C1. Partial MERT Fine-Tuning

Fine-tune only the last one or two MERT blocks with a learning rate much lower
than the regression head. Use short audio windows and gradient accumulation.
Keep the remaining encoder frozen.

This has higher potential than further head-only tuning, but requires more
compute and stronger overfitting controls.

### C2. Add Explicit Musical Features

Concatenate low-dimensional descriptors:

```text
chroma/key/mode
tempo/onset strength
loudness/dynamics
spectral brightness
harmonic-percussive balance
```

This is particularly relevant to Valence and provides interpretable features
for the paper.

### C3. Lyrics Or Semantic Features

For vocal music, add lyric or transcript embeddings as a separate branch.
Audio remains the primary Arousal signal; text may improve Valence.

This requires a clearly defined missing-lyrics policy and should be treated as
a multimodal extension, not mixed into the first acoustic baseline.

## Required Evaluation

Every serious candidate must report:

```text
three random seeds
validation/test RMSE and MAE
Valence and Arousal RMSE separately
Pearson r and CCC
mean and standard deviation
best-single and seed-ensemble results
```

The test set is opened only for final reporting after validation-based
selection.

## Practical Targets

These are internal engineering targets, not guaranteed claims:

```text
minimum useful improvement: validation RMSE below 0.210
strong next milestone: test RMSE below 0.190 with mean CCC above 0.60
preferred Teacher target: test RMSE near or below 0.180 with stable seeds
```

If performance remains around `0.20`, the model can still support
section-level trend analysis with human validation, but it should not be
presented as precise frame-level affect ground truth.

## Graduation Threshold And Feasibility

User-specified graduation target:

```text
Test RMSE <= 0.18
Test MAE <= 0.15
Pearson r >= 0.80
CCC >= 0.80
```

Current validation-selected MERT+BiLSTM test result:

```text
RMSE=0.202452
MAE=0.160708
mean Pearson r=0.576296
mean CCC=0.538263
Valence Pearson r=0.511522
Valence CCC=0.464444
```

Assessment:

- RMSE and MAE targets may be reachable with a stronger representation,
  alignment, fusion, calibration, and ensemble.
- Pearson and CCC targets are substantially harder, especially for Valence.
- Achieving all four under strict song-level, pure-audio evaluation is not
  guaranteed and should not be promised through head-only tuning.
- Historical ground-truth VA inputs could inflate rolling-forecast metrics but
  would change the task and are excluded from the primary model.

Required high-probability route:

1. Search dimension-specific annotation lag using validation only.
2. Extract multiple MERT hidden layers and learn a layer mixture.
3. Fuse MERT with the already extracted 118-dimensional handcrafted features.
4. Use separate Valence/Arousal temporal branches and losses.
5. Optimize MSE + CCC + first-difference trajectory agreement.
6. Partially fine-tune the last MERT blocks with a small learning rate.
7. Add lyric-semantic features for Valence if the pure-audio branch plateaus.
8. Ensemble only validation-selected complementary models.

Evaluation correction:

- The current test set has already been inspected across several experiments.
- It must not be repeatedly optimized against the graduation threshold.
- Final evidence should use nested cross-validation or a newly frozen,
  untouched song-level holdout.

## Recommended Execution Order

1. Complete three-seed MERT+BiLSTM baseline verification.
2. Run annotation-lag search.
3. Run multi-scale context and song-balanced loss experiments.
4. Run MERT layer-mixture experiments.
5. Train the multi-scale temporal head with separate VA outputs.
6. Consider partial MERT fine-tuning only if the frozen-feature experiments
   plateau.
7. Add explicit harmony and lyric features as a later multimodal experiment.

Do not spend another large tuning round only changing hidden size, dropout, and
learning rate. Previous trials show that this produces small gains compared
with the methodological changes above.

## 2026-06-21 Spectral And Handcrafted Ablation Update

Completed in:

```text
research/phase1c_spectral_feature_fusion_va
```

Key validation RMSE:

```text
MERT+BiLSTM=0.219934
log-Mel DAMFF-inspired=0.235476
handcrafted=0.246835
log-Mel+handcrafted fusion=0.239780
```

The ablation does not replace MERT. It shows that handcrafted features improve
Arousal substantially, while MERT remains strongest for Valence. The next
representation experiment should therefore be MERT + handcrafted fusion, not a
larger standalone log-Mel CNN.
