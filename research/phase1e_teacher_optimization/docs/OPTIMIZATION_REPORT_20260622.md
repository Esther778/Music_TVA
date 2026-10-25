# Teacher Optimization Report - 2026-06-22

## Scope

This round tested whether intermediate and final MERT representations are
complementary for DEAM dynamic Valence-Arousal prediction. The test split
remained locked throughout.

## Implemented Changes

1. Added a three-input model:
   - MERT layer 6;
   - MERT final layer;
   - 118-dimensional handcrafted music features.
2. Added separate feature-wise softmax gates for Valence and Arousal.
3. Added separate BiLSTM branches for Valence and Arousal.
4. Ran one controlled regularization follow-up:
   - projection dimension: 160 to 128;
   - BiLSTM hidden dimension: 96 to 80;
   - dropout: 0.35 to 0.40;
   - weight decay: 0.0003 to 0.0005.
5. Added nonnegative validation-only ensemble optimization.
6. Added Gaussian smoothing constrained within each song.

## Validation Results

| System | RMSE | MAE | Mean PCC | Mean CCC |
| --- | ---: | ---: | ---: | ---: |
| Original MERT+BiLSTM baseline | 0.219934 | 0.171226 | 0.674311 | 0.627697 |
| Layer6 Valence-weighted | 0.200590 | 0.152005 | 0.735974 | 0.723410 |
| Layer6+final+handcrafted | 0.195190 | 0.151260 | 0.757289 | 0.742639 |
| Regularized multilayer | 0.193794 | 0.150065 | 0.754582 | 0.736294 |
| All-model dimension ensemble | 0.184625 | 0.141800 | 0.778162 | 0.757515 |
| Ensemble + recommended smoothing | 0.184392 | 0.141596 | 0.778719 | 0.757787 |

Recommended smoothing uses Gaussian sigma 2 frames, equivalent to a
1-second standard deviation at the 0.5-second frame hop.

## Best Dimension Results

| Dimension | RMSE | MAE | PCC | CCC |
| --- | ---: | ---: | ---: | ---: |
| Valence | 0.202668 | 0.154493 | 0.723530 | 0.693303 |
| Arousal | 0.164094 | 0.128698 | 0.833909 | 0.822270 |

Aggregate MAE is 0.141596. Arousal meets the requested criteria; Valence
remains the bottleneck.

## Interpretation

- Layer 6 and the final MERT layer are complementary.
- Reduced capacity improves Arousal generalization but does not solve
  Valence.
- Ensembling provides the largest gain in this round.
- Aggressive smoothing can lower RMSE further to 0.183599, but it reduces
  CCC and risks suppressing meaningful short-term emotional dynamics.
- The requested global thresholds have not yet been reached on validation,
  so test evaluation is premature.

## Next Technical Priority

Further gains should target Valence representation rather than model depth:

1. add explicit key, mode, chord-change and harmonic-tension descriptors;
2. train a Valence-only model with longer temporal context;
3. evaluate target smoothing or robust regression losses using validation
   only;
4. after freezing the final pipeline, run three seeds;
5. evaluate the locked test once, followed by a new blind holdout or nested
   cross-validation for final paper evidence.

## Additional Valence Experiments

The requested minimum validation RMSE was updated to 0.16, with an ideal
target of 0.08. Two single-seed Valence experiments were completed without
using test data.

### Harmonic Context TCN-BiLSTM

The existing 118-dimensional handcrafted representation already included
local chroma and tonnetz. A new 215-dimensional representation added:

- major/minor template strength and mode margin;
- chroma entropy and tonal concentration;
- chord-change and tonnetz-change magnitude;
- 8-second and 16-second rolling context.

Standalone validation results:

| Metric | Valence |
| --- | ---: |
| RMSE | 0.218055 |
| MAE | 0.167847 |
| PCC | 0.666425 |
| CCC | 0.624306 |

The model was weaker alone but received about 9% weight in the final
Valence ensemble.

### Training-Target Smoothing

Valence labels were smoothed only within training songs using Gaussian
sigma 2 frames. Validation labels remained unchanged.

| Metric | Valence |
| --- | ---: |
| RMSE | 0.213617 |
| MAE | 0.163569 |
| PCC | 0.702392 |
| CCC | 0.665702 |

This did not improve the primary Valence model.

### Updated Recommended Validation Result

| Dimension | RMSE | MAE | PCC | CCC |
| --- | ---: | ---: | ---: | ---: |
| Valence | 0.202493 | 0.153771 | 0.724139 | 0.692638 |
| Arousal | 0.164094 | 0.128698 | 0.833909 | 0.822270 |
| Mean/combined | 0.184297 | 0.141234 | 0.779024 | 0.757454 |

Status against the new criteria:

| Criterion | Required | Current | Status |
| --- | ---: | ---: | --- |
| Validation RMSE | <= 0.16 | 0.184297 | Not reached |
| Validation MAE reference | <= 0.15 | 0.141234 | Reached |
| Mean PCC | >= 0.80 | 0.779024 | Not reached |
| Mean CCC | >= 0.80 | 0.757454 | Not reached |

Test thresholds cannot be assessed because test remains locked.

Further repeated architecture tuning on this validation split should stop.
The next defensible step is either:

1. nested cross-validation or a fresh development holdout for continued
   model selection;
2. a stronger music-pretrained backbone or supervised representation
   adaptation using training data only;
3. label-reliability modeling or multi-annotator uncertainty if raw DEAM
   annotations are available.
