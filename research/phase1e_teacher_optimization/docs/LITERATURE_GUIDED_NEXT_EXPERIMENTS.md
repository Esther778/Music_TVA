# Literature-Guided Next Experiments

Date: 2026-06-23

## Current Diagnosis

Strict five-fold song-level evaluation shows that the present bottleneck is
Valence representation, not prediction-head capacity:

```text
best robust OOF Valence:
RMSE=0.181819
MAE=0.139021
PCC=0.679455
CCC=0.640288
```

Adding a cached residual adapter, metric loss, music-theory auxiliary heads,
or their joint variant did not produce a stable standalone gain large enough
to justify more head-only tuning.

## Methods Worth Adopting

### 1. True Parameter-Efficient MERT Adaptation

MERT's published downstream protocol mainly uses frozen representations. A
small adapter or LoRA path inside transformer layers is a genuinely different
experiment because gradients update the representation before the cached
layer-6 output.

Proposed first trial:

```text
raw waveform
-> frozen MERT convolutional frontend and lower blocks
-> trainable LoRA/adapters in blocks 5-6
-> weighted layer-5/layer-6 fusion
-> accepted global-local Valence head
```

Use a small learning rate for adapters, a larger learning rate for the head,
gradient accumulation, and one predeclared fold before five-fold training.
This must be described as parameter-efficient fine-tuning. The completed
cached adapter must not be described that way.

Sources:

- MERT: https://arxiv.org/abs/2306.00107
- ELP adapters: https://arxiv.org/abs/2407.21066

### 2. Multi-Scale Temporal Valence Modeling

Frame-level Valence changes slowly and depends more on harmonic context than
instantaneous energy. A two-timescale branch is therefore better motivated
than another deeper recurrent head:

```text
local branch: 0.5-2 second acoustic changes
context branch: 10-30 second pooled harmonic trajectory
global branch: song-level tonal and semantic context
```

The three branches should predict local deviation, medium-term trend, and
song mean separately, then reconstruct the frame prediction. Evaluation must
retain frame, 5-second, 10-second, within-song, and song-mean metrics.

### 3. Bounded Dynamic Annotation Alignment

A zero optimum in the previous global shift scan does not exclude
content-dependent human response delay. Continuous-emotion studies use
learned, time-varying alignment.

The safe version for DEAM is a bounded soft alignment over a small range such
as +/-2 seconds. The aligner is fitted only on each training fold, receives a
strong identity penalty, and cannot freely warp the validation labels.

Sources:

- Multi-delay alignment: https://arxiv.org/abs/1907.03050
- Recurrent alignment: https://arxiv.org/abs/2209.10223

## Methods to Use Carefully

### Music-Theory Multitask Learning

Key/mode and chord-change auxiliary tasks improved some folds and contributed
to the ensemble, but their standalone five-fold RMSE gain was only 0.000908.
They should be retained as optional regularizers during true MERT adaptation,
not treated as the main solution.

Source:

- Unified MER with MERT and musical knowledge:
  https://arxiv.org/abs/2502.03979

### Metric Learning

The DEAM literature supports metric learning, but the current batch-hard
song-mean triplet pilot worsened global RMSE and PCC. A later version should
use ordinal pairwise ranking or supervised contrastive learning with soft
distance weights, not repeat the rejected hard thresholds.

Source:

- DEAM triplet regression: https://arxiv.org/abs/2001.09988

### External Datasets and Jukebox

Cross-dataset work suggests that representation and dataset distribution are
major limitations. Static external emotion labels can supervise a global
branch, but must not be presented as dynamic frame labels. Jukebox is a valid
future representation comparison but has substantially higher compute cost.

Source:

- Cross-dataset MER study: https://arxiv.org/abs/2510.04688

## Experimental Order

1. Audit whether raw audio and MERT layer-5/layer-6 hidden states can be
   streamed within current memory limits.
2. Run one-fold true MERT LoRA/adapter pilot.
3. If RMSE improves by at least 0.005 without PCC/CCC degradation, run all
   five folds.
4. Add the multi-scale 10-30 second context branch.
5. Only if within-song PCC remains weak, test bounded dynamic alignment.
6. Freeze architecture before seeds and final locked evaluation.

## Acceptance Rule

A candidate advances only if it:

- improves fold-0 Valence RMSE by at least 0.005, or gives a comparably large
  PCC/CCC gain without RMSE degradation;
- improves at least four of five folds before becoming the teacher;
- does not use legacy validation or test for architecture selection;
- reports RMSE, MAE, PCC, CCC, within-song, song-mean, and fixed-window
  metrics.
