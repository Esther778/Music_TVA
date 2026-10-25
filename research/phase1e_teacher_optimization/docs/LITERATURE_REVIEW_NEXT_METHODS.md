# Literature Review for the Next Valence Experiments

Date: 2026-06-22

## Scope

This review focuses on methods that could improve strict song-level,
time-continuous DEAM Valence prediction. Static whole-track results, models
using historical emotion labels, and frame-random splits are not considered
directly comparable.

## Findings

### 1. MERT Probing Is Not Task Adaptation

The MERT paper evaluates downstream tasks primarily through probing: the
pre-trained backbone is frozen and a simple downstream model is trained.
MERT was designed to encode acoustic and musical-pitch information using
acoustic teachers and a CQT reconstruction teacher.

Implication:

- our frozen layer-6 experiments are consistent with the original probing
  protocol;
- the next genuinely different step is parameter-efficient representation
  adaptation, not another prediction head on fixed embeddings.

Source:

- https://arxiv.org/abs/2306.00107

### 2. Layer Adapters Are Relevant to Emotion

ELP-Adapters introduce encoder, layer, and prompt adapters. Their layer
adapters create paths from transformer layers to downstream heads and are
reported as useful for non-linguistic tasks including emotion recognition.
The approach can match or exceed full fine-tuning with substantially fewer
trainable parameters.

Implication:

- use a layer adapter or low-rank residual path around MERT layer 6;
- initially freeze the convolutional frontend and most transformer blocks;
- compare against the frozen global-local temporal model under the same
  song-level folds.

This is speech evidence, not direct DEAM evidence, so it must be treated as
a transferable method rather than a guaranteed result.

Source:

- https://arxiv.org/abs/2407.21066

### 3. Metric Learning Has DEAM Valence Evidence

Regression-based triplet neural networks were evaluated on DEAM and reported
a 9% relative Valence improvement over their baseline while greatly reducing
feature dimensionality.

Implication:

- add a song-aware metric loss to the global embedding;
- positive pairs should have similar song-level mean Valence;
- negative pairs should differ by a predefined Valence margin;
- keep the dynamic local branch and frame loss unchanged.

The paper's exact split and static/dynamic evaluation must be checked before
comparing absolute metrics. Its relative result supports the method, not a
numeric target for our pipeline.

Source:

- https://arxiv.org/abs/2001.09988

### 4. Key, Chord, Multitask Learning, and Distillation

A recent unified MER framework combines MERT embeddings with key/chord
features, trains across categorical and dimensional emotion datasets, and
uses knowledge distillation. The authors report that musical features,
multitask learning, and distillation improve performance across datasets
including DEAM.

Implication:

- explicit music-theory auxiliary tasks are more defensible than adding more
  generic Librosa statistics;
- useful auxiliary targets include key/mode and chord-change prediction;
- static emotion datasets may supervise the global song-mean branch while
  DEAM dynamic labels supervise the local branch.

This is not directly comparable to dynamic DEAM frame prediction.

Source:

- https://arxiv.org/abs/2502.03979

### 5. Distribution Shift Is a Known MER Limitation

Recent cross-dataset analysis demonstrates severe MER distribution gaps.
The study finds Jukebox embeddings stronger than MERT in its static setup and
shows that combining Jukebox with chroma and diverse training datasets can
improve cross-dataset generalization.

Implication:

- our observed Train/Validation Valence distribution shift is not an isolated
  implementation issue;
- chroma is useful as a relatively genre-stable feature;
- external static datasets should be used only for global representation or
  auxiliary learning, not as frame-level dynamic labels;
- Jukebox is a legitimate later representation baseline, but is expensive
  and should follow the MERT-adapter experiment.

Source:

- https://arxiv.org/abs/2510.04688

### 6. Dynamic Annotation Delay Can Be Learned

Continuous emotion research shows that annotation delay can be
time-varying. Multi-delay alignment networks and recurrent alignment models
learn non-stationary delays rather than applying one global shift.

Implication:

- our global shift scan selecting zero does not rule out local,
  content-dependent annotation delay;
- a small differentiable local aligner may be tested after representation
  adaptation;
- the alignment range must be tightly constrained and trained only on
  training folds to avoid label warping.

The published evidence is mostly speech emotion, so this is a secondary
experiment rather than the first priority.

Sources:

- https://arxiv.org/abs/1907.03050
- https://arxiv.org/abs/2209.10223

### 7. Separate Valence and Arousal Branches Are Established

Earlier continuous music emotion work used separate recurrent branches for
Valence and Arousal and reported Valence as substantially harder than
Arousal.

Implication:

- our dimension-specific design is methodologically supported;
- Arousal should remain frozen while Valence representation is adapted.

Source:

- https://arxiv.org/abs/1706.02292

## Recommended Experimental Order

### Experiment A: MERT Layer-6 Parameter-Efficient Adapter

Keep the accepted global-local temporal head. Add a bottleneck adapter to
MERT layer 6 or the top portion leading into layer 6. Freeze all other MERT
parameters.

Acceptance:

- five-fold Valence RMSE improvement >= 0.005;
- at least four of five folds improve;
- within-song PCC and frame PCC do not degrade.

### Experiment B: Song-Aware Metric Learning

Add low-weight triplet/ranking loss to the global song embedding. Use only
training-fold song means to construct positive and negative pairs.

Purpose:

- improve global Valence ordering and PCC;
- reduce regression-to-the-mean behavior.

### Experiment C: Music-Theory Auxiliary Tasks

Add key/mode and chord-change prediction heads. Auxiliary targets can be
derived from audio without emotion-label leakage.

Purpose:

- improve harmonic and tonal representation relevant to Valence;
- regularize across genres.

### Experiment D: Local Delay Aligner

Only if A-C improve representation but within-song PCC remains weak. Use a
bounded differentiable delay module, not an unconstrained label warp.

### Deferred

- MERT-330M: expensive; run after adapter evidence.
- Jukebox: valuable representation baseline, but expensive and static-focused.
- multi-dataset training: use static external labels only for the global
  branch or auxiliary teacher.
- full end-to-end MERT fine-tuning: high overfitting and compute risk.

## Decision

The immediate next experiment should be a MERT layer-adapter pilot using one
predeclared cross-validation fold. The previously planned adapter on cached
embeddings is useful only as a cheap screening step; it must not be described
as MERT fine-tuning because gradients cannot update the frozen backbone from
cached features.
