# MIR Frontier Research for Full-Song Dynamic VA

Date: 2026-06-23

## Research Question

How should the current DEAM Teacher be improved when the actual deployment
target is an unseen complete song rather than a randomly held frame or only a
45-second excerpt?

No training was started during this literature and GitHub review.

## Main Diagnosis

DEAM contains:

```text
1,744 clips of approximately 45 seconds
58 complete songs with an average duration of approximately 4 minutes
```

The current project split distributes the 58 complete songs across train,
validation, and test. Recent dynamic MER work instead treats all 58 complete
songs as a separate target domain. This confirms that the remaining problem
is a short-clip-to-full-song domain shift, not merely insufficient MLP or
BiLSTM capacity.

The current project evidence agrees:

```text
best extended OOF stack, short clips:
RMSE=0.162024, PCC=0.729679, CCC=0.705219

same stack, long songs:
RMSE=0.257576, PCC=0.508516, CCC=0.370302
```

## Relevant Frontier Methods

### 1. MusicFM and MuQ Representation Screening

MusicFM was pretrained with 30-second inputs and explicitly reports that
training input duration is important for downbeat and structure tasks. It
outputs 25 Hz frame representations and provides open checkpoints and code.
This is directly relevant because the current MERT representation was not
pretrained for complete-song dynamic emotion.

MuQ is a 12-layer, approximately 310M-parameter Conformer trained with
30-second inputs and 25 Hz representations. On the MARBLE EmoMusic task,
MuQ's best reported model improves Valence R2 over MERT from 61.2 to 62.8
and provides competitive Arousal results. The public checkpoint and Python
package are available.

Recommended use:

1. extract frozen MusicFM-MSD and MuQ representations;
2. pool each to the DEAM 0.5-second grid;
3. run the existing strict Ridge and global-local Ridge probes;
4. compare long-song folds before building another deep head.

This is a low-scientific-risk experiment because it changes only the
representation while retaining the accepted split and evaluation code.

Sources:

- MusicFM paper and code:
  https://arxiv.org/abs/2311.03318
  https://github.com/minzwon/musicfm
- MuQ paper and code:
  https://arxiv.org/abs/2501.01108
  https://github.com/tencent-ailab/MuQ
- MARBLE benchmark:
  https://arxiv.org/abs/2306.10548
  https://github.com/a43992899/MARBLE-Benchmark

### 2. PDMER Dual-Scale Attention

The AAAI 2025 PDMER work provides public code and evaluates dynamic emotion
on all 58 DEAM complete songs. Its DSAML architecture combines:

```text
frozen global audio representation
+ trainable local spectrogram representation
-> feature fusion
-> local-window attention
+ global-window attention
-> dynamic VA sequence
```

The paper reports the following complete-song DEAM results:

```text
Arousal: RMSE=0.202, PCC=0.675, CCC=0.390
Valence: RMSE=0.260, PCC=0.221, CCC=0.104
```

The absolute Valence score is not a target to copy. The useful contribution
is the explicit local/global attention design and the official full-song
protocol. Ablations show that both local and global attention contribute.

Recommended adaptation:

```text
MERT/MusicFM/MuQ 0.5-second embeddings
-> 128-dimensional projection
-> local attention mask: approximately 5-10 seconds
-> global attention mask: approximately 30-60 seconds or full song
-> separate Valence and Arousal branches
```

Unlike the rejected trend-head experiment, this changes information flow:
each frame can directly attend to short-term and long-term contexts rather
than merely receiving additional smoothed-target losses.

Sources:

- Paper: https://arxiv.org/abs/2412.19200
- Official code: https://github.com/Littleor/Personalized-DMER

### 3. Dynamic Multi-Dataset Pretraining

PMEmo provides 794 variable-length songs with dynamic mean and standard
deviation annotations. PDMER's official repository already contains DEAM
and PMEmo loaders.

Recommended strategy:

1. normalize each dataset's VA scale explicitly;
2. pretrain the local/global temporal head on PMEmo and DEAM training clips;
3. fine-tune on DEAM training data;
4. evaluate only on held complete DEAM songs;
5. include a dataset embedding or domain-specific output calibration.

Do not merge EmoMusic blindly with DEAM because of dataset lineage and
possible overlap. Song IDs and audio fingerprints must be audited first.

### 4. Short-to-Long Domain Generalization

A 2025 cross-dataset MER study demonstrates strong distribution gaps across
DEAM, PMEmo, EmoMusic, WTC, and WCMED. It reports improved cross-dataset
generalization from diverse training data and Jukebox plus chroma features.

For the current task, the domains are:

```text
source domain: 45-second excerpts
target domain: complete songs
```

Recommended low-cost methods:

- sample equal numbers of songs rather than equal numbers of frames;
- construct 45-60 second windows from complete songs during training;
- use a duration/domain token;
- apply a small CORAL or MMD penalty between excerpt and complete-song
  embeddings;
- train with a curriculum: excerpts first, then mixed excerpt/full-song
  windows;
- retain a full-song global token at inference.

The earlier hard short/long expert split was high variance because only
29-32 complete songs were available in each training fold. A shared encoder
with a weak domain-specific residual head is safer than two independent
models.

Source:

- https://arxiv.org/abs/2510.04688

### 5. Multi-Annotator and Uncertainty Modeling

DEAM's 58 complete songs contain individual dynamic annotations. For 1,000
of the 45-second clips, annotator identities are also available. PDMER uses
annotator-specific tasks and MAML; its public code provides a concrete data
loader and implementation reference.

The group-average Teacher does not need to become personalized. A lower-risk
use is:

```text
shared audio encoder
-> mean VA head
-> annotation variance head
-> optional annotator residual heads during training only
```

Useful losses:

- MSE/Huber on the group mean;
- Gaussian negative log likelihood using annotation variance;
- consistency between the average of annotator predictions and the group
  mean;
- lower training weight for high-disagreement frames.

This may reduce label-noise overfitting and improve CCC, especially for
Valence. It requires auditing the raw individual annotation files before
implementation.

### 6. Semantic Valence Features

Valence is more semantic than Arousal. MuQ-MuLan provides a bilingual
music-text embedding space and supports mood descriptions. A frozen global
semantic branch can compute similarities against a controlled prompt bank:

```text
positive / negative
hopeful / despairing
warm / cold
tense / peaceful
```

The resulting similarities should be fused only into the global and
30-second Valence branches. They should not replace frame-level acoustic
features. MuQ-MuLan is approximately 700M parameters, so this is a
high-cost, later experiment.

Source:

- https://github.com/tencent-ailab/MuQ

### 7. Long-Sequence State-Space Models

Audio Mamba and SSAMBA show that state-space models can model long audio
sequences more efficiently than quadratic self-attention. This is relevant
to complete songs, but the current DEAM sequences contain at most 1,223
half-second frames. A projected local/global Transformer is already
computationally feasible on cached embeddings.

Therefore Mamba is not the first experiment. It becomes useful only if the
dual-scale attention model is limited by sequence length or memory.

Sources:

- Audio Mamba: https://arxiv.org/abs/2405.13636
- SSAMBA: https://arxiv.org/abs/2405.11831

## Recommended Evaluation Protocol

Two protocols should be reported:

### Literature-Compatible Full-Song Test

```text
train: 1,744 short DEAM clips
test: all 58 DEAM complete songs
```

This matches PDMER and directly measures short-to-long generalization.

### Development Protocol for Model Selection

```text
outer split: reserve complete songs by song_id
inner split: train on short clips plus only the allowed complete songs
model selection: inner complete-song validation only
final report: untouched outer complete songs
```

The current mixed five-fold OOF result remains useful, but it is not enough
as the sole complete-song deployment claim.

## Prioritized Experiment Order

### Experiment 1: Frozen Backbone Screening

Compare:

```text
MERT-95M layer 6
MusicFM-MSD selected layers
MuQ selected layers
MERT + MusicFM
MERT + MuQ
```

Use Ridge/global-local Ridge first. Stop a backbone if it does not improve
complete-song Valence RMSE by at least 0.005 or PCC/CCC materially.

### Experiment 2: Dual-Scale Attention on the Best Frozen Features

Use local and global masks with separate V/A branches. Keep the model small
and train entirely from cached embeddings.

### Experiment 3: PMEmo Dynamic Pretraining

Pretrain the temporal head on PMEmo plus DEAM clips, then fine-tune on DEAM.
Audit overlap and scale conversion first.

### Experiment 4: Domain-Generalized Full-Song Training

Add domain token, song-balanced sampling, mixed-duration windows, and a weak
CORAL/MMD loss.

### Experiment 5: Annotation-Uncertainty Training

Use individual annotations and variance only after confirming the raw files
and timestamps.

## Expected Value

Highest expected return:

1. frozen MusicFM/MuQ screening;
2. dual-scale local/global attention;
3. PMEmo dynamic pretraining.

Medium expected return:

4. short-to-long domain alignment;
5. multi-annotator uncertainty modeling.

High cost or lower priority:

6. MuQ-MuLan semantic prompt branch;
7. Jukebox;
8. Mamba;
9. further raw-audio MERT LoRA.

The previously tested raw-audio MERT LoRA should remain rejected: it was
computationally expensive and worsened fold-0 RMSE.
