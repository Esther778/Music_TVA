# Phase 1E Run Log

## 2026-06-21

Created after the DEAM multi-level evaluation and Phase 1C feature ablation.

First experiment:

```text
trial=mert_handcrafted_dual_gate_mse
seed=2026
test evaluation disabled during architecture selection
```

The experiment combines MERT and handcrafted features with separate modality
gates and separate temporal branches for Valence and Arousal.

### Trial 1 Completed

```text
trial=mert_handcrafted_dual_gate_mse
seed=2026
best_epoch=7
test_evaluated=false
```

Validation comparison:

| Metric | MERT+BiLSTM baseline | Gated fusion |
| --- | ---: | ---: |
| RMSE | 0.219934 | 0.213873 |
| MAE | 0.171226 | 0.164517 |
| Mean PCC | 0.674311 | 0.686224 |
| Mean CCC | 0.627697 | 0.644727 |
| Valence PCC | 0.581611 | 0.589793 |
| Arousal PCC | 0.767011 | 0.782655 |
| Valence CCC | 0.504360 | 0.520545 |
| Arousal CCC | 0.751032 | 0.768910 |

Interpretation:

- MERT and handcrafted gated fusion improves all primary validation metrics.
- The model overfits rapidly after epoch 7.
- Arousal is close to the `0.80` PCC/CCC target.
- Valence remains the dominant bottleneck.

Next trial:

- keep the same evidence-supported modalities,
- increase dropout and weight decay,
- add low-weight CCC, Pearson, and first-difference losses,
- keep test locked.

### Trial 2 Completed

```text
trial=mert_handcrafted_dual_gate_composite
best_epoch=10
test_evaluated=false
```

Validation:

```text
RMSE=0.214544
Mean PCC=0.690364
Mean CCC=0.658940
Valence CCC=0.544814
Arousal CCC=0.773066
```

Trial 1 remains the lowest-RMSE single model. Trial 2 is retained as a
correlation/CCC and ensemble candidate.

### Validation-Only Ensemble Search

Best shared weights:

```text
MERT baseline=0.35
gated MSE=0.25
gated composite=0.40
validation RMSE=0.208087
Mean PCC=0.710353
Mean CCC=0.661300
test_used=false
```

Dimension-specific search reached:

```text
Arousal RMSE=0.178408
Arousal PCC=0.799291
Arousal CCC=0.780173
```

Valence remains the limiting dimension.

### MERT Multilayer Pilot

Pilot data:

```text
train songs=100
validation songs=60
test used=false
layers=3,6,9,12
```

Lightweight Ridge screening ranked:

| Candidate | RMSE | Mean PCC | Mean CCC |
| --- | ---: | ---: | ---: |
| layer 6 | 0.206815 | 0.707360 | 0.686744 |
| mean layers | 0.209023 | 0.694797 | 0.662371 |
| layer 9 | 0.223881 | 0.647101 | 0.623613 |
| layer 3 | 0.237979 | 0.624415 | 0.620251 |
| layer 12 | 0.239429 | 0.598723 | 0.577885 |

Decision: extract full train/validation layer 6. The model will be truncated
after layer 6 to reduce extraction cost. Test remains unextracted and locked.

### Full Layer 6 Extraction Started

```text
scope=train+validation only
songs=1621
layer=6
pooling=0.5-second mean
storage=float16
test extracted=false
```

The follow-up model uses the established handcrafted features and dual-gate
BiLSTM architecture. Layer 6 frames are matched to DEAM targets by exact
`time_sec` values; array-position truncation is not permitted. Feature
normalization is fitted on training songs only.

### Full Layer 6 Audit and Trial

Audit:

```text
files=1621
train=1441
validation=180
aligned frames=114675
frame range per song=60..1223
missing/extra files=0/0
alignment or finite-value failures=0
```

Layer 6 gated fusion, seed 2026, test locked:

| Metric | Validation |
| --- | ---: |
| RMSE | 0.205030 |
| MAE | 0.154035 |
| Mean PCC | 0.727846 |
| Mean CCC | 0.698420 |
| Valence RMSE / PCC / CCC | 0.228450 / 0.648059 / 0.599883 |
| Arousal RMSE / PCC / CCC | 0.178566 / 0.807633 / 0.796956 |

Best epoch was 12. The model improves over both the previous best single
model and the old three-model ensemble on validation.

### Layer 6 Validation Ensemble

Dimension-specific four-model weights reached:

```text
RMSE=0.198234
MAE=0.150415
Mean PCC=0.745355
Mean CCC=0.698683
Valence RMSE/PCC/CCC=0.223236/0.668312/0.592103
Arousal RMSE/PCC/CCC=0.169585/0.822398/0.805262
test used=false
```

Arousal now satisfies the target thresholds on validation. Valence is the
remaining bottleneck. The next trial increases the Valence loss weight and
uses low-weight correlation and derivative losses; test remains locked.

### Layer 6 Valence-Weighted Trial

Best epoch 21, seed 2026, test locked:

```text
Validation RMSE=0.200590
Validation MAE=0.152005
Mean PCC=0.735974
Mean CCC=0.723410
Valence RMSE/MAE/PCC/CCC=0.212076/0.156764/0.692505/0.673642
Arousal RMSE/MAE/PCC/CCC=0.188405/0.147246/0.779443/0.773178
```

The Valence-weighted loss substantially improves Valence but weakens Arousal.
Combining this model's Valence with the earlier validation-selected Arousal
ensemble gives:

```text
RMSE=0.192009
MAE=0.144502
Mean PCC=0.757452
Mean CCC=0.739452
Valence RMSE/PCC/CCC=0.212076/0.692505/0.673642
Arousal RMSE/PCC/CCC=0.169585/0.822398/0.805262
test used=false
```

This meets the MAE target on validation, while RMSE, mean PCC, and mean CCC
remain short of the requested thresholds. Test remains locked because model
selection is not finished.

## 2026-06-22

### Multilayer Valence Fusion Trial Started

The next experiment adds the existing final-layer MERT sequence to layer 6
and handcrafted features. Valence and Arousal use separate three-way,
feature-wise softmax gates followed by separate BiLSTM branches.

```text
inputs=MERT layer 6 + MERT final layer + handcrafted
selection=validation RMSE
test used=false
seed=2026
```

All three inputs are aligned by `time_sec`; normalization statistics are
fitted from training songs only.

### Multilayer Valence Fusion Result

Best epoch 27, seed 2026, test locked:

```text
Validation RMSE=0.195190
Validation MAE=0.151260
Mean PCC=0.757289
Mean CCC=0.742639
Valence RMSE/MAE/PCC/CCC=0.210590/0.163336/0.707783/0.687046
Arousal RMSE/MAE/PCC/CCC=0.178467/0.139183/0.806794/0.798231
Train RMSE=0.069336
```

The final MERT layer provides a small but consistent Valence improvement.
The large train-validation gap indicates overfitting. One controlled
follow-up reduces projection and recurrent dimensions and increases dropout
and weight decay; no further architecture expansion is planned.

### Regularized Multilayer Fusion

The controlled follow-up reduced parameters from 1,126,862 to 780,206.
Best epoch 15, seed 2026, test locked:

```text
Validation RMSE=0.193794
Validation MAE=0.150065
Mean PCC=0.754582
Mean CCC=0.736294
Valence RMSE/MAE/PCC/CCC=0.215091/0.167307/0.685787/0.654271
Arousal RMSE/MAE/PCC/CCC=0.169848/0.132823/0.823377/0.818316
Train RMSE=0.085113
```

Regularization improves overall RMSE through Arousal but does not improve
Valence. The model is retained as an Arousal ensemble member.

### All-Model Validation Ensemble

Seven available validation predictions were combined with nonnegative
weights constrained to sum to one. Weights were optimized on validation
frame-level RMSE, separately for Valence and Arousal.

```text
Validation RMSE=0.184625
Validation MAE=0.141800
Mean PCC=0.778162
Mean CCC=0.757515
Valence RMSE/PCC/CCC=0.202849/0.723029/0.693043
Arousal RMSE/PCC/CCC=0.164393/0.833294/0.821987
test used=false
```

Relative to the original MERT+BiLSTM validation baseline, RMSE falls from
0.219934 to 0.184625 and MAE falls from 0.171226 to 0.141800. Valence remains
the limiting dimension.

### Song-Bounded Ensemble Smoothing

Gaussian smoothing was applied independently within each song. No values
were mixed across song boundaries.

The strict RMSE optimum was:

```text
sigma=16 frames (8-second standard deviation)
RMSE=0.183599
MAE=0.140915
Mean PCC=0.780490
Mean CCC=0.751937
```

This setting lowers RMSE but weakens CCC and is considered overly aggressive
for dynamic-curve interpretation. The recommended Pareto setting is:

```text
sigma=2 frames (1-second standard deviation)
RMSE=0.184392
MAE=0.141596
Mean PCC=0.778719
Mean CCC=0.757787
Valence RMSE/PCC/CCC=0.202668/0.723530/0.693303
Arousal RMSE/PCC/CCC=0.164094/0.833909/0.822270
```

Current status against the requested validation thresholds:

```text
RMSE <= 0.18: not reached
MAE <= 0.15: reached
Mean PCC >= 0.80: not reached
Mean CCC >= 0.80: not reached
Arousal thresholds: reached
Valence thresholds: not reached
```

Test remains locked. These are model-selection results, not final test
results.

### Updated Target Criteria

The requested criteria are now:

| Metric | Ideal | Minimum |
| --- | ---: | ---: |
| Validation RMSE | <= 0.08 | <= 0.16 |
| Test RMSE | <= 0.09 | <= 0.18 |
| Test MAE | <= 0.07 | <= 0.15 |
| Mean PCC | >= 0.90 | >= 0.80 |
| Mean CCC | >= 0.90 | >= 0.80 |

The current strict song-level validation result is below the MAE threshold
but not the validation RMSE, PCC, or CCC thresholds. Random seeds remain
deferred.

### Valence Harmonic Context Experiment Started

Audit found that the existing 118-dimensional features contain local chroma
and tonnetz statistics, but not explicit key/mode strength, chord-change
rate, tonal stability, or long harmonic context. A new 215-dimensional
feature set was derived for all 1,621 train/validation songs:

```text
local chroma and tonnetz statistics
major/minor template strength and mode margin
chroma entropy and concentration
chord and tonal-change magnitude
8-second rolling mean/std
16-second rolling mean/std
```

The Valence-only model uses layer 6, final-layer MERT, harmonic context,
feature-wise gates, a multi-dilation TCN, and a BiLSTM. Selection is based
only on validation Valence RMSE. Test remains locked.

The first launch stopped before the first optimizer step because the loss
configuration used the key `pearson` while the script read `pcc`. The key was
corrected to `pearson`; no checkpoint or metric from the failed launch is
treated as an experiment result.

The completed harmonic-context trial reached:

```text
Valence RMSE=0.218055
Valence MAE=0.167847
Valence PCC=0.666425
Valence CCC=0.624306
best epoch=29
test used=false
```

It is weaker as a standalone model. In a validation-only Valence ensemble it
received 9.3% weight and changed Valence RMSE from 0.202849 to 0.202709, a
small complementary gain. Further tuning of this architecture is stopped.

### Training-Target Smoothing Trial Started

The regularized multilayer model is trained with Valence labels smoothed
within each training song using Gaussian sigma 2 frames. Validation labels
remain original and unsmoothed. Checkpoint selection uses validation Valence
RMSE rather than joint VA RMSE. Test remains locked.

The target-smoothed trial did not improve Valence:

```text
Valence RMSE=0.213617
Valence MAE=0.163569
Valence PCC=0.702392
Valence CCC=0.665702
best epoch=16
test used=false
```

Training-target smoothing is therefore rejected as the primary model.

### Final Validation Ensemble After Valence Experiments

The harmonic-context and target-smoothed candidates were added to the
nonnegative dimension-specific ensemble. Recommended song-bounded Gaussian
smoothing remains sigma 2 frames.

```text
Validation RMSE=0.184297
Validation MAE=0.141234
Mean PCC=0.779024
Mean CCC=0.757454
Valence RMSE/MAE/PCC/CCC=0.202493/0.153771/0.724139/0.692638
Arousal RMSE/MAE/PCC/CCC=0.164094/0.128698/0.833909/0.822270
test used=false
```

Compared with the previous recommended result, RMSE improves from 0.184392
to 0.184297. The gain is real but too small to justify further tuning on the
same validation split. The minimum validation RMSE threshold of 0.16 is not
reached. Random seeds and test evaluation remain deferred.

### Shadow Validation Split

A fixed song-level split was created from the original 1,441 training songs:

```text
train_sub=1261 songs / 89616 frames
shadow_val=180 songs / 11632 frames
overlap=0
seed=2026
```

The split is stratified by song-level Valence and Arousal quantiles.
Train-sub/shadow mean Valence is 0.09945/0.10097 and standard deviation is
0.23430/0.23693. Existing validation and test remain frozen.

The next comparison retrains a common three-input encoder on train_sub:

1. a raw-Valence baseline;
2. a 5-second/10-second trend plus residual model.

Both models are selected only on shadow-validation Valence RMSE.

### Shadow Trend-Residual Result

Fair comparison on the same train-sub/shadow split:

| Model | Shadow RMSE | MAE | PCC | CCC |
| --- | ---: | ---: | ---: | ---: |
| Raw-Valence baseline | 0.185229 | 0.144518 | 0.671930 | 0.663702 |
| Trend + residual | 0.185096 | 0.142694 | 0.671715 | 0.663185 |

The trend-residual model improves RMSE by only 0.000133 and does not improve
PCC or CCC. It fails the predefined 0.003 improvement threshold.

Shadow-only post-processing selected:

```text
baseline/trend weights=0.5/0.5
shift=0 frames
Gaussian sigma=8 frames
calibration=false
shadow RMSE/MAE/PCC/CCC=0.180815/0.139727/0.683143/0.671636
```

Positive affine calibration reduced shadow RMSE to 0.177199 but reduced CCC
to 0.636380, so calibration was rejected.

### Full-Train Refit and Frozen Validation

The shadow-selected baseline was refit on all 1,441 training songs for 21
epochs. The trend-residual model was refit for 36 epochs. Epoch counts and
post-processing parameters were fixed before frozen validation evaluation.

Frozen validation:

```text
Valence RMSE=0.224760
Valence MAE=0.169021
Valence PCC=0.669015
Valence CCC=0.606852
test used=false
```

This is substantially worse than the existing selected Valence ensemble:

```text
Valence RMSE/MAE/PCC/CCC=0.202493/0.153771/0.724139/0.692638
```

The trend-residual candidate is rejected.

Frame-weighted label distribution audit:

| Split | Valence mean | Valence std |
| --- | ---: | ---: |
| Full train | 0.084464 | 0.247172 |
| Train-sub | 0.082711 | 0.247700 |
| Shadow validation | 0.097969 | 0.242644 |
| Frozen validation | 0.025175 | 0.290057 |

The frozen validation split has a lower mean and wider Valence distribution
than training and shadow validation. This distribution shift helps explain
why the shadow-selected candidate did not transfer. No parameters were
retuned on frozen validation, and test remains locked.

### Five-Fold MERT Representation Screening Started

The next experiment uses only the original 1,441 DEAM training songs. A
fixed stratified five-fold song-level manifest is used. Existing validation
and test are not used.

The low-capacity screening compares:

```text
MERT final layer
25% layer6 + 75% final
50% layer6 + 50% final
75% layer6 + 25% final
MERT layer6
```

All candidates use Ridge alpha 100, selected previously in the multilayer
pilot. Each song has equal total training weight. Evaluation includes frame
micro, song macro, 5-second window, and 10-second window Valence metrics.

### Five-Fold MERT Representation Results

| Representation | Micro RMSE | MAE | PCC | CCC | Better than final |
| --- | ---: | ---: | ---: | ---: | ---: |
| Final layer | 0.215333 | 0.167478 | 0.511855 | 0.461936 | - |
| 25% layer6 | 0.205420 | 0.158858 | 0.570203 | 0.528005 | 5/5 |
| 50% layer6 | 0.198452 | 0.152759 | 0.607996 | 0.571240 | 5/5 |
| 75% layer6 | 0.195065 | 0.149203 | 0.625832 | 0.592134 | 5/5 |
| Layer6 | 0.193463 | 0.147204 | 0.634325 | 0.602373 | 5/5 |

The result is monotonic: increasing layer-6 weight improves all primary
micro metrics. No fixed layer6/final mixture outperforms pure layer 6.

Dynamic decomposition of layer-6 OOF predictions:

```text
song-mean RMSE/PCC/CCC=0.159158/0.734419/0.703161
within-song centered RMSE/PCC/CCC=0.089303/0.041133/0.040586
first-difference RMSE/PCC/CCC=0.070568/0.004340/0.001429
```

Layer 6 is strong for between-song Valence but weak for within-song dynamic
trajectory. Additional MERT layer extraction is deferred because the
existing final layer only weakens the fixed mixtures and the pilot layers
3/9 were substantially worse.

### Five-Fold Global-Local Ridge

The layer-6 representation was decomposed into:

```text
global pooled feature -> song mean Valence
song-centered frame feature -> within-song deviation
final prediction = predicted song mean + zero-mean local deviation
```

Five-fold result:

```text
Micro RMSE/MAE/PCC/CCC=0.190874/0.146924/0.650235/0.624113
Within-song RMSE/PCC/CCC=0.058173/0.121603/0.054934
Song-mean RMSE/PCC/CCC=0.172436/0.693359/0.680748
```

Relative to direct layer-6 Ridge:

```text
Micro RMSE improvement=0.002588
folds with lower RMSE=3/5
within-song PCC improvement=0.080470
```

The decomposition improves within-song correlation materially, but it fails
the predefined acceptance rule of at least 0.005 mean RMSE improvement and
4/5 improving folds. It is retained as an architectural direction, not as a
new selected Teacher.

### Global-Local Temporal Pilot Started

A fixed layer-6 deep model is now tested:

```text
attention pooling -> song mean Valence
song-centered projected sequence -> local BiLSTM -> zero-mean deviation
final prediction = global mean + learned local scale * deviation
```

Losses separately constrain frame prediction, song mean, within-song
deviation, local PCC/CCC, first differences, and variance. Fold 0 is used as
a predeclared pilot. The experiment continues to all folds only if it beats
the fold-0 global-local Ridge RMSE of 0.209961 and within-song PCC of
0.141536.

### Global-Local Temporal Five-Fold Result

Fold 0 passed both pilot conditions:

```text
RMSE=0.208149 < Ridge 0.209961
within-song PCC=0.235306 > Ridge 0.141536
```

The remaining folds were then trained with the same fixed configuration.

Five-fold mean:

| Metric | Temporal | Global-local Ridge |
| --- | ---: | ---: |
| Micro RMSE | 0.185384 | 0.190874 |
| Micro MAE | 0.143655 | 0.146924 |
| Micro PCC | 0.666099 | 0.650235 |
| Micro CCC | 0.624040 | 0.624113 |
| Within-song PCC | 0.142098 | 0.121603 |
| Song-mean PCC | 0.726275 | 0.693359 |
| 10-second RMSE | 0.183008 | 0.189345 |

RMSE improves in 5/5 folds with a mean reduction of 0.005490. The model
passes the predeclared RMSE acceptance rule. Within-song PCC improves on
average but only in 3/5 folds, so local-dynamic generalization remains
unstable.

### Cross-Fitted OOF Stacking

Three OOF systems were combined:

```text
direct layer6 Ridge
global-local Ridge
global-local temporal model
```

For each held fold, nonnegative weights summing to one were fitted using only
the other four OOF folds. The held fold was never used to learn its weights.

```text
Valence micro RMSE=0.183046
Valence MAE=0.139923
Valence PCC=0.673847
Valence CCC=0.631831
Within-song PCC=0.133477
Song-mean PCC/CCC=0.732268/0.703598
10-second RMSE/PCC/CCC=0.180933/0.679874/0.636939
```

The temporal model received 0.5168-0.5527 weight across folds, showing
stable complementary value.

Cross-fitted positive affine calibration was also tested. It worsened RMSE
to 0.183305 and reduced PCC/CCC to 0.670851/0.622465, so it was rejected.

Legacy validation and test were not used in any five-fold experiment.

### Literature-Guided Embedding Adapter Pilot

Before attempting expensive audio-side MERT fine-tuning, a cached-embedding
adapter is used as a screening experiment:

```text
MERT layer6 embedding
-> LayerNorm
-> 768-to-64-to-768 bottleneck residual adapter
-> existing global-local temporal head
```

The adapter up-projection is zero-initialized, so the model starts exactly
from the original layer-6 representation. This experiment must be described
as embedding adaptation, not MERT backbone fine-tuning. Fold 0 uses the same
seed, split, head, losses, and stopping rule as the accepted temporal model.

The five-fold adapter result was:

```text
RMSE=0.184267 versus 0.185384 baseline
PCC=0.668740 versus 0.666099
CCC=0.632514 versus 0.624040
within-song PCC=0.129591 versus 0.142098
mean RMSE improvement=0.001117
RMSE improved folds=5/5
within-song PCC improved folds=1/5
```

The adapter fails the predeclared 0.005 RMSE threshold and degrades local
dynamic correlation. Cached embedding adaptation is rejected.

### Song-Aware Metric Learning Pilot Started

The accepted no-adapter global-local model is retained. A low-weight
batch-hard triplet loss is applied to the global song embedding:

```text
positive: song mean Valence distance <= 0.05
negative: song mean Valence distance >= 0.20
triplet margin=0.20
loss weight=0.05
```

This follows DEAM metric-learning evidence and targets song-level ordering
and PCC. Fold 0 is the predeclared pilot; local dynamic metrics must not
materially degrade.

The metric-learning pilot failed:

```text
RMSE=0.210778 versus 0.208149 baseline
PCC=0.629979 versus 0.639057
song-mean PCC=0.721751 versus 0.729928
within-song PCC=0.244374 versus 0.235306
```

The local metric improves, but overall and global metrics degrade. No further
folds or metric-weight tuning are performed.

### Music-Theory Auxiliary Pilot Started

The no-adapter global-local model now adds:

```text
global embedding -> 24-class key/mode prediction
local hidden sequence -> frame chord-change regression
```

Targets are derived from audio chroma, not emotion labels. All 24 key/mode
classes are represented in the training set. Auxiliary weights are fixed at
0.02 for key/mode cross entropy and 0.05 for chord-change SmoothL1.

Five-fold music-auxiliary mean:

```text
RMSE=0.184476
PCC=0.673869
CCC=0.641182
within-song PCC=0.169025
RMSE improved folds=2/5
```

It fails the standalone RMSE acceptance rule but provides complementary
predictions. An extended cross-fitted stack using layer6 Ridge,
global-local Ridge, embedding adapter, and music auxiliary reaches:

```text
RMSE=0.181819
MAE=0.139021
PCC=0.679455
CCC=0.640288
10-second RMSE=0.179733
song-mean PCC/CCC=0.737172/0.711316
```

The no-adapter temporal model receives zero meta weight, while embedding
adapter and music auxiliary receive stable non-zero weights across folds.

### Joint Adapter and Music-Auxiliary Pilot Started

The final cached-feature screening combines the 64-dimensional residual
adapter with key/mode and chord-change auxiliary losses. Fold 0 proceeds only
as a predeclared pilot. It must outperform both adapter-only and
auxiliary-only fold-0 results to justify full cross-validation.

The fold-0 pilot completed at epoch 24:

```text
joint adapter + auxiliary:
RMSE=0.200282
MAE=0.154245
PCC=0.664516
CCC=0.626859
within-song PCC=0.312922

adapter only:
RMSE=0.206676
PCC=0.641140
CCC=0.597402
within-song PCC=0.249126

auxiliary only:
RMSE=0.202388
PCC=0.670165
CCC=0.628671
within-song PCC=0.398827
```

The joint model lowers fold-0 RMSE by 0.002106 relative to the auxiliary-only
model, but slightly lowers PCC and CCC and materially lowers within-song PCC.
It therefore does not dominate both component models under the predeclared
multi-metric rule. No additional folds are run.

This concludes cached-feature architecture screening. The strongest robust
result remains the cross-fitted extended stack:

```text
RMSE=0.181819
MAE=0.139021
PCC=0.679455
CCC=0.640288
```

The next materially different experiment must update the MERT representation
itself from raw audio, or introduce a strictly bounded temporal alignment
mechanism. More heads on fixed layer-6 embeddings are not prioritized.

### True MERT LoRA Feasibility and Fold-0 Pilot

A separate raw-audio training path was added. It truncates MERT-v1-95M after
encoder block 6 and installs rank-8 LoRA modules in the query and value
projections of blocks 5 and 6. The accepted global-local Valence head is
loaded from the matching fold and can be frozen.

Environment audit:

```text
device=Apple M3 MPS
memory=16 GB
transformers=4.44.0
MERT checkpoint=local cache
all DEAM audio paths valid=1802/1802
trainable LoRA parameters=49,152
```

The first implementation used approximately 17-second audio chunks. Its
zero-update fold-0 score was RMSE 0.215742, showing that shorter context
already lost information relative to the accepted 60-second feature
extraction baseline. One joint-head training epoch improved this only to:

```text
RMSE=0.214701
MAE=0.159740
PCC=0.618447
CCC=0.553640
within-song PCC=0.052404
```

This short-context variant was rejected.

The implementation was then changed to exactly reproduce the historical
60-second MERT extraction context. On the same five held songs, raw-audio
recomputation and cached layer-6 evaluation matched within approximately
1e-6 RMSE:

```text
cached RMSE=0.14337502
raw-audio recomputation RMSE=0.14337599
```

The full-context pilot froze the accepted global-local head and trained only
the four LoRA modules for one complete fold-0 epoch:

```text
train songs=1,152
held songs=289
trainable parameters=49,152
training and first full validation time=18,793.6 seconds

accepted frozen-feature baseline:
RMSE=0.208149
PCC=0.639057
CCC=0.582024
within-song PCC=0.235306

60-second MERT LoRA:
RMSE=0.211506
PCC=0.639784
CCC=0.578858
within-song PCC=0.245389
```

LoRA improves within-song PCC by 0.010083 but worsens RMSE by 0.003357 and
slightly lowers CCC. It is rejected and will not proceed to more epochs,
seeds, or five folds. The experiment also demonstrates that repeated
raw-audio adaptation is computationally unsuitable on the current machine;
future MERT-top-block experiments would need cached lower-layer states.

### Short-Clip and Long-Track Error Audit

The DEAM training subset is heterogeneous:

```text
short 60-frame clips=1,402 songs / 84,120 frames
long tracks=39 songs / 17,128 frames
long-track share=2.7% of songs but 16.9% of frames
```

The strongest robust extended OOF stack separates as:

```text
short clips:
RMSE=0.162024
MAE=0.127939
PCC=0.729679
CCC=0.705219

long tracks:
RMSE=0.257576
MAE=0.193215
PCC=0.508516
CCC=0.370302
```

Long tracks are the dominant source of the aggregate RMSE gap.

A cross-fitted clip-type-specific stack was tested, learning separate
nonnegative weights from the same clip type in the other four folds. It
worsened aggregate RMSE from 0.181819 to 0.182816 because each long-track
meta-training subset contained only 29-32 songs. This stack was rejected.

Finally, layer-6 Ridge models were trained separately for short clips and
long tracks under the same five folds and fixed alpha=100:

```text
clip-type direct Ridge:
overall RMSE=0.176925
MAE=0.138954
PCC=0.703481
CCC=0.684054

clip-type global-local Ridge:
overall RMSE=0.177411
MAE=0.139398
PCC=0.703268
CCC=0.686490
```

The direct clip-type Ridge lowers robust OOF RMSE by 0.004894 relative to the
previous best extended stack and is the new lowest strict OOF Valence RMSE.
Its correlation metrics also improve, although the minimum PCC/CCC threshold
of 0.80 is still not reached. This result shows that clip-duration regime is
a meaningful modeling variable, but the 39-song long-track expert remains
high variance across folds.

### Full-Song Evaluation Protocol Frozen

The project inventory was re-audited against the actual audio durations and
valid dynamic-label sequences:

```text
short clips=1,744
complete songs=58
complete-song IDs=2001..2058
song 2000=45-second short clip
missing audio paths=0
missing sequence paths=0
```

Two protocols were generated:

1. PDMER-compatible: all 1,744 short clips train, all 58 complete songs test.
2. Primary nested evaluation: five outer complete-song folds, with complete
   songs used separately for training, validation, and untouched testing;
   all short clips are training-only.

The complete-song folds are balanced by Valence/Arousal quadrant, duration,
and total frames. Ten thousand stratified candidates were searched using
distribution statistics only; no model predictions were inspected. Selected
seed is 7080.

All five nested splits have disjoint train/validation/test song IDs. This
protocol replaces the old mixed-duration five-fold split for future Teacher
selection. Existing results remain valid historical evidence but are not the
final arbitrary-duration-song evaluation.

### Dual-Scale Full-Song VA Fold-0 Pilot

A compact shared-input, dimension-specific model was trained using:

```text
MERT layer 6 + MERT final layer + 118-d handcrafted features
-> separate Valence/Arousal source gates
-> 5-second local window attention
-> 10-second pooled global Transformer context
-> full-song context token
-> separate Valence/Arousal outputs
```

Protocol:

```text
outer fold=0
training=1,744 short clips + 34 complete songs
validation=12 complete songs
outer test=12 complete songs, not loaded
parameters=1,280,890
selection=0.65 * Valence RMSE + 0.35 * Arousal RMSE
elapsed=735.5 seconds
```

Best checkpoint was epoch 23:

| Dimension | RMSE | MAE | PCC | CCC |
| --- | ---: | ---: | ---: | ---: |
| Valence | 0.286758 | 0.198130 | 0.546796 | 0.446766 |
| Arousal | 0.190826 | 0.158357 | 0.757155 | 0.726246 |

Additional diagnostics:

```text
10-second Valence/Arousal RMSE=0.283979/0.183493
song-mean Valence/Arousal RMSE=0.192727/0.137525
within-song Valence/Arousal PCC=0.395671/0.768188
```

The same-fold training-mean baseline gives Valence/Arousal RMSE
0.335492/0.290197, so the model learns useful signal. It nevertheless falls
well short of the Teacher thresholds and is rejected. It will not proceed to
the outer test fold or the other four folds.

Training became non-finite after epoch 27. The best epoch-23 checkpoint is
intact. The training script now stops immediately on non-finite loss or
validation metrics.

Interpretation:

- Arousal transfers to complete songs substantially better than Valence.
- The 1,744 excerpt training samples still dominate the 34 complete-song
  training samples despite song-balanced loss.
- Adding local/global attention alone does not solve excerpt-to-complete-song
  domain shift.

### Leakage Audit Before BiLSTM Transfer

Directly fine-tuning the historical `bilstm_baseline_best.pt` checkpoint was
rejected before training. Its original mixed song-level split had already
used 7 of the current Fold-0 validation complete songs and 6 of the current
Fold-0 outer-test complete songs for training. Reusing that checkpoint would
therefore invalidate the frozen full-song protocol.

The replacement experiment used the same 920,066-parameter BiLSTM structure,
but trained it from scratch under the frozen protocol:

```text
stage 1: pretrain on 1,744 short clips
stage 2: fine-tune on 34 complete songs
validation: 12 complete songs
outer test: 12 complete songs, not loaded
normalization: computed from the 1,778 allowed training songs only
```

### Leakage-Free BiLSTM Full-Song Transfer Fold 0

Configuration:

```text
hidden_dim=128
layers=1
dropout=0.2
pretrain_lr=5e-4
finetune_lr=5e-5
Valence/Arousal loss weights=1.3/1.0
selection=0.65 * Valence RMSE + 0.35 * Arousal RMSE
seed=2026
elapsed=111.0 seconds
```

The short-clip pretraining checkpoint was selected at epoch 4. Complete-song
fine-tuning continued to epoch 29.

| Stage | V RMSE | V MAE | V PCC | V CCC | A RMSE | A MAE | A PCC | A CCC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Short-only pretrain | 0.339458 | 0.249752 | 0.291911 | 0.198306 | 0.190261 | 0.159104 | 0.756904 | 0.713502 |
| Full-song fine-tune | 0.303597 | 0.237861 | 0.399460 | 0.324179 | 0.178970 | 0.146207 | 0.788292 | 0.759716 |

Aggregate complete-song validation:

```text
RMSE=0.249200
MAE=0.192034
Mean PCC=0.593876
Mean CCC=0.541947
```

Decision:

- Complete-song fine-tuning is useful: aggregate RMSE improves by 0.025964,
  Valence RMSE by 0.035861, and Arousal RMSE by 0.011291 relative to the
  leakage-free short-only checkpoint.
- Arousal reaches RMSE below 0.18 and MAE below 0.15 on the Fold-0 validation
  songs, but its PCC and CCC remain slightly below 0.80.
- Valence remains far below the required standard and limits the model.
- This checkpoint is not promoted to Teacher status and the outer test remains
  locked.

### Valence Global-Local Full-Song Expert

A Valence-only global-local expert was trained to address the failure mode
observed above: wrong whole-song Valence baseline. The model uses:

```text
MERT layer 6 + MERT final layer + 118-d handcrafted features
-> source gate
-> global branch: attention pool + mean + std -> song mean Valence
-> local branch: zero-mean BiLSTM residual
-> final Valence = global mean + residual scale * local residual
```

Training protocol:

```text
outer fold=0
short pretrain songs=1,744
full-song fine-tune songs=34
validation complete songs=12
outer test complete songs=12, not loaded
parameters=440,627
```

The short-clip pretraining stage showed negative transfer at first and selected
epoch 11 (`Valence RMSE=0.327036`). Full-song fine-tuning then corrected the
global branch and selected epoch 18.

Valence-only validation comparison:

| Model | V RMSE | V MAE | V PCC | V CCC |
| --- | ---: | ---: | ---: | ---: |
| Dual-scale full-song pilot | 0.286758 | 0.198130 | 0.546796 | 0.446766 |
| Transfer BiLSTM | 0.303597 | 0.237861 | 0.399460 | 0.324179 |
| Global-local Valence expert | **0.210011** | **0.160227** | **0.770921** | **0.750184** |

Additional Valence diagnostics:

```text
song-mean RMSE/MAE/PCC/CCC=0.160297/0.111520/0.819314/0.800286
within-song RMSE/MAE/PCC/CCC=0.134312/0.104879/0.255864/0.255797
source weights layer6/final/handcrafted=0.423472/0.322396/0.254132
```

Combining this Valence expert with the best current Arousal branch from
`bilstm_full_song_transfer_fold0` gives:

| Level | Mean RMSE | Mean MAE | Mean PCC | Mean CCC | V RMSE | A RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| frame | 0.195109 | 0.153217 | 0.779607 | 0.754950 | 0.210011 | 0.178970 |
| 5 s window | 0.189219 | 0.148964 | 0.797416 | 0.768278 | 0.208474 | 0.167769 |
| 10 s window | 0.185729 | 0.146026 | 0.806635 | 0.775701 | 0.206463 | 0.162370 |

Decision:

- This is the strongest strict complete-song Fold-0 result so far.
- It is still not a final Teacher because frame-level RMSE, MAE, PCC, and CCC
  remain just outside the minimum target.
- The outer test remains locked.
- Next optimization should keep the global-local Valence structure and focus
  on full-song-dominant training, not additional generic short-clip pretraining.

### Full-Song-Only Valence Ablation

The same Valence global-local architecture was trained without short-clip
pretraining and with feature normalization computed only from the 34 complete
training songs.

Result:

```text
best epoch=2
Valence RMSE/MAE/PCC/CCC=0.292363/0.232830/0.540656/0.531351
hybrid frame RMSE/MAE/Mean PCC/Mean CCC=0.242391/0.189518/0.664474/0.645533
```

Decision: rejected. Complete-song-only training is too data-poor. The previous
short-pretrain plus complete-song fine-tune model remains the best Valence
model, despite the negative-transfer behavior during the pretraining stage.

### Validation-Selected Smoothing And Calibration

Post-processing was searched on the Fold-0 validation set only. The selected
configuration is:

```text
Valence smoothing=151 frames = 75.5 seconds
Arousal smoothing=41 frames = 20.5 seconds
Valence calibration: pred = 1.075922 * pred - 0.014892
Arousal calibration: pred = 1.204416 * pred - 0.013322
test_used=false
```

Validation result after smoothing and calibration:

| Metric | Valence | Arousal | Mean |
| --- | ---: | ---: | ---: |
| RMSE | 0.194891 | 0.159030 | 0.177867 |
| MAE | 0.148454 | 0.129251 | 0.138852 |
| PCC | 0.805250 | 0.836447 | 0.820848 |
| CCC | 0.786722 | 0.823283 | 0.805003 |

Decision:

- This is the first strict Fold-0 complete-song validation result that meets
  the overall minimum thresholds: `RMSE<=0.18`, `MAE<=0.15`,
  `Mean PCC>=0.80`, and `Mean CCC>=0.80`.
- It still does not meet the stricter per-dimension Valence RMSE target
  (`Valence RMSE<=0.16`), so it should be reported as validation-stage
  progress, not final proof.
- The smoothing windows are long, especially for Valence. This is defensible
  for complete-song emotion trend estimation, but the paper must state that
  it reduces high-frequency dynamics.
- The outer test remains locked. The next step is to apply these frozen
  post-processing parameters to the Fold-0 outer test once the user approves
  test evaluation.

### Target-Smoothed Valence Training Attempt

To test whether the validation benefit from long-range smoothing should be
moved into training, the Valence global-local script was extended with optional
`target_smoothing_frames` in the training loss. The evaluation target remains
the original DEAM label, so this does not smooth the metric itself.

Files added or updated:

```text
configs/valence_global_local_target_smooth.json
scripts/train_valence_global_local_full_song.py
scripts/evaluate_validation_postprocess.py
```

Attempt 1 applied the smoothed target during short-clip pretraining. It was
stopped after five pretraining epochs because validation Valence RMSE degraded
from `0.329671` initial to the `0.36-0.41` range. Decision: rejected because
short clips do not carry enough complete-song context for this objective.

Attempt 2 reused the original leakage-free pretraining checkpoint and applied
the smoothed target only during complete-song fine-tuning:

```text
initial checkpoint=outputs/valence_global_local_full_song_fold0/pretrain_best.pt
skip_pretrain=true
target_smoothing_frames=61
fine-tune songs=34
validation complete songs=12
outer test not loaded
best fine-tune epoch=15
```

Validation comparison:

| Model | V RMSE | V MAE | V PCC | V CCC | Mean RMSE | Mean MAE | Mean PCC | Mean CCC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Original global-local raw | **0.210011** | **0.160227** | **0.770921** | **0.750184** | **0.195109** | **0.153217** | **0.779607** | **0.754950** |
| Target-smoothed fine-tune raw | 0.216441 | 0.162128 | 0.756288 | 0.731866 | 0.198592 | 0.154167 | 0.772290 | 0.745791 |

The target-smoothed model was also evaluated with the same validation-selected
song-bounded smoothing plus linear calibration search:

| Model | V Smooth Frames | A Smooth Frames | V RMSE | A RMSE | Mean RMSE | Mean MAE | Mean PCC | Mean CCC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Original global-local postprocessed | 181 | 41 | **0.192896** | 0.159030 | **0.176776** | **0.138350** | **0.823066** | **0.807627** |
| Target-smoothed fine-tune postprocessed | 181 | 41 | 0.200582 | 0.159030 | 0.181002 | 0.138869 | 0.814328 | 0.797238 |

A two-model validation ensemble between original Valence and target-smoothed
Valence selected `100%` original and `0%` target-smoothed. Decision: the
target-smoothed training path does not improve the current Teacher candidate
and should remain a rejected ablation.

Current best validation-stage candidate remains:

```text
Valence: original global-local model
Arousal: leakage-free BiLSTM transfer model
Postprocess: validation-selected song-bounded smoothing + linear calibration
Valence smoothing=181 frames = 90.5 s
Arousal smoothing=41 frames = 20.5 s
Mean RMSE/MAE/PCC/CCC=0.176776/0.138350/0.823066/0.807627
test_used=false
```

Important caveat: the validation result now meets the overall minimum target,
but it is validation-selected and uses a very long Valence smoothing window.
The outer test is still locked and must be evaluated only once the user
approves applying the frozen model and frozen postprocessing parameters.

### Validation Postprocess V2: Fine Smoothing And Nonlinear Calibration

Following the user's request, two postprocessing-only directions were tested on
Fold-0 validation:

1. Finer smoothing search, including moving average, Gaussian smoothing, and
   Savitzky-Golay smoothing.
2. Nonlinear calibration, including quadratic calibration and isotonic
   calibration.

No model weights were changed and the Fold-0 outer test was not loaded.

Files:

```text
scripts/search_validation_postprocess_v2.py
outputs/valence_global_local_full_song_fold0/validation_postprocess_v2_search.json
outputs/valence_global_local_full_song_fold0/validation_postprocess_v2_no_isotonic_wide_search.json
```

Baseline before this search:

```text
Valence smoothing=181 frames
Arousal smoothing=41 frames
linear calibration
Mean RMSE/MAE/PCC/CCC=0.176776/0.138350/0.823066/0.807627
Valence RMSE/MAE/PCC/CCC=0.192896/0.147450/0.809685/0.791971
Arousal RMSE/MAE/PCC/CCC=0.159030/0.129251/0.836447/0.823283
```

Conservative result, excluding isotonic calibration:

```text
Valence smoothing=moving_average, 321 frames = 160.5 s
Arousal smoothing=gaussian, sigma=14 frames = 7.0 s
Valence calibration=quadratic
Arousal calibration=quadratic
Mean RMSE/MAE/PCC/CCC=0.169705/0.132280/0.837199/0.824150
Valence RMSE/MAE/PCC/CCC=0.181051/0.136789/0.834619/0.821164
Arousal RMSE/MAE/PCC/CCC=0.157545/0.127771/0.839779/0.827136
```

Interpretation:

- Valence CCC improves from `0.791971` to `0.821164`, crossing the 0.80 line.
- Valence RMSE improves from `0.192896` to `0.181051`, but remains `0.001051`
  above the 0.18 target.
- Mean metrics comfortably meet the minimum target.
- The 321-frame Valence smoothing window is very long, so this should be
  described as complete-song emotional trend postprocessing, not fine-grained
  frame dynamics preservation.

Diagnostic result with isotonic calibration:

```text
Valence smoothing=moving_average, 221 frames = 110.5 s
Arousal smoothing=gaussian, sigma=14 frames = 7.0 s
Valence calibration=isotonic
Arousal calibration=isotonic
Mean RMSE/MAE/PCC/CCC=0.160899/0.123552/0.854844/0.844437
Valence RMSE/MAE/PCC/CCC=0.170801/0.126867/0.854383/0.843911
Arousal RMSE/MAE/PCC/CCC=0.150347/0.120238/0.855306/0.844964
```

Interpretation:

- Isotonic calibration makes all validation metrics pass, including Valence
  RMSE.
- Because only 12 complete validation songs are available and the isotonic
  mapping has high flexibility, this result is diagnostic only and should not
  be treated as the formal Teacher unless it generalizes on the locked outer
  test or future folds.

Current recommended reporting hierarchy:

1. Conservative postprocess V2 as the strongest non-isotonic validation
   candidate.
2. Isotonic postprocess as an exploratory upper-bound diagnostic.
3. Original linear postprocess as the safer baseline if the paper needs the
   least aggressive postprocessing story.

### Fold-0 Locked Outer Test: Frozen V2 Evaluation

The user approved continuing to the locked Fold-0 outer test. The frozen
conservative V2 candidate was evaluated once on the 12 complete-song test set.
No test-set parameter selection was performed.

Files:

```text
scripts/evaluate_fold0_locked_test_frozen_v2.py
outputs/fold0_locked_test_frozen_v2/summary.json
outputs/fold0_locked_test_frozen_v2/test_manifest.csv
outputs/fold0_locked_test_frozen_v2/test_raw_pred.npy
outputs/fold0_locked_test_frozen_v2/test_postprocessed_pred.npy
outputs/fold0_locked_test_frozen_v2/test_target.npy
outputs/fold0_locked_test_frozen_v2/test_song_ids.npy
outputs/fold0_locked_test_frozen_v2/test_per_song_diagnostics.csv
```

Test songs:

```text
2006, 2008, 2015, 2016, 2021, 2026, 2036, 2038, 2051, 2055, 2056, 2057
```

Locked test result:

| Version | Mean RMSE | Mean MAE | Mean PCC | Mean CCC | V RMSE | V MAE | V PCC | V CCC | A RMSE | A MAE | A PCC | A CCC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Hybrid raw | 0.231705 | 0.180635 | 0.397999 | 0.371315 | 0.268444 | 0.206488 | 0.271062 | 0.253858 | 0.187915 | 0.154782 | 0.524937 | 0.488772 |
| Previous linear postprocess | 0.228491 | 0.174951 | 0.424376 | 0.393014 | 0.267563 | 0.199566 | 0.269133 | 0.248240 | 0.181180 | 0.150336 | 0.579619 | 0.537787 |
| Conservative V2 postprocess | 0.236773 | 0.174319 | 0.400369 | 0.376076 | 0.285205 | 0.203472 | 0.206967 | 0.198378 | 0.175445 | 0.145165 | 0.593771 | 0.553774 |

Mean baseline using the training-set VA mean:

| Split | Mean RMSE | Mean MAE | V RMSE | V MAE | A RMSE | A MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| validation | 0.313664 | 0.260312 | 0.335493 | 0.281971 | 0.290197 | 0.238653 |
| test | 0.246191 | 0.198987 | 0.270264 | 0.218514 | 0.219493 | 0.179461 |

Interpretation:

- The validation-selected conservative V2 candidate does not generalize to the
  locked test set.
- Arousal improves after V2 postprocessing relative to raw test output
  (`A RMSE 0.187915 -> 0.175445`), but Valence becomes worse
  (`V RMSE 0.268444 -> 0.285205`).
- The previous linear postprocess is less aggressive and is slightly better
  than V2 on test, but it still fails the target thresholds.
- Valence is the fundamental failure mode. Per-song diagnostics show large
  whole-song Valence baseline errors for several test songs, especially 2015,
  2036, and 2051.
- Current conclusion: this Fold-0 candidate must not be promoted as the final
  scientific Teacher. Validation-only success was over-optimistic.

Next research direction after this failure:

1. Stop tuning Fold-0 validation/test postprocessing.
2. Move to multi-fold complete-song cross-validation or a nested protocol so
   validation selection is less dependent on 12 songs.
3. Focus on robust song-level Valence baseline modeling, because within-song
   residual Valence was not the main error source.
4. Treat the current model as an ablation baseline, not the final Teacher.

### Nested Full-Song Fold Diagnostics

A five-fold diagnostic was added to inspect whether the Fold-0 failure was a
single unlucky split or a systematic complete-song issue.

Files:

```text
scripts/diagnose_nested_full_song_folds.py
outputs/nested_full_song_fold_diagnostics/summary.json
outputs/nested_full_song_fold_diagnostics/fold_split_summary.csv
```

Key observation:

- Every held-out complete-song split has a noticeable Valence mean offset
  relative to its training set.
- Fold-0 test is not uniquely problematic. Other folds also show Valence mean
  shifts, for example Fold-1 validation and Fold-2 test.
- Mean baseline test RMSE varies substantially across folds, from about
  `0.246` to `0.320`, showing that a 12-song validation/test split is too
  small and unstable for strong model-selection conclusions.

Decision: future Teacher claims should be based on five-fold outer-test
aggregation, not a single Fold-0 validation/test pair.

### Complete-Song Global Ridge Baseline

A lightweight baseline was added to test whether whole-song VA can be predicted
robustly from complete-song global audio features alone. For each outer fold:

```text
train = complete songs in train folds only
validation = complete songs in validation fold
test = untouched complete songs in outer test fold
model = Ridge predicting song-level mean VA
frame output = constant VA within each song
feature candidates = hand, MERT mean, all mean/std
alpha selected on validation only
```

Files:

```text
scripts/cross_validate_full_song_global_ridge.py
outputs/full_song_global_ridge_cv/summary.json
outputs/full_song_global_ridge_cv/fold_results.csv
outputs/full_song_global_ridge_cv/outer_test_pred.npy
outputs/full_song_global_ridge_cv/outer_test_target.npy
outputs/full_song_global_ridge_cv/outer_test_song_ids.npy
```

Five-fold concatenated outer-test result:

| Model | Mean RMSE | Mean MAE | Mean PCC | Mean CCC | V RMSE | V MAE | V PCC | V CCC | A RMSE | A MAE | A PCC | A CCC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Complete-song global Ridge | 0.242049 | 0.193788 | 0.513339 | 0.393422 | 0.253628 | 0.200424 | 0.490952 | 0.407597 | 0.229888 | 0.187151 | 0.535725 | 0.379247 |

Interpretation:

- The simple global baseline does not solve complete-song generalization.
- Its Valence RMSE is better than the failed Fold-0 deep raw Valence test
  (`0.253628` five-fold Ridge vs `0.268444` Fold-0 deep raw), but it remains
  far from the target.
- Because the output is constant within a song, this baseline is not suitable
  as a final dynamic VA Teacher. It is useful evidence that whole-song Valence
  baseline prediction is the core bottleneck.

Updated conclusion:

- Current data and protocol reveal a stronger limitation than expected:
  complete-song Valence generalization on only 58 DEAM full songs is unstable.
- Continuing to tune postprocessing or small heads on one fold is unlikely to
  produce a scientifically defensible Teacher.
- The next viable path is to change the training/evaluation strategy, not just
  another small hyperparameter tweak.

### Five-Fold Valence Global-Local Nested Evaluation

The original Valence Global-Local full-song expert was trained on all five
nested complete-song folds. Folds 1-4 were trained after Fold 0; the script was
patched so Valence-only folds can write summaries even when the matching
Arousal fold has not yet been trained.

Files:

```text
scripts/evaluate_valence_global_local_nested_cv.py
outputs/valence_global_local_full_song_fold0/
outputs/valence_global_local_full_song_fold1/
outputs/valence_global_local_full_song_fold2/
outputs/valence_global_local_full_song_fold3/
outputs/valence_global_local_full_song_fold4/
outputs/valence_global_local_nested_cv/summary.json
outputs/valence_global_local_nested_cv/fold_results.csv
outputs/valence_global_local_nested_cv/outer_test_valence_pred.npy
outputs/valence_global_local_nested_cv/outer_test_valence_target.npy
outputs/valence_global_local_nested_cv/outer_test_song_ids.npy
```

Per-fold Valence results:

| Fold | Validation RMSE | Validation CCC | Test RMSE | Test CCC |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.210011 | 0.750184 | 0.268444 | 0.253858 |
| 1 | 0.240991 | 0.521889 | 0.274892 | 0.498123 |
| 2 | 0.193668 | 0.803913 | 0.249621 | 0.534948 |
| 3 | 0.252934 | 0.189568 | 0.255243 | 0.592741 |
| 4 | 0.206350 | 0.524659 | 0.286630 | -0.018054 |

Concatenated five-fold outer-test Valence result:

```text
RMSE=0.267009
MAE=0.205546
PCC=0.463024
CCC=0.430917
```

Interpretation:

- The model does not generalize under five-fold complete-song evaluation.
- Validation performance is not a reliable proxy for outer-test performance on
  this 58-song complete-song subset.
- Fold 2 looks promising on validation and is still weak on test; Fold 4 is a
  clear failure case, with negative test CCC.
- The five-fold Global-Local Valence result is weaker than the lightweight
  complete-song Global Ridge baseline on Valence RMSE
  (`0.267009` vs `0.253628`).

Decision:

- The Valence Global-Local full-song expert should be retained only as an
  ablation.
- It must not be promoted to the final Teacher.
- The immediate research priority should shift from architecture tuning to data
  strategy:
  - use DEAM short clips for frame-level teacher evaluation;
  - use complete-song DEAM only as an exploratory generalization diagnostic;
  - add another dynamic VA dataset with more complete songs if strict
    complete-song metrics are required;
  - or redefine Phase 2 as section-aware structural analysis using a
    transparent pseudo-VA teacher with explicitly stated limitations.

### Progress Snapshot And PMEmo Transition

The current good candidates, failed locked-test result, five-fold Valence
checkpoints, and complete-song global Ridge diagnostic have been frozen in:

```text
docs/TEACHER_PROGRESS_SNAPSHOT_20260624.md
```

This snapshot is the handoff point before adding PMEmo. The current decision is
not to continue single-fold DEAM-only hyperparameter tuning. The next stage is:

```text
research/phase1f_external_va_datasets/pmemo
```

PMEmo integration starts with dataset auditing and normalization, not immediate
mixed-dataset training.
