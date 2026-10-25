# Arousal Prediction Model Report Summary

Date: 2026-06-24

## 1. Research Context

The current project aims to train a Valence-Arousal Teacher model that can be
used for arbitrary complete-song VA prediction. The later section-aware study
will use this Teacher to generate frame-level VA trajectories and combine them
with manually reviewed song sections.

The current Arousal milestone is:

```text
Train a model that can predict Arousal for arbitrary complete songs and reaches
the current minimum project metric standard.
```

Canonical reporting scale:

```text
Valence/Arousal range = [-1, 1]
```

## 2. Dataset And Feature Basis

### PMEmo

PMEmo was introduced because it provides more dynamic VA annotations than the
small complete-song DEAM subset currently used in our strict song-level tests.

Current local PMEmo subset:

```text
dynamic songs = 767
dynamic frames = 36434
label step = 0.5 s
original label scale = [0, 1]
canonical label scale = [-1, 1]
```

The label conversion is:

```text
canonical_value = 2 * pmemo_value - 1
```

### OpenSMILE Features

PMEmo provides pre-computed OpenSMILE dynamic acoustic features. In our local
audit:

```text
feature_dim = 260
base = 65 ComParE_2016 low-level descriptors
deltas = 65 ComParE_2016 low-level descriptor deltas
aggregation = mean/std
```

To make arbitrary complete-song prediction possible, the project implemented a
PMEmo-style OpenSMILE feature reproduction:

```text
feature_set = ComParE_2016
feature_level = LowLevelDescriptors + LowLevelDescriptors_Deltas
window = trailing 1.0 s
hop = 0.5 s
```

This reproduction was checked against a PMEmo official feature table. The
flattened feature correlation was about `0.996`. It should therefore be
reported as a high-correlation PMEmo-style reproduction, not bit-identical
official extraction.

## 3. Initial Joint VA Prediction Baseline

The first PMEmo baseline trained Valence and Arousal jointly with the same
model output head.

Models:

```text
PMEmo OpenSMILE MLP
PMEmo OpenSMILE BiLSTM
```

Split:

```text
song-level split
train = 614 songs / 29184 frames
validation = 77 songs / 3912 frames
test = 76 songs / 3338 frames
```

Best joint model:

```text
model = OpenSMILE dynamic features + BiLSTM
input_dim = 260
hidden_dim = 128
layers = 2
dropout = 0.25
optimizer = AdamW
learning_rate = 8e-4
weight_decay = 1e-4
selection = validation mean VA RMSE
```

Joint BiLSTM test result:

| Dimension | RMSE | MAE | PCC | CCC |
| --- | ---: | ---: | ---: | ---: |
| Valence | 0.271199 | 0.212034 | 0.571292 | 0.546814 |
| Arousal | 0.186934 | 0.141942 | 0.837598 | 0.830559 |
| Mean | 0.232909 | 0.176988 | 0.704445 | 0.688687 |

Interpretation:

```text
Arousal was already close to acceptable.
Valence was much weaker and pulled down mean VA metrics.
```

## 4. Why Separate Valence And Arousal

The joint VA model was not abandoned arbitrarily. It showed a clear empirical
asymmetry:

```text
Arousal PCC/CCC = 0.837598 / 0.830559
Valence PCC/CCC = 0.571292 / 0.546814
```

This means the model learned energy/intensity-related emotion variation much
better than positive/negative affect.

This is musically reasonable:

- Arousal is strongly related to audio dynamics such as loudness, energy,
  spectral flux, rhythm density, onset strength, and tempo.
- Valence often depends more on harmony, mode, melody, lyrics, cultural
  association, and listener preference.
- Therefore, a shared output head and shared selection rule may under-optimize
  Arousal while still failing to solve Valence.

The optimization strategy was changed to:

```text
first make Arousal independently usable;
then focus the next stage on Valence.
```

The concrete technical changes were:

```text
old target = joint Valence + Arousal
new target = Arousal only
old selection = validation mean VA RMSE
new selection = validation Arousal RMSE
```

This is aligned with the later section-aware goal because section-level climax
and intensity progression depend heavily on Arousal.

## 5. Arousal-Only Training Results

Training script:

```text
research/phase1f_external_va_datasets/pmemo/scripts/train_pmemo_opensmile_arousal_trials.py
```

Best single Arousal-only model:

```text
trial_id = a_only_h96_l1_do015_mse
model = 1-layer BiLSTM
hidden_dim = 96
dropout = 0.15
loss = MSE
best_epoch = 59
```

Single model test result:

| Metric | Value |
| --- | ---: |
| RMSE | 0.180960 |
| MAE | 0.140092 |
| PCC | 0.852634 |
| CCC | 0.848122 |

This single model passed MAE/PCC/CCC but missed the RMSE threshold by
`0.000960`.

## 6. Final Arousal Ensemble

To reduce variance and improve robustness, a validation-selected ensemble was
created.

Selection rule:

```text
weights selected by validation RMSE only
test set used only for final reporting
```

Ensemble config:

```text
research/phase1f_external_va_datasets/pmemo/outputs/opensmile_arousal_trials/arousal_ensemble_v1.json
```

Members:

| Trial | Weight |
| --- | ---: |
| a_only_h128_l2_do025_mse | 0.15 |
| a_only_h160_l2_do030_mse_ccc | 0.30 |
| a_only_h192_l2_do035_mse_ccc_smooth | 0.20 |
| a_only_h96_l1_do015_mse | 0.35 |

Final Arousal ensemble test result:

| Metric | Current | Minimum pass | Gap to pass | Good | Gap to good | Excellent | Gap to excellent | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| RMSE | 0.176695 | 0.180000 | -0.003305 | 0.150000 | +0.026695 | 0.100000 | +0.076695 | Pass minimum |
| MAE | 0.135837 | 0.150000 | -0.014163 | 0.120000 | +0.015837 | 0.080000 | +0.055837 | Pass minimum |
| PCC | 0.859718 | 0.800000 | -0.059718 | 0.850000 | -0.009718 | 0.900000 | +0.040282 | Pass good |
| CCC | 0.848452 | 0.800000 | -0.048452 | 0.850000 | +0.001548 | 0.900000 | +0.051548 | Pass minimum, near good |

Conclusion:

```text
Current Arousal model = metric-qualified minimum-pass Teacher candidate
Quality level = passable to near-good
Not excellent
```

## 7. Complete-Song Prediction Status

The Arousal ensemble has a complete-song inference path.

Single-song prediction:

```bash
.venv/bin/python research/phase1f_external_va_datasets/pmemo/scripts/predict_full_song_arousal_opensmile_ensemble.py \
  --audio "songs/江南-林俊杰.mp3" \
  --output "research/phase1f_external_va_datasets/pmemo/outputs/opensmile_arousal_trials/fullsong_predictions/jiangnan_arousal_opensmile_ensemble.csv"
```

Batch prediction:

```bash
.venv/bin/python research/phase1f_external_va_datasets/pmemo/scripts/predict_batch_full_song_arousal_opensmile_ensemble.py \
  --audio-dir songs \
  --output-dir research/phase1f_external_va_datasets/pmemo/outputs/opensmile_arousal_trials/fullsong_predictions_batch
```

Verified:

```text
single full-song test = songs/江南-林俊杰.mp3, 534 frames
batch smoke test = 2/2 songs succeeded
```

Output columns:

```text
time_sec
arousal_minus1_1
arousal_0_1
model_id
feature_type
```

## 8. Why The Thresholds Are Defined This Way

The thresholds are project evaluation targets, not universal laws. They are
defined from three sources:

```text
1. statistical interpretation of correlation and concordance;
2. normalized error size on the canonical [-1, 1] VA scale;
3. observed difficulty and recent practice in MER/DMER literature.
```

Current project thresholds:

| Level | RMSE | MAE | PCC | CCC |
| --- | ---: | ---: | ---: | ---: |
| Minimum pass | <= 0.18 | <= 0.15 | >= 0.80 | >= 0.80 |
| Good | <= 0.15 | <= 0.12 | >= 0.85 | >= 0.85 |
| Excellent | <= 0.10 | <= 0.08 | >= 0.90 | >= 0.90 |

Rationale:

1. The canonical VA scale is `[-1, 1]`, total span 2.0. An RMSE of 0.18 is
   about 9% of the full scale, while MAE 0.15 is about 7.5% of the full scale.
   This is a strict but realistic minimum target for a usable Teacher.

2. In applied statistics, Pearson correlation is commonly interpreted as the
   strength of linear association. Reviews such as Schober et al. emphasize
   that interpretation depends on context, but values around 0.80 and above are
   generally in the strong/high association range. Since later section-aware
   analysis depends on emotional trend, not just absolute pointwise error,
   PCC should not merely be moderate.

3. CCC, originally proposed by Lin, is stricter than PCC because it measures
   agreement rather than only association. It penalizes both low correlation
   and mean/variance mismatch. For pseudo-VA generation, CCC is especially
   important because a model that has good trend but biased scale can distort
   section contrast and climax strength.

4. PMEmo and DEAM-style dynamic music emotion recognition are difficult because
   labels are subjective, time-varying, and affected by listener response
   latency. Older and difficult DMER benchmarks often report Arousal around the
   0.20 RMSE level, while recent frontier methods increasingly use stronger
   sequence modeling, local/global features, personalization, and multimodal
   information. Therefore `0.18` is a reasonable minimum for a usable internal
   Teacher, while `0.15` and `0.10` represent stronger and excellent levels.

5. Some very low reported RMSE values in the literature may involve different
   label scales, different splits, short excerpts, fixed test sets, personalized
   settings, or access to emotion-history features. They should not be compared
   directly without checking protocol.

Practical interpretation:

```text
Minimum pass = strong enough for internal pseudo-label generation, with caution.
Good = stable enough to support stronger empirical claims.
Excellent = very strong result under strict split and scale assumptions.
```

## 9. Relation To Current MIR / DMER Frontiers

Current dynamic music emotion recognition research has several relevant trends:

- It treats DMER as sequence prediction rather than only clip-level regression.
- It emphasizes short-term and long-term temporal context.
- It increasingly uses dual-scale or local/global representations.
- It recognizes listener-specific emotional perception as a major source of
  label uncertainty.
- It often uses stronger pretrained audio representations or multimodal
  information.

Our current Arousal model is not a frontier architecture, but it is technically
defensible because:

```text
1. it uses a standard affective computing audio feature family;
2. it uses song-level split;
3. it reports RMSE, MAE, PCC, and CCC;
4. it has an arbitrary complete-song inference path;
5. it does not claim that pseudo-VA equals human ground truth.
```

The model is best positioned as a reliable Arousal Teacher component for
section-aware emotion structure analysis, not as a SOTA DMER claim.

## 10. Current Limitations

Important limitations:

1. The ensemble is validated on PMEmo chorus clips, while real application
   targets complete songs. The inference path works on complete songs, but the
   training distribution is still chorus-focused.

2. OpenSMILE feature reproduction is highly correlated with PMEmo official
   features but not bit-identical.

3. The current Arousal result is not yet multi-seed averaged.

4. Arousal is only one dimension. Valence remains unresolved and much weaker.

5. The model is most trustworthy for pop-like music close to the PMEmo
   distribution.

## 11. Recommendation

Short-term decision:

```text
Freeze pmemo_opensmile_arousal_ensemble_v1 as the current usable Arousal
Teacher candidate.
```

Next priority:

```text
Focus optimization on Valence.
```

Before final paper writing:

```text
1. run 3 seeds or bootstrap stability check for the Arousal ensemble;
2. add a small complete-song sanity evaluation if labeled complete-song Arousal
   becomes available;
3. clearly state that Arousal pseudo-labels are model-generated estimates;
4. use Arousal mainly for trend, climax, and intensity progression features.
```

## 12. References

1. PMEmo dataset and released dynamic annotations/features:
   https://github.com/HuiZhangDB/PMEmo
2. openSMILE Python package and supported feature sets:
   https://audeering.github.io/opensmile-python/
3. PDMER / Personalized Dynamic Music Emotion Recognition:
   https://arxiv.org/abs/2412.19200
4. Lin, L. I-K. Concordance correlation coefficient, Biometrics, 1989.
   https://doi.org/10.2307/2532051
5. Schober, P., Boer, C., Schwarte, L. A. Correlation coefficients:
   appropriate use and interpretation, Anesthesia & Analgesia, 2018.
   https://doi.org/10.1213/ANE.0000000000002864
