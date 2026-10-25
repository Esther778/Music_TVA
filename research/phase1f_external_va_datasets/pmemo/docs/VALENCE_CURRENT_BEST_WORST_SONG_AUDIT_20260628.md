# Valence Current-Best Worst-Song Audit

Report time: 2026-06-28

Current research status: first-stage VA Teacher optimization. Arousal has a
usable single-dimension model; Valence remains the limiting dimension. This
report audits the current best Valence model before further training.

## Current Best Before This Round

Model:

```text
pmemo_valence_prediction_only_rescale_ccc_preserving_20260628
```

Test macro over 3 song-level splits:

| Metric | Current | Minimum Pass | Good | Excellent | Gap to Pass |
| --- | ---: | ---: | ---: | ---: | ---: |
| RMSE | 0.195316 | 0.180000 | 0.150000 | 0.100000 | +0.015316 |
| MAE | 0.151793 | 0.150000 | 0.120000 | 0.080000 | +0.001793 |
| PCC | 0.797766 | 0.800000 | 0.850000 | 0.900000 | -0.002234 |
| CCC | 0.784834 | 0.800000 | 0.850000 | 0.900000 | -0.015166 |

Per split:

| Split | RMSE | MAE | PCC | CCC |
| --- | ---: | ---: | ---: | ---: |
| split_01_seed_101 | 0.181559 | 0.141920 | 0.844993 | 0.835335 |
| split_02_seed_202 | 0.209386 | 0.162401 | 0.762291 | 0.742968 |
| split_03_seed_303 | 0.195004 | 0.151058 | 0.786013 | 0.776199 |

## Audit Outputs

```text
script = scripts/audit_pmemo_valence_current_best_worst_songs.py
output_dir = outputs/valence_current_best_worst_song_audit_20260628
```

Generated files:

```text
subset_frame_metrics.csv
per_song_metrics.csv
worst_songs_overall.csv
split_02_worst_songs.csv
correlations.csv
error_type_summary.csv
gap_to_thresholds.csv
audit_summary.json
```

## Main Finding

The dominant Valence error is song-level mean bias, not local curve shape.

Across all test songs:

```text
corr(song_rmse, abs_mean_error) = 0.996773
```

This means most high-RMSE failures happen when the model predicts the whole song
too positive, too negative, or too close to neutral. Local slope/variance errors
are secondary under the current model.

Error type frequency across 231 test-song rows:

| Error Type | Count | Rate |
| --- | ---: | ---: |
| global_mean_bias | 113 | 48.92% |
| trend_failure | 97 | 41.99% |
| ordinary_error | 56 | 24.24% |
| flat_prediction | 50 | 21.65% |
| polarity_mismatch | 19 | 8.23% |

Worst split remains:

```text
split_02_seed_202
test RMSE/MAE/PCC/CCC = 0.209386/0.162401/0.762291/0.742968
```

The largest split_02 failures are mostly songs where the target mean is strongly
positive or strongly negative while the predicted mean stays near neutral or on
the wrong side.

Examples:

| Song | Artist | Target Mean | Pred Mean | RMSE | Error Type |
| --- | --- | ---: | ---: | ---: | --- |
| Higher Love | Hunter Plake | -0.550722 | 0.133724 | 0.685281 | polarity_mismatch + global_mean_bias |
| Glow | Drake | 0.510122 | -0.026173 | 0.536978 | global_mean_bias |
| Party | Chris Brown | 0.577296 | 0.172052 | 0.405629 | global_mean_bias |
| Kill Em With Kindness | Selena Gomez | 0.689524 | 0.289658 | 0.401465 | global_mean_bias |
| Fire Away | Chris Stapleton | -0.380327 | 0.009003 | 0.391564 | global_mean_bias + flat_prediction |

## Follow-Up Trial: Validation-Safe Song-Mean Microshift

Purpose:

```text
Try a conservative song-level mean correction after the current best Valence
curve, without retraining the main network.
```

Script:

```text
scripts/materialize_pmemo_valence_valsafe_song_mean_microshift.py
```

Output:

```text
outputs/valence_valsafe_song_mean_microshift_20260628
```

Configuration:

```text
input model = pmemo_valence_prediction_only_rescale_ccc_preserving_20260628
feature_set = pred_only
song mean model = rf_pca32
strength = 1.0
max shift cap = 0.010
```

Result:

| Model | RMSE | MAE | PCC | CCC | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| current best | 0.195316 | 0.151793 | 0.797766 | 0.784834 | keep |
| validation-safe microshift | 0.194707 | 0.151853 | 0.798582 | 0.784572 | reject |

Interpretation:

```text
The microshift slightly improves RMSE and PCC, but it worsens MAE and CCC.
Because the project requires multi-metric acceptability, this trial is not
promoted to current best.
```

## Decision

No update to Valence best.

Current Valence best remains:

```text
pmemo_valence_prediction_only_rescale_ccc_preserving_20260628
RMSE/MAE/PCC/CCC = 0.195316/0.151793/0.797766/0.784834
```

## Next Experimental Direction

The next Valence improvement should target song-level polarity and mean
estimation inside the model, not only post-hoc curve shifting.

Recommended next trial:

```text
Train a two-head Valence model:
1. frame head predicts frame-level Valence curve;
2. song-mean head predicts song-level Valence mean from pooled audio features;
3. loss = frame MSE/CCC + song-mean MSE;
4. inference uses the song-mean head as a weak constraint, not a large manual shift.
```

Reason:

```text
The audit shows that current failures are dominated by song-level mean/polarity
mistakes. A bounded post-hoc correction is too weak to fix large mean errors,
but a model trained with an explicit song-mean objective may learn stronger
global Valence evidence without destroying local curve shape.
```

## Attempted Trial: Two-Head Song-Mean Model

Script:

```text
scripts/train_pmemo_valence_two_head_song_mean.py
```

Output:

```text
outputs/valence_two_head_song_mean_20260628/two_head_h80_songw008_blend015_cap004
```

Configuration:

```text
base feature = MERT hidden_6 + harmonic Librosa
frame model = BiLSTM hidden_dim=80
extra head = dynamic song-mean Valence head
song_mean_weight = 0.08
song_consistency_weight = 0.02
mean_blend_weight = 0.15
mean_shift_cap = 0.04
```

Status:

```text
aborted after split_01 completed and split_02 showed clearly poor validation behavior
```

Completed split_01 result:

| Model | Val RMSE | Val MAE | Val PCC | Val CCC | Test RMSE | Test MAE | Test PCC | Test CCC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| current best split_01 | 0.215691 | 0.167271 | 0.717712 | 0.707375 | 0.181559 | 0.141920 | 0.844993 | 0.835335 |
| two-head split_01 | 0.236341 | 0.180385 | 0.659318 | 0.646044 | 0.202286 | 0.157685 | 0.809525 | 0.788182 |

Decision:

```text
Reject this exact two-head configuration. The explicit song-mean head is too
strong under the current setup and damages validation frame-level Valence
prediction. The idea may still be useful, but only with a much weaker auxiliary
loss and no direct output blending at first.
```

## Subsequent Update: Targeted Split-02 Song-Mean Correction

After the broad validation-safe song-mean search, a more conservative targeted
rule was tested.

Script:

```text
scripts/search_pmemo_valence_song_mean_valsafe_selection.py
scripts/materialize_pmemo_valence_targeted_split02_song_mean.py
```

Outputs:

```text
outputs/valence_song_mean_valsafe_selection_20260628
outputs/valence_targeted_split02_song_mean_20260628
```

Selection rule:

```text
Apply the validation-safe song-mean correction only if:
val_rmse_delta <= -0.003 and val_ccc_delta >= 0.003.
Otherwise keep the current best prediction unchanged.
```

This selected only `split_02_seed_202`, which was also the audited worst split.

Selected split_02 correction:

```text
feature_set = mert_all_librosa
model_id = ridge_pca96_a100
strength = 0.5
cap = none
```

Result:

| Model | RMSE | MAE | PCC | CCC | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| previous Valence best | 0.195316 | 0.151793 | 0.797766 | 0.784834 | replaced as CV metric best |
| targeted split_02 correction | 0.192943 | 0.150559 | 0.802080 | 0.785124 | new CV metric best |

Gap to thresholds:

| Metric | Current | Minimum Pass | Good | Excellent | Gap to Pass |
| --- | ---: | ---: | ---: | ---: | ---: |
| RMSE | 0.192943 | 0.180000 | 0.150000 | 0.100000 | +0.012943 |
| MAE | 0.150559 | 0.150000 | 0.120000 | 0.080000 | +0.000559 |
| PCC | 0.802080 | 0.800000 | 0.850000 | 0.900000 | pass |
| CCC | 0.785124 | 0.800000 | 0.850000 | 0.900000 | +0.014876 |

Important limitation:

```text
This is the current best cross-validation metric result, but it is not yet the
packaged arbitrary-MP3 deployment model. The full-song Valence deployment script
still points to the earlier MERT hidden_6 + harmonic ensemble until this
targeted correction is adapted into a deployable all-training pipeline.
```

## Targeted Rescale Check

Script:

```text
scripts/search_pmemo_valence_targeted_rescale_valsafe.py
```

Output:

```text
outputs/valence_targeted_rescale_valsafe_20260628
```

Result:

| Model | RMSE | MAE | PCC | CCC |
| --- | ---: | ---: | ---: | ---: |
| targeted split_02 song-mean | 0.192943 | 0.150559 | 0.802080 | 0.785124 |
| targeted rescale valsafe | 0.192981 | 0.150646 | 0.802475 | 0.785847 |

Decision:

```text
Do not promote the rescale output as the primary best. It is useful as a
CCC-biased alternative, but it worsens RMSE and MAE. The primary Valence CV
metric best remains pmemo_valence_targeted_split02_song_mean_20260628.
```

## Universal Correction Check

The user explicitly reminded that the final Teacher must support arbitrary
complete-song prediction. Therefore, a fixed correction configuration was
materialized across all CV splits.

Script:

```text
scripts/materialize_pmemo_valence_universal_song_mean_correction.py
```

Output:

```text
outputs/valence_universal_song_mean_correction_20260628
```

Configuration:

```text
feature_set = mert_all_librosa
model_id = ridge_pca64_a30
strength = 0.10
cap = none
```

Result:

| Model | RMSE | MAE | PCC | CCC |
| --- | ---: | ---: | ---: | ---: |
| previous Valence best | 0.195316 | 0.151793 | 0.797766 | 0.784834 |
| targeted split_02 correction | 0.192943 | 0.150559 | 0.802080 | 0.785124 |
| universal song-mean correction | 0.193143 | 0.150075 | 0.801781 | 0.786682 |

Decision:

```text
For paper/result reporting, keep both distinctions clear:
1. targeted split_02 correction is the current best CV metric result;
2. universal song-mean correction is the current best deployment-oriented
   Valence candidate because it uses one fixed configuration across splits.
```

## Universal Metric-Optimized Exploratory Candidate

Script:

```text
scripts/materialize_pmemo_valence_universal_song_mean_correction.py
```

Output:

```text
outputs/valence_universal_metric_optimized_exploratory_20260628
```

Fixed configuration:

```text
feature_set = mert_harmonic
model_id = ridge_pca64_a30
strength = 1.0
cap = 0.02
```

Result:

| Model | RMSE | MAE | PCC | CCC |
| --- | ---: | ---: | ---: | ---: |
| previous Valence best | 0.195316 | 0.151793 | 0.797766 | 0.784834 |
| universal song-mean correction | 0.193143 | 0.150075 | 0.801781 | 0.786682 |
| universal metric-optimized exploratory | 0.191822 | 0.149198 | 0.804483 | 0.788190 |

Threshold status:

```text
MAE and PCC pass the minimum line.
RMSE still above pass by 0.011822.
CCC still below pass by 0.011810.
```

Caveat:

```text
The model is structurally compatible with arbitrary-song prediction because the
same fixed correction is applied across all splits. However, the configuration
was identified during exploratory candidate inspection, so it should be
confirmed by an independent holdout or stricter nested validation before being
called the final Teacher.
```

## Universal Nested-Selection Audit

Script:

```text
scripts/evaluate_pmemo_valence_universal_nested_selection.py
```

Output:

```text
outputs/valence_universal_nested_selection_20260628
```

Protocol:

```text
For each held-out split, select the universal correction configuration using
only the validation metrics from the other two splits. Then report the held-out
split test metrics.
```

Result:

| Model | RMSE | MAE | PCC | CCC |
| --- | ---: | ---: | ---: | ---: |
| baseline | 0.195316 | 0.151793 | 0.797766 | 0.784834 |
| nested-selected universal correction | 0.194173 | 0.152107 | 0.798620 | 0.779097 |

Decision:

```text
Reject this nested-selected universal correction as a true improvement because
MAE and CCC are worse than the baseline. The exploratory high-scoring universal
candidate remains useful for analysis, but it is not yet strong enough evidence
for final Teacher selection.
```

## Static-Target Song-Mean Correction

PMEmo static Valence was checked as a cleaner global target.

Pre-check:

```text
corr(static Valence, dynamic song-mean Valence) = 0.991187
```

Script:

```text
scripts/search_pmemo_valence_static_target_song_mean.py
```

Output:

```text
outputs/valence_static_target_song_mean_20260628
```

Result:

| Model | RMSE | MAE | PCC | CCC |
| --- | ---: | ---: | ---: | ---: |
| baseline | 0.195316 | 0.151793 | 0.797766 | 0.784834 |
| static-target correction | 0.192812 | 0.151347 | 0.800882 | 0.785297 |
| metric-optimized exploratory | 0.191822 | 0.149198 | 0.804483 | 0.788190 |

Decision:

```text
Static-target correction improves over the baseline but does not become the new
best. It supports the idea that global Valence supervision is useful, but the
post-hoc correction form is still insufficient for the final Teacher.
```
