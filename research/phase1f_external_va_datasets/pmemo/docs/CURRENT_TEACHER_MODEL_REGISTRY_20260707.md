# Current Teacher Model Registry

## 2026-07-15 Final Executable Teacher And Rigor Audit

The canonical executable models are now:

```text
Valence = pmemo_final_clean_valence_teacher_v1
Arousal = pmemo_final_clean_arousal_teacher_v1
Input = MERT hidden-6 + PMEmo OpenSMILE dynamic, pure audio
Architecture = separate global-local BiGRU models
Training = 652 PMEmo development clips
Final evaluation = 115 frozen PMEmo clips, 5,390 frames
```

Official complete-clip final metrics:

| Dimension | RMSE | MAE | PCC | CCC |
| --- | ---: | ---: | ---: | ---: |
| Valence | 0.186298 | 0.152855 | 0.802852 | 0.800168 |
| Arousal | 0.188573 | 0.146912 | 0.857441 | 0.854803 |

Official 0.5-second frame final metrics:

| Dimension | RMSE | MAE | PCC | CCC |
| --- | ---: | ---: | ---: | ---: |
| Valence | 0.194416 | 0.158386 | 0.810613 | 0.801561 |
| Arousal | 0.200411 | 0.157457 | 0.848749 | 0.839168 |

Audit status:

- predictions were frozen before final label access and metrics reproduce
  exactly;
- no post-test tuning is allowed or has been performed;
- ID overlap is zero, but three exact MP3 duplicates cross development/final;
- the 112-song deduplicated calculation is sensitivity analysis only;
- PMEmo's median labelled span is 20.5 seconds, so complete-song
  generalization is not proven;
- raw-MP3 OpenSMILE reproduction is not bit-equivalent to the official
  features, so arbitrary-song deployment still has a measured feature-domain
  gap.

Canonical audit:
`../final_clean_teacher/FINAL_VA_TEACHER_RIGOR_AUDIT_20260715.md`.

The development metric artifacts below remain useful experiment history, but
must not replace the final executable model or its prospective result.

## 2026-07-15 Development Freeze Update

The canonical current development results and artifact hashes are now frozen
in `VA_TEACHER_DEVELOPMENT_FREEZE_20260715.md`.

At complete-clip aggregation on the 652-song locked development pool:

| Dimension | Model | RMSE | MAE | PCC | CCC |
| --- | --- | ---: | ---: | ---: | ---: |
| Valence | `pmemo_valence_locked_global_shift_amp_guarded_20260715` | 0.178526 | 0.140722 | 0.821365 | 0.805417 |
| Arousal | `pmemo_arousal_locked_opensmile_static_songmean_fine_20260715` | 0.174339 | 0.133155 | 0.870038 | 0.860214 |

These are frozen development OOF artifacts, not prospective final-test
results. At this historical freeze point, their executable final-test chain
had not yet been materialized. The final clean model and completed final test
are now recorded in the top update above. The older sections below are retained
as chronological history and must not override that update.

Written at: 2026-07-07 CST

Current research stage: Phase 1 VA Teacher is still being fixed before Phase 2
section-aware complete-song emotion structure analysis. Arousal has passed the
minimum metric line. Valence remains the bottleneck.

## Reporting Thresholds

| Metric | Minimum pass | Good | Excellent |
| --- | ---: | ---: | ---: |
| RMSE | <= 0.180 | <= 0.150 | <= 0.100 |
| MAE | <= 0.150 | <= 0.120 | <= 0.080 |
| PCC | >= 0.800 | >= 0.850 | >= 0.900 |
| CCC | >= 0.800 | >= 0.850 | >= 0.900 |

## Arousal Teacher

### Metric-Best And Deployable Candidate

```text
model_id = pmemo_arousal_rmse_primary_postprocess_20260629
input_model = a_only_h96_l1_do015_mse
feature = PMEmo-style OpenSMILE dynamic features, 260 dimensions
architecture = Arousal-only BiLSTM
postprocess = validation-selected linear calibration strength 0.5 + moving average window 9
selection = validation candidates with val_pcc >= 0.80 and val_ccc >= 0.80, then minimum val_rmse
output = outputs/opensmile_arousal_rmse_primary_postprocess_20260629
```

| Metric | Current | Minimum pass | Gap to pass | Good | Gap to good | Excellent | Gap to excellent | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| RMSE | 0.179278 | 0.180000 | -0.000722 | 0.150000 | +0.029278 | 0.100000 | +0.079278 | pass |
| MAE | 0.138949 | 0.150000 | -0.011051 | 0.120000 | +0.018949 | 0.080000 | +0.058949 | pass |
| PCC | 0.855885 | 0.800000 | -0.055885 | 0.850000 | -0.005885 | 0.900000 | +0.044115 | pass |
| CCC | 0.848522 | 0.800000 | -0.048522 | 0.850000 | +0.001478 | 0.900000 | +0.051478 | pass |

Deployability status:

```text
status = usable after extracting compatible OpenSMILE dynamic features
single-song scripts =
  scripts/predict_full_song_arousal_opensmile_ensemble.py
  scripts/predict_full_song_arousal.py
batch scripts =
  scripts/predict_batch_full_song_arousal_opensmile_ensemble.py
  scripts/predict_batch_full_song_arousal.py
```

Note:
- The OpenSMILE-based arousal path is the metric-qualified path.
- The librosa arousal path is easier to run from raw MP3 but has weaker metrics.

### Sealed Blind Holdout Audit

```text
split_id = pmemo_blind_20260709
split_file = data/splits/pmemo_blind_20260709/pmemo_blind_song_level_split.csv
train/val/blind_test songs = 536 / 115 / 116
output = outputs/opensmile_arousal_blind_holdout_20260709
trial = a_only_h96_l1_do015_mse
feature = PMEmo official OpenSMILE dynamic features
architecture = 1-layer Arousal-only BiLSTM
hidden_dim = 96
dropout = 0.15
loss = MSE only
best_epoch = 22
selection = validation RMSE only
```

Validation:

```text
RMSE/MAE/PCC/CCC =
  0.198540 / 0.155541 / 0.839615 / 0.819647
```

Blind test raw:

```text
RMSE/MAE/PCC/CCC =
  0.201161 / 0.161095 / 0.838482 / 0.831313
minimum-line status =
  RMSE not pass, gap +0.021161
  MAE not pass, gap +0.011095
  PCC pass, margin 0.038482
  CCC pass, margin 0.031313
```

Validation-selected postprocess:

```text
postprocess = smooth_w1_linear_cal
blind RMSE/MAE/PCC/CCC =
  0.207575 / 0.165906 / 0.838482 / 0.826968
decision = rejected for final reporting because blind RMSE/MAE/CCC worsened
```

Interpretation:
- Arousal is more stable than Valence on the sealed blind split in PCC/CCC.
- It still misses the RMSE and MAE minimum lines on blind holdout.
- Since this blind result has now been inspected, this exact blind split should
  not be used for further Arousal tuning while still being called blind.

## Valence Teacher

### Metric-Best Candidate

Updated 2026-07-08:

```text
model_id = pmemo_valence_guardtopccc_temporal_shift_wide_20260708
output = outputs/valence_guardtopccc_temporal_shift_wide_20260708
base = pmemo_valence_correlated_song_mean_guardtopccc_20260708
base_chain = pmemo_valence_current_best_tonal_correction_20260707 -> static correlated residual song-mean correction
temporal_shift = -9 frames = -4.5 seconds
selection = one global temporal shift selected by validation macro pass-gap score; test labels not used
status = current RMSE-primary Valence candidate; still fails RMSE and CCC minimum line
```

| Metric | Current | Minimum pass | Gap to pass | Good | Gap to good | Excellent | Gap to excellent | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| RMSE | 0.189619 | 0.180000 | +0.009619 | 0.150000 | +0.039619 | 0.100000 | +0.089619 | not pass |
| MAE | 0.147460 | 0.150000 | -0.002540 | 0.120000 | +0.027460 | 0.080000 | +0.067460 | pass |
| PCC | 0.810504 | 0.800000 | -0.010504 | 0.850000 | +0.039496 | 0.900000 | +0.089496 | pass |
| CCC | 0.797385 | 0.800000 | +0.002615 | 0.850000 | +0.052615 | 0.900000 | +0.102615 | not pass |

Updated again 2026-07-08 after smoothing:

```text
model_id = pmemo_valence_current_best_smoothing_20260708
output = outputs/valence_current_best_smoothing_20260708
input = pmemo_valence_guardtopccc_temporal_shift_wide_20260708
postprocess = moving average window 21, blend 1.0
selection = smoothing parameters selected by validation macro pass-gap score
status = current RMSE-primary Valence candidate; still fails RMSE and CCC minimum line
```

| Metric | Current | Minimum pass | Gap to pass | Good | Gap to good | Excellent | Gap to excellent | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| RMSE | 0.189553 | 0.180000 | +0.009553 | 0.150000 | +0.039553 | 0.100000 | +0.089553 | not pass |
| MAE | 0.147438 | 0.150000 | -0.002562 | 0.120000 | +0.027438 | 0.080000 | +0.067438 | pass |
| PCC | 0.810636 | 0.800000 | -0.010636 | 0.850000 | +0.039364 | 0.900000 | +0.089364 | pass |
| CCC | 0.797476 | 0.800000 | +0.002524 | 0.850000 | +0.052524 | 0.900000 | +0.102524 | not pass |

Previous metric-best retained for provenance:

```text
model_id = pmemo_valence_current_best_tonal_correction_20260707
output = outputs/valence_current_best_tonal_correction_20260707
base = pmemo_valence_current_best_song_mean_global_policy_20260707
base_features = MERT hidden_4 + hidden_6 + hidden_8 + Librosa harmonic
global_policy_feature = pred_only song-level statistics from the base predicted Valence curve
global_policy_model = Ridge alpha=3.0
global_policy_strength = 0.35
global_policy_cap = 0.020
selection = one shared correction configuration across all splits; validation guard vs the previous global policy; test labels not used
previous_metric_best = pmemo_valence_current_best_song_mean_global_policy_20260707
note = tonal/chroma/tonnetz features were tested in the same search, but the validation-selected best still used pred-only song-level statistics
```

| Metric | Current | Minimum pass | Gap to pass | Good | Gap to good | Excellent | Gap to excellent | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| RMSE | 0.190678 | 0.180000 | +0.010678 | 0.150000 | +0.040678 | 0.100000 | +0.090678 | not pass |
| MAE | 0.148681 | 0.150000 | -0.001319 | 0.120000 | +0.028681 | 0.080000 | +0.068681 | pass |
| PCC | 0.808054 | 0.800000 | -0.008054 | 0.850000 | +0.041946 | 0.900000 | +0.091946 | pass |
| CCC | 0.796190 | 0.800000 | +0.003810 | 0.850000 | +0.053810 | 0.900000 | +0.103810 | not pass |

Deployability status:

```text
status = raw-MP3 full-song predictor now exists, but metric-best CV chain and production approximation are not identical
production_model_id = pmemo_valence_current_best_tonal_correction_fullsong_20260708
single-song script = scripts/predict_full_song_valence_latest_refined.py
batch script = scripts/predict_batch_full_song_valence_latest_refined.py
policy params = outputs/valence_latest_refinement_params_20260708/pred_only_ridge_a3_strength035_cap020.json
formal CV reference RMSE/MAE/PCC/CCC = 0.190678 / 0.148681 / 0.808054 / 0.796190
materialized production reference RMSE/MAE/PCC/CCC = 0.190731 / 0.148601 / 0.807820 / 0.796043
reason = production script uses materialized train-fold Ridge parameters after CV selection, so it is close to but not numerically identical to the formal CV metric-best
```

Important distinction:
- `pmemo_valence_current_best_tonal_correction_20260707` is the current
  metric-best Valence candidate.
- `pmemo_valence_current_best_song_mean_global_policy_20260707` is the previous
  metric-best and the base input to the current refinement.
- `pmemo_valence_parsimony_output_guarded_20260707` is the deeper base model
  before the two song-level global policies.
- `scripts/predict_full_song_valence_global_policy.py` is the current raw-MP3
  deployable Valence predictor with the global song-mean policy.
- `scripts/predict_full_song_valence_current_best_approx.py` remains the base
  raw-MP3 approximation to the previous metric-best chain.
- `scripts/predict_full_song_valence_mert_ensemble.py` is the older deployable
  Valence script and still points to the hidden_6 + harmonic ensemble path:

```text
script model_id = pmemo_valence_mert_hidden6_harmonic_bilstm_ensemble_global_rescale_v1
feature = MERT hidden_6 + Librosa harmonic
model_dir = outputs/valence_multisplit_best_20260627
global_rescale = mean_gain 1.0, std_gain 0.45, blend 1.0
related metric reference = valence_global_prediction_only_rescale_20260628
macro test RMSE/MAE/PCC/CCC = 0.195388 / 0.152277 / 0.797910 / 0.783420
```

This means:
- The current metric-best Valence model is still not a single clean checkpoint.
- The new current-best approximation script can produce Valence curves from raw
  MP3, but it must be reported as a deployable approximation rather than a new
  independently validated metric result.
- The older hidden_6 deployable script is close to, but weaker than, the
  current metric-best Valence candidate.

### Current-Best Approximate Full-Song Valence Script

```text
script = scripts/predict_full_song_valence_latest_refined.py
batch_script = scripts/predict_batch_full_song_valence_latest_refined.py
model_id = pmemo_valence_current_best_tonal_correction_fullsong_20260708
feature = MERT hidden_4 + hidden_6 + hidden_8 + Librosa harmonic
base_script = scripts/predict_full_song_valence_global_policy.py
global_policy = previous pred_only Ridge(alpha=1.0), strength 0.5, cap 0.070
latest_refinement = pred_only Ridge(alpha=3.0), strength 0.35, cap 0.020
policy_params = outputs/valence_latest_refinement_params_20260708/pred_only_ridge_a3_strength035_cap020.json
materialized reference RMSE/MAE/PCC/CCC = 0.190731 / 0.148601 / 0.807820 / 0.796043
```

Smoke test:

```text
audio = songs/江南-林俊杰.mp3
range = first 20 seconds
output = outputs/valence_fullsong_latest_refined/smoke_jiangnan_20s.csv
frames = 40
status = success
```

## Current Overall Status

| Dimension | Current practical status | Metric status | Deployability status |
| --- | --- | --- | --- |
| Arousal | usable first | passes minimum line | deployable with OpenSMILE-compatible extraction |
| Valence | still bottleneck but closer | MAE/PCC pass; RMSE misses by 0.010678 and CCC misses by 0.003810 | metric-best global policy and production script not yet fully aligned |

2026-07-08 correction after deployability review:
- The table above reflects the earlier production approximation state.
- The current Valence status must now be read with two separate bests:
  metric-best artifact and deployable raw-MP3 candidate.
- For Phase 2 arbitrary-song pseudo-VA generation, deployability is a hard
  promotion gate. A candidate that depends on official PMEmo/DEAM feature
  tables is diagnostic only unless the same features can be reproduced from
  a new MP3.

Current Valence metric-best artifact:

```text
model_id = valence_extra_wide_rmse_shift_smoothing_20260708
output = outputs/valence_extra_wide_rmse_shift_smoothing_20260708
base = valence_correlated_song_mean_guardtopccc_20260708
postprocess = global temporal shift -21 frames (-10.5s) + moving average window 21
RMSE/MAE/PCC/CCC = 0.188791 / 0.146457 / 0.812194 / 0.797279
status = MAE/PCC pass; RMSE/CCC still fail
deployability = metric artifact only until its correlated song-mean/static feature dependency is reproduced from raw audio
```

Current Valence deployable raw-MP3 candidate:

```text
model_id = pmemo_valence_deployable_tonal_shift_smoothing_affine_fullsong_20260708
single-song script = scripts/predict_full_song_valence_deployable_tonal_shifted.py
base full-song script = scripts/predict_full_song_valence_latest_refined.py
postprocess = temporal shift -22 frames (-11.0s) + moving average window 21 + affine blend 0.03
CV reference RMSE/MAE/PCC/CCC = 0.188912 / 0.146588 / 0.811721 / 0.796749
status = MAE/PCC pass; RMSE misses by 0.008912; CCC misses by 0.003251
deployability = raw-MP3 deployable approximation; no official PMEmo static CSV dependency
```

Deployable smoke test:

```text
audio = songs/0001.大籽 - 白月光与朱砂痣.mp3
range = first 20 seconds
output = outputs/valence_fullsong_deployable_tonal_shifted/smoke_0001_20s.csv
frames = 40
status = success
```

## 2026-07-08 Valence Optimization Update

Formal RMSE-primary best after 2026-07-08 smoothing:

```text
model_id = pmemo_valence_current_best_smoothing_20260708
RMSE/MAE/PCC/CCC = 0.189553 / 0.147438 / 0.810636 / 0.797476
status = MAE and PCC pass; RMSE and CCC still fail
```

Target-line-closest variant found after additional calibration:

```text
model_id = pmemo_valence_opensmile_song_mean_affine_calibration_20260708
output = outputs/valence_opensmile_song_mean_affine_calibration_20260708
base = outputs/valence_current_best_opensmile_song_mean_20260708
selected_config = affine, global, blend=0.55, no smoothing
RMSE/MAE/PCC/CCC = 0.190769 / 0.148779 / 0.808449 / 0.797472
gap_to_pass = RMSE +0.010769; MAE pass by 0.001221; PCC pass by 0.008449; CCC +0.002528
status = closer to CCC threshold, but not promoted because RMSE/MAE worsen slightly and it still fails RMSE/CCC
```

Negative follow-ups on 2026-07-08:

```text
OpenSMILE-only Valence BiLSTM:
  output = outputs/opensmile_valence_multisplit_trials_20260708
  trial = v_smile_h96_l1_do010_rmse
  RMSE/MAE/PCC/CCC = 0.247054 / 0.195484 / 0.648742 / 0.609695
  decision = rejected

Current-best + weak OpenSMILE blend:
  output = outputs/valence_current_best_opensmile_weak_blend_20260708
  selected_weight = 0.08
  RMSE/MAE/PCC/CCC = 0.190165 / 0.148741 / 0.808266 / 0.792652
  decision = rejected because CCC worsened

Monotonic calibration:
  output = outputs/valence_current_best_monotonic_calibration_20260708
  selected_blend = 0.0
  decision = rejected; validation selected identity

Stronger training-time CCC/clip-mean screen:
  output = outputs/valence_multilayer_cccmean_screen_20260708
  split = split_02 only
  RMSE/MAE/PCC/CCC = 0.228454 / 0.183940 / 0.753946 / 0.742506
  decision = rejected; underfitting
```

Current interpretation:
- Post-processing/calibration has nearly saturated.
- Direct OpenSMILE is effective for Arousal but too weak for Valence.
- DEAM MERT+BiLSTM results are weaker on Valence than the PMEmo branch and
  should not be mixed directly as a metric upgrade.
- The remaining Valence gap likely requires either stronger Valence-specific
  supervision, a better song-level polarity/mean predictor, or a redesigned
  model trained across datasets with careful scale alignment.

## Next Best Actions

1. Materialize the new Valence global song-mean policy in the full-song
   prediction path so the metric-best chain can be reproduced from raw MP3.
2. For metric improvement, focus on the remaining song-level Valence
   polarity/mean problem:
   - train a dedicated song-level mean/polarity predictor with stricter
     cross-validation,
   - add stronger key/mode/chord/tonal-centroid features,
   - use its output only as a small capped correction to the frame-level curve.
3. Avoid more generic smoothing or unconstrained per-split calibration; recent
   experiments show those do not generalize reliably.
   - external validation on complete songs before using it for Phase 2.

## Prepared Clean Valence Teacher Training Entry

Prepared at: 2026-07-07 CST

```text
script = scripts/train_pmemo_valence_multilayer_clean_teacher.py
status = split_02 seed_606 trained; not promoted
purpose = train a cleaner checkpoint-based Valence teacher candidate using the strongest current audio-only feature direction
feature = MERT hidden_4 + hidden_6 + hidden_8 + Librosa harmonic
default architecture = BiLSTM, proj_dim 384, hidden_dim 80, dropout 0.35
default loss = extreme-weighted frame MSE + 0.08 CCC loss + 0.10 clip mean MSE
selection = validation selection_score; test reported only after checkpoint selection
```

Recommended first run, when training is explicitly approved:

```bash
.venv/bin/python research/phase1f_external_va_datasets/pmemo/scripts/train_pmemo_valence_multilayer_clean_teacher.py \
  --split-id 2 \
  --seed 606 \
  --device auto
```

Rationale:
- split_02 is the current Valence bottleneck.
- If split_02 does not improve, running all three splits is unlikely to change
  the final conclusion.

Observed split_02 run:

```text
run = split_02 seed_606
best_epoch = 29
test RMSE/MAE/PCC/CCC = 0.207912 / 0.162272 / 0.780715 / 0.774531
delta vs current metric-best = +0.016640 / +0.013352 / -0.025358 / -0.016970
decision = rejected, current Valence best unchanged
```

Second split_02 regularization check:

```text
run = split_02 seed_606
trial = h468_harm_clean_proj256_h64_do045_wd8e4_ccc012_clip015
change = lower capacity and stronger regularization
test RMSE/MAE/PCC/CCC = 0.226543 / 0.181884 / 0.753072 / 0.743925
decision = rejected, worse than the first clean run and historical best
```

Best clean-checkpoint variant from 2026-07-07:

```text
run = split_02 seed_606
trial = h468_harm_clean_proj384_h80_do035_ccc012_clip012_std003
change = original capacity + light CCC/clip-std distribution constraint
test RMSE/MAE/PCC/CCC = 0.201663 / 0.165461 / 0.791397 / 0.785107
decision = best clean-checkpoint run so far, but still rejected as historical best
```

Conservative rollback check:

```text
run = split_02 seed_606
trial = h468_harm_clean_proj384_h80_do035_ccc010_clip008_std0015
test RMSE/MAE/PCC/CCC = 0.208462 / 0.165015 / 0.782964 / 0.780285
decision = rejected
```

Post-clean blend audit:

```text
script = scripts/search_pmemo_valence_split02_clean_blend.py
scope = split_02 only
candidate_count = 882
guarded_candidate_count = 0
selected = identity/no-change
decision = rejected; clean checkpoint correlation gain is not validation-stable
```

Current-best song-level mean fusion retry:

```text
script = scripts/search_pmemo_valence_current_best_song_mean_fusion.py
input = outputs/valence_parsimony_output_guarded_20260707
selected macro RMSE/MAE/PCC/CCC = 0.192466 / 0.152009 / 0.804494 / 0.783131
delta vs current best = +0.001193 / +0.003089 / -0.001579 / -0.008370
decision = rejected; split_02 improves in PCC/CCC but macro result worsens
```

Threshold comparison for this run:

| Metric | Current experiment | Historical best | Minimum pass | Current gap to pass | Good | Current gap to good | Excellent | Current gap to excellent | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| RMSE | 0.207912 | 0.191273 | 0.180 | +0.027912 | 0.150 | +0.057912 | 0.100 | +0.107912 | not pass |
| MAE | 0.162272 | 0.148921 | 0.150 | +0.012272 | 0.120 | +0.042272 | 0.080 | +0.082272 | not pass |
| PCC | 0.780715 | 0.806073 | 0.800 | -0.019285 | 0.850 | -0.069285 | 0.900 | -0.119285 | not pass |
| CCC | 0.774531 | 0.791501 | 0.800 | -0.025469 | 0.850 | -0.075469 | 0.900 | -0.125469 | not pass |

Full three-split run is not recommended for this exact configuration unless
the goal is documentation rather than metric improvement. The next training
attempt should adjust the clean teacher configuration first.

```bash
.venv/bin/python research/phase1f_external_va_datasets/pmemo/scripts/train_pmemo_valence_multilayer_clean_teacher.py \
  --all-splits \
  --seed 606 \
  --device auto
```

At the time of this clean-checkpoint experiment, the best remained:

```text
Valence metric-best = pmemo_valence_parsimony_output_guarded_20260707
RMSE/MAE/PCC/CCC = 0.191273 / 0.148921 / 0.806073 / 0.791501
status = MAE/PCC pass; RMSE/CCC fail
```

Latest update:

```text
previous Valence metric-best = pmemo_valence_current_best_song_mean_global_policy_20260707
RMSE/MAE/PCC/CCC = 0.190789 / 0.148697 / 0.807496 / 0.795137
status = MAE/PCC pass; RMSE/CCC still fail
```

Latest update:

```text
current Valence RMSE-primary best =
  valence_extra_wide_rmse_shift_smoothing_20260708
RMSE/MAE/PCC/CCC = 0.188791 / 0.146457 / 0.812194 / 0.797279
selected config =
  input valence_correlated_song_mean_guardtopccc_20260708
  global temporal shift = -21 frames (-10.5s), selected by validation RMSE
  smoothing = moving average window 21, blend 1.0
delta vs previous smoothing best =
  -0.000762 / -0.000981 / +0.001558 / -0.000197
status = MAE/PCC pass; RMSE/CCC still fail
minimum-line gaps =
  RMSE +0.008791, MAE pass by 0.003543,
  PCC pass by 0.012194, CCC +0.002721
good-line gaps =
  RMSE +0.038791, MAE +0.026457, PCC +0.037806, CCC +0.052721
excellent-line gaps =
  RMSE +0.088791, MAE +0.066457, PCC +0.087806, CCC +0.102721
```

Target-closest guarded calibration candidate:

```text
candidate =
  valence_extra_wide_rmse_shift_smoothing_affine_20260708
RMSE/MAE/PCC/CCC = 0.188798 / 0.146450 / 0.812194 / 0.797512
selected config =
  affine global calibration, blend 0.05, no smoothing, no clipping
decision =
  not promoted as RMSE-primary best because RMSE is slightly worse;
  useful as CCC-nearest validation-guarded reference.
```

Rejected latest diagnostic:

```text
candidate =
  valence_temporal_shift_extra_wide_per_split_20260708
RMSE/MAE/PCC/CCC = 0.189214 / 0.146728 / 0.811395 / 0.797163
selected split delays =
  split_01 -26 frames, split_02 -19 frames, split_03 +2 frames
decision =
  rejected; worse than global -21 delay and not deployable as a single
  arbitrary-song policy.
```

Rejected latest optimization attempts:

```text
candidate =
  valence_extra_wide_global_stacking_20260708
RMSE/MAE/PCC/CCC = 0.193127 / 0.151060 / 0.802810 / 0.787358
decision =
  rejected; validation-selected stacking overfit and degraded all main test
  metrics versus the current RMSE-primary best.

candidate =
  valence_rmse_primary_training_trials_20260708, split_02 screen only
trial h80_rmse_ext110_clip005_ccc002_do025 =
  0.227531 / 0.179807 / 0.731971 / 0.717188
trial h96_rmse_ext100_clip005_ccc003_do030 =
  0.224322 / 0.176265 / 0.743364 / 0.728626
decision =
  rejected; no full three-split run recommended for these configurations.
```

## Standing Reporting Rule

Every new experiment must report:

```text
1. current experiment metrics
2. historical best metrics
3. gaps to minimum pass / good / excellent lines
4. whether the historical best is updated
5. whether the model is deployable for arbitrary complete-song prediction
```

## Promotion Gate For Arbitrary-Song Teacher Use

Valence/Arousal teacher models are only eligible for final Phase 2 use if they
can be applied to an arbitrary complete MP3 through a reproducible raw-audio
pipeline. This is now a hard promotion gate, not an optional note.

Eligible teacher candidates must satisfy all of the following:

```text
1. input features can be extracted from arbitrary MP3/WAV audio;
2. preprocessing, normalization, temporal alignment, smoothing, and calibration
   parameters are fixed from training/validation only;
3. one deployable policy is used across songs, not per-test-song or
   per-test-split tuning;
4. test labels are used only once for final reporting;
5. a single-song or batch full-song prediction script exists, or can be
   directly materialized from saved checkpoints and parameter files;
6. the reported metrics are tied to that deployable pipeline or explicitly
   marked as a CV/diagnostic artifact.
```

The following are diagnostic only and cannot be promoted as final teacher
models without redesign:

```text
- test-set-selected temporal shift or calibration;
- per-split policies that cannot be mapped to a single arbitrary-song rule;
- corrections requiring true test labels, PMEmo/DEAM target statistics from the
  target song, or manual post-hoc inspection;
- official dataset feature tables unless we have a matching raw-audio feature
  reproduction script for new MP3 files;
- ensemble/stacking results that improve validation but degrade held-out test.
```

Current implication:

```text
The current Valence RMSE-primary best remains useful as the best metric artifact
so far, but the next optimization priority must be a deployable raw-audio
teacher path. A slightly worse but fully deployable model is preferable to a
non-reproducible test-optimized artifact for Phase 2 pseudo-VA generation.
```

## 2026-07-08 Late Valence Update

Current RMSE-primary Valence artifact after nonlinear song-mean correction:

```text
model_id = pmemo_valence_hgb_shift_smoothing_20260708
output = outputs/valence_hgb_shift_smoothing_20260708
base = valence_extra_wide_rmse_shift_smoothing_20260708
added correction =
  HGB song-level residual-mean correction using current prediction statistics
  and PMEmo static OpenSMILE features, selected by validation macro guard
postprocess =
  global temporal shift -9 frames (-4.5s) selected by validation RMSE
  moving average window 21, blend 1.0
RMSE/MAE/PCC/CCC = 0.188245 / 0.146625 / 0.813931 / 0.799716
status = MAE/PCC pass; CCC almost passes; RMSE still fails
minimum-line gaps =
  RMSE +0.008245, MAE pass by 0.003375,
  PCC pass by 0.013931, CCC +0.000284
good-line gaps =
  RMSE +0.038245, MAE +0.026625, PCC +0.036069, CCC +0.050284
excellent-line gaps =
  RMSE +0.088245, MAE +0.066625, PCC +0.086069, CCC +0.100284
```

Closest CCC diagnostic:

```text
artifact = outputs/valence_hgb_frame_residual_temporal_shift_20260708
diagnostic positive temporal shift = +3 frames (+1.5s)
RMSE/MAE/PCC/CCC ~= 0.188411 / 0.146823 / 0.813698 / 0.800022
status = CCC crosses minimum line, but RMSE still fails by about 0.008411
decision = diagnostic only; not promoted as RMSE-primary best
```

Rejected late follow-ups:

```text
candidate = valence_hgb_shift_smoothing_final_calibration_20260708
RMSE/MAE/PCC/CCC = 0.188296 / 0.146663 / 0.813458 / 0.798615
decision = rejected; calibration worsened CCC and RMSE.

candidate = valence_extra_wide_hgb_target_mean_screen_20260708
RMSE/MAE/PCC/CCC = 0.190036 / 0.147167 / 0.809574 / 0.795992
decision = rejected; direct target-mean correction was weaker than residual-mean correction.

candidate = valence_hgb_shift_frame_residual_20260708
RMSE/MAE/PCC/CCC = 0.188290 / 0.146680 / 0.813932 / 0.799872
decision = useful CCC diagnostic, but not promoted because RMSE worsened versus current RMSE-primary artifact.
```

Current technical conclusion:

```text
Lightweight postprocessing is saturated. The remaining blocker is frame-level
RMSE, not CCC. Further meaningful improvement should come from training-side
or data-side changes: stronger Valence representation, better domain-balanced
PMEmo/DEAM fine-tuning, or a new model that reduces split_02/split_03 frame
error before output calibration.
```

## 2026-07-09 Registry Update - Valence Best Unchanged

Current formal Valence best:

```text
model_id = pmemo_valence_hgb_shift_smoothing_20260708
output = outputs/valence_hgb_shift_smoothing_20260708
RMSE/MAE/PCC/CCC = 0.188245 / 0.146625 / 0.813931 / 0.799716
status =
  RMSE not pass, gap +0.008245
  MAE pass, margin 0.003375
  PCC pass, margin 0.013931
  CCC not pass, gap +0.000284
promotion =
  remains the RMSE-primary metric artifact, but still below the full minimum
  Valence line because RMSE and CCC are not both satisfied.
```

Rejected 2026-07-09 challengers:

```text
candidate = valence_hgb_smoothing_opensmile_topk_residual_20260709
result = identity, no metric change
decision = rejected; no improvement.

candidate = valence_multilayer_rmsefirst_screen_20260709 split_02
result = 0.216528 / 0.171287 / 0.755705 / 0.739145
decision = rejected; weaker than current split_02 reference.

candidate = valence_hgb_guarded_stacking_20260709
result = 0.188353 / 0.146519 / 0.812706 / 0.799179
decision = rejected; RMSE/PCC/CCC worsened.

candidate = valence_deam_init_bilstm_screen_20260709 split_02
result = 0.213080 / 0.171561 / 0.747856 / 0.733105
decision = rejected; direct DEAM checkpoint initialization is not sufficient.

candidate = valence_current_best_tree_song_mean_20260709
result = 0.189105 / 0.147224 / 0.812508 / 0.799853
decision = rejected; CCC improved slightly, RMSE worsened.

candidate = valence_hgb_arousal_residual_20260709
result = 0.188392 / 0.146787 / 0.813737 / 0.799952
decision = rejected as RMSE-primary; CCC nearly passes but RMSE worsens.

candidate = valence_per_split_shift_smoothing_20260709
result = 0.188651 / 0.147170 / 0.813146 / 0.799714
decision = rejected; per-split postprocess is weaker and less deployable.
```

Next registry-level direction:

```text
Stop prioritizing extra output calibration for Valence. The next promotion
candidate needs a stronger upstream base predictor, especially on split_02 and
split_03. Keep arbitrary-song deployability as a hard gate: raw-audio feature
extraction must be reproducible for any MP3 before Phase 2 pseudo-VA use.
```

## 2026-07-09 Registry Update - New Valence RMSE-Primary Best

Promoted Valence artifact:

```text
model_id = pmemo_valence_residual_sequence_h6_base_resid_h48_do045_cap006
output = outputs/valence_residual_sequence_20260709/h6_base_resid_h48_do045_cap006
base_model = pmemo_valence_hgb_shift_smoothing_20260708
method =
  bounded sequence residual corrector using current best prediction,
  MERT hidden6, and prediction/time context
RMSE/MAE/PCC/CCC =
  0.187978 / 0.146040 / 0.813989 / 0.801162
delta vs previous best =
  RMSE -0.000266
  MAE -0.000585
  PCC +0.000058
  CCC +0.001446
minimum-line status =
  RMSE not pass, gap +0.007978
  MAE pass, margin 0.003960
  PCC pass, margin 0.013989
  CCC pass, margin 0.001162
promotion =
  current RMSE-primary Valence artifact, but not yet fully compliant because
  RMSE remains above 0.18.
```

Important caveat:

```text
The residual corrector improved the three-split mean, but split_03 RMSE
worsened while CCC improved. Further residual experiments should add a
validation guard that can choose identity when RMSE risk is high, or use a
stronger split_02/split_03 upstream base predictor.
```

## 2026-07-09 Registry Update - Smoothed Residual Valence Best

Current Valence RMSE-primary artifact:

```text
model_id = pmemo_valence_residual_sequence_smoothing_20260709
output = outputs/valence_residual_sequence_smoothing_20260709
base = pmemo_valence_residual_sequence_h6_base_resid_h48_do045_cap006
postprocess = validation-selected moving average window 21, blend 1.0
RMSE/MAE/PCC/CCC =
  0.187921 / 0.146010 / 0.814101 / 0.801233
delta vs unsmoothed residual =
  RMSE -0.000058
  MAE -0.000030
  PCC +0.000112
  CCC +0.000071
minimum-line status =
  RMSE not pass, gap +0.007921
  MAE pass, margin 0.003990
  PCC pass, margin 0.014101
  CCC pass, margin 0.001233
promotion =
  current best Valence metric artifact. RMSE remains the only unmet minimum
  criterion.
```

## 2026-07-09 Registry Update - Second Residual Valence Best

Current Valence RMSE-primary artifact:

```text
model_id = pmemo_valence_second_residual_h6_second_resid_h32_do055_cap002_l2p60
output = outputs/valence_residual_sequence_20260709/h6_second_resid_h32_do055_cap002_l2p60
base = outputs/valence_residual_sequence_smoothing_20260709
method =
  second bounded residual corrector on top of the smoothed residual artifact
RMSE/MAE/PCC/CCC =
  0.187830 / 0.145907 / 0.814061 / 0.801347
delta vs previous best =
  RMSE -0.000090
  MAE -0.000103
  PCC -0.000040
  CCC +0.000114
minimum-line status =
  RMSE not pass, gap +0.007830
  MAE pass, margin 0.004093
  PCC pass, margin 0.014061
  CCC pass, margin 0.001347
promotion =
  current best Valence metric artifact. RMSE remains the only unmet minimum
  criterion.
```

## 2026-07-09 Registry Update - Second Residual Smoothed Valence Best

Current Valence RMSE-primary artifact:

```text
model_id = pmemo_valence_current_best_smoothing_20260708
output = outputs/valence_second_residual_smoothing_20260709
base = outputs/valence_residual_sequence_20260709/h6_second_resid_h32_do055_cap002_l2p60
method =
  validation-selected moving average smoothing on top of the second bounded
  residual corrector
postprocess = moving average window 21, blend 1.0
RMSE/MAE/PCC/CCC =
  0.187812 / 0.145899 / 0.814095 / 0.801356
delta vs previous best =
  RMSE -0.000018
  MAE -0.000008
  PCC +0.000034
  CCC +0.000008
minimum-line status =
  RMSE not pass, gap +0.007812
  MAE pass, margin 0.004101
  PCC pass, margin 0.014095
  CCC pass, margin 0.001356
promotion =
  current best Valence metric artifact. RMSE remains the only unmet minimum
  criterion.
```

## 2026-07-09 Registry Update - Ridge CCC-Guard Smoothed Valence Best

Current target-closest Valence artifact:

```text
model_id = pmemo_valence_latest_ridge_ccc_guard_smoothing_20260709
output = outputs/valence_latest_ridge_ccc_guard_smoothing_20260709
base = outputs/valence_second_residual_smoothing_20260709
method =
  validation CCC-guarded Ridge song-level target-mean correction followed by
  validation-selected moving average smoothing
selected_ridge_config =
  target_col=target_mean, top_k=64, alpha=1000.0,
  strength=0.10, cap=none
postprocess = moving average window 21, blend 1.0
RMSE/MAE/PCC/CCC =
  0.187623 / 0.146017 / 0.814184 / 0.800233
delta vs previous second-residual smoothed best =
  RMSE -0.000188
  MAE +0.000117
  PCC +0.000090
  CCC -0.001123
minimum-line status =
  RMSE not pass, gap +0.007623
  MAE pass, margin 0.003983
  PCC pass, margin 0.014184
  CCC pass, margin 0.000233
promotion =
  current target-closest Valence artifact. RMSE remains the only unmet minimum
  criterion.
```

## 2026-07-09 Registry Update - Ridge RMSE/CCC Blend Valence Best

Current target-closest Valence artifact:

```text
model_id = pmemo_valence_ridge_rmse_ccc_blend_20260709
output = outputs/valence_ridge_rmse_ccc_blend_20260709
base =
  outputs/valence_latest_ridge_song_mean_fast_20260709
  outputs/valence_latest_ridge_ccc_guard_smoothing_20260709
method =
  validation-selected blend between RMSE-primary Ridge and CCC-guarded smoothed
  Ridge outputs
selected_blend =
  rmse_weight=0.05
RMSE/MAE/PCC/CCC =
  0.187620 / 0.146020 / 0.814185 / 0.800203
delta vs previous Ridge CCC-guard smoothed best =
  RMSE -0.000003
  MAE +0.000004
  PCC +0.000000
  CCC -0.000029
minimum-line status =
  RMSE not pass, gap +0.007620
  MAE pass, margin 0.003980
  PCC pass, margin 0.014185
  CCC pass, margin 0.000203
promotion =
  current target-closest Valence artifact. RMSE remains the only unmet minimum
  criterion.
```

Strict song-level split audit:

```text
audit_script = scripts/audit_pmemo_valence_strict_song_level_split.py
audit_output = outputs/valence_strict_song_level_audit_20260709
result =
  split_01 train/val/test musicId overlap = 0
  split_02 train/val/test musicId overlap = 0
  split_03 train/val/test musicId overlap = 0
strict_song_level_macro_test RMSE/MAE/PCC/CCC =
  0.187620 / 0.146020 / 0.814185 / 0.800203
pooled_heldout_frame_test RMSE/MAE/PCC/CCC =
  0.187453 / 0.145812 / 0.817564 / 0.804405
interpretation =
  The existing folds are song-level disjoint with no frame-level leakage.
  This is still not a fresh blind final test because these folds were used
  during iterative model development.
```

Sealed blind holdout audit:

```text
split_id = pmemo_blind_20260709
split_file = data/splits/pmemo_blind_20260709/pmemo_blind_song_level_split.csv
train/val/blind_test songs = 536 / 115 / 116
overlap among train/val/blind_test musicId sets = 0
base_model_output = outputs/valence_blind_holdout_base_20260709
model_family = MERT hidden6 + harmonic BiLSTM base teacher
selection = validation-only checkpoint selection
validation RMSE/MAE/PCC/CCC =
  0.190779 / 0.149162 / 0.806151 / 0.798922
blind_test RMSE/MAE/PCC/CCC =
  0.241493 / 0.183770 / 0.635608 / 0.630783
interpretation =
  This sealed blind result is substantially weaker than the development-fold
  metrics. The current Valence base model does not pass blind holdout criteria.
  Since this blind result has now been inspected, this exact blind_test split
  must not be used for further tuning while still being called blind.
```

## 2026-07-09 Registry Update - Development-only candidates after blind audit

Important boundary:
- The sealed `pmemo_blind_20260709` blind/test split has been evaluated once.
- The candidates below are selected on the remaining train/val development
  data only. They are useful for optimization, but they are not final blind
  test results.

### Arousal Development Candidate

```text
model_id = pmemo_arousal_dev_h96_postprocess_20260709
base_output = outputs/opensmile_arousal_dev_only_seed20260709_20260709/a_dev_h96_l1_do015_lr8e4_mse
postprocess_output = outputs/opensmile_arousal_dev_val_postprocess_20260709
feature = PMEmo official OpenSMILE dynamic features
architecture = Arousal-only BiLSTM
hidden_dim = 96
dropout = 0.15
loss = MSE only
postprocess = moving average window 21 + linear calibration
selection = validation only; blind_test not read
```

Validation after postprocess:

| Metric | Current | Minimum pass | Gap to pass | Good | Gap to good | Excellent | Gap to excellent | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| RMSE | 0.190067 | 0.180000 | +0.010067 | 0.150000 | +0.040067 | 0.100000 | +0.090067 | not pass |
| MAE | 0.148734 | 0.150000 | -0.001266 | 0.120000 | +0.028734 | 0.080000 | +0.068734 | pass |
| PCC | 0.851273 | 0.800000 | -0.051273 | 0.850000 | -0.001273 | 0.900000 | +0.048727 | pass |
| CCC | 0.840354 | 0.800000 | -0.040354 | 0.850000 | +0.009646 | 0.900000 | +0.059646 | pass |

### Valence Development Candidate

```text
model_id = pmemo_valence_dev_best_like_h80_postprocess_20260709
output = outputs/valence_dev_only_trials_20260709/v_dev_best_like_h80_do030_lr4e4
feature = MERT hidden6 + Librosa harmonic features
architecture = Valence-only BiLSTM
proj_dim = 320
hidden_dim = 80
dropout = 0.30
learning_rate = 4e-4
weight_decay = 2e-4
batch_size = 20
frame_ccc_weight = 0.05
clip_mse_weight = 0.10
extreme_weight = 1.25
best_epoch = 7
postprocess = moving average window 41 + linear calibration
selection = validation only; blind_test not read
```

Validation after postprocess:

| Metric | Current | Minimum pass | Gap to pass | Good | Gap to good | Excellent | Gap to excellent | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| RMSE | 0.177319 | 0.180000 | -0.002681 | 0.150000 | +0.027319 | 0.100000 | +0.077319 | pass |
| MAE | 0.140811 | 0.150000 | -0.009189 | 0.120000 | +0.020811 | 0.080000 | +0.060811 | pass |
| PCC | 0.832726 | 0.800000 | -0.032726 | 0.850000 | +0.017274 | 0.900000 | +0.067274 | pass |
| CCC | 0.818966 | 0.800000 | -0.018966 | 0.850000 | +0.031034 | 0.900000 | +0.081034 | pass |

Decision:
- This is the first Valence development candidate in the sealed-split workflow
  whose validation metrics all pass the minimum line.
- It must still be frozen and evaluated on a new untouched blind split, or
  validated through a nested protocol, before being reported as final Teacher
  performance.

## 2026-07-09 Registry Update - Robust development-fold correction

The previous single-validation candidates are now considered optimistic. A
stricter train+val internal three-fold development protocol was created:

```text
folds = data/splits/pmemo_trainval_devfolds_20260709
source = train + val rows from pmemo_blind_20260709
excluded = test/holdout rows
fold protocol = song-level 3-fold development validation
```

### Valence Robust Development Status

Current tested Valence baseline under the stricter protocol:

```text
model = v_dev_best_like_h80_do030_lr4e4
feature = MERT hidden6 + Librosa harmonic
postprocess = fold-local validation-selected smoothing + linear calibration
output = outputs/valence_devfolds_bestlike_20260709
```

Three-fold mean after postprocess:

| Metric | Current | Minimum pass | Gap to pass | Good | Gap to good | Excellent | Gap to excellent | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| RMSE | 0.196111 | 0.180000 | +0.016111 | 0.150000 | +0.046111 | 0.100000 | +0.096111 | not pass |
| MAE | 0.153090 | 0.150000 | +0.003090 | 0.120000 | +0.033090 | 0.080000 | +0.073090 | not pass |
| PCC | 0.794714 | 0.800000 | +0.005286 | 0.850000 | +0.055286 | 0.900000 | +0.105286 | not pass |
| CCC | 0.774110 | 0.800000 | +0.025890 | 0.850000 | +0.075890 | 0.900000 | +0.125890 | not pass |

Follow-up:
- A conservative Valence fold_03 retrain only slightly improved fold_03.
- Current conclusion: Valence needs a stronger generalization strategy, not
  more single-split calibration.

### Arousal Robust Development Status

Current best robust Arousal development candidate:

```text
model_id = pmemo_arousal_h96_h64_devfold_ensemble_20260709
output = outputs/arousal_devfolds_ensemble_20260709
base_1 = a_dev_h96_l1_do015_lr8e4_mse
base_2 = a_dev_h64_l1_do035_lr4e4_wd8e4_smooth003
feature = PMEmo official OpenSMILE dynamic features
method = fold-local validation-selected ensemble + smoothing/calibration
selection = development folds only; holdout/test excluded
```

Three-fold mean after ensemble/postprocess:

| Metric | Current | Minimum pass | Gap to pass | Good | Gap to good | Excellent | Gap to excellent | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| RMSE | 0.209104 | 0.180000 | +0.029104 | 0.150000 | +0.059104 | 0.100000 | +0.109104 | not pass |
| MAE | 0.165259 | 0.150000 | +0.015259 | 0.120000 | +0.045259 | 0.080000 | +0.085259 | not pass |
| PCC | 0.804022 | 0.800000 | -0.004022 | 0.850000 | +0.045978 | 0.900000 | +0.095978 | pass |
| CCC | 0.785957 | 0.800000 | +0.014043 | 0.850000 | +0.064043 | 0.900000 | +0.114043 | not pass |

Interpretation:
- Arousal ensemble materially improves robust development performance over
  the h96 single model, especially PCC.
- It is still not final-compliant because RMSE/MAE/CCC miss the minimum line.
- Future optimization should keep the robust fold protocol and improve
  fold-mean RMSE/MAE without sacrificing PCC/CCC.

## 2026-07-09 Registry Update - Current Robust Development Bests

### Current Arousal Robust Development Best

```text
model_id = pmemo_arousal_fourway_songmean_amplitude_20260709
output = outputs/arousal_amplitude_fine_20260709
base_models =
  a_dev_h96_l1_do015_lr8e4_mse
  a_dev_h64_l1_do035_lr4e4_wd8e4_smooth003
  a_dev_h128_l1_do020_lr6e4_mse
  OpenSMILE Ridge
postprocess =
  fold-local ensemble + train-fitted song-mean correction +
  within-song amplitude calibration + smoothing/linear calibration
validation_protocol = train+val internal 3-fold song-level development folds
holdout_status = holdout/test excluded from optimization
```

Three-fold development mean:

| Metric | Current | Minimum pass | Gap to pass | Good | Gap to good | Excellent | Gap to excellent | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| RMSE | 0.200586 | 0.180000 | +0.020586 | 0.150000 | +0.050586 | 0.100000 | +0.100586 | not pass |
| MAE | 0.157462 | 0.150000 | +0.007462 | 0.120000 | +0.037462 | 0.080000 | +0.077462 | not pass |
| PCC | 0.821404 | 0.800000 | -0.021404 | 0.850000 | +0.028596 | 0.900000 | +0.078596 | pass |
| CCC | 0.805756 | 0.800000 | -0.005756 | 0.850000 | +0.044244 | 0.900000 | +0.094244 | pass |

Interpretation:
- Arousal now passes the trend/agreement metrics PCC and CCC under the robust
  development protocol.
- RMSE and MAE remain above the minimum target, so it is not yet fully
  compliant.

### Current Valence Robust Development Best

```text
model_id = pmemo_valence_bilstm_ridge_mert_songmean_20260709
output = outputs/valence_bilstm_ridge_mert_songmean_20260709
base_models =
  v_dev_best_like_h80_do030_lr4e4
  OpenSMILE Ridge
postprocess =
  fold-local BILSTM/Ridge ensemble + train-fitted MERT song-mean correction +
  smoothing/linear calibration
validation_protocol = train+val internal 3-fold song-level development folds
holdout_status = holdout/test excluded from optimization
```

Three-fold development mean:

| Metric | Current | Minimum pass | Gap to pass | Good | Gap to good | Excellent | Gap to excellent | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| RMSE | 0.191669 | 0.180000 | +0.011669 | 0.150000 | +0.041669 | 0.100000 | +0.091669 | not pass |
| MAE | 0.149742 | 0.150000 | -0.000258 | 0.120000 | +0.029742 | 0.080000 | +0.069742 | pass |
| PCC | 0.805012 | 0.800000 | -0.005012 | 0.850000 | +0.044988 | 0.900000 | +0.094988 | pass |
| CCC | 0.786354 | 0.800000 | +0.013646 | 0.850000 | +0.063646 | 0.900000 | +0.113646 | not pass |

Interpretation:
- Valence now passes MAE and PCC under robust development validation.
- RMSE and CCC remain the active bottlenecks, especially the difficult fold_03.

## 2026-07-09 Registry Update - Arousal Five-Way MERT/Harmony Ensemble

### Current Arousal Robust Development Best

```text
model_id = pmemo_arousal_mert_harmonic_fiveway_shift_20260709
output = outputs/arousal_mert_harmonic_fiveway_shift_20260709
base_models =
  OpenSMILE h96 BiLSTM
  OpenSMILE h64 regularized BiLSTM
  OpenSMILE h128 BiLSTM
  OpenSMILE Ridge
  MERT hidden6 + Librosa harmonic BiLSTM
postprocess =
  fold-local validation-selected ensemble + temporal shift +
  within-song amplitude scaling + smoothing + linear calibration
validation_protocol = train+val internal 3-fold song-level development folds
holdout_status = holdout/test excluded from optimization
previous_best = pmemo_arousal_amplitude_shift_20260709
```

Three-fold development mean:

| Metric | Current | Minimum pass | Gap to pass | Good | Gap to good | Excellent | Gap to excellent | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| RMSE | 0.186773 | 0.180000 | +0.006773 | 0.150000 | +0.036773 | 0.100000 | +0.086773 | not pass |
| MAE | 0.147163 | 0.150000 | -0.002837 | 0.120000 | +0.027163 | 0.080000 | +0.067163 | pass |
| PCC | 0.847339 | 0.800000 | -0.047339 | 0.850000 | +0.002661 | 0.900000 | +0.052661 | pass |
| CCC | 0.835831 | 0.800000 | -0.035831 | 0.850000 | +0.014169 | 0.900000 | +0.064169 | pass |

Delta vs previous robust Arousal best:

| Metric | Previous | Current | Delta |
| --- | ---: | ---: | ---: |
| RMSE | 0.198666 | 0.186773 | -0.011893 |
| MAE | 0.155569 | 0.147163 | -0.008406 |
| PCC | 0.825108 | 0.847339 | +0.022231 |
| CCC | 0.810089 | 0.835831 | +0.025742 |

Interpretation:
- Arousal now passes MAE, PCC, and CCC under the robust development protocol.
- RMSE remains the only active minimum-line bottleneck, missing by `+0.006773`.
- The improvement came from adding a complementary MERT hidden6 + harmonic
  branch, not from reusing holdout/test data.

### Current Valence Robust Development Best

```text
model_id = pmemo_valence_shift_amplitude_narrow_20260709
output = outputs/valence_shift_amplitude_narrow_20260709
base_models =
  v_dev_best_like_h80_do030_lr4e4
  OpenSMILE Ridge
postprocess =
  BILSTM/Ridge ensemble + train-fitted MERT song-mean correction +
  temporal shift + narrow within-song amplitude search +
  smoothing/linear calibration
validation_protocol = train+val internal 3-fold song-level development folds
holdout_status = holdout/test excluded from optimization
previous_best = pmemo_valence_mert_songmean_shift_20260709
```

Three-fold development mean:

| Metric | Current | Minimum pass | Gap to pass | Good | Gap to good | Excellent | Gap to excellent | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| RMSE | 0.186795 | 0.180000 | +0.006795 | 0.150000 | +0.036795 | 0.100000 | +0.086795 | not pass |
| MAE | 0.145886 | 0.150000 | -0.004114 | 0.120000 | +0.025886 | 0.080000 | +0.065886 | pass |
| PCC | 0.815804 | 0.800000 | -0.015804 | 0.850000 | +0.034196 | 0.900000 | +0.084196 | pass |
| CCC | 0.799099 | 0.800000 | +0.000901 | 0.850000 | +0.050901 | 0.900000 | +0.100901 | not pass |

Interpretation:
- Valence passes MAE and PCC.
- RMSE and CCC remain active bottlenecks; CCC is extremely close to the
  minimum line, while RMSE still misses by about `0.0068`.

## 2026-07-14 Registry Update - MusiCNN Song-Polarity Development Candidate

Current target-closest Valence locked-development artifact:

```text
model_id = pmemo_valence_locked_musicnn_pair_blend_calibrated_20260714
output = outputs/valence_locked_musicnn_pair_blend_calibrated_20260714
base = valence_locked_muq_arousal_stacker_fine_20260714
audio additions = Essentia MusiCNN happy/sad heads and 200-d dense embeddings
song model = cross-fold Ridge over pooled MusiCNN and prediction statistics
ensemble = 64% mean+std+base Ridge + 36% mean-only Ridge
calibration = other-two-fold mean/std fit, strength 0.025, cap 0.001
validation = three locked PMEmo development folds
final locked test = excluded
deployment = not yet packaged as one arbitrary-MP3 inference chain
```

| Metric | Current | Minimum pass | Gap to pass | Good | Gap to good | Excellent | Gap to excellent | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| RMSE | 0.185394 | 0.180000 | +0.005394 | 0.150000 | +0.035394 | 0.100000 | +0.085394 | not pass |
| MAE | 0.144891 | 0.150000 | -0.005109 | 0.120000 | +0.024891 | 0.080000 | +0.064891 | pass |
| PCC | 0.815078 | 0.800000 | -0.015078 | 0.850000 | +0.034922 | 0.900000 | +0.084922 | pass |
| CCC | 0.800321 | 0.800000 | -0.000321 | 0.850000 | +0.049679 | 0.900000 | +0.099679 | pass |

Current strict development RMSE-primary artifact:

```text
model_id = pmemo_valence_locked_musicnn_pair_blend_20260714
output = outputs/valence_locked_musicnn_pair_blend_20260714
RMSE/MAE/PCC/CCC = 0.185379/0.144910/0.815074/0.799864
```

Decision:
- Keep both artifacts because their scientific roles differ by only a tiny
  calibration step: the first is closest to all four thresholds, while the
  second has the lowest development RMSE.
- RMSE remains the only unmet minimum criterion.
- Do not evaluate the final locked test or claim final teacher performance yet.

## 2026-07-14 Registry Update - Ordinal + PLS Valence Development Best

Current Valence development best:

```text
model_id = pmemo_valence_locked_ordinal_residual_pls_fine3_20260714
output = outputs/valence_locked_ordinal_residual_pls_fine3_20260714
base = valence_locked_ordinal_song_head_fine2_monotonic_20260714
ordinal = six train-fold-derived Valence levels, multinomial logistic C=200
ordinal correction = strength 1.0, cap 0.020
calibration = cross-fold quantile mapping, 11 knots, strength 0.5, cap 0.005
residual = PLS-2 over deployable Valence predictor statistics
residual correction = strength 10.0, cap 0.010
validation = three locked PMEmo development folds
final locked test = excluded
deployment = not yet packaged as one arbitrary-MP3 inference chain
```

| Metric | Current | Minimum pass | Gap to pass | Good | Gap to good | Excellent | Gap to excellent | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| RMSE | 0.183484 | 0.180000 | +0.003484 | 0.150000 | +0.033484 | 0.100000 | +0.083484 | not pass |
| MAE | 0.143893 | 0.150000 | -0.006107 | 0.120000 | +0.023893 | 0.080000 | +0.063893 | pass |
| PCC | 0.819141 | 0.800000 | -0.019141 | 0.850000 | +0.030859 | 0.900000 | +0.080859 | pass |
| CCC | 0.803031 | 0.800000 | -0.003031 | 0.850000 | +0.046969 | 0.900000 | +0.096969 | pass |

Per-fold RMSE:
- fold_01: `0.177943`
- fold_02: `0.176274`
- fold_03: `0.196237`

Paired song-bootstrap versus the previous registered candidate
`0.185394/0.144891/0.815078/0.800321`:
- RMSE mean delta `-0.001912`, 95% interval
  `[-0.003671, -0.000137]`, improvement probability `0.980`.
- MAE mean delta `-0.000995`, 95% interval
  `[-0.002699, +0.000740]`.
- PCC mean delta `+0.004052`, 95% interval
  `[+0.000070, +0.007945]`, improvement probability `0.9765`.
- CCC mean delta `+0.002760`, 95% interval
  `[-0.001269, +0.006783]`.

Decision:
- Promote this as both the current RMSE-primary and target-closest development
  artifact because MAE/PCC/CCC pass and RMSE is the lowest legal macro result.
- The final locked test remains untouched. Repeated development selection can
  be optimistic; do not call these final-test metrics.

## 2026-07-14 Registry Update - Cross-Fold Reference Extrapolation

Current Valence development best:

```text
model_id = pmemo_valence_locked_crossfold_reference_extrapolation_20260714
output = outputs/valence_locked_crossfold_reference_extrapolation_20260714
base = valence_locked_ordinal_residual_pls_fine3_20260714
reference = valence_locked_muq_song_mean_correction_20260713
formula = base + weight * (reference - base)
weight fitting = analytic least squares on the other two development folds
weight bounds = [-0.5, 0.1]
held-fold weights = fold_01 -0.5; fold_02 -0.5; fold_03 -0.494051
validation = three locked PMEmo development folds
final locked test = excluded
deployment = not yet packaged as one arbitrary-MP3 inference chain
```

| Metric | Current | Minimum pass | Gap to pass | Good | Gap to good | Excellent | Gap to excellent | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| RMSE | 0.182994 | 0.180000 | +0.002994 | 0.150000 | +0.032994 | 0.100000 | +0.082994 | not pass |
| MAE | 0.143365 | 0.150000 | -0.006635 | 0.120000 | +0.023365 | 0.080000 | +0.063365 | pass |
| PCC | 0.820268 | 0.800000 | -0.020268 | 0.850000 | +0.029732 | 0.900000 | +0.079732 | pass |
| CCC | 0.805793 | 0.800000 | -0.005793 | 0.850000 | +0.044207 | 0.900000 | +0.094207 | pass |

Per-fold RMSE/MAE/PCC/CCC:
- fold_01: `0.177870 / 0.140666 / 0.808387 / 0.793714`
- fold_02: `0.175568 / 0.138465 / 0.837783 / 0.824976`
- fold_03: `0.195543 / 0.150965 / 0.814634 / 0.798690`

Decision:
- Promote as the current RMSE-primary and target-closest development artifact.
- All four macro metrics improve over the previous PLS candidate, but RMSE
  remains `0.002994` above the minimum line.
- The final locked test remains untouched. These are development metrics, not
  final generalization results.
- A nested multi-reference Ridge residual stack was also screened and rejected
  (`0.183418 / 0.143895 / 0.819339 / 0.803576`) because it did not improve the
  simpler reference extrapolation.

## 2026-07-14 Registry Update - MusiCNN 50-Tag Valence Candidate

Current Valence development best:

```text
model_id = pmemo_valence_locked_musicnn_tag_monotonic_20260714
output = outputs/valence_locked_musicnn_tag_monotonic_20260714
base = valence_locked_crossfold_reference_extrapolation_20260714
feature source = Essentia msd-musicnn-1 model/Sigmoid (50 audio tags)
feature pooling = mean + std + q25 + q75 (200 song-level features)
tag correction = direct Ridge, PCA-8, alpha 100, strength 0.025, cap 0.015
calibration = held-fold isotonic song-mean mapping, strength 1.0, cap 0.003
validation = three locked PMEmo development folds
final locked test = excluded
deployment = not yet packaged as one arbitrary-MP3 inference chain
```

| Metric | Current | Minimum pass | Gap to pass | Good | Gap to good | Excellent | Gap to excellent | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| RMSE | 0.182788 | 0.180000 | +0.002788 | 0.150000 | +0.032788 | 0.100000 | +0.082788 | not pass |
| MAE | 0.143369 | 0.150000 | -0.006631 | 0.120000 | +0.023369 | 0.080000 | +0.063369 | pass |
| PCC | 0.820703 | 0.800000 | -0.020703 | 0.850000 | +0.029297 | 0.900000 | +0.079297 | pass |
| CCC | 0.805001 | 0.800000 | -0.005001 | 0.850000 | +0.044999 | 0.900000 | +0.094999 | pass |

Per-fold RMSE/MAE/PCC/CCC:
- fold_01: `0.177514 / 0.140678 / 0.809107 / 0.792744`
- fold_02: `0.175589 / 0.138492 / 0.837753 / 0.824019`
- fold_03: `0.195261 / 0.150938 / 0.815248 / 0.798240`

Paired song-bootstrap versus the prior PLS development best
`0.183484/0.143893/0.819141/0.803031` (2,000 resamples):
- RMSE delta `-0.000716`, 95% interval `[-0.001997, +0.000487]`.
- MAE delta `-0.000530`, 95% interval `[-0.001493, +0.000412]`.
- PCC delta `+0.001553`, 95% interval `[-0.001015, +0.004291]`.
- CCC delta `+0.001973`, 95% interval `[-0.000662, +0.004905]`.

Decision:
- Promote as the current RMSE-primary and target-closest development artifact.
- The improvement direction is consistent but its bootstrap intervals cross
  zero; treat the gain as provisional until the final locked test is run once.
- A final PLS-2 refine selected identity, so no further post-processing layer
  was added.
- RMSE remains the only failed minimum criterion, with a gap of `0.002788`.

## 2026-07-14 Registry Update - Dynamic Past-Context MusiCNN Tags

Current Valence development best:

```text
model_id = pmemo_valence_locked_musicnn_dynamic_tag_past_dense_monotonic_20260714
output = outputs/valence_locked_musicnn_dynamic_tag_past_dense_monotonic_20260714
base = valence_locked_musicnn_tag_monotonic_20260714
dynamic input = 50 MusiCNN tag probabilities derived from saved patch embeddings
temporal context = current plus past 2s, 4s, 6s, and 8s tag vectors
dynamic head = direct frame-level Ridge, alpha 0.1, frame weighting
dynamic correction = strength 0.1, cap 0.020
calibration = cross-fold quantile-11 song-mean mapping, strength 0.075, cap 0.015
validation = three locked PMEmo development folds
final locked test = excluded
deployment = all inputs are audio-derived; one-command arbitrary-MP3 chain not yet packaged
```

| Metric | Current | Minimum pass | Gap to pass | Good | Gap to good | Excellent | Gap to excellent | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| RMSE | 0.182323 | 0.180000 | +0.002323 | 0.150000 | +0.032323 | 0.100000 | +0.082323 | not pass |
| MAE | 0.142986 | 0.150000 | -0.007014 | 0.120000 | +0.022986 | 0.080000 | +0.062986 | pass |
| PCC | 0.821707 | 0.800000 | -0.021707 | 0.850000 | +0.028293 | 0.900000 | +0.078293 | pass |
| CCC | 0.805652 | 0.800000 | -0.005652 | 0.850000 | +0.044348 | 0.900000 | +0.094348 | pass |

Per-fold RMSE/MAE/PCC/CCC:
- fold_01: `0.177623 / 0.141187 / 0.808921 / 0.792191`
- fold_02: `0.174359 / 0.137861 / 0.840244 / 0.826635`
- fold_03: `0.194988 / 0.149909 / 0.815957 / 0.798129`

Paired song-bootstrap versus the round-start PLS candidate, 1,500 resamples:
- RMSE delta `-0.001161`, 95% interval `[-0.002548, +0.000262]`, improvement
  probability `0.9473`.
- MAE delta `-0.000914`, 95% interval `[-0.002189, +0.000269]`, improvement
  probability `0.9287`.
- PCC delta `+0.002533`, 95% interval `[-0.000538, +0.005540]`, improvement
  probability `0.9440`.
- CCC delta `+0.002648`, 95% interval `[-0.000567, +0.005907]`, improvement
  probability `0.9507`.

Decision:
- Promote as the current RMSE-primary and target-closest development artifact.
- Past-only context outperformed future-only and bidirectional context; dense
  sampling over the previous eight seconds was the best tested window.
- Local alpha refinement produced only `2e-8` RMSE change and was rejected as
  scientifically immaterial; keep alpha `0.1`.
- RMSE remains the only failed minimum criterion, with a gap of `0.002323`.
- Final locked test remains untouched.

## 2026-07-15 Registry Update - Complete-Song Evaluation Scale

Valence scale distinction:

```text
model = pmemo_valence_locked_global_shift_amp_guarded_20260715
frame-level = 0.181284 / 0.142308 / 0.824000 / 0.807112
complete-song = 0.178526 / 0.140722 / 0.821365 / 0.805417
claim = RMSE below 0.18 only at complete-song scale
```

Arousal complete-song result:

```text
model = pmemo_arousal_locked_opensmile_static_songmean_fine_20260715
three-fold macro = 0.174339 / 0.133155 / 0.870038 / 0.860214
minimum = all pass
good = PCC and CCC pass; RMSE and MAE do not
fold_01_rmse = 0.183445
```

These remain locked-development results. Final test remains untouched.

## 2026-07-15 Registry Update - MERT Attention Gate Rejected

- New gate: frozen MERT hidden-4/6/8 attention statistics pooling plus EffNet
  mood/theme context for Valence song-mean prediction.
- Direct-target fold-03 correction selected identity.
- Zero-initialized residual fold-03 frame result:
  `0.194013 / 0.149134 / 0.818697 / 0.798955`.
- Fold-03 base:
  `0.193754 / 0.149194 / 0.818762 / 0.799365`.
- Promotion: rejected; no three-fold expansion.
- Registered bests remain unchanged and the final 115-song test is untouched.

## 2026-07-15 Registry Update - Valence-Only Residual Gates Rejected

- Arousal remains frozen at
  `0.174062 / 0.135122 / 0.867330 / 0.859597`.
- Valence low-rank residual fold-03:
  `0.197734 / 0.152182 / 0.811467 / 0.792872`.
- Valence conservative multi-scale residual fold-03:
  `0.194659 / 0.149761 / 0.817052 / 0.797974`.
- Both are weaker than the fold-03 robust base
  `0.193754 / 0.149194 / 0.818762 / 0.799365` and are rejected.
- Valence robust best remains
  `0.181284 / 0.142308 / 0.824000 / 0.807112` on the three locked development
  folds. Its clip-level metrics are
  `0.178526 / 0.140722 / 0.821365 / 0.805417`.
- Final locked test remains untouched.

Rejected nonlinear follow-ups:
- Causal residual TCN (`channels=32`, dilations `1/2/4`, correction cap `0.04`):
  `0.183467 / 0.143645 / 0.819871 / 0.803865`. All held-fold selection runs
  chose epoch `1`, indicating immediate overfitting. Positive blend with the
  registered best selected zero TCN weight.
- Past-context nonlinear tree head, selected RandomForest with leaf size `20`:
  `0.182493 / 0.143283 / 0.821464 / 0.802942`. Cross-fold blend with the
  registered best worsened RMSE to `0.182644`.
- Dynamic 200-d MusiCNN dense Ridge recheck selected identity.
- Decision: keep the registered linear past-context model unchanged.

## 2026-07-14 Registry Update - Discogs-EffNet Valence Candidate

Current Valence development RMSE-primary candidate:

```text
model_id = pmemo_valence_locked_discogs_effnet_dynamic_fine_20260714
output = outputs/valence_locked_discogs_effnet_dynamic_fine_20260714
base = valence_locked_discogs_effnet_fine_20260714
backbone = Essentia Discogs-EffNet bs64, PartitionedCall:1, 1280 dimensions
model sha256 = 3ed9af50d5367c0b9c795b294b00e7599e4943244f4cbd376869f3bfc87721b1
song polarity = mean embedding, PCA-18, residual Ridge alpha 0.001,
  strength 1.5, cap 0.009
dynamic correction = direct Ridge alpha 60, blend 0.20, cap 0.0075
validation = three locked PMEmo development folds
final locked test = excluded
deployment = all inputs are audio-derived; arbitrary-MP3 chain not yet packaged
```

| Metric | Current | Minimum pass | Gap to pass | Good | Gap to good | Excellent | Gap to excellent | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| RMSE | 0.181872 | 0.180000 | +0.001872 | 0.150000 | +0.031872 | 0.100000 | +0.081872 | not pass |
| MAE | 0.142449 | 0.150000 | -0.007551 | 0.120000 | +0.022449 | 0.080000 | +0.062449 | pass |
| PCC | 0.822796 | 0.800000 | -0.022796 | 0.850000 | +0.027204 | 0.900000 | +0.077204 | pass |
| CCC | 0.805583 | 0.800000 | -0.005583 | 0.850000 | +0.044417 | 0.900000 | +0.094417 | pass |

Per-fold RMSE/MAE/PCC/CCC:
- fold_01: `0.177141 / 0.140916 / 0.810156 / 0.791662`
- fold_02: `0.173797 / 0.136890 / 0.841385 / 0.828258`
- fold_03: `0.194677 / 0.149540 / 0.816848 / 0.796829`

Paired song bootstrap versus the previous MusiCNN dynamic-tag best, 3,000
resamples:
- RMSE delta `-0.000445`, 95% interval `[-0.001194, +0.000330]`, improvement
  probability `0.8697`.
- MAE delta `-0.000532`, 95% interval `[-0.001259, +0.000209]`, improvement
  probability `0.9113`.
- PCC delta `+0.001066`, 95% interval `[-0.000610, +0.002719]`, improvement
  probability `0.8873`.
- CCC delta `-0.000078`, 95% interval `[-0.001945, +0.001710]`; effectively
  neutral and still above the minimum line.

Decision:
- Register as the new development RMSE-primary candidate because RMSE, MAE,
  and PCC improve while all non-RMSE minimum criteria remain passed.
- The improvement is provisional: every 95% bootstrap interval crosses zero,
  and the same development folds have been used repeatedly for selection.
- Stop local EffNet hyperparameter refinement. Do not unlock the final test.

## 2026-07-14 Registry Update - Balanced EffNet Mood/Theme Candidate

Superseding provisional Valence development candidate:

```text
model_id = pmemo_valence_locked_effnet_moodtheme_tags_balanced_final_20260714
output = outputs/valence_locked_effnet_moodtheme_tags_balanced_final_20260714
base = valence_locked_discogs_effnet_dynamic_fine_20260714
semantic head = mtg_jamendo_moodtheme-discogs-effnet-1, 56 tags
semantic context = current plus past 2s, 4s, 6s, and 8s
semantic correction = residual Ridge, alpha 1000, frame weighting,
  strength 0.18, cap 0.12
selection guard = RMSE primary; MAE, PCC, and CCC may not degrade versus base
validation = three locked PMEmo development folds
final locked test = excluded
```

| Metric | Current | Minimum pass | Gap to pass | Good | Gap to good | Excellent | Gap to excellent | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| RMSE | 0.181546 | 0.180000 | +0.001546 | 0.150000 | +0.031546 | 0.100000 | +0.081546 | not pass |
| MAE | 0.142319 | 0.150000 | -0.007681 | 0.120000 | +0.022319 | 0.080000 | +0.062319 | pass |
| PCC | 0.823404 | 0.800000 | -0.023404 | 0.850000 | +0.026596 | 0.900000 | +0.076596 | pass |
| CCC | 0.806874 | 0.800000 | -0.006874 | 0.850000 | +0.043126 | 0.900000 | +0.093126 | pass |

Per-fold RMSE/MAE/PCC/CCC:
- fold_01: `0.177576 / 0.141693 / 0.809154 / 0.790821`
- fold_02: `0.173097 / 0.136170 / 0.842781 / 0.830492`
- fold_03: `0.193966 / 0.149095 / 0.818276 / 0.799309`

Paired song bootstrap versus the prior MusiCNN dynamic-tag best, 3,000
resamples:
- RMSE delta `-0.000753`, improvement probability `0.8687`.
- MAE delta `-0.000658`, improvement probability `0.8680`.
- PCC delta `+0.001644`, improvement probability `0.8707`.
- CCC delta `+0.001189`, improvement probability `0.7783`.
- All 95% intervals cross zero; the promotion remains provisional.

Decision:
- Promote over the raw EffNet dynamic candidate because all four aggregate
  metrics improve under an explicit non-degradation guard.
- RMSE remains the only unmet minimum criterion, with a gap of `0.001546`.
- Freeze this mood/theme search range. Do not evaluate the final locked test.

## 2026-07-14 Registry Audit - Post-Promotion Robustness Gates

The following follow-ups do **not** supersede the balanced EffNet mood/theme
candidate:

| Gate | RMSE | MAE | PCC | CCC | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| Registered balanced EffNet mood/theme base | 0.181546 | 0.142319 | 0.823404 | 0.806874 | keep |
| EffNet tag song-mean, nested selection | 0.181749 | 0.142735 | 0.822878 | 0.806167 | reject |
| MusiCNN happy/sad correction recheck | 0.181553 | 0.142321 | 0.823399 | 0.806866 | reject |
| MAEST PCA-8 song-mean correction | 0.181531 | 0.142528 | 0.823504 | 0.805975 | reject |
| MAEST supervised top-k/Ridge | 0.181546 | 0.142319 | 0.823404 | 0.806874 | identity selected |
| Joint semantic BiGRU, zero-init | 0.181739 | 0.142294 | 0.823049 | 0.807020 | reject |
| Dynamic MAEST nested Ridge | 0.181693 | 0.142459 | 0.823097 | 0.806637 | reject |

Audit notes:
- A nonnested EffNet tag song-mean search reached RMSE `0.181297`, but nested
  other-fold selection degraded to RMSE `0.181749`; the apparent improvement
  is therefore not registered.
- MAEST used the official `discogs-maest-30s-pw-129e` model, layer 7, with full
  non-overlapping 30-second coverage of all 652 development songs. This branch
  is pure audio and compatible with arbitrary-length songs, but it did not add
  robust complementary Valence information through the tested song-mean heads.
- Final locked test remains `115` songs and untouched.
- The joint semantic BiGRU was trained with strict outer-song exclusion and a
  conservative zero-initialized residual design. It improved MAE and CCC only
  fractionally while worsening RMSE and PCC; all outer-fold RMSE values were
  weaker than the corresponding registered base.
- Dynamic MAEST correction used nested bidirectional selection on the other two
  folds. Two held-fold policies selected identity, and its aggregate result was
  weaker on all four metrics.

Registry decision:
- Keep
  `pmemo_valence_locked_effnet_moodtheme_tags_balanced_final_20260714` as the
  provisional development candidate.
- Do not promote gains that appear only in nonnested aggregate searches.
- Stop local post-hoc correction searches on these development folds. The next
  defensible experiment must change training-time representation/objective and
  use nested selection before any final-test evaluation.

## 2026-07-15 Registry Update - Local VA Development Bests

Validation boundary:

```text
development_pool = 652 PMEmo songs
split = three song-level development folds
final_locked_test = 115 songs, not evaluated or used for selection
```

Current Arousal point-estimate development best:

```text
model_id = pmemo_arousal_locked_effnet_moodtheme_fine_complete_20260715
output = outputs/arousal_locked_effnet_moodtheme_fine_complete_20260715
base = pmemo_arousal_locked_mert_smile_refine_20260713
new_component = cross-fold Discogs-EffNet mood/theme residual Ridge
mode = current-time tags
ridge_alpha = 30
residual_strength = 0.09
correction_cap = 0.04
rmse = 0.1795997171
mae = 0.1396304461
pcc = 0.8583583524
ccc = 0.8488259148
status = all minimum criteria pass on locked development folds
caveat = only CCC improvement versus the prior best had a bootstrap CI excluding zero
```

Current Valence point-estimate development best:

```text
model_id = pmemo_valence_locked_global_amp_final_complete_20260715
output = outputs/valence_locked_global_amp_final_complete_20260715
base = pmemo_valence_locked_effnet_moodtheme_tags_balanced_final_20260714
postprocess = retain predicted song mean and scale within-song frame deviations
amplitude = 0.82
smoothing = none
rmse = 0.1813722720
mae = 0.1423092205
pcc = 0.8238088897
ccc = 0.8071090246
status = MAE/PCC/CCC pass; RMSE misses minimum by 0.0013722720
caveat = the underlying base remains a multi-stage OOF development artifact
```

Promotion policy:
- These are the latest development point-estimate bests, not final-test
  results.
- The Arousal gain is small; the Valence calibration is deployable but does not
  solve the remaining RMSE gap.
- Complete artifacts restore five dropped terminal frames and cover all
  31,044 expected development frames.
- Do not continue local post-processing searches on these same development
  predictions.

## 2026-07-15 Registry Update - Song-Mean and Delay Models

Current Arousal point-estimate development best:

```text
model_id = pmemo_arousal_locked_opensmile_static_songmean_fine_20260715
output = outputs/arousal_locked_opensmile_static_songmean_fine_20260715
base = pmemo_arousal_locked_song_mean_fusion_20260715
major_song_mean_family = prediction stats + EffNet mood/theme + MAEST
major_strength = 0.50
major_cap = 0.20
refinement = PCA32 OpenSMILE full-song static-aux Ridge
refinement_strength = 0.02
refinement_cap = 0.15
rmse = 0.1740616882
mae = 0.1351224412
pcc = 0.8673304930
ccc = 0.8595965795
minimum_status = all pass
good_status = PCC and CCC pass; RMSE and MAE do not pass
bootstrap_note = major fusion improves all four significantly; refinement is small and only CCC CI excludes zero
```

Current Valence robust development best:

```text
model_id = pmemo_valence_locked_global_shift_amp_guarded_20260715
output = outputs/valence_locked_global_shift_amp_guarded_20260715
base = pmemo_valence_locked_effnet_moodtheme_tags_balanced_final_complete_20260715
global_shift = -0.5 seconds
within_song_amplitude = 0.75
rmse = 0.1812840926
mae = 0.1423075867
pcc = 0.8240002853
ccc = 0.8071124068
minimum_status = MAE/PCC/CCC pass; RMSE misses by 0.0012840926
bootstrap_note = RMSE and PCC improve; MAE and CCC unchanged
```

Valence RMSE-primary exploratory artifact:

```text
model_id = pmemo_valence_locked_global_shift_delay_20260715
output = outputs/valence_locked_global_shift_delay_20260715
global_shift = -2.0 seconds
within_song_amplitude = 0.78
rmse = 0.1811625401
mae = 0.1423102545
pcc = 0.8242576294
ccc = 0.8069844851
promotion = no; CCC is lower than the guarded candidate and bootstrap evidence is inconclusive
```

Final locked test:
- Still untouched.
- None of these development metrics may be reported as final-test performance.

## 2026-07-15 Registry Update - Rejected Follow-up Checks

- Static/dynamic auxiliary song-label mixtures selected identity for both
  Arousal and Valence under nested other-fold selection.
- Valence RMSE-primary extended-amplitude result:
  `0.181131 / 0.142397 / 0.824339 / 0.807015` at a common `-2.0 s` delay and
  `0.70` within-song amplitude.
- Promotion: rejected because MAE and CCC were lower than the robust candidate.
- Registered models are unchanged: Arousal point best
  `0.174062 / 0.135122 / 0.867330 / 0.859597`; Valence robust best
  `0.181284 / 0.142308 / 0.824000 / 0.807112`.
- Final locked test remains untouched.
