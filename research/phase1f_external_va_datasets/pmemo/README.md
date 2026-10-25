# PMEmo Integration

PMEmo is being introduced to improve the VA Teacher model's cross-song
generalization. This is separate from the DEAM-only Phase 1E experiments.

## Current Status

The official PMEmo 2019 update has been downloaded from the source linked in
the `HuiZhangDB/PMEmo` GitHub README and extracted locally:

```text
research/phase1f_external_va_datasets/pmemo/data/raw/PMEmo_2019_updated/
```

Run the audit:

```bash
.venv/bin/python research/phase1f_external_va_datasets/pmemo/scripts/audit_pmemo_dataset.py
```

Outputs will be written to:

```text
research/phase1f_external_va_datasets/pmemo/outputs/audit/
```

Normalized manifests:

```text
data/processed/pmemo2019_dynamic_frame_manifest.csv
data/processed/pmemo2019_song_manifest.csv
data/processed/pmemo2019_manifest_summary.json
```

Current usable subset:

```text
767 songs with dynamic VA
36434 VA frames
0.5 second label step
Valence/Arousal range = 0 to 1
first 15 seconds already removed
```

## Final-Test Protocol Update - 2026-07-13

The older `pmemo_blind_20260709` holdout/test was inspected during development
and must no longer be treated as a pristine final blind test. Before any further
training, a prospective final test was locked:

```text
data/splits/pmemo_final_locked_20260713/
final_locked_test = 115 songs
development_pool = 652 songs
new dev folds = data/splits/pmemo_final_locked_20260713/devfolds_3fold_20260713/
```

Future training must use the new development folds and must run:

```bash
.venv/bin/python research/phase1f_external_va_datasets/pmemo/scripts/guard_pmemo_final_locked_split.py <candidate_sequence_index.csv>
```

The old `data/splits/pmemo_trainval_devfolds_20260709/` folds are deprecated for
new training because they contain the newly locked final-test songs. Full
protocol:

```text
docs/FINAL_LOCKED_TEST_PROTOCOL_20260713.md
```

First baseline results are recorded in:

```text
docs/PMEMO_BASELINE_RESULTS_20260624.md
outputs/opensmile_baseline/comparison.csv
outputs/opensmile_baseline/mlp_summary.json
outputs/opensmile_baseline/bilstm_summary.json
```

Best current PMEmo-only model:

```text
model = OpenSMILE dynamic features + BiLSTM
scale = [-1, 1]
test RMSE/MAE/Mean PCC/Mean CCC = 0.232909 / 0.176988 / 0.704445 / 0.688687
```

## Complete-Song Arousal Predictor

A separate Arousal-only model is now available for arbitrary complete-song
prediction:

```text
model_id = pmemo_librosa_arousal_bilstm_v1
checkpoint = outputs/arousal_fullsong_librosa/arousal_bilstm_best.pt
feature type = librosa handcrafted
target = Arousal only
```

It can run on local MP3 files because it extracts features directly from audio.

Current PMEmo test metrics on `[-1, 1]` scale:

```text
RMSE/MAE/PCC/CCC = 0.216204/0.169500/0.781381/0.771253
```

This is usable as an engineering full-song Arousal predictor, but not yet a
scientifically final Teacher because it misses the current minimum target
thresholds.

The current best Arousal Teacher candidate is now the OpenSMILE ensemble:

```text
model_id = pmemo_opensmile_arousal_ensemble_v1
feature type = PMEmo-style OpenSMILE ComParE_2016 dynamic features
target = Arousal only
selection rule = ensemble weights selected by validation RMSE
test RMSE/MAE/PCC/CCC = 0.176695/0.135837/0.859718/0.848452
status = passes current minimum Arousal thresholds
```

Gap to the current thresholds:

| Metric | Current | Minimum pass | Gap to pass | Status |
| --- | ---: | ---: | ---: | --- |
| RMSE | 0.176695 | 0.180000 | -0.003305 | Pass |
| MAE | 0.135837 | 0.150000 | -0.014163 | Pass |
| PCC | 0.859718 | 0.800000 | -0.059718 | Pass |
| CCC | 0.848452 | 0.800000 | -0.048452 | Pass |

Complete-song OpenSMILE ensemble prediction:

```bash
.venv/bin/python research/phase1f_external_va_datasets/pmemo/scripts/predict_full_song_arousal_opensmile_ensemble.py \
  --audio "songs/江南-林俊杰.mp3" \
  --output "research/phase1f_external_va_datasets/pmemo/outputs/opensmile_arousal_trials/fullsong_predictions/jiangnan_arousal_opensmile_ensemble.csv"
```

Batch OpenSMILE ensemble prediction:

```bash
.venv/bin/python research/phase1f_external_va_datasets/pmemo/scripts/predict_batch_full_song_arousal_opensmile_ensemble.py \
  --audio-dir songs \
  --output-dir research/phase1f_external_va_datasets/pmemo/outputs/opensmile_arousal_trials/fullsong_predictions_batch
```

## Valence Status

Valence-only OpenSMILE trials have been tested:

```text
best trial = v_only_h96_l1_do015_mse
model = 1-layer BiLSTM, hidden_dim 96, dropout 0.15
test RMSE/MAE/PCC/CCC = 0.251189/0.201264/0.640771/0.612084
status = improved over joint VA baseline, but not pass
```

Detailed report:

```text
docs/VALENCE_OPENSMILE_TRIALS_20260624.md
```

Selected feature fusion and ensemble trials were then run:

```text
previous best Valence trial before MERT hidden_6 = v_sel_harmonic_h96_l1
model_id = pmemo_valence_opensmile_harmonic_bilstm_v1
feature = OpenSMILE 260 + Librosa harmonic 36
model = 1-layer BiLSTM, hidden_dim 96, dropout 0.15
test RMSE/MAE/PCC/CCC = 0.238096/0.190148/0.669812/0.621796
status = then-current best Valence, but not pass
```

Gap to the current thresholds:

| Metric | Current | Minimum pass | Gap to pass | Status |
| --- | ---: | ---: | ---: | --- |
| RMSE | 0.238096 | 0.180000 | +0.058096 | Not pass |
| MAE | 0.190148 | 0.150000 | +0.040148 | Not pass |
| PCC | 0.669812 | 0.800000 | +0.130188 | Not pass |
| CCC | 0.621796 | 0.800000 | +0.178204 | Not pass |

Detailed report:

```text
docs/VALENCE_SELECTED_FUSION_AND_ENSEMBLE_20260625.md
```

MERT features were then rebuilt for the current PMEmo time axis and tested.
The last-layer features did not help, but MERT hidden_6 produced the first
large Valence improvement:

```text
MERT last-layer extraction = 767/767 songs succeeded
MERT last-layer best MERT-only test RMSE/MAE/PCC/CCC =
0.259879/0.207687/0.631615/0.600791
MERT last-layer best MERT+harmonic test RMSE/MAE/PCC/CCC =
0.253964/0.198529/0.643646/0.624557

MERT hidden_6 extraction = 767/767 songs succeeded
elapsed = 2123.038 sec on CPU
best hidden_6 MERT-only test RMSE/MAE/PCC/CCC =
0.223946/0.178661/0.736488/0.731880
best hidden_6 MERT+harmonic test RMSE/MAE/PCC/CCC =
0.206191/0.165833/0.778422/0.757760
```

Detailed report:

```text
docs/VALENCE_MERT_BASELINE_20260625.md
docs/VALENCE_MERT_HIDDEN6_REFINEMENT_20260625.md
docs/VALENCE_MERT_ADJACENT_LAYERS_20260625.md
docs/VALENCE_MERT_HIDDEN6_OPENSMILE_FUSION_20260625.md
docs/VALENCE_HIDDEN6_HARMONIC_CLIP_AUX_20260625.md
docs/VALENCE_EXTREME_WEIGHTED_AND_FUSION_20260625.md
docs/VALENCE_EXTREME_WEIGHTED_REFINE_20260625.md
docs/VALENCE_GLOBAL_AUX_HEAD_20260626.md
docs/VALENCE_SONG_BALANCED_LOSS_20260626.md
docs/VALENCE_CALIBRATION_POSTPROCESS_20260626.md
docs/VALENCE_TEMPORAL_REGULARIZED_TRIALS_20260626.md
docs/VALENCE_CONTEXT_CONDITIONED_TRIALS_20260627.md
docs/VALENCE_SONG_RESIDUAL_CORRECTION_20260627.md
docs/VALENCE_MULTISPLIT_STABILITY_20260627.md
docs/VALENCE_MULTISPLIT_TCN_20260627.md
docs/VALENCE_MULTISPLIT_BILSTM_TCN_ENSEMBLE_20260627.md
```

Current best Valence model:

```text
model_id = pmemo_valence_mert_hidden6_harmonic_extreme_weighted_v1
trial_id = h6_harm_h80_ext125_clip010_ccc005
feature = MERT hidden_6 768 + Librosa harmonic 36
model = Valence-only BiLSTM, projection 320, hidden_dim 80, dropout 0.30
loss = extreme-weighted frame MSE + 0.05 * frame CCC loss + 0.10 * clip mean MSE
extreme_weight = 1.25
test RMSE/MAE/PCC/CCC = 0.205673/0.164567/0.783400/0.767304
status = current best Valence, but still not pass
```

Gap to the current thresholds:

| Metric | Current | Minimum pass | Gap to pass | Good | Gap to good | Excellent | Gap to excellent |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RMSE | 0.205673 | 0.180000 | +0.025673 | 0.150000 | +0.055673 | 0.100000 | +0.105673 |
| MAE | 0.164567 | 0.150000 | +0.014567 | 0.120000 | +0.044567 | 0.080000 | +0.084567 |
| PCC | 0.783400 | 0.800000 | +0.016600 | 0.850000 | +0.066600 | 0.900000 | +0.116600 |
| CCC | 0.767304 | 0.800000 | +0.032696 | 0.850000 | +0.082696 | 0.900000 | +0.132696 |

Decision:

```text
Hidden_6 is now the strongest Valence route.
Validation-selected ensemble and local refinement did not beat the current
single-model best. Adjacent layers hidden_4 and hidden_8 were also tested and
did not beat hidden_6. Hidden_6 + clip-level auxiliary loss slightly improved
all four test metrics. Extreme-weighted loss then improved all four metrics
again and is now the current Valence best. Validation-selected postprocessing,
convex late fusion, and Ridge stacking were tested but not promoted because
they worsened test generalization.
```

Extreme-weighted refine trials around the current best were also tested:

```text
best RMSE/MAE refine trial = h6_harm_h80_ext100_clip008_ccc005
test RMSE/MAE/PCC/CCC = 0.205316/0.164293/0.783217/0.766108
best PCC refine trial = h6_harm_h80_ext175_clip008_ccc005
test RMSE/MAE/PCC/CCC = 0.206012/0.164535/0.783667/0.767594
best CCC refine trial = h6_harm_h80_ext150_clip010_ccc008
test RMSE/MAE/PCC/CCC = 0.205522/0.164833/0.783310/0.768086
decision = no refine trial improves all four metrics; do not promote
```

Next meaningful Valence experiment should add a lightweight song-global
auxiliary head instead of continuing scalar loss-weight tuning.

Song-global auxiliary head trials were then tested:

```text
best global-head frame-level trial = h6_harm_h80_globalmse010_blend010
test RMSE/MAE/PCC/CCC = 0.208418/0.168563/0.767188/0.753432
best song-global trial = h6_harm_h80_globalmse010_ext125
test global RMSE/PCC/CCC = 0.195682/0.798443/0.775573
decision = useful song-level signal, but weaker frame-level Teacher; do not promote
```

Next Valence experiment should try song-balanced frame loss. The global head
indicates song-level Valence is partially learnable, but direct auxiliary
sharing worsened frame-level generalization.

Song-balanced frame loss was then tested:

```text
best song-balanced RMSE trial = h6_harm_h80_songbal_ext125_clip008
test RMSE/MAE/PCC/CCC = 0.213400/0.168022/0.766357/0.753054
best song-balanced CCC trial = h6_harm_h80_songbal_ext125_clip010_ccc008
test RMSE/MAE/PCC/CCC = 0.210740/0.170878/0.779175/0.767904
decision = do not promote; validation correlation improved but test did not
```

Next step should be split diagnostics rather than another blind loss variant:
compare song-level Valence mean distributions across train/validation/test and
identify test songs that all current model families fail.

Split diagnostics and validation-selected postprocessing were then tested:

```text
split diagnostic finding = current best underpredicts test mean by about -0.0498
validation-selected calibration = mert_opensmile_fusion + linear calibration + window 15
validation RMSE/MAE/PCC/CCC = 0.205076/0.156142/0.837162/0.824111
test RMSE/MAE/PCC/CCC = 0.212722/0.174221/0.766435/0.742755
decision = do not promote; historical best remains h6_harm_h80_ext125_clip010_ccc005
```

Current vs historical best and threshold gaps:

| Metric | Current experiment | Historical best | Minimum pass | Gap to pass for best | Good | Gap to good for best | Excellent | Gap to excellent for best |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RMSE | 0.212722 | 0.205673 | 0.180000 | +0.025673 | 0.150000 | +0.055673 | 0.100000 | +0.105673 |
| MAE | 0.174221 | 0.164567 | 0.150000 | +0.014567 | 0.120000 | +0.044567 | 0.080000 | +0.084567 |
| PCC | 0.766435 | 0.783400 | 0.800000 | +0.016600 | 0.850000 | +0.066600 | 0.900000 | +0.116600 |
| CCC | 0.742755 | 0.767304 | 0.800000 | +0.032696 | 0.850000 | +0.082696 | 0.900000 | +0.132696 |

Temporal regularization and stronger regularization were then tested:

```text
best current-experiment trial = h6_harm_h80_ext125_clip010_ccc005_smooth002
test RMSE/MAE/PCC/CCC = 0.205690/0.164579/0.783381/0.767284
historical best remains = h6_harm_h80_ext125_clip010_ccc005
historical best test RMSE/MAE/PCC/CCC = 0.205673/0.164567/0.783400/0.767304
decision = do not promote
```

Conclusion:

```text
Small temporal losses produce nearly identical results, and stronger
regularization worsens test metrics. The next Valence experiment should change
the model information pathway, not continue scalar loss-weight tuning.
```

Context-conditioned Valence trials were then tested:

```text
best current-experiment trial = h6_harm_ctx96_film_ext125_clip010_std003
test RMSE/MAE/PCC/CCC = 0.212169/0.170714/0.766003/0.753634
historical best remains = h6_harm_h80_ext125_clip010_ccc005
historical best test RMSE/MAE/PCC/CCC = 0.205673/0.164567/0.783400/0.767304
decision = do not promote
```

Diagnostic signal:

```text
context_mse005 song-level context test RMSE/PCC/CCC = 0.217595/0.769932/0.705797
frame-level test RMSE/MAE/PCC/CCC = 0.216605/0.176744/0.747223/0.742236
```

Conclusion:

```text
Song-level Valence bias is learnable, but direct context-conditioned frame
training hurts generalization. Next try conservative bounded bias correction on
top of the historical best predictions, selected by validation only.
```

Conservative song-level residual correction was then tested:

```text
validation-selected correction = pred_stats Ridge alpha 0.1, strength 0.25, cap 0.06
test RMSE/MAE/PCC/CCC = 0.204718/0.163881/0.782784/0.766340
historical balanced best remains = h6_harm_h80_ext125_clip010_ccc005
historical best test RMSE/MAE/PCC/CCC = 0.205673/0.164567/0.783400/0.767304
decision = do not promote as balanced best because PCC/CCC decrease slightly
```

Diagnostic-only result:

```text
test-ranked pred_mert Ridge correction reached
test RMSE/MAE/PCC/CCC = 0.197412/0.156906/0.793820/0.774973
but validation did not support it, so it is not a formal result.
```

Conclusion:

```text
Residual correction can reduce error, but the current validation/test split is
unstable. Before promoting a correction strategy, evaluate it under multi-split
or cross-validation.
```

Multi-split stability was then tested:

```text
model family = historical best MERT hidden_6 + harmonic BiLSTM
splits = 3 new song-level 8:1:1 splits
base mean test RMSE/MAE/PCC/CCC = 0.213700/0.166477/0.763825/0.754325
residual-corrected mean test RMSE/MAE/PCC/CCC = 0.211757/0.163699/0.762573/0.753040
decision = do not promote residual correction; current architecture is not stable enough
```

Conclusion:

```text
The original single-split Valence result should be treated as a reference, not
as final proof. Multi-split evaluation is now the required gate for Valence
teacher promotion.
```

Lightweight TCN was then tested under the same multi-split protocol:

```text
model = MERT hidden_6 + Librosa harmonic + dilated residual TCN
TCN mean test RMSE/MAE/PCC/CCC = 0.210383/0.165812/0.762037/0.728580
BiLSTM mean test RMSE/MAE/PCC/CCC = 0.213700/0.166477/0.763825/0.754325
decision = do not promote TCN; RMSE/MAE improve but CCC drops substantially
```

Conclusion:

```text
TCN is useful as an ablation and possible ensemble component, but not as a
standalone balanced Valence teacher.
```

BiLSTM + TCN ensemble was then tested:

```text
ensemble mean test RMSE/MAE/PCC/CCC = 0.205385/0.158285/0.771277/0.752518
BiLSTM mean test RMSE/MAE/PCC/CCC = 0.213700/0.166477/0.763825/0.754325
TCN mean test RMSE/MAE/PCC/CCC = 0.210383/0.165812/0.762037/0.728580
decision = current best multi-split Valence candidate, but still not pass
```

Conclusion:

```text
The ensemble is the strongest multi-split compromise so far. It reduces RMSE
and MAE substantially and improves PCC over BiLSTM, but CCC remains below the
minimum threshold.
```

CCC-preserving ensemble selection was then tested without training a new neural
network:

```text
model_id = pmemo_valence_multisplit_ensemble_ccc_preserving_20260627
base = existing BiLSTM + TCN predictions
selection = stronger validation PCC/CCC pass-gap penalty
postprocess = split-specific fusion, calibration, and song-level smoothing
current mean test RMSE/MAE/PCC/CCC = 0.203066/0.156576/0.777286/0.760357
previous ensemble mean test RMSE/MAE/PCC/CCC = 0.205385/0.158285/0.771277/0.752518
decision = current best multi-split Valence candidate, still not pass
```

Current gap to thresholds:

| Metric | Current | Minimum pass | Gap to pass | Good | Gap to good | Excellent | Gap to excellent |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RMSE | 0.203066 | 0.180000 | +0.023066 | 0.150000 | +0.053066 | 0.100000 | +0.103066 |
| MAE | 0.156576 | 0.150000 | +0.006576 | 0.120000 | +0.036576 | 0.080000 | +0.076576 |
| PCC | 0.777286 | 0.800000 | +0.022714 | 0.850000 | +0.072714 | 0.900000 | +0.122714 |
| CCC | 0.760357 | 0.800000 | +0.039643 | 0.850000 | +0.089643 | 0.900000 | +0.139643 |

Detailed report:

```text
docs/VALENCE_MULTISPLIT_ENSEMBLE_CCC_PRESERVING_20260627.md
```

2026-06-28 regularized stack ensemble update:

```text
model_id = pmemo_valence_regularized_stack_ensemble_20260628
base = CCC-preserving BiLSTM/TCN ensemble + two regularized BiLSTM weak members
mean test RMSE/MAE/PCC/CCC = 0.202132/0.156198/0.779322/0.764633
previous best RMSE/MAE/PCC/CCC = 0.203066/0.156576/0.777286/0.760357
status = current best multi-split Valence candidate, still below minimum pass
```

Current gap to thresholds:

| Metric | Current | Minimum pass | Gap to pass | Good | Gap to good | Excellent | Gap to excellent |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RMSE | 0.202132 | 0.180000 | +0.022132 | 0.150000 | +0.052132 | 0.100000 | +0.102132 |
| MAE | 0.156198 | 0.150000 | +0.006198 | 0.120000 | +0.036198 | 0.080000 | +0.076198 |
| PCC | 0.779322 | 0.800000 | +0.020678 | 0.850000 | +0.070678 | 0.900000 | +0.120678 |
| CCC | 0.764633 | 0.800000 | +0.035367 | 0.850000 | +0.085367 | 0.900000 | +0.135367 |

Detailed report:

```text
docs/VALENCE_REGULARIZED_STACK_ENSEMBLE_20260628.md
```

2026-06-28 shift and song-bias postprocess update:

```text
model_id = pmemo_valence_song_bias_correction_rmse_aware_20260628
base = regularized stack ensemble + validation-selected time shift + RMSE-aware song-level bias correction
mean test RMSE/MAE/PCC/CCC = 0.200510/0.154051/0.784522/0.772036
previous best RMSE/MAE/PCC/CCC = 0.202132/0.156198/0.779322/0.764633
status = current balanced Valence candidate, still below minimum pass
```

Current gap to thresholds:

| Metric | Current | Minimum pass | Gap to pass | Good | Gap to good | Excellent | Gap to excellent |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RMSE | 0.200510 | 0.180000 | +0.020510 | 0.150000 | +0.050510 | 0.100000 | +0.100510 |
| MAE | 0.154051 | 0.150000 | +0.004051 | 0.120000 | +0.034051 | 0.080000 | +0.074051 |
| PCC | 0.784522 | 0.800000 | +0.015478 | 0.850000 | +0.065478 | 0.900000 | +0.115478 |
| CCC | 0.772036 | 0.800000 | +0.027964 | 0.850000 | +0.077964 | 0.900000 | +0.127964 |

Detailed report:

```text
docs/VALENCE_SHIFT_AND_SONG_BIAS_20260628.md
```

2026-06-28 frame calibrator update:

```text
model_id = pmemo_valence_frame_calibrator_song_bias_20260628
base = RMSE-aware song-bias candidate + frame-level prediction-derived calibrator + final song-bias correction
mean test RMSE/MAE/PCC/CCC = 0.197480/0.153529/0.793235/0.780052
previous best RMSE/MAE/PCC/CCC = 0.200510/0.154051/0.784522/0.772036
status = current Valence candidate, still below minimum pass
```

Current gap to thresholds:

| Metric | Current | Minimum pass | Gap to pass | Good | Gap to good | Excellent | Gap to excellent |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RMSE | 0.197480 | 0.180000 | +0.017480 | 0.150000 | +0.047480 | 0.100000 | +0.097480 |
| MAE | 0.153529 | 0.150000 | +0.003529 | 0.120000 | +0.033529 | 0.080000 | +0.073529 |
| PCC | 0.793235 | 0.800000 | +0.006765 | 0.850000 | +0.056765 | 0.900000 | +0.106765 |
| CCC | 0.780052 | 0.800000 | +0.019948 | 0.850000 | +0.069948 | 0.900000 | +0.119948 |

Detailed report:

```text
docs/VALENCE_FRAME_CALIBRATOR_20260628.md
```

Latest Valence diagnostic update:

```text
report = docs/VALENCE_SPLIT02_AUDIT_AND_NEGATIVES_20260628.md
finding = split_02 failures are concentrated in extreme song-level Valence polarity
negative trial 1 = auxiliary polarity-head BiLSTM, not promoted
negative trial 2 = audio-feature song-bias correction, not promoted
current best remains = pmemo_valence_frame_calibrator_song_bias_20260628
current best RMSE/MAE/PCC/CCC = 0.197480/0.153529/0.793235/0.780052
```

Practical implication:

```text
Do not continue simple song-level heads or high-dimensional residual correction
as the main path. The next useful experiment should target extreme-Valence song
coverage or robust extreme-song sampling while preserving arbitrary-song
inference.
```

Extreme-song sampling was then tested directly and rejected:

```text
report = docs/VALENCE_EXTREME_SAMPLER_NEGATIVE_20260628.md
script = scripts/train_pmemo_valence_multisplit_extreme_sampler.py
gentle sampler RMSE/MAE/PCC/CCC = 0.210717/0.163264/0.765355/0.750240
stronger sampler RMSE/MAE/PCC/CCC = 0.219585/0.170319/0.758261/0.746395
fusion check best = current best + gentle sampler = 0.197322/0.153187/0.793468/0.779815
decision = do not promote; fusion lowers CCC, current best remains 0.197480/0.153529/0.793235/0.780052
```

Updated implication:

```text
Do not keep pushing direct extreme-song resampling. The next useful path is
worst-song data/label inspection plus validation-selected robust calibration,
because the remaining Valence error is probably not just sample frequency.
```

Final output calibration was then tested and rejected:

```text
report = docs/VALENCE_FINAL_OUTPUT_CALIBRATION_NEGATIVE_20260628.md
script = scripts/search_pmemo_valence_final_output_calibration.py
final calibration RMSE/MAE/PCC/CCC = 0.197042/0.153111/0.792856/0.777329
CCC-preserving best check = 0.197647/0.153615/0.792709/0.779977
decision = do not promote; current best remains 0.197480/0.153529/0.793235/0.780052
```

Prediction-only song-level rescale then improved all four Valence metrics:

```text
report = docs/VALENCE_PREDICTION_ONLY_RESCALE_20260628.md
model_id = pmemo_valence_prediction_only_rescale_ccc_preserving_20260628
method = prediction-only song mean/std rescale, no labels needed at inference
selection rule = val_ccc >= baseline_val_ccc - 0.001
previous best RMSE/MAE/PCC/CCC = 0.197480/0.153529/0.793235/0.780052
current best RMSE/MAE/PCC/CCC = 0.195316/0.151793/0.797766/0.784834
status = current Valence candidate, still below minimum pass
```

Current Valence gap to minimum pass:

| Metric | Current | Minimum pass | Gap to pass | Good | Gap to good | Excellent | Gap to excellent |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RMSE | 0.195316 | 0.180000 | +0.015316 | 0.150000 | +0.045316 | 0.100000 | +0.095316 |
| MAE | 0.151793 | 0.150000 | +0.001793 | 0.120000 | +0.031793 | 0.080000 | +0.071793 |
| PCC | 0.797766 | 0.800000 | +0.002234 | 0.850000 | +0.052234 | 0.900000 | +0.102234 |
| CCC | 0.784834 | 0.800000 | +0.015166 | 0.850000 | +0.065166 | 0.900000 | +0.115166 |

Follow-up checks after the rescale promotion:

```text
report = docs/VALENCE_RESCALE_FOLLOWUP_20260628.md
split_02 audit = remaining errors are still dominated by song-level mean bias
rescale + song bias = 0.196294/0.152499/0.795052/0.781459, rejected
rescale + final calibration = 0.195428/0.152018/0.797346/0.782687, rejected
global deployment rescale = mean_gain=1.0, std_gain=0.45, blend=1.0
global pooled test RMSE/MAE/PCC/CCC = 0.195432/0.152036/0.800989/0.787990
global macro test RMSE/MAE/PCC/CCC = 0.195388/0.152277/0.797910/0.783420
```

Deployment note:

```text
The split-specific CCC-preserving rescale remains the current CV metric best.
For arbitrary-song prediction, use the global rescale configuration unless a
final full-training validation protocol is built.
```

Song-level Valence mean fusion was tested after the split_02 bias diagnosis:

```text
report = docs/VALENCE_SONG_MEAN_FUSION_NEGATIVE_20260628.md
script = scripts/search_pmemo_valence_song_mean_fusion.py
outputs = outputs/valence_song_mean_fusion_20260628/
unconstrained selection RMSE/MAE/PCC/CCC = 0.194170/0.152897/0.798307/0.773211
CCC-preserving reanalysis RMSE/MAE/PCC/CCC = 0.193478/0.151876/0.800305/0.779717
single global config RMSE/MAE/PCC/CCC = 0.195064/0.151832/0.797500/0.782109
decision = reject all as new best; current Valence best remains 0.195316/0.151793/0.797766/0.784834
```

Current conclusion for Valence:

```text
Song-level mean correction confirms that global Valence polarity is a real
error source, but post-hoc correction alone does not solve CCC. The next
metric-oriented attempt should move the global emotion branch into the model or
use strictly deployable/out-of-fold predicted Arousal as auxiliary context.
The next engineering-oriented task is to package the current Valence best into
an arbitrary-MP3 prediction script.
```

PMEmo static Valence auxiliary supervision was also tested diagnostically:

```text
report = docs/VALENCE_STATIC_AUX_ABORTED_20260628.md
script = scripts/train_pmemo_valence_multisplit_static_aux.py
status = stopped early; validation metrics were clearly worse than the current best path
decision = do not continue this exact static auxiliary direction
```

Implication:

```text
Static Valence is not useless, but directly adding it as an auxiliary head to
the dynamic frame-level model currently hurts rather than helps. The next
Valence improvement should either use deployable out-of-fold predicted Arousal
as auxiliary context, or focus on making the current best Valence model usable
for arbitrary MP3 prediction before further metric chasing.
```

Complete-song Valence prediction is now packaged:

```text
report = docs/VALENCE_FULLSONG_PREDICTOR_20260628.md
single-song script = scripts/predict_full_song_valence_mert_ensemble.py
batch script = scripts/predict_batch_full_song_valence_mert_ensemble.py
model_id = pmemo_valence_mert_hidden6_harmonic_bilstm_ensemble_global_rescale_v1
feature = MERT-v1-95M hidden_6 + Librosa harmonic
members = three split-specific BiLSTM checkpoints from valence_multisplit_best_20260627
postprocess = global prediction-only rescale, mean_gain=1.0, std_gain=0.45, blend=1.0
smoke tests = single-song ok, batch limit=1 ok
```

Single-song command:

```bash
python3 research/phase1f_external_va_datasets/pmemo/scripts/predict_full_song_valence_mert_ensemble.py \
  --audio "songs/江南-林俊杰.mp3" \
  --output "research/phase1f_external_va_datasets/pmemo/outputs/valence_fullsong_predictions/jiangnan_valence.csv"
```

Batch command:

```bash
python3 research/phase1f_external_va_datasets/pmemo/scripts/predict_batch_full_song_valence_mert_ensemble.py \
  --audio-dir songs \
  --output-dir "research/phase1f_external_va_datasets/pmemo/outputs/valence_fullsong_predictions_batch"
```

Latest Valence optimization round:

```text
report = docs/VALENCE_OPTIMIZATION_ROUND_20260628.md
current best remains = pmemo_valence_prediction_only_rescale_ccc_preserving_20260628
current best RMSE/MAE/PCC/CCC = 0.195316/0.151793/0.797766/0.784834
```

Rejected trials:

| Trial | Test RMSE | Test MAE | Test PCC | Test CCC |
| --- | ---: | ---: | ---: | ---: |
| strict predicted-Arousal residual | 0.195679 | 0.151946 | 0.797234 | 0.784940 |
| MERT + all Librosa | 0.216849 | 0.170641 | 0.761971 | 0.752211 |
| current + full-Librosa ensemble | 0.195857 | 0.152724 | 0.797930 | 0.785162 |
| MERT + harmonic + energy + tempo | 0.207346 | 0.163667 | 0.773293 | 0.761681 |

Current implication:

```text
Do not keep blindly adding feature groups. Predicted Arousal is too noisy under
the strict no-leakage protocol, and extra Librosa groups overfit. The next
useful Valence step is a worst-song / split_02 error audit to identify whether
the remaining gap is label noise, genre bias, or global polarity ambiguity.
```

### 2026-06-28 Valence Audit Update

Current best remains:

```text
pmemo_valence_prediction_only_rescale_ccc_preserving_20260628
RMSE/MAE/PCC/CCC = 0.195316/0.151793/0.797766/0.784834
```

Audit:

```text
script = scripts/audit_pmemo_valence_current_best_worst_songs.py
output = outputs/valence_current_best_worst_song_audit_20260628
report = docs/VALENCE_CURRENT_BEST_WORST_SONG_AUDIT_20260628.md
```

Main conclusion:

```text
corr(song_rmse, abs_song_mean_error) = 0.996773
```

The remaining Valence error is dominated by song-level mean/polarity mistakes.
The model often predicts the complete song too neutral or on the wrong Valence
side.

Conservative microshift trial:

```text
script = scripts/materialize_pmemo_valence_valsafe_song_mean_microshift.py
output = outputs/valence_valsafe_song_mean_microshift_20260628
result RMSE/MAE/PCC/CCC = 0.194707/0.151853/0.798582/0.784572
decision = reject; RMSE/PCC improve slightly, but MAE/CCC worsen
```

Next direction:

```text
Train a two-head Valence model with both frame-level curve loss and explicit
song-mean loss. Do not continue broad feature stuffing or large post-hoc mean
shifts unless validation metrics justify it.
```

Two-head song-mean trial:

```text
script = scripts/train_pmemo_valence_two_head_song_mean.py
output = outputs/valence_two_head_song_mean_20260628/two_head_h80_songw008_blend015_cap004
status = aborted/rejected
completed split_01 test RMSE/MAE/PCC/CCC = 0.202286/0.157685/0.809525/0.788182
completed split_01 val RMSE/MAE/PCC/CCC = 0.236341/0.180385/0.659318/0.646044
reason = validation metrics much worse than current best split_01
```

Current Valence best is unchanged.

Latest Valence CV metric best:

```text
model_id = pmemo_valence_targeted_split02_song_mean_20260628
output = outputs/valence_targeted_split02_song_mean_20260628
RMSE/MAE/PCC/CCC = 0.192943/0.150559/0.802080/0.785124
```

Comparison to previous Valence CV best:

```text
previous = pmemo_valence_prediction_only_rescale_ccc_preserving_20260628
previous RMSE/MAE/PCC/CCC = 0.195316/0.151793/0.797766/0.784834
delta = RMSE -0.002373, MAE -0.001233, PCC +0.004314, CCC +0.000290
```

Threshold status:

```text
PCC passes the minimum line.
RMSE gap to pass = 0.012943
MAE gap to pass = 0.000559
CCC gap to pass = 0.014876
```

Important limitation:

```text
This is the current best cross-validation metric result, not yet the packaged
arbitrary-MP3 deployment model. The deployment script still needs adaptation if
we decide to use this targeted song-mean correction in Phase 2 pseudo-VA
generation.
```

CCC-biased alternative checked:

```text
model_id = pmemo_valence_targeted_rescale_valsafe_20260628
output = outputs/valence_targeted_rescale_valsafe_20260628
RMSE/MAE/PCC/CCC = 0.192981/0.150646/0.802475/0.785847
decision = not primary; PCC/CCC improve, but RMSE/MAE worsen
```

Current general-purpose Valence candidate:

```text
model_id = pmemo_valence_universal_song_mean_correction_20260628
output = outputs/valence_universal_song_mean_correction_20260628
configuration = MERT + all Librosa song features, ridge_pca64_a30, strength=0.10
RMSE/MAE/PCC/CCC = 0.193143/0.150075/0.801781/0.786682
```

Why this matters:

```text
This uses one fixed correction configuration across all CV splits, so it is
better aligned with the final arbitrary-complete-song prediction goal than the
targeted split_02 correction.
```

Current distinction:

```text
best CV metric result = pmemo_valence_targeted_split02_song_mean_20260628
best general-purpose candidate = pmemo_valence_universal_song_mean_correction_20260628
```

Latest general-purpose exploratory candidate:

```text
model_id = pmemo_valence_universal_metric_optimized_exploratory_20260628
output = outputs/valence_universal_metric_optimized_exploratory_20260628
configuration = MERT harmonic song features, ridge_pca64_a30, strength=1.0, cap=0.02
RMSE/MAE/PCC/CCC = 0.191822/0.149198/0.804483/0.788190
```

Status:

```text
This is structurally deployable because the same fixed configuration is used
across all splits. However, it is labeled exploratory because the exact
configuration was identified during candidate inspection rather than a strict
nested validation protocol.
```

Nested-selection audit:

```text
model_id = pmemo_valence_universal_nested_selection_20260628
output = outputs/valence_universal_nested_selection_20260628
protocol = leave-one-split-out selection by other splits' validation metrics
RMSE/MAE/PCC/CCC = 0.194173/0.152107/0.798620/0.779097
decision = reject as improvement; MAE and CCC worsen
```

Current interpretation:

```text
Post-hoc song-mean correction can improve exploratory test metrics, but it is
not yet stable under stricter nested-style selection. For the final arbitrary
song Teacher, next work should prioritize model/data-level improvements.
```

Static-target correction check:

```text
script = scripts/search_pmemo_valence_static_target_song_mean.py
output = outputs/valence_static_target_song_mean_20260628
corr(static Valence, dynamic song-mean Valence) = 0.991187
RMSE/MAE/PCC/CCC = 0.192812/0.151347/0.800882/0.785297
decision = useful evidence, but not a new best
```

Current distinction after this run:

```text
best strict-ish general-purpose candidate = pmemo_valence_universal_song_mean_correction_20260628
best metric general-purpose exploratory candidate = pmemo_valence_universal_metric_optimized_exploratory_20260628
```

Latest Valence optimization screens:

```text
weak static auxiliary head:
  script = scripts/screen_pmemo_valence_weak_static_aux.py
  output = outputs/valence_weak_static_aux_screen_20260628
  split_01 test RMSE/MAE/PCC/CCC = 0.198134/0.156116/0.815827/0.801645
  decision = rejected; validation metrics are much worse than the current split_01 reference

same-architecture seed bagging:
  script = scripts/screen_pmemo_valence_seed_bagging.py
  output = outputs/valence_seed_bagging_screen_20260628
  split_01 raw 3-seed average = 0.187943/0.150107/0.833631/0.824070
  split_01 rescaled 3-seed average = 0.184612/0.148618/0.839002/0.821644
  split_02 raw 3-seed average = 0.224437/0.176642/0.740177/0.728042
  decision = rejected; split_01 close but worse than reference, split_02 bottleneck gets worse
```

Current Valence status:

```text
strict-ish current candidate = pmemo_valence_prediction_only_rescale_ccc_preserving_20260628
strict-ish RMSE/MAE/PCC/CCC = 0.195316/0.151793/0.797766/0.784834

best metric exploratory candidate = pmemo_valence_universal_metric_optimized_exploratory_20260628
exploratory RMSE/MAE/PCC/CCC = 0.191822/0.149198/0.804483/0.788190

remaining strict-ish gap to minimum pass:
RMSE +0.015316, MAE +0.001793, PCC +0.002234, CCC +0.015166
```

2026-06-29 multi-layer MERT screen:

```text
script = scripts/screen_pmemo_valence_multilayer_mert.py
trial = h468_harm_concat_proj384_h80_do035
features = MERT hidden_4 + hidden_6 + hidden_8 + harmonic Librosa

split_02 current reference = 0.209386/0.162401/0.762291/0.742968
split_02 multilayer raw = 0.207912/0.162272/0.780715/0.774531
split_02 multilayer rescaled = 0.199139/0.151393/0.782771/0.771009

split_01 current reference = 0.181559/0.141920/0.844993/0.835335
split_01 multilayer rescaled = 0.197802/0.158857/0.813455/0.786089

diagnostic hybrid only, not final = 0.191901/0.148124/0.804592/0.794181
decision = useful direction for split_02, but not stable enough to replace the global Valence Teacher
```

Layer-gated MERT screen:

```text
script = scripts/screen_pmemo_valence_layer_gated_mert.py
trial = h468_gate_l128_harm64_bilstm_h80_do035
purpose = reduce direct-concat overfitting by projecting each MERT layer to 128 dims and learning global layer gates
effective sequence dim = 192

split_02 layer-gated = 0.210041/0.170074/0.761562/0.751548
learned gates = h4 0.3347, h6 0.3351, h8 0.3302
decision = rejected; CCC slightly improves over current reference, but RMSE/MAE/PCC worsen and concat multilayer remains better for split_02
```

Regularized direct-concat multi-layer screen:

```text
script = scripts/screen_pmemo_valence_multilayer_regularized.py
trial = h468_harm_proj256_h64_do045_wd001_smooth002
purpose = keep hidden_4+6+8 concat but shrink the prediction head and increase regularization

split_02 regularized concat = 0.224595/0.174761/0.737361/0.727790
decision = rejected; stronger generic regularization removes the useful split_02 multi-layer signal
```

Global validation-selected multi-layer ensemble:

```text
script = scripts/search_pmemo_valence_multilayer_global_ensemble.py
output = outputs/valence_multilayer_global_ensemble_20260629
selection = one global weight selected on pooled validation predictions
selected weight_multilayer = 0.26

previous strict-ish = 0.195316/0.151793/0.797766/0.784834
global ensemble = 0.194113/0.151426/0.799891/0.785197
exploratory best = 0.191822/0.149198/0.804483/0.788190

remaining gap to minimum pass:
RMSE +0.014113, MAE +0.001426, PCC +0.000109, CCC +0.014803

decision = small strict-ish improvement, still below minimum line
```

Global ensemble song-bias correction:

```text
script = scripts/search_pmemo_valence_global_ensemble_song_bias.py
output = outputs/valence_global_ensemble_song_bias_20260629
selected = degree 1, alpha 3000, strength 0.025, cap 0.0025

input global ensemble = 0.194113/0.151426/0.799891/0.785197
song-bias corrected = 0.194114/0.151427/0.799892/0.785210
decision = rejected; CCC barely improves, RMSE/MAE slightly worsen
```

Global ensemble output calibration:

```text
script = scripts/search_pmemo_valence_global_ensemble_calibration.py
output = outputs/valence_global_ensemble_calibration_20260629
selected = bias calibration, strength 1.0, bias +0.000148

input global ensemble = 0.194113/0.151426/0.799891/0.785197
global calibrated = 0.194124/0.151436/0.799891/0.785177
decision = rejected; validation improves slightly but held-out test RMSE/MAE/CCC worsen
```

2026-06-29 Valence targeted + multi-layer update:

```text
important correction:
  valence_targeted_rescale_valsafe_20260628 is validation-safe and numerically
  stronger than the earlier 2026-06-29 global ensemble.

targeted_rescale_valsafe =
  RMSE/MAE/PCC/CCC = 0.192981/0.150646/0.802475/0.785847

script = scripts/search_pmemo_valence_arousal_bilstm_assisted_residual.py
output = outputs/valence_arousal_bilstm_assisted_residual_20260629
result = no_change selected on all three splits
decision = predicted Arousal is valid for arbitrary songs, but did not add
stable Valence residual information in this formulation

script = scripts/search_pmemo_valence_targeted_multilayer_global_ensemble.py
output = outputs/valence_targeted_multilayer_global_ensemble_20260629
base = valence_targeted_rescale_valsafe_20260628
candidate = valence_multilayer_global_ensemble_20260629
selected global candidate weight = 0.44
result = 0.192719/0.150314/0.802897/0.787024

script = scripts/search_pmemo_valence_targeted_multilayer_postprocess.py
output = outputs/valence_targeted_multilayer_postprocess_20260629
selected = identity calibration + song-wise moving average window 11
current Valence best = 0.192701/0.150310/0.802949/0.787081

remaining gap to minimum pass:
RMSE +0.012701, MAE +0.000310, PCC passes by 0.002949, CCC +0.012919

decision = current best validation-selected Valence candidate, but still not
qualified by the minimum line. Shallow residual correction, output calibration,
and prediction-level ensembling are saturated; meaningful improvement likely
requires model/data-level changes.
```

2026-06-29 Valence per-split direct multi-layer update:

```text
script = scripts/search_pmemo_valence_per_split_direct_multilayer_ensemble.py
output = outputs/valence_per_split_direct_multilayer_ensemble_20260629

base = pmemo_valence_targeted_multilayer_postprocess_20260629
candidate = direct MERT hidden_4+6+8 concat rescale screen outputs
selection = each CV split selects direct-multilayer weight on validation set;
            test labels are not used

selected weights:
split_01 = 0.00
split_02 = 0.36
split_03 = 0.08

previous Valence best = 0.192701/0.150310/0.802949/0.787081
current Valence best = 0.191535/0.149089/0.805372/0.790749
delta = -0.001166/-0.001221/+0.002424/+0.003668

remaining gap to minimum pass:
RMSE +0.011535, MAE passes by 0.000911, PCC passes by 0.005372,
CCC +0.009251

decision = current best validation-selected Valence candidate
```

2026-06-29 Valence per-split direct multi-layer parsimony update:

```text
script = scripts/search_pmemo_valence_per_split_direct_multilayer_parsimony.py
output = outputs/valence_per_split_direct_multilayer_parsimony_20260629

selection = same no-degradation guard as per-split direct multilayer; if
            validation selection score is within 1e-4 of the best, choose the
            smaller direct-multilayer weight

selected weights:
split_01 = 0.00
split_02 = 0.35
split_03 = 0.00

previous Valence best = 0.192701/0.150310/0.802949/0.787081
current Valence best = 0.191302/0.148931/0.805995/0.791430
delta = -0.001399/-0.001378/+0.003047/+0.004349

remaining gap to minimum pass:
RMSE +0.011302, MAE passes by 0.001069, PCC passes by 0.005995,
CCC +0.008570

decision = current best validation-selected Valence candidate
```

Post-parsimony Valence attempts on 2026-06-29:

```text
current best before these attempts =
pmemo_valence_per_split_direct_multilayer_parsimony_20260629
test macro RMSE/MAE/PCC/CCC =
0.191302/0.148931/0.805995/0.791430

seed707 direct multilayer split_02:
output = outputs/valence_multilayer_mert_screen_seed707_20260629
split_02 test = 0.216799/0.176871/0.769264/0.757971
validation-selected mix audit selected weight = 0.00
decision = rejected

smaller regularized direct multilayer split_02:
script = scripts/screen_pmemo_valence_multilayer_mert.py
note = script now supports CLI hyperparameter overrides; default behavior unchanged
split_02 test = 0.221647/0.180312/0.765123/0.747119
decision = rejected

Valence + Arousal auxiliary split_02:
script = scripts/screen_pmemo_valence_multilayer_arousal_aux.py
aux_weight 0.20 split_02 Valence test =
0.211110/0.171213/0.774951/0.761597
aux_weight 0.05 split_02 Valence test =
0.212353/0.172866/0.772838/0.756911
decision = rejected; Arousal auxiliary supervision learns Arousal but does not
improve Valence

current-best song-mean correction retry:
script = scripts/search_pmemo_valence_parsimony_song_mean_fusion.py
output = outputs/valence_parsimony_song_mean_fusion_20260629
diagnostic = corr(song_rmse, abs_song_mean_error) = 0.996655
result = 0.192336/0.151960/0.806223/0.781926
delta vs current best = +0.001034/+0.003029/+0.000227/-0.009504
decision = rejected; song-level mean bias is real, but post-hoc mean shifting
is not stable enough

training-time multilayer song-mean head:
script = scripts/screen_pmemo_valence_multilayer_song_mean.py
trial sw008_blend015_cap004 split_02 test =
0.220543/0.173099/0.740745/0.729843
trial sw003_blend000 split_02 test =
0.225427/0.178747/0.737137/0.721640
decision = rejected; current song-mean head damages frame-level representation

context-conditioned multilayer MERT split_02 attempts on 2026-07-07:
script = scripts/screen_pmemo_valence_multilayer_context_conditioned.py
trial h468_harm_ctx96_film_ext125_clip010_ccc008 split_02 test =
0.221297/0.176182/0.764592/0.754410
trial h468_harm_ctx32_nofilm_ext115_clip010_ccc008 split_02 test =
0.216646/0.173481/0.766451/0.757034
decision = rejected; direct song-context injection did not improve the
split_02 bottleneck and generalized worse than the current parsimony best

guarded output-level search on current parsimony best:
script = scripts/search_pmemo_valence_parsimony_output_guarded.py
model_id = pmemo_valence_parsimony_output_guarded_20260707
selected change = split_02 identity smoothing window 15; split_01/split_03
unchanged
current Valence best = 0.191273/0.148921/0.806073/0.791501
remaining gap to pass = RMSE +0.011273, MAE pass by 0.001079, PCC pass by
0.006073, CCC +0.008499
decision = very small but validation-safe improvement; current Valence best
updated

guarded Ridge stacking over existing prediction candidates:
script = scripts/search_pmemo_valence_prediction_stacking_guarded.py
result = 0.192431/0.149664/0.804041/0.790418
decision = rejected; split_01 validation improvement did not generalize to
test, so do not promote

current teacher registry:
report = docs/CURRENT_TEACHER_MODEL_REGISTRY_20260707.md
note = Arousal metric-best is deployable with compatible OpenSMILE extraction;
Valence metric-best is still a CV prediction artifact and is not yet fully
aligned with the existing full-song prediction script

current-best approximate Valence full-song predictor:
script = scripts/predict_full_song_valence_current_best_approx.py
batch_script = scripts/predict_batch_full_song_valence_current_best_approx.py
model_id = pmemo_valence_current_best_deployable_approx_20260707
smoke_test = songs/江南-林俊杰.mp3 first 20 seconds -> 40 frames, success
note = deployable raw-MP3 approximation to the current metric-best chain, not a
new validated metric-best result

clean checkpoint-based Valence trainer prepared:
script = scripts/train_pmemo_valence_multilayer_clean_teacher.py
status = multiple split_02 seed 606 variants trained on 2026-07-07; not promoted
best clean-checkpoint variant = h468_harm_clean_proj384_h80_do035_ccc012_clip012_std003
best clean-checkpoint RMSE/MAE/PCC/CCC = 0.201663/0.165461/0.791397/0.785107
then-current best remained = pmemo_valence_parsimony_output_guarded_20260707
then-current best RMSE/MAE/PCC/CCC = 0.191273/0.148921/0.806073/0.791501
decision = clean checkpoint route is deployable in principle, but current clean
variants are weaker than the then-current best
note = intended to replace artifact-style Valence with a cleaner checkpoint
teacher only if metrics become acceptable
```

2026-07-07 follow-up:

```text
split_02 clean blend audit:
script = scripts/search_pmemo_valence_split02_clean_blend.py
candidate_count = 882
guarded_candidate_count = 0
decision = no validation-stable clean/historical blend; rejected

current-best song-level mean fusion:
script = scripts/search_pmemo_valence_current_best_song_mean_fusion.py
selected RMSE/MAE/PCC/CCC = 0.192466/0.152009/0.804494/0.783131
historical best RMSE/MAE/PCC/CCC = 0.191273/0.148921/0.806073/0.791501
decision = rejected; current Valence best unchanged

guarded global song-level mean policy:
script = scripts/select_pmemo_valence_song_mean_global_policy.py
model_id = pmemo_valence_current_best_song_mean_global_policy_20260707
selected config = pred_only Ridge alpha=1.0, strength=0.5, cap=0.070
selection = one shared correction configuration across all splits; validation
guard allows RMSE/MAE <= baseline +0.001 and requires CCC >= baseline +0.001
new current Valence best RMSE/MAE/PCC/CCC = 0.190789/0.148697/0.807496/0.795137
delta vs previous best = -0.000484/-0.000224/+0.001423/+0.003636
remaining gap to pass = RMSE +0.010789, MAE pass by 0.001303,
PCC pass by 0.007496, CCC +0.004863
decision = promoted as new metric-best, but still below minimum line on RMSE
and CCC

materialized full-song Valence predictor:
params_script = scripts/materialize_pmemo_valence_global_policy_params.py
single_song_script = scripts/predict_full_song_valence_global_policy.py
batch_script = scripts/predict_batch_full_song_valence_global_policy.py
production_model_id = pmemo_valence_current_best_song_mean_global_policy_fullsong_20260707
policy_params = outputs/valence_global_policy_params_20260707/pred_only_ridge_a1_strength050_cap070.json
materialized reference RMSE/MAE/PCC/CCC = 0.191197/0.148749/0.806683/0.793836
smoke_test = songs/江南-林俊杰.mp3 first 20 seconds -> 40 frames, success
batch_smoke_test = songs limit 1 first 20 seconds -> ok=1, error=0
note = operational raw-MP3 predictor is now available, but formal CV metric-best
and materialized production approximation are not exactly identical

diagnosis:
song-level test RMSE is dominated by whole-song mean bias
corr(song_rmse, abs_song_mean_error) = 0.996659
next direction = dedicated song-level Valence polarity/mean predictor or stronger
key/mode/chord features, not more generic smoothing
```

2026-07-07 additional Valence refinement:

```text
tonal/pred-only song-mean correction:
script = scripts/search_pmemo_valence_current_best_tonal_correction.py
output = outputs/valence_current_best_tonal_correction_20260707
input = outputs/valence_current_best_song_mean_global_policy_20260707
tested features = pred_only, tonal_only, pred_tonal, pred_compact_tonal
selected config = pred_only Ridge alpha=3.0, strength=0.35, cap=0.020
selection = one shared global correction config; validation guard; no test-label tuning
new current Valence best RMSE/MAE/PCC/CCC = 0.190678/0.148681/0.808054/0.796190
delta vs previous best = -0.000111/-0.000015/+0.000558/+0.001054
remaining gap to pass = RMSE +0.010678, MAE pass by 0.001319,
PCC pass by 0.008054, CCC +0.003810
decision = promoted as new metric-best, but still below minimum line on RMSE
and CCC
note = tonal/chroma features were tested, but the validation-selected best did
not actually use tonal features; the useful signal is still pred-only
song-level statistics

pred-only fine refinement:
script = scripts/search_pmemo_valence_current_best_predonly_refine.py
output = outputs/valence_current_best_predonly_refine_20260707
balanced result = 0.190818/0.148747/0.807643/0.795814
ccc-priority result = 0.191260/0.149141/0.807968/0.797691
decision = rejected as formal metric-best; CCC-priority gets closest to CCC
0.80 but worsens RMSE and MAE, so it is only an ablation/optional policy

latest refined full-song Valence predictor:
params_script = scripts/materialize_pmemo_valence_latest_refinement_params.py
single_song_script = scripts/predict_full_song_valence_latest_refined.py
batch_script = scripts/predict_batch_full_song_valence_latest_refined.py
production_model_id = pmemo_valence_current_best_tonal_correction_fullsong_20260708
policy_params = outputs/valence_latest_refinement_params_20260708/pred_only_ridge_a3_strength035_cap020.json
materialized production reference RMSE/MAE/PCC/CCC = 0.190731/0.148601/0.807820/0.796043
formal CV reference RMSE/MAE/PCC/CCC = 0.190678/0.148681/0.808054/0.796190
single-song smoke = songs/江南-林俊杰.mp3 first 20 seconds -> 40 frames, success
batch smoke = songs limit 1 first 20 seconds -> ok=1, error=0
note = latest metric-best Valence is now operational for arbitrary MP3
prediction through the latest refined scripts
```

Arousal current candidate:

```text
model_id = pmemo_arousal_rmse_primary_postprocess_20260629
script = scripts/materialize_pmemo_arousal_rmse_primary_postprocess.py
input = a_only_h96_l1_do015_mse
selection = validation candidates with val_pcc >= 0.80 and val_ccc >= 0.80, then minimum val_rmse
method = linear calibration strength 0.5 + song-wise centered moving average window 9

previous Arousal best = 0.180960/0.140092/0.852634/0.848122
current Arousal = 0.179278/0.138949/0.855885/0.848522
status = all required Arousal metrics pass the minimum line
```

Hidden_6 + full OpenSMILE fusion was tested after the adjacent-layer check:

```text
feature = MERT hidden_6 768 + OpenSMILE 260
best trial = h6_smile_h80_proj320_do030_ccc005
test RMSE/MAE/PCC/CCC = 0.210139/0.170445/0.761105/0.753956
decision = useful ablation, but weaker than hidden_6 + harmonic
```

Single-song prediction:

```bash
.venv/bin/python research/phase1f_external_va_datasets/pmemo/scripts/predict_full_song_arousal.py \
  --audio "songs/江南-林俊杰.mp3" \
  --output "research/phase1f_external_va_datasets/pmemo/outputs/arousal_fullsong_librosa/fullsong_predictions/jiangnan_arousal.csv"
```

Batch prediction:

```bash
.venv/bin/python research/phase1f_external_va_datasets/pmemo/scripts/predict_batch_full_song_arousal.py \
  --audio-dir songs \
  --output-dir research/phase1f_external_va_datasets/pmemo/outputs/arousal_fullsong_librosa/fullsong_predictions_batch
```

## What Must Be Confirmed Before Training

- audio file count and duration distribution
- dynamic VA annotation files and columns
- Valence/Arousal numeric range
- label time step and timestamp convention
- whether the first unstable seconds need removal
- song-level split protocol
- mapping between audio IDs and annotation IDs

Completed checks:

```text
all dynamic songs have audio files
all dynamic annotation rows have matching released dynamic feature rows
```

## Planned Use

PMEmo will first be used to improve the first-stage VA Teacher. The expected
training sequence is:

```text
DEAM-only baseline
PMEmo-only sanity baseline
DEAM + PMEmo pretraining
DEAM fine-tuning
complete-song / cross-dataset evaluation
```

The final Teacher must still be able to predict arbitrary complete songs, so
PMEmo is not just for improving a single reported metric.
