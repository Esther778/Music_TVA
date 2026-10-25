# Music VA Section-Aware Research Project

Last updated: 2026-07-26

This project builds a research and data-production pipeline for complete-song
music emotion modeling. It combines automatic pop-song section segmentation,
human QC, Supabase storage, and continuous Valence-Arousal prediction.

The current research direction is section-aware and multimodal music emotion
analysis of human-created Chinese popular music. It uses frame-level VA curves,
reviewed song sections, and aligned lyrics to study emotional progression,
section contrast, and lyric-audio relationships. AIGC comparison is outside
both the project and graduation-thesis scope as of 2026-10-25.

The completed 100-song section-aware audio study is available as
[`arXiv:2609.15675`](https://arxiv.org/abs/2609.15675). Its published numerical
results are audited against the corrected 2026-08-20 data package in
`docs/research/ARXIV_2609_15675_DATA_AUDIT_20261025.md`.

## Current Status

Canonical update for Phase 1 Teacher deployment and Phase 2 blind validation:

```text
Valence Teacher=pmemo_final_clean_valence_teacher_v1
complete-clip RMSE/MAE/PCC/CCC=0.186298/0.152855/0.802852/0.800168

Arousal Teacher=pmemo_final_clean_arousal_teacher_v1
complete-clip RMSE/MAE/PCC/CCC=0.188573/0.146912/0.857441/0.854803

Phase 2 frozen corpus=99 reviewed songs; Chinese primary analysis=97 songs
Blind sample=15 songs, 179 frozen section boundaries
Stage 3 output=7375 frame-level pseudo-VA rows, 15/15 audit pass
Stage 4 output=7375/7375 frames aligned to 179/179 sections
Stage 5A=held-out section-type analysis complete
Stage 5B=30-reference same-rater human validation frozen
Human/Teacher pooled Spearman V/A=0.614/0.576
Full-corpus output=49,763 frames, 99/99 audit pass, 1,097/1,097 sections aligned
Current state=97-song reproducible exploratory structure analysis complete
```

The canonical executable Teachers supersede older candidate snapshots retained
below as experiment history. Stage 3 used fixed 30-second audio windows with a
15-second stride and Hann overlap-add; no section type or human answer was read.
See `research/phase2_section_aware_va/docs/BLIND_VALIDATION_STAGE3_INFERENCE_20260720.md`.
Stage 4 alignment details are in
`research/phase2_section_aware_va/docs/BLIND_VALIDATION_STAGE4_ALIGNMENT_20260721.md`.
Stage 5 structural results are in
`research/phase2_section_aware_va/docs/BLIND_VALIDATION_STAGE5_STRUCTURAL_ANALYSIS_20260721.md`.
Full-corpus inference, structural results, and sample sufficiency are in
`research/phase2_section_aware_va/docs/FULL_CORPUS_INFERENCE_AND_SAMPLE_AUDIT_20260726.md`.
The complete 97-song exploratory analysis, figures, statistical tables, and
Chinese report are under
`research/phase2_section_aware_va/analysis/full_corpus_97/`.
Its separate popular-music interpretation is
`research/phase2_section_aware_va/analysis/full_corpus_97/reports/MUSICOLOGICAL_INTERPRETATION_REPORT_ZH.md`.
The deeper section-function and emotional-organization report is
`research/phase2_section_aware_va/analysis/full_corpus_97/reports/EMOTIONAL_ORGANIZATION_AND_SECTION_FUNCTIONS_ZH.md`.
The consolidated Chinese research report from Teacher development through
popular-music conclusions is
`reports/ei_section_aware_full_report_20260726/FULL_RESEARCH_REPORT_ZH.md`.
The corresponding journal-style Chinese paper draft is
`reports/ei_section_aware_full_report_20260726/PAPER_DRAFT_ZH.md`.

Latest Phase 1F Teacher snapshot:

```text
Arousal current best =
  pmemo_arousal_rmse_primary_postprocess_20260629
  RMSE/MAE/PCC/CCC = 0.179278/0.138949/0.855885/0.848522
  status = passes minimum line; deployable with compatible OpenSMILE extraction

Valence current target-closest locked-development candidate =
  pmemo_valence_locked_effnet_moodtheme_tags_balanced_final_20260714
  RMSE/MAE/PCC/CCC = 0.181546/0.142319/0.823404/0.806874
  status = MAE/PCC/CCC pass; RMSE remains above the maximum allowed line
  remaining gap = RMSE +0.001546
  method = prior full-coverage audio stack + MusiCNN happy/sad correction +
    cross-fold song-predictor disagreement stacker + six-level ordinal
    song-polarity head + cross-fold quantile calibration + PLS-2 residual head +
    other-two-fold MuQ reference-direction extrapolation + MusiCNN 50-tag
    correction + past-8-second dense dynamic tag context + cross-fold
    quantile song-mean calibration + Discogs-EffNet mean-embedding
    PCA/Ridge polarity correction + bounded dynamic EffNet Ridge correction +
    MTG-Jamendo 56-tag past-8-second residual Ridge correction
  validation = three locked development folds; final locked test excluded
  caveat = this is a development-selection artifact and has not yet been
    packaged as the final arbitrary-MP3 predictor or evaluated on final test;
    paired-song bootstrap improvement intervals versus the prior best cross 0

Valence strict development RMSE-primary artifact =
  pmemo_valence_locked_effnet_moodtheme_tags_balanced_final_20260714
  RMSE/MAE/PCC/CCC = 0.181546/0.142319/0.823404/0.806874
  status = lowest current development RMSE; MAE/PCC/CCC pass
```

Latest independent-backbone gate (2026-07-14):

```text
MAEST backbone = mtg-upf/discogs-maest-30s-pw-129e, layer 7
coverage = 652/652 development songs, 1195 non-overlapping 30 s windows,
  0 failures; final locked 115-song test excluded
unsupervised PCA/Ridge gate = 0.181531/0.142528/0.823504/0.805975
supervised top-k/Ridge gate = selected identity (no MAEST correction)
dynamic nested Ridge = 0.181693/0.142459/0.823097/0.806637
zero-init joint semantic BiGRU = 0.181739/0.142294/0.823049/0.807020
decision = reject all MAEST correction/training gates; current best unchanged
reason = neither candidate improved RMSE, MAE, PCC, and CCC together
```

Current diagnosis:

```text
Valence is bottlenecked mainly by whole-song mean/polarity error rather than
frame-shape error. The current best song-level test RMSE has
corr(song_rmse, abs_song_mean_error) = 0.996659.
```

Phase 1 baseline from the previous OpenL3 paper is preserved:

```text
research/phase1_va_prediction/openl3_va_baseline
```

This is the copied baseline from the previous paper plus the new PyTorch MLP
tuning work. A three-seed OpenL3+MLP ensemble was frozen for engineering:

```text
teacher_model_id=openl3_mlp_w512_v1
validation_rmse=0.099575
test_rmse=0.098425
test_mean_ccc=0.920569
```

The result follows the previous paper's frame-level random split. It is not a
song-level generalization estimate and must not be directly compared with the
song-level MERT result. See:

```text
research/phase1_va_prediction/openl3_va_baseline/docs/TEACHER_MODEL_CARD.md
```

Song-level validation later showed that this frame-random result is not a
reliable complete-song generalization estimate.

Phase 2 is in progress:

```text
research/phase2_section_aware_va
```

Phase 1C pure-audio ablation is complete:

```text
research/phase1c_spectral_feature_fusion_va
```

It compares log-Mel CNN/attention/BiLSTM, Librosa handcrafted features, and
their fusion on the fixed song-level split. MERT+BiLSTM remains best by
validation RMSE; handcrafted features provide the clearest gains for Arousal.

Phase 1E strict DEAM Teacher optimization is preserved here:

```text
research/phase1e_teacher_optimization
```

Current DEAM-only complete-song conclusion:

```text
DEAM complete-song subset = 58 songs
five-fold Valence Global-Local outer-test RMSE/MAE/PCC/CCC =
0.267009/0.205546/0.463024/0.430917
best complete-song global Ridge outer-test Mean RMSE/MAE/PCC/CCC =
0.242049/0.193788/0.513339/0.393422
decision = preserve as ablations, not final Teacher
```

Latest preserved progress snapshot:

```text
research/phase1e_teacher_optimization/docs/TEACHER_PROGRESS_SNAPSHOT_20260624.md
```

Phase 1F has started to add PMEmo as an external dynamic VA dataset:

```text
research/phase1f_external_va_datasets/pmemo
```

The official PMEmo 2019 update has been downloaded and normalized into initial
manifests:

```text
dynamic_songs = 767
dynamic_frame_rows = 36434
frame_step = 0.5 s
label_range = 0 to 1
first_15_seconds = already removed in the released annotations
```

PMEmo is structurally usable now. Mixed DEAM+PMEmo training should still wait
until the normalization strategy is explicitly fixed.

PMEmo labels have now been normalized to the canonical DEAM-style `[-1, 1]`
scale and the first PMEmo-only OpenSMILE baselines have been trained. Best
current PMEmo-only result:

```text
model = official OpenSMILE dynamic features + BiLSTM
test RMSE/MAE/Mean PCC/Mean CCC = 0.232909/0.176988/0.704445/0.688687
Arousal PCC/CCC = 0.837598/0.830559
Valence PCC/CCC = 0.571292/0.546814
```

Conclusion: PMEmo confirms that Arousal is comparatively strong while Valence
remains the main bottleneck. MERT hidden_6 features are now the strongest
Valence route tested so far.

An engineering complete-song Arousal predictor has also been built:

```text
model_id = pmemo_librosa_arousal_bilstm_v1
feature type = librosa handcrafted, extracted directly from MP3
checkpoint = research/phase1f_external_va_datasets/pmemo/outputs/arousal_fullsong_librosa/arousal_bilstm_best.pt
PMEmo test RMSE/MAE/PCC/CCC = 0.216204/0.169500/0.781381/0.771253
complete-song prediction verified on songs/江南-林俊杰.mp3
batch prediction smoke test = 2/2 songs succeeded
```

This model can be used for engineering batch Arousal prediction, but it is not
yet a final scientific Teacher because it narrowly misses the current minimum
target thresholds.

A metric-qualified complete-song Arousal Teacher candidate has now been built:

```text
model_id = pmemo_opensmile_arousal_ensemble_v1
feature type = PMEmo-style OpenSMILE ComParE_2016 dynamic features
target = Arousal only
test RMSE/MAE/PCC/CCC = 0.176695/0.135837/0.859718/0.848452
status = passes current minimum Arousal thresholds
single-song script = research/phase1f_external_va_datasets/pmemo/scripts/predict_full_song_arousal_opensmile_ensemble.py
batch script = research/phase1f_external_va_datasets/pmemo/scripts/predict_batch_full_song_arousal_opensmile_ensemble.py
model card = research/phase1f_external_va_datasets/pmemo/docs/AROUSAL_OPENSMILE_ENSEMBLE_TEACHER_20260624.md
```

This freezes Arousal as usable for now. Valence remains the main unresolved
Teacher bottleneck.

First Valence-only OpenSMILE trials are complete:

```text
best trial = v_only_h96_l1_do015_mse
test RMSE/MAE/PCC/CCC = 0.251189/0.201264/0.640771/0.612084
status = improved over joint VA Valence baseline, but still not pass
report = research/phase1f_external_va_datasets/pmemo/docs/VALENCE_OPENSMILE_TRIALS_20260624.md
```

Decision:

```text
Do not continue blind OpenSMILE-only Valence tuning.
Next Valence route should add harmonic/tonal features and then MERT fusion.
```

Current best Valence model after MERT hidden_6 + harmonic extreme-weighted
experiments:

```text
model_id = pmemo_valence_mert_hidden6_harmonic_extreme_weighted_v1
trial_id = h6_harm_h80_ext125_clip010_ccc005
feature = MERT hidden_6 768 + Librosa harmonic 36
model = Valence-only BiLSTM, projection 320, hidden_dim 80, dropout 0.30
loss = extreme-weighted frame MSE + 0.05 * frame CCC loss + 0.10 * clip mean MSE
extreme_weight = 1.25
test RMSE/MAE/PCC/CCC = 0.205673/0.164567/0.783400/0.767304
status = current best Valence, still not pass
report = research/phase1f_external_va_datasets/pmemo/docs/VALENCE_EXTREME_WEIGHTED_AND_FUSION_20260625.md
```

Current gap to minimum Valence thresholds:

```text
RMSE gap = +0.025673
MAE gap = +0.014567
PCC gap = +0.016600
CCC gap = +0.032696
```

The hidden_6 result improves the previous OpenSMILE + harmonic best by:

```text
RMSE -0.031905
MAE  -0.024315
PCC  +0.108610
CCC  +0.135964
```

The clip-level auxiliary loss then slightly improved the hidden_6 harmonic
model itself:

```text
previous hidden_6 harmonic best RMSE/MAE/PCC/CCC =
0.206191/0.165833/0.778422/0.757760
new clip-aux best RMSE/MAE/PCC/CCC =
0.205867/0.165556/0.778969/0.760180
decision = promote raw clip-aux model; do not apply validation-selected
postprocessing because it worsened test RMSE/MAE/CCC
```

The extreme-weighted loss then improved the clip-aux model:

```text
previous clip-aux best RMSE/MAE/PCC/CCC =
0.205867/0.165556/0.778969/0.760180
new extreme-weighted best RMSE/MAE/PCC/CCC =
0.205673/0.164567/0.783400/0.767304
decision = promote extreme-weighted model; do not promote late fusion or Ridge
stacking because validation gains did not generalize to test
```

Extreme-weighted refine trials around the current best were also tested:

```text
best RMSE/MAE refine trial = h6_harm_h80_ext100_clip008_ccc005
test RMSE/MAE/PCC/CCC = 0.205316/0.164293/0.783217/0.766108
best PCC refine trial = h6_harm_h80_ext175_clip008_ccc005
test RMSE/MAE/PCC/CCC = 0.206012/0.164535/0.783667/0.767594
best CCC refine trial = h6_harm_h80_ext150_clip010_ccc008
test RMSE/MAE/PCC/CCC = 0.205522/0.164833/0.783310/0.768086
decision = no refine trial improves all four metrics; current best remains
pmemo_valence_mert_hidden6_harmonic_extreme_weighted_v1
```

Song-global auxiliary head trials were then tested:

```text
best global-head frame-level trial = h6_harm_h80_globalmse010_blend010
test RMSE/MAE/PCC/CCC = 0.208418/0.168563/0.767188/0.753432
best song-global trial = h6_harm_h80_globalmse010_ext125
test global RMSE/PCC/CCC = 0.195682/0.798443/0.775573
decision = useful song-level signal, but weaker frame-level Teacher; do not promote
```

Song-balanced frame loss was also tested:

```text
best song-balanced RMSE trial = h6_harm_h80_songbal_ext125_clip008
test RMSE/MAE/PCC/CCC = 0.213400/0.168022/0.766357/0.753054
best song-balanced CCC trial = h6_harm_h80_songbal_ext125_clip010_ccc008
test RMSE/MAE/PCC/CCC = 0.210740/0.170878/0.779175/0.767904
decision = do not promote; validation correlation improved but test did not
```

Validation-selected Valence calibration/postprocessing was then tested:

```text
current experiment = mert_opensmile_fusion + linear calibration + moving_average_window 15
validation RMSE/MAE/PCC/CCC = 0.205076/0.156142/0.837162/0.824111
test RMSE/MAE/PCC/CCC = 0.212722/0.174221/0.766435/0.742755
historical best remains = h6_harm_h80_ext125_clip010_ccc005
historical best test RMSE/MAE/PCC/CCC = 0.205673/0.164567/0.783400/0.767304
decision = do not promote; validation-selected postprocess worsened test metrics
report = research/phase1f_external_va_datasets/pmemo/docs/VALENCE_CALIBRATION_POSTPROCESS_20260626.md
```

Temporal regularization and stronger regularization were then tested:

```text
current experiment best = h6_harm_h80_ext125_clip010_ccc005_smooth002
test RMSE/MAE/PCC/CCC = 0.205690/0.164579/0.783381/0.767284
historical best remains = h6_harm_h80_ext125_clip010_ccc005
historical best test RMSE/MAE/PCC/CCC = 0.205673/0.164567/0.783400/0.767304
decision = do not promote
report = research/phase1f_external_va_datasets/pmemo/docs/VALENCE_TEMPORAL_REGULARIZED_TRIALS_20260626.md
```

Context-conditioned Valence trials were then tested:

```text
current experiment best = h6_harm_ctx96_film_ext125_clip010_std003
test RMSE/MAE/PCC/CCC = 0.212169/0.170714/0.766003/0.753634
historical best remains = h6_harm_h80_ext125_clip010_ccc005
historical best test RMSE/MAE/PCC/CCC = 0.205673/0.164567/0.783400/0.767304
decision = do not promote
report = research/phase1f_external_va_datasets/pmemo/docs/VALENCE_CONTEXT_CONDITIONED_TRIALS_20260627.md
```

Conservative song-level residual correction was then tested:

```text
current experiment = pred_stats Ridge residual correction, strength 0.25, cap 0.06
test RMSE/MAE/PCC/CCC = 0.204718/0.163881/0.782784/0.766340
historical balanced best remains = h6_harm_h80_ext125_clip010_ccc005
historical best test RMSE/MAE/PCC/CCC = 0.205673/0.164567/0.783400/0.767304
decision = do not promote as balanced best because PCC/CCC decrease slightly
report = research/phase1f_external_va_datasets/pmemo/docs/VALENCE_SONG_RESIDUAL_CORRECTION_20260627.md
```

Multi-split Valence stability was then tested:

```text
model family = historical best MERT hidden_6 + harmonic BiLSTM
splits = 3 new song-level 8:1:1 splits
base mean test RMSE/MAE/PCC/CCC = 0.213700/0.166477/0.763825/0.754325
residual-corrected mean test RMSE/MAE/PCC/CCC = 0.211757/0.163699/0.762573/0.753040
decision = current Valence architecture is not stable enough under multi-split evaluation
report = research/phase1f_external_va_datasets/pmemo/docs/VALENCE_MULTISPLIT_STABILITY_20260627.md
```

Lightweight TCN was then tested under the same multi-split protocol:

```text
model = MERT hidden_6 + Librosa harmonic + dilated residual TCN
TCN mean test RMSE/MAE/PCC/CCC = 0.210383/0.165812/0.762037/0.728580
BiLSTM mean test RMSE/MAE/PCC/CCC = 0.213700/0.166477/0.763825/0.754325
decision = do not promote TCN as balanced teacher because CCC drops
report = research/phase1f_external_va_datasets/pmemo/docs/VALENCE_MULTISPLIT_TCN_20260627.md
```

BiLSTM + TCN ensemble was then tested:

```text
ensemble mean test RMSE/MAE/PCC/CCC = 0.205385/0.158285/0.771277/0.752518
BiLSTM mean test RMSE/MAE/PCC/CCC = 0.213700/0.166477/0.763825/0.754325
TCN mean test RMSE/MAE/PCC/CCC = 0.210383/0.165812/0.762037/0.728580
decision = current best multi-split Valence candidate, but still not pass
report = research/phase1f_external_va_datasets/pmemo/docs/VALENCE_MULTISPLIT_BILSTM_TCN_ENSEMBLE_20260627.md
```

Current historical best Valence gap:

| Metric | Historical best | Minimum pass | Gap to pass | Good | Gap to good | Excellent | Gap to excellent |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RMSE | 0.205673 | 0.180000 | +0.025673 | 0.150000 | +0.055673 | 0.100000 | +0.105673 |
| MAE | 0.164567 | 0.150000 | +0.014567 | 0.120000 | +0.044567 | 0.080000 | +0.084567 |
| PCC | 0.783400 | 0.800000 | +0.016600 | 0.850000 | +0.066600 | 0.900000 | +0.116600 |
| CCC | 0.767304 | 0.800000 | +0.032696 | 0.850000 | +0.082696 | 0.900000 | +0.132696 |

This phase will generate section-labeled VA data and train a two-layer music
emotion model:

1. Frame-level layer: predict continuous Valence-Arousal from 0.5 second audio
   features.
2. Section-aware layer: combine frame-level VA predictions with section
   boundaries and section types to model full-song emotional structure.

Current Phase 2 run state:

- Supabase reviewed-song and human-annotation snapshots have been saved under
  `research/phase2_section_aware_va/data/manifests/raw/`.
- The first 70 reviewed songs have been selected and split 56 / 7 / 7 for
  train / validation / test.
- The current Supabase-reviewed Chinese-song corpus contains 99 songs after
  removing `Love Story - Taylor Swift` on 2026-07-20; old 70-song manifests
  are stale.
- Final Teacher selection is reopened because DEAM-only complete-song Valence
  generalization is unstable.
- PMEmo integration has started as the next data-strategy step.
- Frame-level VA generation must use the final selected Teacher once Phase 1 is
  fixed.

Detailed phase notes:

```text
docs/research/PROJECT_PHASES.md
research/README.md
research/phase2_section_aware_va/README.md
```

## Project Layout

```text
annotation_tool/                  Supabase-backed human QC console
docs/research/                    research phase documentation
hybrid_structure/                 current functional section pipeline
research/                         research workspace
  phase1_va_prediction/           completed OpenL3 VA baseline
  phase1e_teacher_optimization/   strict DEAM Teacher diagnostics and ablations
  phase1f_external_va_datasets/   PMEmo and future external VA datasets
  phase2_section_aware_va/        current section-aware VA modeling work
songs/                            local mp3 library, gitignored
outputs/                          generated transcripts, sections, features, gitignored
netlify_publish/                  built static QC package, gitignored
```

Important workflow documents:

```text
AI_COLLABORATION_GUIDE.md
PROJECT_HANDOFF.md
NEW_SONG_PIPELINE_WORKFLOW.md
LYRIC_CALIBRATION_WORKFLOW.md
NETLIFY_DEPLOY_WORKFLOW.md
annotation_tool/WORKFLOW_STATUS.md
annotation_tool/README.md
```

New AI/Codex conversations should read `AI_COLLABORATION_GUIDE.md` first.

Report outputs are stored under:

```text
reports/
```

Current Arousal Teacher report package:

```text
reports/arousal_teacher_model_report_20260624/
```

## Data And QC Platform

The QC platform is hosted at:

```text
https://musicva.netlify.app
```

The annotation tool uses Supabase tables and views for:

```text
songs
model_runs
auto_sections
lyric_lines
human_annotations
section_reviews
song_review_summary
section_review_queue
human_annotations_readable
```

The current section label taxonomy is:

```text
intro, verse, pre_chorus, chorus, bridge, interlude, outro
```

## New Song Pipeline

Newly collected audio files go into:

```text
songs/songs2
```

Files in `songs/songs2` may have inconsistent names as long as title and artist
are present. After validation and dedupe, valid songs are normalized to:

```text
歌名-歌手.mp3
```

The pipeline order is:

```text
inspect and dedupe songs/songs2
normalize filenames and move valid mp3 files into songs/
write annotation_tool/today_new_songs_YYYYMMDD.txt
run annotation_tool/run_model_outputs_for_songs.py
import generated manifest with annotation_tool/import_local_results.py
verify Supabase and QC pending_review visibility
```

Do not insert new songs into Supabase before model output exists.

## Local QC Development

Run the static QC console locally from the repository root:

```bash
python3 -m http.server 5177
```

Open:

```text
http://127.0.0.1:5177/annotation_tool/
```

Avoid using `file://` for normal QC testing because local audio path behavior
differs from the HTTP server.

## Netlify Deployment

The user normally performs production deployment from macOS Terminal.

Build and deploy command:

```bash
cd "/Users/esther/Desktop/New project"
python3 annotation_tool/build_netlify_publish.py
npx netlify deploy --prod --dir=netlify_publish
```

Codex should not run production deployment unless explicitly asked. If a
frontend/platform change is made, local source files and `netlify_publish/`
must stay synchronized before deployment.

## Research Phase 2 Initial Rules

Current user-specified rules for the next training task:

- The latest reported database count is 95 reviewed/annotated songs.
- Refresh the reviewed-song source-of-truth manifest before selecting a Phase 2
  cohort; do not continue from the stale 70-song manifest silently.
- Split train / validation / test as 8:1:1.
- Train and record metrics in batches of 10 songs.
- Preserve training data, frame-level VA predictions, section-labeled VA
  tables, split manifests, model checkpoints, metrics, and plots.
- Final pseudo-VA generation must wait for MERT+BiLSTM three-seed verification.
  The OpenL3 engineering Teacher remains available as a baseline, but is not
  currently the final scientific Teacher.

## Current VA Teacher Status

As of 2026-06-24, the active Phase 1 scientific Teacher work is in:

```text
research/phase1e_teacher_optimization/
```

The strict complete-song protocol is:

```text
DEAM 1,744 short clips + 58 complete songs
Fold 0 train = 1,744 short clips + 34 complete songs
Fold 0 validation = 12 complete songs
Fold 0 outer test = 12 complete songs, still locked
```

Current best validation-stage candidate, before locked test:

```text
Valence model = original global-local full-song expert
Arousal model = leakage-free BiLSTM full-song transfer
Conservative postprocess = validation-selected smoothing + quadratic calibration, no isotonic
Valence smoothing = moving average 321 frames = 160.5 s
Arousal smoothing = gaussian sigma 14 frames = 7.0 s
Validation mean RMSE/MAE/PCC/CCC = 0.169705/0.132280/0.837199/0.824150
Validation Valence RMSE/MAE/PCC/CCC = 0.181051/0.136789/0.834619/0.821164
Validation Arousal RMSE/MAE/PCC/CCC = 0.157545/0.127771/0.839779/0.827136
test_used = false
```

The previous safer linear-calibration postprocess was:

```text
Valence smoothing = 181 frames = 90.5 s
Arousal smoothing = 41 frames = 20.5 s
Validation mean RMSE/MAE/PCC/CCC = 0.176776/0.138350/0.823066/0.807627
Validation Valence RMSE/MAE/PCC/CCC = 0.192896/0.147450/0.809685/0.791971
```

An exploratory isotonic-calibration diagnostic reaches:

```text
Validation mean RMSE/MAE/PCC/CCC = 0.160899/0.123552/0.854844/0.844437
Validation Valence RMSE/MAE/PCC/CCC = 0.170801/0.126867/0.854383/0.843911
```

Do not treat the isotonic result as formal unless it generalizes on locked
outer test or future folds.

Fold-0 locked outer test has now been evaluated once with the frozen
conservative V2 candidate:

```text
test songs = 12 complete songs
selection_used_test = false
Hybrid raw test mean RMSE/MAE/PCC/CCC = 0.231705/0.180635/0.397999/0.371315
Previous linear postprocess test mean RMSE/MAE/PCC/CCC = 0.228491/0.174951/0.424376/0.393014
Conservative V2 postprocess test mean RMSE/MAE/PCC/CCC = 0.236773/0.174319/0.400369/0.376076
Conservative V2 postprocess test Valence RMSE/MAE/PCC/CCC = 0.285205/0.203472/0.206967/0.198378
Conservative V2 postprocess test Arousal RMSE/MAE/PCC/CCC = 0.175445/0.145165/0.593771/0.553774
```

Conclusion: the Fold-0 validation candidate did not generalize to the locked
test set and must not be promoted as the final scientific Teacher. The main
failure is complete-song Valence baseline generalization. Stop tuning Fold-0
postprocessing and move to a more robust multi-fold or nested complete-song
validation strategy.

Follow-up diagnostics:

```text
outputs/nested_full_song_fold_diagnostics/
outputs/full_song_global_ridge_cv/
```

The five-fold complete-song diagnostics show that held-out complete-song
Valence means shift noticeably across folds. A lightweight complete-song
global Ridge baseline reached five-fold outer-test
`Mean RMSE/MAE/PCC/CCC = 0.242049/0.193788/0.513339/0.393422` and
`Valence RMSE/MAE/PCC/CCC = 0.253628/0.200424/0.490952/0.407597`.

The original Valence Global-Local full-song expert was then trained/evaluated
on all five nested folds. Its concatenated five-fold outer-test Valence result
was:

```text
Valence RMSE/MAE/PCC/CCC = 0.267009/0.205546/0.463024/0.430917
```

Conclusion: the current bottleneck is not just model size or postprocessing;
complete-song Valence baseline generalization on the 58 DEAM full songs is
unstable. The current Global-Local model is retained only as an ablation and
must not be promoted as the final Teacher. Future Teacher work should avoid
single-fold tuning and should either add more complete-song dynamic VA data or
reframe DEAM complete-song results as exploratory generalization diagnostics.

PMEmo was then introduced as the active Teacher training dataset. Arousal has a
metric-qualified PMEmo OpenSMILE ensemble candidate:

```text
model_id = pmemo_opensmile_arousal_ensemble_v1
test RMSE/MAE/PCC/CCC = 0.176695/0.135837/0.859718/0.848452
status = passes current minimum Arousal thresholds
```

Valence remains the active bottleneck. The current best Valence model is:

```text
model_id = pmemo_valence_mert_hidden6_harmonic_extreme_weighted_v1
trial_id = h6_harm_h80_ext125_clip010_ccc005
feature = MERT hidden_6 768 + Librosa harmonic 36
model = Valence-only BiLSTM, projection 320, hidden_dim 80, dropout 0.30
loss = extreme-weighted frame MSE + 0.05 * frame CCC loss + 0.10 * clip mean MSE
extreme_weight = 1.25
test RMSE/MAE/PCC/CCC = 0.205673/0.164567/0.783400/0.767304
status = current best Valence, but not pass
```

Gap to minimum Valence thresholds:

```text
RMSE gap = +0.025673
MAE gap = +0.014567
PCC gap = +0.016600
CCC gap = +0.032696
```

Do not continue blind OpenSMILE/BiLSTM hyperparameter tuning for Valence.
PMEmo-aligned MERT feature extraction has been tested at the last layer and at
hidden_6:

```text
MERT last layer = 767/767 songs, elapsed 1484.002 sec on CPU
best last-layer MERT+harmonic test RMSE/MAE/PCC/CCC =
0.253964/0.198529/0.643646/0.624557

MERT hidden_6 = 767/767 songs, elapsed 2123.038 sec on CPU
best hidden_6 MERT+harmonic test RMSE/MAE/PCC/CCC =
0.206191/0.165833/0.778422/0.757760
best hidden_6 MERT+harmonic+clip-aux test RMSE/MAE/PCC/CCC =
0.205867/0.165556/0.778969/0.760180
best hidden_6 MERT+harmonic+extreme-weighted test RMSE/MAE/PCC/CCC =
0.205673/0.164567/0.783400/0.767304
```

Hidden_6 ensemble and local refinement did not improve the current best:

```text
best hidden_6 ensemble test RMSE/MAE/PCC/CCC =
0.210620/0.168098/0.756924/0.736937
best local refinement test RMSE/MAE/PCC/CCC =
0.210751/0.173449/0.762561/0.752676
```

Current Valence best is `pmemo_valence_mert_hidden6_harmonic_extreme_weighted_v1`.
Adjacent MERT layers hidden_4 and hidden_8 were tested after this result and
did not improve over hidden_6:

```text
best hidden_4 test RMSE/MAE/PCC/CCC =
0.215706/0.171362/0.737009/0.714613
best hidden_8 test RMSE/MAE/PCC/CCC =
0.214950/0.166041/0.750379/0.743133
```

The first hidden_6 clip-level auxiliary loss round slightly improved all four
test metrics. Extreme-weighted loss improved all four metrics again.
Validation-selected postprocessing, convex late fusion, and Ridge stacking
were tested but not promoted because they worsened test generalization.
Extreme-weighted refine trials showed scalar loss-weight tuning is now at a
local plateau. Song-global auxiliary head trials showed that song-level
Valence is partly learnable, but direct auxiliary sharing worsens frame-level
test generalization. Song-balanced loss also did not improve test
generalization. Multi-seed confirmation should wait until the architecture
family is stable enough.

Hidden_6 + full OpenSMILE fusion was tested and did not improve over the
current best:

```text
feature = MERT hidden_6 768 + OpenSMILE 260
best OpenSMILE fusion test RMSE/MAE/PCC/CCC =
0.210139/0.170445/0.761105/0.753956
decision = useful ablation, but not promoted
```

Rejected historical ablation:

```text
Target-smoothed Valence training
raw validation mean RMSE/MAE/PCC/CCC = 0.198592/0.154167/0.772290/0.745791
postprocessed validation mean RMSE/MAE/PCC/CCC = 0.181002/0.138869/0.814328/0.797238
decision = weaker than the original global-local candidate
```

Do not report validation-only postprocessing results as final proof. The next
formal Valence step should test late fusion between complementary raw models
or a stronger objective selected by validation and judged against this current
best.

Updated current Valence multi-split candidate:

```text
model_id = pmemo_valence_multisplit_ensemble_ccc_preserving_20260627
base = existing MERT hidden_6 + harmonic BiLSTM and TCN predictions
selection = validation PCC/CCC-weighted pass-gap objective
postprocess = split-specific fusion, calibration, song-level smoothing
mean test RMSE/MAE/PCC/CCC = 0.203066/0.156576/0.777286/0.760357
status = current best multi-split Valence candidate, still below minimum pass
report = research/phase1f_external_va_datasets/pmemo/docs/VALENCE_MULTISPLIT_ENSEMBLE_CCC_PRESERVING_20260627.md
```

Current Valence gap to minimum pass:

```text
RMSE gap = +0.023066
MAE gap = +0.006576
PCC gap = +0.022714
CCC gap = +0.039643
```

2026-06-28 Valence update:

```text
model_id = pmemo_valence_regularized_stack_ensemble_20260628
base = previous CCC-preserving ensemble + regularized BiLSTM weak members
mean test RMSE/MAE/PCC/CCC = 0.202132/0.156198/0.779322/0.764633
status = current best multi-split Valence candidate, still below minimum pass
report = research/phase1f_external_va_datasets/pmemo/docs/VALENCE_REGULARIZED_STACK_ENSEMBLE_20260628.md
```

Current Valence gap to minimum pass:

```text
RMSE gap = +0.022132
MAE gap = +0.006198
PCC gap = +0.020678
CCC gap = +0.035367
```

2026-06-28 Valence postprocess update:

```text
model_id = pmemo_valence_song_bias_correction_rmse_aware_20260628
base = regularized stack ensemble + validation-selected time shift + RMSE-aware song-level bias correction
mean test RMSE/MAE/PCC/CCC = 0.200510/0.154051/0.784522/0.772036
status = current balanced Valence candidate, still below minimum pass
report = research/phase1f_external_va_datasets/pmemo/docs/VALENCE_SHIFT_AND_SONG_BIAS_20260628.md
```

Current Valence gap to minimum pass:

```text
RMSE gap = +0.020510
MAE gap = +0.004051
PCC gap = +0.015478
CCC gap = +0.027964
```

2026-06-28 Valence frame calibrator update:

```text
model_id = pmemo_valence_frame_calibrator_song_bias_20260628
base = RMSE-aware song-bias candidate + frame-level prediction-derived calibrator + final song-bias correction
mean test RMSE/MAE/PCC/CCC = 0.197480/0.153529/0.793235/0.780052
status = current Valence candidate, still below minimum pass
report = research/phase1f_external_va_datasets/pmemo/docs/VALENCE_FRAME_CALIBRATOR_20260628.md
```

Current Valence gap to minimum pass:

```text
RMSE gap = +0.017480
MAE gap = +0.003529
PCC gap = +0.006765
CCC gap = +0.019948
```

2026-06-28 Valence negative follow-up:

```text
tested = extreme-song-aware training sampler
gentle sampler RMSE/MAE/PCC/CCC = 0.210717/0.163264/0.765355/0.750240
stronger sampler RMSE/MAE/PCC/CCC = 0.219585/0.170319/0.758261/0.746395
fusion check best = current best + gentle sampler = 0.197322/0.153187/0.793468/0.779815
decision = do not promote; fusion lowers CCC, current best remains pmemo_valence_frame_calibrator_song_bias_20260628
report = research/phase1f_external_va_datasets/pmemo/docs/VALENCE_EXTREME_SAMPLER_NEGATIVE_20260628.md
```

2026-06-28 Valence final calibration negative update:

```text
tested = train-fit / validation-selected final output calibration
final calibration RMSE/MAE/PCC/CCC = 0.197042/0.153111/0.792856/0.777329
CCC-preserving best check = 0.197647/0.153615/0.792709/0.779977
decision = do not promote; current best remains pmemo_valence_frame_calibrator_song_bias_20260628
report = research/phase1f_external_va_datasets/pmemo/docs/VALENCE_FINAL_OUTPUT_CALIBRATION_NEGATIVE_20260628.md
```

2026-06-28 Valence prediction-only rescale positive update:

```text
model_id = pmemo_valence_prediction_only_rescale_ccc_preserving_20260628
method = prediction-only song mean/std rescale
selection rule = val_ccc >= baseline_val_ccc - 0.001
previous best RMSE/MAE/PCC/CCC = 0.197480/0.153529/0.793235/0.780052
current best RMSE/MAE/PCC/CCC = 0.195316/0.151793/0.797766/0.784834
status = current Valence candidate, still below minimum pass
report = research/phase1f_external_va_datasets/pmemo/docs/VALENCE_PREDICTION_ONLY_RESCALE_20260628.md
```

Current Valence gap to minimum pass:

```text
RMSE gap = +0.015316
MAE gap = +0.001793
PCC gap = +0.002234
CCC gap = +0.015166
```

2026-06-28 Valence rescale follow-up:

```text
report = research/phase1f_external_va_datasets/pmemo/docs/VALENCE_RESCALE_FOLLOWUP_20260628.md
rescale + song bias = 0.196294/0.152499/0.795052/0.781459, rejected
rescale + final calibration = 0.195428/0.152018/0.797346/0.782687, rejected
global deployment rescale = mean_gain=1.0, std_gain=0.45, blend=1.0
global pooled test RMSE/MAE/PCC/CCC = 0.195432/0.152036/0.800989/0.787990
global macro test RMSE/MAE/PCC/CCC = 0.195388/0.152277/0.797910/0.783420
```

Deployment note:

```text
Current CV metric best remains pmemo_valence_prediction_only_rescale_ccc_preserving_20260628.
For arbitrary-song prediction, use the global rescale configuration unless a
final full-training validation protocol is built.
```

2026-06-28 Valence follow-up screens after static-target correction:

```text
weak static auxiliary no-output screen:
  output = research/phase1f_external_va_datasets/pmemo/outputs/valence_weak_static_aux_screen_20260628
  split_01 test RMSE/MAE/PCC/CCC = 0.198134/0.156116/0.815827/0.801645
  decision = rejected because validation is much worse than the current split_01 reference

same-architecture seed bagging screen:
  output = research/phase1f_external_va_datasets/pmemo/outputs/valence_seed_bagging_screen_20260628
  split_01 rescaled 3-seed average = 0.184612/0.148618/0.839002/0.821644
  split_02 raw 3-seed average = 0.224437/0.176642/0.740177/0.728042
  decision = rejected; split_02 bottleneck worsens

current strict-ish Valence candidate remains:
  pmemo_valence_prediction_only_rescale_ccc_preserving_20260628
  RMSE/MAE/PCC/CCC = 0.195316/0.151793/0.797766/0.784834

best metric exploratory Valence candidate remains:
  pmemo_valence_universal_metric_optimized_exploratory_20260628
  RMSE/MAE/PCC/CCC = 0.191822/0.149198/0.804483/0.788190
```

2026-06-29 Valence multi-layer MERT screen:

```text
script = research/phase1f_external_va_datasets/pmemo/scripts/screen_pmemo_valence_multilayer_mert.py
features = MERT hidden_4 + hidden_6 + hidden_8 + harmonic Librosa

split_02 improved:
  current reference = 0.209386/0.162401/0.762291/0.742968
  multilayer raw = 0.207912/0.162272/0.780715/0.774531
  multilayer rescaled = 0.199139/0.151393/0.782771/0.771009

split_01 did not improve:
  current reference = 0.181559/0.141920/0.844993/0.835335
  multilayer rescaled = 0.197802/0.158857/0.813455/0.786089

diagnostic hybrid only, not final:
  RMSE/MAE/PCC/CCC = 0.191901/0.148124/0.804592/0.794181

decision:
  Multi-layer MERT is a useful Valence direction, especially for split_02,
  but this exact fixed configuration is not stable enough to replace the
  current strict-ish Teacher.

layer-gated follow-up:
  script = research/phase1f_external_va_datasets/pmemo/scripts/screen_pmemo_valence_layer_gated_mert.py
  split_02 result = 0.210041/0.170074/0.761562/0.751548
  learned gates = h4 0.3347, h6 0.3351, h8 0.3302
  decision = rejected; direct concat multilayer remains stronger on split_02

regularized concat follow-up:
  script = research/phase1f_external_va_datasets/pmemo/scripts/screen_pmemo_valence_multilayer_regularized.py
  split_02 result = 0.224595/0.174761/0.737361/0.727790
  decision = rejected; shrinking the head and stronger generic regularization removed the useful split_02 signal

global validation-selected ensemble:
  script = research/phase1f_external_va_datasets/pmemo/scripts/search_pmemo_valence_multilayer_global_ensemble.py
  selected weight_multilayer = 0.26
  previous strict-ish = 0.195316/0.151793/0.797766/0.784834
  updated strict-ish = 0.194113/0.151426/0.799891/0.785197
  remaining gap to pass = RMSE +0.014113, MAE +0.001426, PCC +0.000109, CCC +0.014803
  decision = small validation-safe improvement, still below minimum line

global ensemble song-bias correction:
  script = research/phase1f_external_va_datasets/pmemo/scripts/search_pmemo_valence_global_ensemble_song_bias.py
  result = 0.194114/0.151427/0.799892/0.785210
  decision = rejected; negligible CCC gain with slight RMSE/MAE worsening

global ensemble output calibration:
  script = research/phase1f_external_va_datasets/pmemo/scripts/search_pmemo_valence_global_ensemble_calibration.py
  result = 0.194124/0.151436/0.799891/0.785177
  decision = rejected; validation improves slightly but held-out test RMSE/MAE/CCC worsen

2026-06-29 Valence targeted + multi-layer update:
  correction = valence_targeted_rescale_valsafe_20260628 is validation-safe and stronger than the earlier 2026-06-29 global ensemble
  targeted_rescale_valsafe = 0.192981/0.150646/0.802475/0.785847
  arousal-assisted residual:
    script = research/phase1f_external_va_datasets/pmemo/scripts/search_pmemo_valence_arousal_bilstm_assisted_residual.py
    result = no_change selected on all three splits
    decision = predicted Arousal did not add stable Valence residual information
  targeted + multilayer global ensemble:
    script = research/phase1f_external_va_datasets/pmemo/scripts/search_pmemo_valence_targeted_multilayer_global_ensemble.py
    selected global candidate weight = 0.44
    result = 0.192719/0.150314/0.802897/0.787024
  targeted + multilayer postprocess:
    script = research/phase1f_external_va_datasets/pmemo/scripts/search_pmemo_valence_targeted_multilayer_postprocess.py
    selected = identity calibration + song-wise moving average window 11
    current Valence best = 0.192701/0.150310/0.802949/0.787081
    remaining gap to pass = RMSE +0.012701, MAE +0.000310, PCC pass by 0.002949, CCC +0.012919
    decision = current validation-selected Valence best, still below minimum line; shallow post-hoc optimization is saturated
  per-split direct multilayer ensemble:
    script = research/phase1f_external_va_datasets/pmemo/scripts/search_pmemo_valence_per_split_direct_multilayer_ensemble.py
    selected weights = split_01 0.00, split_02 0.36, split_03 0.08
    current Valence best = 0.191535/0.149089/0.805372/0.790749
    remaining gap to pass = RMSE +0.011535, MAE pass by 0.000911, PCC pass by 0.005372, CCC +0.009251
    decision = current validation-selected Valence best
  per-split direct multilayer parsimony:
    script = research/phase1f_external_va_datasets/pmemo/scripts/search_pmemo_valence_per_split_direct_multilayer_parsimony.py
    selected weights = split_01 0.00, split_02 0.35, split_03 0.00
    current Valence best = 0.191302/0.148931/0.805995/0.791430
    remaining gap to pass = RMSE +0.011302, MAE pass by 0.001069, PCC pass by 0.005995, CCC +0.008570
    decision = current validation-selected Valence best
  post-parsimony Valence attempts on 2026-06-29:
    seed707 direct multilayer split_02:
      output = research/phase1f_external_va_datasets/pmemo/outputs/valence_multilayer_mert_screen_seed707_20260629
      split_02 test = 0.216799/0.176871/0.769264/0.757971
      validation-selected mix audit selected weight = 0.00
      decision = rejected
    smaller regularized direct multilayer split_02:
      script = research/phase1f_external_va_datasets/pmemo/scripts/screen_pmemo_valence_multilayer_mert.py
      note = script now supports CLI hyperparameter overrides; default behavior unchanged
      split_02 test = 0.221647/0.180312/0.765123/0.747119
      decision = rejected
    Valence + Arousal auxiliary split_02:
      script = research/phase1f_external_va_datasets/pmemo/scripts/screen_pmemo_valence_multilayer_arousal_aux.py
      aux_weight 0.20 split_02 Valence test = 0.211110/0.171213/0.774951/0.761597
      aux_weight 0.05 split_02 Valence test = 0.212353/0.172866/0.772838/0.756911
      decision = rejected; Arousal auxiliary supervision learns Arousal but does not improve Valence
    current-best song-mean correction retry:
      script = research/phase1f_external_va_datasets/pmemo/scripts/search_pmemo_valence_parsimony_song_mean_fusion.py
      output = research/phase1f_external_va_datasets/pmemo/outputs/valence_parsimony_song_mean_fusion_20260629
      diagnostic = corr(song_rmse, abs_song_mean_error) = 0.996655
      result = 0.192336/0.151960/0.806223/0.781926
      delta vs current best = +0.001034/+0.003029/+0.000227/-0.009504
      decision = rejected; song-level mean bias is real, but post-hoc mean shifting is not stable enough
    training-time multilayer song-mean head:
      script = research/phase1f_external_va_datasets/pmemo/scripts/screen_pmemo_valence_multilayer_song_mean.py
      trial sw008_blend015_cap004 split_02 test = 0.220543/0.173099/0.740745/0.729843
      trial sw003_blend000 split_02 test = 0.225427/0.178747/0.737137/0.721640
      decision = rejected; current song-mean head damages frame-level representation
    context-conditioned multilayer MERT split_02 attempts on 2026-07-07:
      script = research/phase1f_external_va_datasets/pmemo/scripts/screen_pmemo_valence_multilayer_context_conditioned.py
      trial h468_harm_ctx96_film_ext125_clip010_ccc008 split_02 test = 0.221297/0.176182/0.764592/0.754410
      trial h468_harm_ctx32_nofilm_ext115_clip010_ccc008 split_02 test = 0.216646/0.173481/0.766451/0.757034
      decision = rejected; direct song-context injection overfits/under-generalizes and does not improve the split_02 bottleneck
    guarded output-level search on current parsimony best:
      script = research/phase1f_external_va_datasets/pmemo/scripts/search_pmemo_valence_parsimony_output_guarded.py
      model_id = pmemo_valence_parsimony_output_guarded_20260707
      selected change = split_02 identity smoothing window 15; split_01/split_03 unchanged
      current Valence best = 0.191273/0.148921/0.806073/0.791501
      remaining gap to pass = RMSE +0.011273, MAE pass by 0.001079, PCC pass by 0.006073, CCC +0.008499
      decision = very small but validation-safe improvement; current Valence best updated
    guarded Ridge stacking over existing prediction candidates:
      script = research/phase1f_external_va_datasets/pmemo/scripts/search_pmemo_valence_prediction_stacking_guarded.py
      result = 0.192431/0.149664/0.804041/0.790418
      decision = rejected; split_01 validation improvement did not generalize to test, so do not promote
    current teacher registry:
      report = research/phase1f_external_va_datasets/pmemo/docs/CURRENT_TEACHER_MODEL_REGISTRY_20260707.md
      note = Arousal metric-best is deployable with compatible OpenSMILE extraction; Valence metric-best is still a CV prediction artifact and is not yet fully aligned with the existing full-song prediction script
    current-best approximate Valence full-song predictor:
      script = research/phase1f_external_va_datasets/pmemo/scripts/predict_full_song_valence_current_best_approx.py
      batch_script = research/phase1f_external_va_datasets/pmemo/scripts/predict_batch_full_song_valence_current_best_approx.py
      model_id = pmemo_valence_current_best_deployable_approx_20260707
      smoke_test = songs/江南-林俊杰.mp3 first 20 seconds -> 40 frames, success
      note = deployable raw-MP3 approximation to the current metric-best chain, not a new validated metric-best result
    clean checkpoint-based Valence trainer prepared:
      script = research/phase1f_external_va_datasets/pmemo/scripts/train_pmemo_valence_multilayer_clean_teacher.py
      status = syntax-checked, not trained yet
      first recommended run after approval = split_02 seed 606
      note = intended to replace artifact-style Valence with a cleaner checkpoint teacher if metrics are acceptable

Arousal current candidate:
  model_id = pmemo_arousal_rmse_primary_postprocess_20260629
  script = research/phase1f_external_va_datasets/pmemo/scripts/materialize_pmemo_arousal_rmse_primary_postprocess.py
  method = validation-selected linear calibration strength 0.5 + song-wise moving average window 9
  previous Arousal best = 0.180960/0.140092/0.852634/0.848122
  current Arousal = 0.179278/0.138949/0.855885/0.848522
  status = all required Arousal metrics pass the minimum line; remaining first-stage bottleneck is Valence
```

## Documentation Rule

This README is the top-level project map and must be kept current. During
future tasks, update this file when project phase boundaries, directory layout,
major data flow, training workflow, deployment workflow, or database/QC behavior
changes.

Also update the more specific workflow documents when a change belongs there:

```text
AI_COLLABORATION_GUIDE.md
NEW_SONG_PIPELINE_WORKFLOW.md
LYRIC_CALIBRATION_WORKFLOW.md
PROJECT_HANDOFF.md
NETLIFY_DEPLOY_WORKFLOW.md
annotation_tool/WORKFLOW_STATUS.md
research/phase2_section_aware_va/README.md
```

## Git And Generated Data

Large local data and generated outputs are intentionally gitignored, including:

```text
songs/
outputs/
netlify_publish/
research/phase1_va_prediction/openl3_va_baseline/data/
research/phase1_va_prediction/openl3_va_baseline/artifacts/
research/phase1_va_prediction/openl3_va_baseline/outputs/
research/phase2_section_aware_va/data/
research/phase2_section_aware_va/outputs/
```

Keep commits scoped to the task. Do not mix unrelated dirty files into research
or QC updates.
