# VA Song-Mean and Delay Optimization

Report time: 2026-07-15 CST

Stage: Phase 1 VA Teacher development. All experiments used only the 652-song
development pool with three song-level folds. The 115-song final locked test
was excluded.

## Main diagnosis

Oracle error decomposition showed that frame-level shape was not the dominant
error source. Replacing only each predicted song mean with the true song mean,
while preserving frame deviations, reduced diagnostic RMSE to about `0.046`
for Arousal and `0.041` for Valence. This oracle is not a model result; it
identified unseen-song global emotion level as the next training target.

## Arousal

Base:
- `pmemo_arousal_locked_effnet_moodtheme_fine_complete_20260715`
- RMSE/MAE/PCC/CCC:
  `0.179600 / 0.139630 / 0.858358 / 0.848826`

Nested song-mean fusion:
- Inputs: current prediction statistics, EffNet mood/theme tags, CLAP, and
  MAEST full-song audio embeddings.
- Every held fold used only the other two folds for feature-family, target-mode,
  and Ridge-alpha selection.
- Selected family: prediction statistics + mood/theme tags + MAEST.
- Selected bounded correction: strength `0.50`, cap `0.20`.
- Result:
  `0.174356 / 0.135597 / 0.866822 / 0.858846`.
- Paired song bootstrap versus the base supported all four improvements:
  RMSE CI `[-0.008948, -0.001629]`,
  MAE CI `[-0.007287, -0.000807]`,
  PCC CI `[+0.002736, +0.014833]`,
  CCC CI `[+0.004131, +0.016591]`.

OpenSMILE static-functionals refinement:
- Added a nested 32-component PCA + Ridge full-song branch with static
  Arousal as training-only auxiliary supervision.
- Point-estimate best:
  `0.174062 / 0.135122 / 0.867330 / 0.859597`.
- Versus the first song-mean fusion, only the CCC bootstrap interval clearly
  excluded zero; this is a small refinement, not another major gain.

Current threshold status:
- Minimum: all four pass.
- Good: PCC and CCC pass.
- RMSE misses good by `0.024062`.
- MAE misses good by `0.015122`.

## Valence

Deep full-song and auxiliary checks:
- Nested EffNet/CLAP/MAEST song-mean fusion selected identity.
- The improved predicted Arousal curve as an auxiliary feature also selected
  identity.
- OpenSMILE static functionals, with and without static-Valence auxiliary
  supervision, selected identity after nested PCA/Ridge validation.

Global reaction-delay calibration:
- A common delay and within-song amplitude were selected across all
  development folds; no fold-specific calibration was used.
- RMSE-primary exploratory optimum:
  shift `-2.0 s`, amplitude `0.78`,
  `0.181163 / 0.142310 / 0.824258 / 0.806984`.
- Because its CCC was slightly below the previous robust best and bootstrap
  intervals crossed zero, this is retained as an exploratory RMSE-primary
  artifact.
- Guarded robust candidate:
  shift `-0.5 s`, amplitude `0.75`,
  `0.181284 / 0.142308 / 0.824000 / 0.807112`.
- Versus the prior robust best, paired song bootstrap supported RMSE and PCC
  improvement; MAE and CCC were effectively unchanged.

Current threshold status:
- MAE, PCC, and CCC pass.
- RMSE remains above the minimum line by `0.001284`.
- Valence has not reached all four minimum criteria.

## Decision

- Promote the Arousal song-mean fusion as a statistically supported major
  improvement; retain its OpenSMILE refinement as the latest point-estimate
  best with a small-effect caveat.
- Promote the guarded Valence delay/amplitude candidate as the robust
  development best.
- Do not claim that Arousal has reached the full good line: only PCC and CCC
  are good.
- Do not claim that Valence has passed: RMSE remains above `0.18`.
- The final locked test remains untouched.

## Artifacts

- `scripts/train_pmemo_va_locked_song_mean_fusion.py`
- `scripts/search_pmemo_valence_locked_global_smooth_amp.py`
- `outputs/arousal_locked_song_mean_fusion_20260715`
- `outputs/arousal_locked_opensmile_static_songmean_fine_20260715`
- `outputs/valence_locked_global_shift_amp_guarded_20260715`
- `outputs/valence_locked_global_shift_delay_20260715`

## Continued Auxiliary-Supervision and Amplitude Checks

Static/dynamic song-label mixing:
- Added training-only targets that mix each song's dynamic-label mean with its
  mapped PMEmo static label at weights `0.25`, `0.50`, and `0.75`.
- The static label was never used as an inference feature. All held-fold models
  still received audio-derived features only.
- Arousal output:
  `outputs/arousal_locked_static_mix_songmean_20260715`.
- Valence output:
  `outputs/valence_locked_static_mix_songmean_20260715`.
- Strict nested selection chose identity for both dimensions. This supervision
  did not improve unseen-song performance and is rejected.

Extended Valence amplitude check:
- Output: `outputs/valence_locked_global_amp_extended_20260715`.
- Search covered one common delay from `-2.0` to `0.0` seconds and stronger
  within-song amplitude contraction from `0.35` to `0.75`.
- RMSE-primary point estimate selected `-2.0` seconds and amplitude `0.70`:
  `0.181131 / 0.142397 / 0.824339 / 0.807015`.
- It reduced RMSE and increased PCC relative to its input, but worsened MAE and
  CCC. The four-metric guard therefore rejected it.
- Valence remains short of the minimum RMSE line by `0.001131` even under this
  exploratory RMSE-primary policy.

Updated conclusion:
- Arousal point best remains
  `0.174062 / 0.135122 / 0.867330 / 0.859597`; only PCC and CCC meet the good
  line.
- Valence robust best remains
  `0.181284 / 0.142308 / 0.824000 / 0.807112`.
- The final 115-song locked test remains untouched.

## MERT Attention Song-Mean Training Gate

Purpose:
- Test a genuine training-time song-mean model instead of another output-only
  calibration.
- Preserve arbitrary-MP3 deployability by using frozen MERT hidden-4/6/8 audio
  frames and EffNet mood/theme audio tags only.

Implementation:
- Added `scripts/train_pmemo_valence_locked_mert_attention_songmean.py`.
- Architecture: frame projection, attention mean/variance, ordinary mean/max
  pooling, EffNet tag context, and a compact song-level head.
- Protocol: stratified inner validation for epoch and correction selection,
  outer song-fold evaluation, and a hard zero-overlap guard against the final
  115 songs.
- Five previously documented terminal-edge songs required feature-time
  interpolation with edge hold: `6, 126, 150, 862, 992`.

Fold-03 direct-target gate:
- Inner song RMSE: `0.217770`.
- Held song RMSE: `0.223998`.
- Inner selection chose identity (`strength=0`).
- Frame result remained the base:
  `0.193754 / 0.149194 / 0.818762 / 0.799365`.

Fold-03 zero-initialized residual gate:
- Started exactly from the base song mean and learned a bounded residual.
- Inner selection: epoch `2`, strength `0.5`, cap `0.005`.
- Held song RMSE: `0.186471`.
- Held frame RMSE/MAE/PCC/CCC:
  `0.194013 / 0.149134 / 0.818697 / 0.798955`.
- RMSE, PCC, and CCC were worse than the fold-03 base. The gate is rejected
  and was not expanded to all three folds.

Decision:
- Frozen MERT multi-layer attention pooling does not improve unseen-song
  Valence mean estimation over the current multi-stage base.
- Current robust Valence best remains unchanged.
- Further meaningful improvement requires either trainable backbone adaptation
  with suitable accelerator resources or genuinely additional aligned
  Valence supervision; another song-mean head on the same frozen features is
  not justified.
