# VA Teacher Development Freeze Manifest

Frozen at: 2026-07-15 CST

Research stage: Phase 1 VA Teacher development is closed. The prospective
115-song final locked test has not been evaluated at the time of this freeze.

## Freeze Purpose

This manifest preserves the best development evidence before any final-test
inference. It separates two concepts that must not be conflated:

- `development_metric_best`: the best three-fold OOF prediction artifact;
- `final_test_executable_model`: the inference pipeline that will produce the
  115-song final predictions. This remains pending until its complete
  dependency chain is materialized and audited.

The final labels must remain sealed until the second item is frozen.

## Canonical Development Results

Common protocol: PMEmo 2019, labels in `[-1, 1]`, 652 development clips,
three mutually exclusive song folds, complete-clip mean aggregation, then
three-fold macro metrics.

| Dimension | Development metric best | RMSE | MAE | PCC | CCC |
| --- | --- | ---: | ---: | ---: | ---: |
| Valence | `pmemo_valence_locked_global_shift_amp_guarded_20260715` | 0.178526 | 0.140722 | 0.821365 | 0.805417 |
| Arousal | `pmemo_arousal_locked_opensmile_static_songmean_fine_20260715` | 0.174339 | 0.133155 | 0.870038 | 0.860214 |

Frame-level reference:

| Dimension | RMSE | MAE | PCC | CCC | Minimum status |
| --- | ---: | ---: | ---: | ---: | --- |
| Valence | 0.181284 | 0.142308 | 0.824000 | 0.807112 | RMSE misses by 0.001284 |
| Arousal | 0.174062 | 0.135122 | 0.867330 | 0.859597 | all pass |

The statement `Valence RMSE < 0.18` is valid only for complete-clip
aggregation. These are development-fold results, not final-test results and
not multi-minute commercial-song results.

## Selected Development Policies

Valence final low-degree policy:

```text
input = valence_locked_effnet_moodtheme_tags_balanced_final_complete_20260715
shift_frames = -1
shift_sec = -0.5
smoothing_window = 1
smoothing_blend = 0.0
within-song amplitude = 0.75
```

Arousal final song-mean refinement:

```text
input = arousal_locked_song_mean_fusion_20260715
feature_family = pred_opensmile_static
strength = 0.02
cap = 0.15
```

## Traceability Hashes

Repository HEAD at freeze: `41cf9d9f54e80a2e28bad3763cbd654244d17f25`.
The worktree contained pre-existing uncommitted research and application files;
therefore file hashes below, not the Git commit alone, are the authoritative
artifact identifiers.

Split files:

```text
65971f15ee892d60c09d778968188f9ed5a1fe18974d6f8c0dd2bc114d8932a2  final_locked_test_index.csv
d3fa3577e6f543dd2f6b5c60ae995ab638be735e5730c31bbef8645413bd98e7  fold_01/sequence_index.csv
521a69e9f069340162da16d34f5a866ec46fec2ef70e143d7a11fdd75d2264d1  fold_02/sequence_index.csv
241d60889daaf521f1aa4e16e752e479ec77883f31463f373155dab83e4cebda  fold_03/sequence_index.csv
```

Valence artifacts:

```text
da050dc41a1a5dd01c2e63b69364f1561b5852806e586a94c28fe3345dc6bbb5  summary.json
5529bc6f197f429858922d5fb8503df22f1c3a5dc16956be2bc3b4cff4d8e3f7  fold_01_val_predictions.csv
a0be4c8b28c103ebd398a14f6c0b1212938da370adba44038bd208022be8f95a  fold_02_val_predictions.csv
282d7d7ca1e275c421538140b59b1e5b5d7503acfda706fcfc2c6f9eb0468435  fold_03_val_predictions.csv
```

Arousal artifacts:

```text
56d91ca1f460d94d75a6b49245eb5f18af804520af5b4198c046f5ea76e0b76c  summary.json
8e2ae072bd34aa8f49e722e4d3397fb10eeee2b811c671946cae88036e90ed75  fold_01_val_predictions.csv
e076654b1df602d46545a19450ac293b117770978fa21cf7579d3df85384edac  fold_02_val_predictions.csv
a69e182aff636d03bbfe516fbb90074e6bed906d99d54c06652148cf00a4fbae  fold_03_val_predictions.csv
```

## Executability Audit

Both registered bests are multi-stage OOF artifacts rather than single clean
checkpoints. Several song-level correction estimators were evaluated through
cross-fold prediction tables but were not serialized for arbitrary-song
inference. In addition, EffNet/MAEST/MusiCNN auxiliary features currently cover
the 652-song development pool only, while MERT features cover all 767 PMEmo
clips.

Consequences:

1. The historical raw-MP3 approximation scripts must not be presented as the
   exact registered bests.
2. Final-test labels remain sealed while an executable fixed inference chain is
   materialized.
3. If a reduced deployable chain is required, its development OOF metrics must
   be reported separately before final-test inference.
4. Once label-free final predictions are hashed, final labels may be opened
   exactly once. No model, calibration, ensemble weight, smoothing, delay, or
   threshold may then change in response to the result.

## Freeze Status

```text
development_metric_best = frozen
development_artifact_hashes = frozen
final_test_executable_model = pending materialization
final_test_predictions = not generated
final_test_labels = sealed
final_test_metrics = not computed
```
