# OpenL3 Song-Level Split Validation

Last updated: 2026-06-21

## Purpose

This experiment measures OpenL3+MLP generalization to unseen DEAM songs and
provides a split-compatible comparison with the current MERT experiments.

The experiment does not overwrite the published frame-level dataset,
checkpoints, or the previously frozen engineering configuration.

## Dataset Protocol

The OpenL3 experiment uses the exact MERT song split manifest:

```text
research/phase1b_bert_deam_va_training/data/manifests/split_manifest.csv
```

| Split | Songs | Frames | Feature dimension |
| --- | ---: | ---: | ---: |
| train | 1441 | 101248 | 512 |
| validation | 180 | 13427 | 512 |
| test | 181 | 15320 | 512 |

Audit results:

- the song and frame counts exactly match the MERT dataset,
- train/validation/test song overlap is zero,
- all retained feature and VA values are finite,
- OpenL3 and DEAM labels use the published alignment convention
  `features[-T:]` with `labels[:T]`.

## Model Protocol

The MLP configuration matches the tuned frame-level Teacher head:

```text
hidden=[512,256]
dropout=0.10
learning_rate=0.0002
weight_decay=0.00005
batch_size=512
max_epochs=120
early_stopping_patience=18
seeds=2026,3407,614
```

Checkpoint epochs were selected using validation RMSE only.

Actual local runtime:

```text
dataset build and audit: approximately 2 seconds
seed 2026 training: approximately 17 seconds
seeds 3407 and 614 in parallel: approximately 16 seconds
total experiment execution: under 1 minute, excluding documentation
device=CPU
```

## OpenL3 Results

| Seed | Best epoch | Validation RMSE | Test RMSE | Test mean CCC |
| ---: | ---: | ---: | ---: | ---: |
| 2026 | 2 | 0.222834 | 0.210084 | 0.500731 |
| 3407 | 2 | 0.227587 | 0.210436 | 0.479568 |
| 614 | 5 | 0.225824 | 0.214057 | 0.479821 |
| mean | - | 0.225415 | 0.211525 | 0.486707 |
| standard deviation | - | 0.002403 | 0.002199 | 0.012146 |

Three-seed prediction ensemble:

| Metric | Validation | Test |
| --- | ---: | ---: |
| RMSE | 0.222275 | 0.208309 |
| MAE | 0.170606 | 0.163822 |
| Valence RMSE | 0.253810 | 0.218854 |
| Arousal RMSE | 0.185453 | 0.197201 |
| Valence Pearson r | 0.514971 | 0.420032 |
| Arousal Pearson r | 0.783268 | 0.653468 |
| Valence CCC | 0.422644 | 0.374708 |
| Arousal CCC | 0.746526 | 0.613654 |
| Mean CCC | 0.584585 | 0.494181 |

## Comparison With MERT+BiLSTM

Both rows below use the same song IDs and frame counts.

| Metric | OpenL3+MLP ensemble | MERT+BiLSTM baseline |
| --- | ---: | ---: |
| Validation RMSE | 0.222275 | **0.219934** |
| Test RMSE | 0.208309 | **0.202452** |
| Test MAE | 0.163822 | **0.160708** |
| Test Valence RMSE | 0.218854 | **0.205797** |
| Test Arousal RMSE | **0.197201** | 0.199050 |
| Test Pearson r(V) | 0.420032 | **0.511522** |
| Test Pearson r(A) | **0.653468** | 0.641069 |
| Test CCC(V) | 0.374708 | **0.464444** |
| Test CCC(A) | **0.613654** | 0.612082 |
| Test mean CCC | 0.494181 | **0.538263** |

MERT+BiLSTM reduces validation RMSE by `0.002342` (`1.05%`) and test RMSE
by `0.005857` (`2.81%`) relative to the OpenL3 ensemble. MERT is notably
stronger for Valence, while OpenL3 is slightly stronger on several Arousal
metrics.

This is a pipeline-level comparison, not a pure backbone ablation, because the
heads differ: OpenL3 uses an MLP and MERT uses a BiLSTM. A pure feature-backbone
comparison additionally requires matched heads and tuning budgets.

## Interpretation

- The frame-level random split Teacher test RMSE was `0.098425`.
- Under song-level splitting, the corresponding ensemble test RMSE is
  `0.208309`, an increase of approximately `111.64%`.
- The published-style frame split therefore substantially overestimates
  performance on unseen songs.
- The old result remains valid as a reproduction of the published protocol,
  but it must not be used as evidence of complete-song generalization.
- Current validation evidence favors MERT+BiLSTM for scientific Teacher
  selection, but MERT has not yet received the same three-seed verification.

## Decision

Do not generate the final 95-song pseudo-VA corpus yet.

The previously frozen OpenL3 ensemble remains an engineering-compatible
configuration, but final scientific Teacher selection is reopened. The next
required experiment is a three-seed MERT+BiLSTM baseline run, followed by the
same validation-only choice between its best single model and seed ensemble.

## Exact 15-Second Alignment Correction

A later timestamp audit found that the initial song-level OpenL3 dataset used
the inherited tail-alignment rule `features[-T:]`. For most 45-second songs,
this selected OpenL3 timestamps `15.5-45.0s`, while the labels represented
`15.0-44.5s`.

The corrected dataset selects OpenL3 timestamp index 30, exactly `15.0s`, and
pairs it with `sample_15000ms`.

Only seed `2026` was rerun by user direction:

| Metric | Old alignment | Exact alignment |
| --- | ---: | ---: |
| validation RMSE | 0.222834 | 0.220933 |
| test RMSE | 0.210084 | 0.208569 |
| test Valence RMSE | 0.221901 | 0.220053 |
| test Arousal RMSE | 0.197562 | 0.196414 |
| test Pearson V | 0.414900 | 0.422190 |
| test Pearson A | 0.654082 | 0.655921 |
| test mean CCC | 0.500731 | 0.501581 |

The correction improves most metrics, but test RMSE falls by only `0.001515`
or `0.72%`. Additional OpenL3 seeds are paused. The result strengthens the
decision to prioritize MERT timing, context, and representation experiments.

Corrected generated artifacts:

```text
artifacts/datasets/mlp_song_level_aligned15/
outputs/models/mlp_song_level_aligned15/
outputs/results/mlp_song_level_aligned15/
```

## Reproduction

```bash
.venv_openl3_mlp/bin/python \
  research/phase1_va_prediction/openl3_va_baseline/scripts/build_song_level_mlp_dataset.py

.venv_openl3_mlp/bin/python \
  research/phase1_va_prediction/openl3_va_baseline/scripts/train_song_level_mlp.py \
  --seed 2026 --device cpu

.venv_openl3_mlp/bin/python \
  research/phase1_va_prediction/openl3_va_baseline/scripts/aggregate_song_level_mlp.py
```

Generated data, checkpoints, histories, and metric summaries are preserved
under:

```text
artifacts/datasets/mlp_song_level/
outputs/models/mlp_song_level/
outputs/results/mlp_song_level/
```
