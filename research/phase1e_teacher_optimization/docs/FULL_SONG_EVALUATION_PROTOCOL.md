# Full-Song Dynamic VA Evaluation Protocol

Date: 2026-06-23

Status: frozen for the next representation and model experiments.

## Purpose

The deployment goal is dynamic Valence-Arousal prediction for an unseen song
of arbitrary duration. A frame-random split or a development score dominated
by 45-second excerpts does not prove this capability.

All future Teacher candidates must therefore report complete-song
generalization.

## DEAM Inventory

The audited local DEAM copy contains:

```text
total songs=1,802
short clips=1,744
complete songs=58
short-clip IDs <= 2000
complete-song IDs=2001..2058
```

Song 2000 is a 45-second excerpt and must not be treated as one of the 58
complete songs.

After removing invalid dynamic-label rows:

```text
short-clip frames=104,640
complete-song frames=25,355
label hop=0.5 seconds
```

All audio and sequence paths exist.

## Protocol A: PDMER-Compatible Short-to-Full Test

```text
training domain=all 1,744 short clips
test domain=all 58 complete songs
song overlap=0
```

This protocol answers:

> Can a model trained only on excerpts generalize to complete songs?

It is directly comparable to the principal DEAM setup in the AAAI 2025
PDMER paper. Hyperparameters must be selected without inspecting the 58
complete-song labels, for example through song-level validation among the
short clips.

Manifest:

```text
data/full_song_protocol/pdmer_short_to_full_manifest.csv
```

## Protocol B: Primary Nested Full-Song Evaluation

The 58 complete songs are assigned to five outer folds. Assignment is
stratified by median Valence and Arousal quadrants. Among 10,000 stratified
candidates, the selected split minimizes imbalance in:

- mean song Valence;
- mean song Arousal;
- audio duration;
- total valid frames.

Selected split seed:

```text
7080
```

For outer fold `k`:

```text
test complete songs=fold k
validation complete songs=fold (k+1) mod 5
training complete songs=remaining three folds
training short clips=all 1,744
```

Per-fold counts:

| Outer fold | Train short | Train full | Validation full | Test full |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 1,744 | 34 | 12 | 12 |
| 1 | 1,744 | 34 | 12 | 12 |
| 2 | 1,744 | 35 | 11 | 12 |
| 3 | 1,744 | 36 | 11 | 11 |
| 4 | 1,744 | 35 | 12 | 11 |

The role sets are disjoint within every outer fold. The old 80/10/10 split
is retained only as provenance and must not be used by this protocol.

Manifests:

```text
data/full_song_protocol/full_song_outer_folds.csv
data/full_song_protocol/nested_full_song_protocol.csv
```

## Required Metrics

Valence and Arousal must be reported separately:

- frame RMSE;
- frame MAE;
- frame PCC;
- frame CCC;
- 5-second window metrics;
- 10-second window metrics;
- song-mean metrics;
- within-song centered PCC and CCC.

The primary model-selection criterion is validation complete-song frame
RMSE. PCC and CCC cannot materially degrade.

The final five-fold result is calculated by concatenating predictions from
the five untouched outer test folds. Per-fold mean and standard deviation
must also be retained.

## Model-Selection Rules

1. Feature backbone and architecture choices use validation complete songs.
2. The outer test fold is evaluated only after the candidate is frozen.
3. Short clips may improve representation learning but cannot dominate
   validation or test selection.
4. Every song has equal sampling weight during training; long songs must not
   receive more optimization weight only because they contain more frames.
5. No frame from a complete song may appear in more than one role within an
   outer fold.
6. Random seeds are run only after a single-seed architecture passes the
   complete-song validation gate.

## Generated Files

```text
data/full_song_protocol/song_inventory.csv
data/full_song_protocol/full_song_outer_folds.csv
data/full_song_protocol/pdmer_short_to_full_manifest.csv
data/full_song_protocol/nested_full_song_protocol.csv
data/full_song_protocol/protocol_audit.json
scripts/build_full_song_protocol.py
```

Regenerate and verify:

```bash
.venv/bin/python \
  research/phase1e_teacher_optimization/scripts/build_full_song_protocol.py
```
