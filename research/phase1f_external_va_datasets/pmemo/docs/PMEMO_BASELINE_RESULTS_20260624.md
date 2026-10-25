# PMEmo Label Normalization And OpenSMILE Baseline Results

Date: 2026-06-24

## Purpose

This experiment starts PMEmo integration for the Phase 1 VA Teacher. It does
not yet mix PMEmo with DEAM, because the currently available PMEmo feature path
uses the official OpenSMILE dynamic features, while the current DEAM Teacher
path uses MERT features.

The purpose of this first run is:

```text
1. normalize PMEmo labels to the same canonical VA scale as DEAM;
2. verify PMEmo dynamic VA training works under song-level split;
3. produce a PMEmo-only baseline before cross-dataset fusion.
```

## Label Scale

PMEmo released labels are in `[0, 1]`.

The current canonical Teacher scale is DEAM-style `[-1, 1]`.

Conversion:

```text
deam_scale = 2 * pmemo_scale - 1
```

Normalized PMEmo dynamic label range:

| Dimension | Min | Max |
| --- | ---: | ---: |
| Valence | -0.786875 | 0.859375 |
| Arousal | -0.809125 | 0.942650 |

Files:

```text
data/processed/pmemo2019_dynamic_frame_manifest_normalized.csv
data/processed/pmemo2019_song_manifest_normalized.csv
data/processed/pmemo2019_label_normalization_summary.json
scripts/normalize_pmemo_labels.py
```

## Data Used

```text
dataset = PMEmo 2019 updated version
feature source = official dynamic_features.csv
feature type = OpenSMILE dynamic acoustic features
feature dim = 260
label source = annotations/dynamic_annotations.csv
label step = 0.5 s
first 15 seconds = already removed in released annotations
dynamic songs = 767
dynamic frames = 36434
```

Song-level split:

| Split | Songs | Frames |
| --- | ---: | ---: |
| Train | 614 | 29184 |
| Validation | 77 | 3912 |
| Test | 76 | 3338 |

## Models

Two simple baselines were trained:

```text
MLP frame-level regressor
BiLSTM sequence regressor
```

Both use canonical `[-1, 1]` labels. For approximate PMEmo original `[0, 1]`
scale values, divide RMSE and MAE by 2. PCC and CCC are unchanged by linear
scale conversion.

## Results On Canonical [-1, 1] Scale

| Model | Val RMSE | Aro RMSE | Mean RMSE | Val MAE | Aro MAE | Mean MAE | Val PCC | Aro PCC | Mean PCC | Val CCC | Aro CCC | Mean CCC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PMEmo OpenSMILE MLP | 0.270793 | 0.218576 | 0.246074 | 0.222314 | 0.174021 | 0.198168 | 0.560607 | 0.772857 | 0.666732 | 0.483330 | 0.734497 | 0.608913 |
| PMEmo OpenSMILE BiLSTM | 0.271199 | 0.186934 | 0.232909 | 0.212034 | 0.141942 | 0.176988 | 0.571292 | 0.837598 | 0.704445 | 0.546814 | 0.830559 | 0.688687 |

## Approximate Results On Original [0, 1] PMEmo Scale

| Model | Val RMSE | Aro RMSE | Mean RMSE | Val MAE | Aro MAE | Mean MAE | Mean PCC | Mean CCC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PMEmo OpenSMILE MLP | 0.135397 | 0.109288 | 0.123037 | 0.111157 | 0.087011 | 0.099084 | 0.666732 | 0.608913 |
| PMEmo OpenSMILE BiLSTM | 0.135599 | 0.093467 | 0.116455 | 0.106017 | 0.070971 | 0.088494 | 0.704445 | 0.688687 |

## Interpretation

BiLSTM is better than MLP, mainly because it improves Arousal:

```text
Arousal test PCC/CCC = 0.837598 / 0.830559
```

Valence remains the weak dimension:

```text
Valence test PCC/CCC = 0.571292 / 0.546814
```

This mirrors the DEAM experiments: Arousal can be learned reasonably from audio
dynamics, while Valence generalization remains harder.

## Decision

This PMEmo-only OpenSMILE baseline is useful as a sanity check and auxiliary
baseline, but it is not the final Teacher path.

Next high-value step:

```text
extract MERT features for PMEmo chorus clips
train PMEmo MERT BiLSTM baseline
then test PMEmo pretraining -> DEAM fine-tuning
```

This keeps the final Teacher aligned with the MERT-based DEAM path and the
arbitrary-song prediction goal.
