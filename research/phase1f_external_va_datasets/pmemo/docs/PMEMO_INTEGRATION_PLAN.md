# PMEmo Integration Plan

Date: 2026-06-24

## Why PMEmo Is Being Added

Current DEAM-only complete-song experiments show that Valence generalization is
the main bottleneck. The best validation-selected models did not generalize on
the locked complete-song test fold, and five-fold complete-song evaluation
confirmed instability.

PMEmo is added to increase dynamic VA supervision and reduce overfitting to the
small DEAM complete-song subset.

## Stage A: Dataset Audit

Goal: confirm that PMEmo can be normalized into the same VA Teacher training
format as DEAM.

Status: completed for the official 2019 update.

Checks:

- audio files and duration
- annotation file formats
- candidate song ID columns
- candidate timestamp columns
- candidate Valence/Arousal columns
- numeric label ranges
- missing values
- approximate label time step

Output:

```text
outputs/audit/pmemo_audit_summary.json
outputs/audit/pmemo_audio_inventory.csv
outputs/audit/pmemo_table_inventory.csv
```

Observed:

```text
dynamic_songs = 767
dynamic frame rows = 36434
frame step = 0.5 s
label range = 0 to 1
dynamic label columns = Arousal(mean), Valence(mean)
first 15 seconds = already removed in the released annotation file
```

## Stage B: Normalized Manifest

Goal: create a clean PMEmo manifest without changing the raw files.

Status: initial manifests completed.

Target fields:

```text
dataset
song_id
audio_path
duration_sec
time_sec
valence
arousal
label_range
split_group
```

The target training scale should be chosen after auditing. If PMEmo labels are
not already in the same range as DEAM, normalization must be recorded explicitly.

Current outputs:

```text
data/processed/pmemo2019_dynamic_frame_manifest.csv
data/processed/pmemo2019_song_manifest.csv
data/processed/pmemo2019_manifest_summary.json
```

## Stage C: Baselines

Run PMEmo-only and DEAM-only sanity experiments with the same evaluation code:

```text
MERT/available feature backbone + MLP
MERT/available feature backbone + BiLSTM
MERT/available feature backbone + Transformer or TCN
```

Report Valence and Arousal separately:

```text
RMSE, MAE, PCC, CCC
```

## Stage D: Fusion Training

Preferred first fusion strategy:

```text
pretrain on PMEmo + DEAM short/dynamic data
fine-tune on DEAM with song-level split
evaluate on complete-song folds and fixed-window/clip-level summaries
```

Do not tune on locked test folds.

## Expected Benefit

PMEmo may help if it provides more full-song or dynamic VA coverage than the
small DEAM complete-song subset. The expected gain is mainly in Valence
stability and cross-song baseline prediction, not merely a lower frame-random
metric.

## Risks

- label scale mismatch
- different annotation protocol
- different music distribution
- missing or inconsistent song ID mapping
- improvement on PMEmo that does not transfer to DEAM complete songs

These risks are why the audit must happen before mixed training.
