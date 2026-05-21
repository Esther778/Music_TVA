# Music Section Segmenter

Python tool for automatic music section segmentation, designed for MER/VA
emotion trajectory analysis. It outputs structural section boundaries that can
be aligned with later valence-arousal curves.

This version uses MSAF for candidate boundaries, then applies MIR
post-processing for pop-song section labels. The post-processing uses adaptive
novelty splitting plus section-level energy, vocal-activity proxy, onset
strength, optional timestamped lyrics, position, and pop-function logic; it does
not assume a fixed phrase length.

## Setup

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run

```bash
python segment_music.py --input songs/song.mp3 --output outputs/sections.csv
```

The default mode is `mir-pop`, which outputs semantic labels such as `Intro`,
`Verse`, `Pre-chorus`, `Chorus`, `Bridge`, and `Outro`, plus reusable
functional labels:

```text
A = Intro / Outro
B = Verse
C = Pre-chorus
D = Chorus
E = Bridge
```

The JSON file is written next to the CSV by default:

```text
outputs/sections.json
```

You can also set it explicitly:

```bash
python segment_music.py \
  --input songs/song.mp3 \
  --output outputs/sections.csv \
  --json-output outputs/sections.json
```

Optional timestamped lyrics can be used as soft evidence:

```bash
python segment_music.py \
  --input songs/song.mp3 \
  --lyrics lyrics/song.lrc \
  --output outputs/sections.csv
```

Lyrics do not create hard section boundaries. A lyric line usually marks vocal
text onset, while melody, harmony, or arrangement changes may happen slightly
before or after the lyric. The script therefore boosts nearby acoustic novelty
peaks within a tolerance window instead of cutting exactly at the lyric time.
When lyrics are unavailable, the script still uses an audio-derived vocal
activity proxy to help distinguish instrumental intro/outro/bridge regions from
vocal narrative and chorus regions.

If an acoustic boundary falls in the gap between two lyric lines, the script
treats that gap as an ambiguous transition fragment. Short fragments are
assigned to the neighboring section whose audio material is more similar, so a
between-line fill, pickup, or breath does not become a separate function label
by accident.

Supported lyric formats:

```text
[00:12.80]第一句歌词
[00:29.50]第二句歌词
```

or CSV:

```text
time,text
0:12.80,第一句歌词
0:29.50,第二句歌词
```

MSAF algorithm choices can be changed:

```bash
python segment_music.py \
  --input songs/song.mp3 \
  --output outputs/sections.csv \
  --boundaries-id sf \
  --labels-id fmc2d
```

To inspect raw MSAF labels without semantic post-processing:

```bash
python segment_music.py \
  --input songs/song.mp3 \
  --output outputs/raw_msaf.csv \
  --mode raw-msaf
```

## Output Fields

```text
section_id,start_time,end_time,duration,label,structure_label
1,0:00.00,0:12.50,0:12.50,Intro,A
2,0:12.50,0:43.00,0:30.50,Verse,B
3,0:43.00,1:08.00,0:25.00,Chorus,D
```

Raw MSAF mode preserves repeated labels as `A/B/C/...`.

## Tuning

The experiment harness compares candidate methods against a hand annotation:

```bash
python experiments/section_tuning.py
```

For the current reference song, the best tested default is `sf/fmc2d` with
`mir-pop` and `--long-section-factor 1.5`.
