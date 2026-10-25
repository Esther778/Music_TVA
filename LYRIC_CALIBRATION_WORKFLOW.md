# Lyric Calibration Workflow

Goal: calibrate future new-song timed lyric text while preserving timing and
section data. Already-imported songs should not be recalibrated retroactively
unless the user explicitly requests a specific song.

## Hard Rules

1. Prefer updating only `lyric_lines.text` for existing lyric rows.
2. Do not update existing `lyric_lines.start_time_sec`, `lyric_lines.end_time_sec`, `line_index`, `song_id`, or `source`.
3. If full-song coverage is missing, insert a new `lyric_lines` coverage marker row instead of shifting or retiming existing rows. Use `line_index=-1` for a missing opening `前奏` row, and `last_line_index + 1` for a missing ending `尾奏` row.
4. Do not update `auto_sections`, `model_runs`, `human_annotations`, or `section_reviews` during lyric text correction.
5. Work in auditable batches and verify each batch before moving on.
6. Do not copy one song's lyric into another song. Every row must be tied to one `song_id` and `line_index`.
7. Do not paste full lyrics into chat replies. Keep source evidence as URLs and local review artifacts.
8. The timed lyric rows for each song must cover from 0 seconds to the final second of the song. If the opening has no sung lyric, write `前奏`; if the ending has no sung lyric, write `尾奏`.
9. For future new songs, lyric calibration should fix obvious transcription
   recognition errors, including homophone, near-homophone, wrong-character,
   and truncated-phrase errors. Examples: `我脚奥的飞远` -> `我骄傲地飞远`;
   `原谅我这一生慌荡不羁爱自` -> `原谅我这一生放荡不羁爱自由`.

## Batch Process

### 1. Export A Snapshot

```bash
cd "/Users/esther/Desktop/New project"
python3 annotation_tool/calibrate_lyrics.py export \
  --out outputs/lyric_calibration/snapshot_YYYYMMDD.json
```

Keep this snapshot for rollback comparison and audit.

### 2. Search Reliable Lyrics

For each song, search the web by exact title and artist.

Preferred source order:

1. Official artist, label, distributor, music platform, or official MV pages when lyrics are shown.
2. Dedicated lyric pages with timed LRC and stable URLs.
3. Secondary lyric mirrors only when two sources agree.

Record the source URL in the review file. If sources disagree, mark the song as blocked and do not update it yet.

### 3. Build A Review CSV

Use one row per lyric line:

```text
song_id,line_index,current_text,corrected_text,source_url,review_note
```

Rules:

- `line_index` must match the existing database line.
- `corrected_text` may change Chinese lyric characters and punctuation only when needed, including obvious transcription fixes such as homophones, wrong characters, or missing trailing words.
- Keep the existing line count and order.
- Preserve full-song coverage. Non-sung opening rows should be `前奏`, non-sung ending rows should be `尾奏`, and instrumental rows in the middle should use a concise Chinese function label instead of unrelated metadata or source noise.
- Leave uncertain rows unchanged and note why.

### 4. Dry Run

```bash
python3 annotation_tool/calibrate_lyrics.py apply \
  --input outputs/lyric_calibration/batch_YYYYMMDD.csv \
  --snapshot outputs/lyric_calibration/snapshot_YYYYMMDD.json \
  --dry-run
```

Read the summary:

- changed line count should match the reviewed rows,
- no timing or section drift should be reported,
- no duplicate `song_id,line_index` pairs should exist.

### 5. Apply One Small Batch

```bash
python3 annotation_tool/calibrate_lyrics.py apply \
  --input outputs/lyric_calibration/batch_YYYYMMDD.csv \
  --snapshot outputs/lyric_calibration/snapshot_YYYYMMDD.json
```

After applying, rerun verification:

```bash
python3 annotation_tool/calibrate_lyrics.py verify \
  --snapshot outputs/lyric_calibration/snapshot_YYYYMMDD.json
```

### 6. Model Rerun Decision

Rerun model output for a song when lyric correction changes structure evidence:

- repeated chorus/hook text changed,
- verse/pre-chorus/chorus paragraph identity changed,
- many lines changed,
- line content was previously Whisper nonsense,
- section evidence in latest `auto_sections.lyric_evidence` is now stale.

Import rerun outputs with `--force-model-output` so the newest model run becomes the machine version while human history is preserved.

### 7. New Song Rule

All future songs must be model-run before database import:

```bash
python3 annotation_tool/run_model_outputs_for_songs.py \
  --song-list annotation_tool/today_new_songs_YYYYMMDD.txt \
  --filename-order title-artist \
  --no-lyric-search \
  --continue-on-error \
  --write-manifest outputs/functional_sections/today_new_songs_YYYYMMDD_generated_manifest.csv

python3 annotation_tool/import_local_results.py \
  --manifest outputs/functional_sections/today_new_songs_YYYYMMDD_generated_manifest.csv \
  --skip-song-discovery
```

The QC platform should never show a newly added song without a model run.
