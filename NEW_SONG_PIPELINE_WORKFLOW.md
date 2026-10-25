# New Song Pipeline Workflow

This workflow is for future batches added under `songs/songs2`.

## First: Cheaper Time And Token Plan

Do not use manual ChatGPT turns to inspect every song one by one. The economical path is:

1. Use deterministic scripts first: filename normalization, duplicate detection, audio validity checks, model-run manifests, and database imports should be command-driven.
2. Do not insert songs into Supabase before model run output exists. This keeps the QC platform free of songs that show "no model run".
3. For a new batch, run Whisper/model once with transcript fallback to produce a usable QC baseline quickly.
4. For lyric correction, avoid rewriting timestamps. Only replace the text after existing timestamps in `lyric_lines` or in a timed lyric CSV.
5. Batch lyric lookup by song list and cache sources/results locally. Reuse cached lyric files before searching again.
6. After lyric correction, rerun model only for songs whose lyric text changed enough to affect sectioning. Use the new `--force-model-output` import mode so the latest model run becomes the machine version without deleting human review history.
7. Deploy only after local files and `netlify_publish/` are updated. If Codex cannot deploy because of DNS/permission timeout, the user runs the exact Netlify terminal command below.

Practical consequence: for each batch, Codex should spend most time on scripts and verification, not on conversational per-song reasoning.

## Documentation Rule

Any change to project behavior must update the relevant workflow Markdown in the same pass. This includes pipeline logic, section label taxonomy, import rules, lyric calibration rules, QC UI behavior, Netlify deployment, and database validation.

- Remove stale instructions when a rule changes.
- Add new rules where future agents will look first.
- Prefer updating `NEW_SONG_PIPELINE_WORKFLOW.md`, `LYRIC_CALIBRATION_WORKFLOW.md`, `PROJECT_HANDOFF.md`, `NETLIFY_DEPLOY_WORKFLOW.md`, `annotation_tool/WORKFLOW_STATUS.md`, and `annotation_tool/README.md` when relevant.

## Section Label Taxonomy

New songs must use the seven-label taxonomy:

```text
intro, verse, pre_chorus, chorus, bridge, interlude, outro
```

Use these distinctions:

- `interlude`:无人声或基本无人声的间奏。It is a short instrumental transition, breath, or buffer between main sections. It can occur multiple times, often after a chorus before the next verse/chorus. It should not imply a new lyrical idea.
- `bridge`:桥段。It is an independent contrast or turn, usually later in the song and often before the final chorus. It usually has lyrics or a distinct new melodic/harmonic/emotional idea. Middle 8 is a bridge subtype.

One-line rule: `interlude` is transition or breath; `bridge` is contrast or turn.

Pipeline consequence:

- Mid-song no-vocal, low-vocal, or long lyric-free instrumental gaps should default to `interlude`.
- Keep `bridge` for lyrical or clearly functional contrast: new material, emotional turn, harmonic/sectional contrast, or tension before returning to chorus.
- Do not collapse `interlude` back into `bridge` merely because it appears between choruses.
- Supabase check constraints must allow the same seven labels in both
  `auto_sections.section_type` and `human_annotations.section_type`. If
  `interlude` submit/import fails, run
  `supabase_section_type_interlude_constraint.sql` in Supabase SQL Editor before
  retrying the submit or migration.
- Transcript parsing must accept Chinese and English lyric text. Do not filter
  transcript rows with a CJK-only predicate, otherwise English songs can produce
  empty `agentic_sections.csv` files and show as `0/0` in the QC console.

## Existing 94-Song Bridge Migration

The 94 songs that were already written to Supabase before `interlude` existed
must not be fully retranscribed or fully rerun just for this taxonomy change.
Their lyrics and non-bridge section decisions are already calibrated and should
be preserved.

Use the targeted bridge migration script:

First run `supabase_section_type_interlude_constraint.sql` in Supabase SQL
Editor. Then run:

```bash
cd "/Users/esther/Desktop/New project"
python3 annotation_tool/reclassify_existing_bridge_sections.py
python3 annotation_tool/reclassify_existing_bridge_sections.py --apply
```

Rules:

- Only rows whose current `section_type` is `bridge` are candidates.
- The script checks the latest `auto_sections` run per song and the current
  `lyric_lines`.
- A bridge becomes `interlude` only when evidence says no-vocal/low-vocal/
  instrumental transition, or when the section has no effective lyric lines.
- Lyrical, contrasting, or uncertain bridge sections stay `bridge`.
- Linked `human_annotations` are updated with the same bridge-to-interlude
  decision so the QC table and readable view stay aligned.
- The script does not create extra Supabase tables or views.

## Input Convention

Put newly collected files here:

```bash
/Users/esther/Desktop/New project/songs/songs2
```

Files collected under `songs/songs2` may have inconsistent names as long as
the title and artist are both present. Do not require manual filename cleanup at
the collection step.

After dedupe and audio validation, rename files into the normalized final
format before moving them into `songs/`:

```text
歌名-歌手.mp3
```

Do not keep `.wav`, `.downloading`, or cloud placeholder files in the final `songs/` root.

## Batch Steps

### 1. Inspect And Deduplicate

Compare `songs/` and `songs/songs2/` by normalized title/artist and by content hash.

Rules:

- Never delete existing files already in `songs/`.
- If duplicate exists in both places, remove the duplicate from `songs/songs2/`.
- If a file cannot be played or decoded, move it to a quarantine folder and report the name for later re-download.
- After dedupe and validation, normalize each valid new mp3 filename to `歌名-歌手.mp3`, move it from `songs/songs2/` into `songs/`, then remove empty `songs/songs2/`.

Quarantine path used today:

```bash
/private/tmp/new_project_removed_unrunnable_20260605
```

### 2. Create A New-Batch Manifest List

Write the exact new filenames to a dated text file, one per line:

```bash
annotation_tool/today_new_songs_YYYYMMDD.txt
```

Example:

```text
遇见-孙燕姿.mp3
小幸运-田馥甄.mp3
```

### 3. Run Model Outputs Before DB Import

For `歌名-歌手.mp3` naming:

```bash
cd "/Users/esther/Desktop/New project"
python3 annotation_tool/run_model_outputs_for_songs.py \
  --song-list annotation_tool/today_new_songs_YYYYMMDD.txt \
  --filename-order title-artist \
  --no-lyric-search \
  --continue-on-error \
  --write-manifest outputs/functional_sections/today_new_songs_YYYYMMDD_generated_manifest.csv
```

Use `--import-after-each` only when Supabase connectivity is stable. If DNS is unstable, skip it and import once at the end.

Expected success signal:

```text
generated N model outputs
failed M model outputs
```

Only generated rows should be shown in the QC platform.

### 4. Import Generated Model Runs To Supabase

When the generated manifest exists:

```bash
cd "/Users/esther/Desktop/New project"
python3 annotation_tool/import_local_results.py \
  --manifest outputs/functional_sections/today_new_songs_YYYYMMDD_generated_manifest.csv \
  --skip-song-discovery
```

Expected success signal:

```text
imported local_xxxxxxxxxxxx run=...
```

If a song already exists and should receive a new final machine version after lyric correction, use:

```bash
cd "/Users/esther/Desktop/New project"
python3 annotation_tool/import_local_results.py \
  --manifest outputs/functional_sections/today_new_songs_YYYYMMDD_generated_manifest.csv \
  --skip-song-discovery \
  --force-model-output
```

`--force-model-output` creates a new latest `model_runs` row and imports new `auto_sections`. It does not delete `human_annotations` or `section_reviews`. It sets `songs.status` back to `pending_review`.

### 5. Verify Supabase Before QC

For a successful batch:

- `songs.status` should be `pending_review` for new songs.
- each imported song should have at least one `model_runs` row.
- the latest `model_runs.id` should have `auto_sections`.
- `lyric_lines` should exist when a lyric/transcript CSV exists.
- the Netlify QC page should show the songs in `未质检`.

QC URL:

```text
https://musicva.netlify.app
```

### 6. Lyric Correction

Goal: for future new songs, correct only lyric text, not timestamps. Do not
retroactively recalibrate already-imported songs unless the user explicitly
requests that specific song.

Allowed:

- Replace wrong Chinese characters after the existing timestamp.
- Correct Whisper homophone, near-homophone, and phrase recognition errors when
  the intended lyric is clear from reliable sources, for example `我脚奥的飞远`
  -> `我骄傲地飞远`, or `原谅我这一生慌荡不羁爱自` -> `原谅我这一生放荡不羁爱自由`.
- Use reliable lyric sources for reference.
- Keep line count and line order when editing via the frontend bulk lyric editor.

Not allowed:

- Changing timestamps during lyric correction.
- Changing sections during lyric correction.
- Copying one song's lyric into many songs without per-song verification.

After lyric correction, decide whether sectioning depends on the corrected lyric:

- If only a few characters changed and structure is unaffected, no rerun is needed.
- If repeated phrases, chorus labels, paragraph grouping, or line content changed substantially, rerun model for that song and import with `--force-model-output`.

### 7. Rerun Final Machine Version After Lyric Correction

For corrected timed lyric CSV/LRC files, use the orchestrator's lyric input support (`--lyrics-csv`, `--lyrics-file`, or `--lyrics-url`) when running per song. Then import the resulting manifest with `--force-model-output`.

The final state before QC should be:

- latest model run uses corrected lyrics,
- latest `auto_sections` are the machine version,
- song is visible as `未质检`,
- prior section review counts are preserved unless the user intentionally resets them.

## Frontend Change And Netlify Rule

Every platform code change must update both:

- local source under `annotation_tool/`
- publish folder under `netlify_publish/`

Run local checks:

```bash
cd "/Users/esther/Desktop/New project"
node --check annotation_tool/app.js
git diff --check -- annotation_tool/app.js annotation_tool/index.html annotation_tool/styles.css
python3 annotation_tool/build_netlify_publish.py
```

Then deploy from the user's Terminal if Codex cannot deploy:

```bash
cd "/Users/esther/Desktop/New project"
npx netlify deploy --prod --dir=netlify_publish
```

Live site:

```text
https://musicva.netlify.app
```

## Today-Specific Logic Notes

- `标为未质检` must only update `songs.status` to `pending_review`.
- It must not delete or reset `section_reviews`.
- It must not delete or reset `human_annotations`.
- Section-level review counts must stay visible after marking a song pending.
- The page must not auto-change a manually pending song back to reviewed merely because all sections already have review records.
