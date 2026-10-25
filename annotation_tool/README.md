# Section Annotation Console

Static Supabase-backed review UI for pop-song section segmentation.

Current section labels:

```text
intro, verse, pre_chorus, chorus, bridge, interlude, outro
```

`interlude` is the no-vocal or mostly no-vocal instrumental transition/breath.
`bridge` is the contrasting or turning section, usually with lyrics or a clear
new musical/emotional idea.

Run from the repository root:

```bash
python3 -m http.server 5177
```

Open:

```text
http://localhost:5177/annotation_tool/
```

The app uses the Supabase publishable key and writes to these tables:

- `songs`
- `model_runs`
- `auto_sections`
- `human_annotations`
- `section_reviews`
- `lyric_lines`
- `audio_features_2hz`

## Database Shape

- `songs`: one row per song. Stores the Chinese title, artist, local audio path,
  duration, and review status.
- `model_runs`: one row per automatic pipeline run for a song. Keeps model and
  pipeline versions, params, and import notes so different runs can be compared.
- `auto_sections`: model-generated section candidates. This is the machine
  output: boundary, type, evidence, confidence, and review flag.
- `lyric_lines`: sentence-level timed lyrics from Whisper or curated timed
  lyrics. The annotation UI can use it as lyric context.
- `human_annotations`: final or reviewer-adjusted section labels. This is the
  human-approved result table.
- `section_reviews`: action log for edits, approvals, rejects, and unsure
  decisions. It is empty until reviewers start saving work, but should be kept
  for auditability.
- `audio_features_2hz`: reserved for 2 Hz valence/arousal and acoustic feature
  sequences. It is currently empty, but matches the planned model-training data
  path.
- `section_review_queue`: view, not a base table. It joins `songs`,
  `model_runs`, and `auto_sections` so Supabase can show Chinese song titles and
  sorted review rows instead of raw ids.
- `song_review_summary`: view, not a base table. It powers the web selector and
  shows whether each song has been fully written to `human_annotations`, plus
  review action counts.
- `human_annotations_readable`: view, not a base table. It joins
  `human_annotations` to `songs` so Supabase inspection shows readable song
  titles and chronological section rows for every section submitted from the QC
  desk.

There are no disposable database tables at this stage. Do not add extra
inspection views unless explicitly required; `human_annotations_readable` is the
single readable human-annotation view.

Human review results are overwritten by `source_auto_section_id`: saving the
same section again updates `human_annotations` instead of creating another final
row. `section_reviews` remains append-only so review counts and edit history are
available.

Times are still stored as numeric seconds for sorting, playback, and training,
with generated display columns such as `start_time_label`, `end_time_label`, and
`duration_label` for `M:SS` display.

## Import Local Results

Song import configuration lives in:

```text
annotation_tool/songs_manifest.csv
```

For new songs, do not insert database rows before model output exists. Follow
[`../NEW_SONG_PIPELINE_WORKFLOW.md`](../NEW_SONG_PIPELINE_WORKFLOW.md): run
`annotation_tool/run_model_outputs_for_songs.py` first, write a generated
manifest, then import that generated manifest:

```bash
python3 annotation_tool/import_local_results.py \
  --manifest outputs/functional_sections/today_new_songs_YYYYMMDD_generated_manifest.csv \
  --skip-song-discovery
```

Use `--replace` only for an intentional destructive rebuild of known test data.
Do not use it for normal batch imports or lyric calibration.

## Existing Bridge To Interlude Migration

For the 94 songs that were already in Supabase before `interlude` existed, do
not rerun the full model or regenerate lyrics. Reclassify only the existing
`bridge` sections:

```bash
python3 annotation_tool/reclassify_existing_bridge_sections.py
python3 annotation_tool/reclassify_existing_bridge_sections.py --apply
```

The first command writes a CSV report without changing Supabase. The second
updates matching `auto_sections` and linked `human_annotations` from `bridge` to
`interlude` when the evidence indicates a lyric-free or low-vocal instrumental
transition.

Before saving or importing `interlude` rows, the Supabase section-type check
constraints must allow the seven-label taxonomy. Run:

```sql
-- Supabase SQL Editor
-- see ../supabase_section_type_interlude_constraint.sql
```

Prototype security note: current Supabase RLS policies allow anon read/write so
the local browser can save annotations. Before deployment, replace them with
authenticated reviewer policies.
