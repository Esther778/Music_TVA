# Section Annotation Console

Static Supabase-backed review UI for pop-song section segmentation.

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

There are no disposable database tables at this stage. The only denormalized
object is `section_review_queue`, and it is intentionally a view so it does not
duplicate data.

## Import Local Results

Song import configuration lives in:

```text
annotation_tool/songs_manifest.csv
```

Add new songs there, then run from the repository root:

```bash
.venv/bin/python annotation_tool/import_local_results.py --replace
```

Prototype security note: current Supabase RLS policies allow anon read/write so
the local browser can save annotations. Before deployment, replace them with
authenticated reviewer policies.
