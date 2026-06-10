# Music Section QC Project Handoff

Last updated: 2026-06-07

This file is the first file a new agent should read. It summarizes the project
goal, current progress, experiment logic, UI behavior, and recurring user
requirements.

## Project Goal

The project has two connected goals:

1. Build a reusable pipeline for pop-song section segmentation.
2. Build a Supabase-backed human QC web tool so machine segmentation can be
   reviewed, corrected, and saved for later modeling.

The target section labels are:

```text
intro, verse, pre_chorus, chorus, bridge, interlude, outro
```

The final use case is not only the current test songs. The method and UI should
scale to many songs, eventually thousands, with human QC and database storage.

Future training work will use reviewed section labels together with 2 Hz
valence/arousal and acoustic feature sequences for BiLSTM/Transformer-style
models.

## Current Working URL

Run the static server from the repository root:

```bash
python3 -m http.server 5177
```

Open:

```text
http://127.0.0.1:5177/annotation_tool/index.html?v=20260530-local29
```

Do not use `file:///.../annotation_tool/index.html` for normal testing. It can
load the page, but local audio path behavior differs from the HTTP server.

## Current Test Song

Current main test song:

```text
song_id: anohana
title: 一期一会《未闻花名》
artist: 周深
audio_path: /songs/周深-一期一会《未闻花名》.mp3
```

The UI is connected to Supabase and uses the following tables/views:

- `songs`: song metadata and review status.
- `model_runs`: automatic segmentation runs.
- `auto_sections`: machine-generated section candidates.
- `lyric_lines`: timed lyric lines.
- `human_annotations`: final human-approved section result.
- `section_reviews`: review action history.
- `song_review_summary`: selector summary view.
- `section_review_queue`: joined queue view.
- `audio_features_2hz`: reserved for future VA/acoustic features.

## Segmentation Method Direction

The current preferred logic is:

1. Use lyrics/vocal structure to infer section function.
2. Use acoustic features to adjust boundaries and challenge NLP when evidence is
   strong.
3. Use structural constraints to avoid impossible labels.
4. Output confidence and evidence so humans can QC, rather than claiming perfect
   automatic labels.

Important modeling principle:

- Lyrics/vocals answer: what function does this segment have?
- Acoustic evidence answers: where does the boundary happen?

Rules to preserve:

- `intro` only at the beginning and should not contain full narrative lyrics.
- `outro` only at the end and should not be used for mid-song instrumental
  parts.
- `pre_chorus` is a buildup before chorus and should not be merged into verse
  when the label system supports it.
- `chorus` is the central repeated hook/theme, but high energy alone is not
  enough.
- `interlude` is a no-vocal or mostly no-vocal instrumental transition/breath
  between main sections. It smooths, buffers, or resets the song without adding
  a new lyrical idea. It may appear multiple times.
- `bridge` is an independent contrast/turn, usually later in the song and often
  before the final chorus. It usually has lyrics or a clearly distinct
  melodic/harmonic/emotional idea. Middle 8 is a bridge subtype.
- One-line rule: `interlude` is transition or breath; `bridge` is contrast or
  turn.
- Repeated lyrics are strong type evidence, but not absolute.
- Acoustic features can override semantic labeling only when confidence is
  strong enough and structurally plausible.

## Annotation Tool Purpose

The annotation tool is a human QC interface, not a marketing page. It should
feel like a compact audio/section editor:

- Machine result is shown per section.
- Current final result is shown per section.
- Human can edit section type and start/end time.
- Human can submit one section or the whole song.
- Later edits overwrite the previous final human annotation.
- Review actions are recorded in `section_reviews`.

## Current UI Requirements

General style:

- Use low-saturation pastel colors, especially pale pink page background.
- Avoid overly dark or saturated section colors.
- Keep typography consistent within one functional area.
- Avoid unnecessary explanatory text in the UI.
- Prefer compact, professional editor layout.

Audio behavior:

- Main progress bar should not be permanently fixed to the top.
- When the main transport scrolls out of view, show a compact bottom player like
  QQ Music PC, so playback is always reachable.
- Spacebar toggles play/pause unless the user is typing in an input, textarea,
  select, or editing lyrics.
- Clicking a section card should seek audio to that section start and play a
  short preview.
- Pulling/dragging the progress bar should update current time, lyrics, and
  active section.

Lyrics behavior:

- Lyrics display as simplified Chinese.
- Compact lyrics viewport should show about three lines.
- Current lyric line is highlighted during playback.
- Double-click a lyric line for inline correction.
- The lyrics panel has a `放大` button for full-song lyric editing.
- Opening the full-song lyric editor highlights and scrolls to the lyric line
  matching the current playback time.
- Full-song lyric editing can update lyric start/end time and text, add rows,
  and delete rows. Saving writes the visible row set back to `lyric_lines`.
- The full-song lyric editor has playback control; clicking a lyric row seeks
  to that row's current start time and starts playback.
- The full-song lyric editor panel is resizable. `新增歌词行` inserts under the
  currently focused/highlighted lyric row. Space toggles playback when focus is
  not inside an input.
- During playback, lyric highlighting and scrolling follow the lyric start
  timestamps. Enter in a lyric editor input keeps the current edit and moves to
  the next row's same field.
- During playback, the lyric editor updates highlighting without auto-scrolling
  the expanded lyric list, while the compact lyric viewport continues to scroll
  to the current line. The expanded lyric panel can be dragged by the header and
  proportionally resized from the bottom-right handle.
- Editing a lyric row's start/end time links adjacent rows: start updates the
  previous row's end, and end updates the next row's start.
- The lyric editor supports Command+Z/Ctrl+Z undo for up to 10 local edit steps.

Section list behavior:

- Section list should be tall enough to use most of the page down to near the
  bottom, but keep bottom whitespace.
- Section rows can be reordered with the left drag handle or up/down step
  buttons. Reordering renumbers visible sections and can be undone.
- Section/editor undo is capped at 10 local steps.
- Deleting a section pushes an undo entry. Command+Z/Ctrl+Z restores the hidden
  machine section or removed local section in the prior order.
- Section cards should show concise data only:
  - section type/title
  - section index
  - review count
  - machine row
  - current final row
  - submit button
- Do not put long reasons in each card. Put reasons in `Model evidence`.
- The currently adopted row must be obvious:
  - if machine is accepted, highlight machine row.
  - if human changed it, highlight changed row.
- Text `人工已修改` should be shortened to `已修改`.
- Locally added sections should show as `新增 Section n` before submit; after
  submit, show the chosen section type with a note that it is newly added.

Editor button behavior:

Current editor buttons:

- `跳到开头`: seek to current section start and play.
- `恢复机器`: fill editor fields from the machine section.
- When focus remains on section type controls, Space or Enter submits the
  current section. If a different type button has focus, the shortcut first
  selects that type, then submits.
- `新增 Section`: create a new local section after current section.
- `删除段落`: remove section from current reviewed result.
- `重置当前section`: delete this section's human annotation/review records and
  reload the song from Supabase.
- `重置整首歌`: delete this song's human annotations/reviews and reload the song
  from Supabase.
- `提交本段`: mark the current section submitted and persist that confirmation.
  If the values are unchanged machine values, it still writes the machine values
  to `human_annotations` and appends one `section_reviews` row because the
  reviewer explicitly confirmed it.
- `提交整首歌`: mark the song reviewed only after every active section button is
  gray `已提交`. It must upsert every active section in the current UI order
  before setting `songs.status = reviewed`.

Removed buttons:

- `上一段`
- `下一段`

These were removed because clicking them caused unwanted scrolling/jumping.

Sidebar behavior:

- `全部歌曲`, `未质检`, and `已质检` summary cards must be clickable.
- Clicking a summary card switches the song directory below.
- Song directory search must support fuzzy matching by song title, artist, or
  `song_id`.
- Refresh button must show its function on hover: `刷新 Supabase 数据`.

## Database Write Semantics

Per-section submit:

- Unchanged machine sections are human confirmations. They must upsert
  `human_annotations` and append `section_reviews`.
- If a reviewer edits a section and changes it back to the exact machine
  type/start/end, save it as machine accepted (`qc_status = machine_accepted`,
  comment `默认采用机器标注`). It is future evidence that the machine section was
  correct, not an artificial human modification.
- For machine sections, `human_annotations` is upserted by
  `source_auto_section_id`.
- For local added sections, `human_annotations` has no `source_auto_section_id`.
- Every submitted section appends one row to `section_reviews`.
- Automatic boundary linking caused by a human edit persists the affected
  neighboring section as a human-modified annotation.
- A green `提交` button is the source of truth for incomplete UI work. The
  current song must not display as `已质检` and `提交整首歌` must not write
  `songs.status = reviewed` while any active section remains green.
- Add/delete operations are committed operations. They write Supabase
  immediately and must read back successfully before the UI shows success.
  Deleted machine sections are persisted as `section_reviews.action = edit`
  rows with `after_json.deleted = true` and restored from those tombstones on reload.
- `human_annotations_readable` is the visual inspection view for QC desk output.
  It must expose song title, artist, whole-song reviewed timestamp, song status,
  and section rows, with song title, artist, and section fields at the front. It
  is the only readable inspection view and is the section-level visible source
  of truth: every submitted section must appear there, including songs that are
  still `in_review`. Sort by whole-song reviewed timestamp descending; inside
  each song, sort sections by playback time.
- The QC dashboard reviewed-song count and `已质检` labels must match
  `human_annotations_readable`. Do not
  use `songs.status = reviewed` alone to mark a song reviewed in the frontend;
  status can drift from actual final annotation rows.
- Dragging or using up/down buttons changes frontend order only. It must not
  write Supabase until the reviewer clicks `提交整首歌`.
- Dragging remaps section functions into the reordered chronological slots.
  Start/end times shown in the UI must update with the new neighbors so the
  song remains a continuous timeline before it is submitted.
- Do not show a database-style success message for drag-only changes. Use
  wording that says the order is not written yet.
- Full-song submit must freeze the current `song_id`, `run_id`, and active
  section snapshot before writing. Every annotation/review/status write must
  use that frozen context and must stop if selection changes mid-submit.
- Section-level operations must use the same frozen context rule. A section
  delete writes any required adjacent boundary sync, deletes the removed
  section's final annotation, writes the delete tombstone, and verifies readback
  before hiding the row.
- After a full-song submit, reload the current song from Supabase. Refresh
  display must use latest `human_annotations.section_index` before falling back
  to original `auto_sections.section_index`, so frontend and backend match.
- Section submits must verify `human_annotations` and `section_reviews` before
  showing a success toast. Whole-song submits must verify
  `human_annotations_readable` before showing a success toast. Database write
  without the expected readback is not a user-facing success.

Reset current section:

- Deletes `section_reviews` for that auto section and associated annotation ids.
- Deletes the matching `human_annotations` row.
- Reloads the current song from Supabase.

Reset whole song:

- Deletes all `section_reviews` for the song.
- Deletes all `human_annotations` for the song.
- Clears local additions/deletions in the UI.
- Reloads the current song from Supabase.

Important implementation note:

- Delete `section_reviews` before deleting `human_annotations`, otherwise
  foreign key constraints can make reset appear ineffective.

## Important Files

- `annotation_tool/index.html`: static HTML shell.
- `annotation_tool/styles.css`: UI layout and pastel styling.
- `annotation_tool/app.js`: Supabase logic, playback, section editing, reset,
  lyric editing.

Documentation rule:

- Any future change to QC submit/database behavior must update this Markdown
  file and `annotation_tool/WORKFLOW_STATUS.md` in the same pass as the code
  change.
- `annotation_tool/WORKFLOW_STATUS.md`: narrower status note for the annotation
  tool.
- `annotation_tool/import_local_results.py`: imports generated sections/lyrics
  into Supabase.
- `annotation_tool/songs_manifest.csv`: import manifest.
- `hybrid_structure/`: section segmentation pipeline code.
- `outputs/functional_sections/`: generated CSVs and candidates.
- `songs/`: local MP3 files.

## Verification Checklist

Before giving a link back to the user:

1. Run:

```bash
node --check annotation_tool/app.js
```

2. Confirm the local server is alive:

```bash
curl -I 'http://127.0.0.1:5177/annotation_tool/index.html?v=VERSION'
```

3. Use a cache-busting query string in the URL, such as:

```text
?v=20260530-local29
```

4. If using browser verification, check:

- no `window.__errors`
- section cards exist
- clicking a section card changes audio current time
- bottom player appears after transport scrolls out of viewport
- summary cards switch directories
- reset buttons show busy text and either complete or show a Supabase error

## Current Known Caveats

- True drag-resize section boundaries are not implemented yet.
- Reset actions are destructive for current reviewed data. Do not test reset
  destructively unless the user accepts clearing current review records.
- The browser may still show an older query string. Always send the newest HTTP
  URL.
- Netlify deployment was discussed, but the active workflow is still local
  `http.server` unless the user explicitly asks to resume deployment.
