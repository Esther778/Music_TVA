# Section QC Workflow Status

Last updated: 2026-06-07

## Goal

Build a Supabase-backed web tool for human QC of automatic pop-song section segmentation.
The tool is for reviewing machine-generated `intro / verse / pre_chorus / chorus / bridge / interlude / outro`
boundaries, correcting mistakes, and writing final human-reviewed results back to Supabase.

## Current Data Scope

- Current Supabase production set: 94 reviewed/QC-ready songs.
- All existing lyric rows were calibrated to cover from 0 seconds to the final
  second, using `前奏` and `尾奏` where no lyric is present.
- Existing songs should not be fully retranscribed or fully rerun just to add
  `interlude`. Use `annotation_tool/reclassify_existing_bridge_sections.py` to
  revisit only existing `bridge` sections and split them into `bridge` or
  `interlude`.
- If a song appears as `0/0` in the QC console, first check whether it has
  local `*_agentic_sections.csv` rows. Re-import the generated manifest without
  `--force-model-output`; the importer preserves songs that already have
  `auto_sections` and fills only songs missing machine sections.
- English transcripts must not be filtered by a CJK-only predicate. Transcript
  rows with Chinese characters or Latin letters are valid lyric rows.

## How To Open

Preferred URL:

```text
http://127.0.0.1:5177/annotation_tool/index.html?v=20260530-local19
```

Start the static server from the repository root if needed:

```text
python3 -m http.server 5177
```

Avoid using `file:///.../annotation_tool/index.html` for normal testing.
The page can render in file mode, but browser path resolution is different:
`/songs/name.mp3` becomes `file:///songs/name.mp3`, which is not the project `songs/`
folder. The frontend now includes a file-mode fallback, but the HTTP URL is still the
reliable path.

## Current UI Behavior

- Left sidebar shows song counts, searchable song directory, and selected song.
- Clicking `全部歌曲 / 未质检 / 已质检` switches the song directory and keeps the search box
  available for fuzzy matching by song title, artist, or song id.
- The refresh button has a hover tooltip: `刷新 Supabase 数据`.
- Main area shows:
  - custom audio transport
  - section timeline
  - zoom control
  - compact synced lyrics panel
  - section result cards
  - current-section editor
- Clicking a section card selects it, seeks to that section start, and plays a short preview.
- `上一段 / 下一段` are adjacent in the editor and jump-play the target section.
- Section card labels use `Section 1`, `Section 2`, etc. instead of `#1`.
- Section rows support pointer drag reordering from the left handle, plus up/down step buttons
  for precise moves.
- Automatic whole-song completion persists the current visible section order before setting the
  song to reviewed.
- Lyrics are displayed through a simplified-Chinese conversion layer, limited to a compact
  three-line viewport, and support double-click inline correction. The expanded lyric editor
  highlights the current playback line, supports lyric start/end/text edits plus row add/delete,
  can play/pause audio, uses Space for playback when not typing, seeks from clicked lyric rows,
  inserts new rows under the current lyric row, moves Enter to the next row, scrolls by lyric
  start timestamps during playback. The compact lyric viewport follows the current line, while the
  expanded list only updates highlight and does not auto-scroll. It links adjacent lyric row times
  while editing, can be moved and proportionally resized, and saves the resulting rows back to
  `lyric_lines` with a post-save database readback check.
- Lyrics and section/editor local undo use Command+Z/Ctrl+Z and keep at most 10 steps.
- Section delete undo restores the deleted/hidden section and prior ordering locally.
- The section timeline supports horizontal drag scrolling when zoomed in.
- The main audio panel is not fixed. Once the transport scrolls above the viewport, a compact
  bottom player appears, following the QQ Music style requirement.
- Each section can be submitted individually.
- A green `提交` button means that section is not submitted yet. A song must not
  display as `已质检` while any active section still has a green `提交` button.
- Clicking `提交` on an unchanged machine section is still a human confirmation.
  It must upsert `human_annotations` with the machine values and append one
  `section_reviews` row.
- `human_annotations` and `section_reviews` are written for every submitted
  active section: machine-confirmed, human-edited, locally added, and
  boundary-linked sections.
- Individually submitted human-edited sections can be submitted again; later
  submissions overwrite the prior final annotation for that section.
- The whole song can be submitted only after every active section is submitted.
- `重置当前section` deletes that section's human annotation/review records and reloads the current
  song from Supabase. `重置整首歌` deletes the song's human annotations/reviews, resets local UI
  state, and reloads the current song.
- The visible editor keeps only section type and start/end time. QC status, confidence, and
  reviewer comment are now inferred by the action to keep the page simpler.
- Editor action buttons are arranged in a 3-column grid.
- Locally added sections are titled `新增 Section 1`, `新增 Section 2`, etc. until submitted;
  after submission the local added-section label is preserved in the section card.
- Section type options include `interlude`.
- If the user submits a section without changing the machine type/start/end, or
  changes it back to the exact machine values, the section card displays the
  machine result line and the saved QC status is treated as machine accepted.
  This confirmation still writes `human_annotations`/`section_reviews` and is
  future evidence that the machine section was correct.
- Section add/delete animations are deliberately slightly slower so the reviewer
  can see what changed, but should still feel quick.
- Sections can be reordered by drag.
- Add/delete section operations are committed operations. They write Supabase
  immediately and must read back successfully before the UI shows success.
- Drag reordering moves section functions into the reordered chronological
  slots. The visible start/end times are remapped to keep the full song
  continuous; it must not leave a reordered row with a time range that belongs
  to its old position.
- After drag reordering, the UI must say the order is not written yet. It must
  not show a success message that implies database persistence.
- `提交整首歌` must freeze the current `song_id`, `run_id`, and active section
  snapshot before writing. Every write must use that frozen context; if song
  selection, refresh, or model_run changes during submit, the submit must stop.
- `提交整首歌` must upsert every active section into `human_annotations` in the
  current UI order before updating `songs.status = reviewed`. Deleted machine
  sections are persisted as `section_reviews.action = delete` tombstones when
  the delete operation is performed and are excluded on the next load. Only
  after the final database writes and reload succeed may the UI show
  `整首歌提交成功`.
- `提交本段` must read back `human_annotations` and `section_reviews` before
  showing success.
- `提交整首歌` must read back `human_annotations_readable` before showing
  success. If the readable view is not updated, show an error instead of a
  success toast.
- On refresh, section order must be reconstructed from latest
  `human_annotations.section_index` first, not from the original
  `auto_sections.section_index`, so frontend display matches backend state.
- Deleting a section writes immediately. It writes any required adjacent
  boundary sync, deletes the removed section's final annotation, writes a
  `section_reviews.action = delete` tombstone, verifies the database readback,
  and only then hides the section in the UI.
- Existing machine sections can still be submitted with adjusted start/end
  boundaries; submission automatically syncs the adjacent section boundary.
- Locally added sections cannot be submitted if they leave an empty time gap
  between themselves and the previous/next section.
- Whole-song submission requires a continuous timeline from 0 to
  `songs.duration_sec`.

## Database Tables Used

- `songs`: song metadata and local audio path.
- `model_runs`: one model run per imported segmentation result.
- `auto_sections`: machine-generated section candidates.
- `lyric_lines`: timed lyric lines.
- `human_annotations`: final per-section human result, including unchanged
  machine sections that the reviewer explicitly confirmed.
- `section_reviews`: append-only review action history for every submitted
  section confirmation or edit.
- `song_review_summary`: view used by the UI selector.
- `section_review_queue`: view for joined review context.
- `human_annotations_readable`: readable view for Supabase inspection. Keep this
  as the only extra human-readable annotation view unless a new one is
  explicitly required. It should show only songs whose whole-song QC was
  completed (`songs.status = reviewed`). Put song title, artist, and section
  fields first. Rows are ordered by whole-song reviewed timestamp descending,
  then by section start time within each song.
- The QC dashboard's reviewed-song count must match `human_annotations_readable`.
  A song is displayed as `已质检` only when readable rows exist for that song;
  `songs.status = reviewed` alone is not sufficient because it can drift from
  the final annotation rows.

## Important Current Caveats

- `human_annotations.qc_status` may not exist in all database snapshots. The frontend sends
  `qc_status` when supported and falls back without it if Supabase rejects the field.
- New/delete section editing writes immediately. Locally added sections persist
  into `human_annotations` with no `source_auto_section_id`; deleted machine sections are excluded
  from the current review result, their matching human annotations are removed on delete, and
  a `section_reviews.action = delete` tombstone is written so reloads can preserve the deletion.
- For long audio, timeline zoom exists, but true drag-resize section boundaries is still a next step.
- When the Codex sandbox cannot reach Supabase/Netlify because of DNS, do not
  keep retrying. Give the user the exact terminal command to run outside the
  sandbox.
- Any future change to QC submit/database behavior must update this Markdown
  file in the same pass as the code change.

## Next UI Improvements

- Add direct drag handles on section boundaries.
- Add a stronger audio-load error message when a file path cannot be resolved.
- Add true browser-level click verification for destructive reset flows only when the user confirms
  that clearing current review data is acceptable.
