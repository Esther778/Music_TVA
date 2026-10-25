# Blind Section-Level Emotion Annotation Guide

Updated: 2026-07-25

Current 30-reference file for a fresh annotation round:
`outputs/human_validation/blind_sample_15_human_validation_console_anchored_v2_30refs.html`.

Open the HTML file directly in a browser. Each of its 179 frozen section rows
has an integrated player with play/pause, a seekable progress bar, and
current/total time. Required fields use single-click segmented buttons,
section types remain hidden, progress is saved in the browser, and the finished
annotations can be exported as CSV. The accompanying `audio_clips/`,
`calibration_candidates/`, and `calibration_candidates_v2_supplement/`
directories must remain beside the HTML file so its relative audio paths
continue to work.

This page uses the independent browser storage key
`musicva_blind15_human_validation_anchored_v2_30refs` and does not import
answers from the original or anchored-v1 pages. A result may be described as a
30-reference annotation only after all 179 sections have been freshly rated in
this page. The original page, anchored-v1 page, and completed response files
remain separate historical/formal artifacts and must not be overwritten.

The 2026-07-24 delayed-review result was collected with anchored-v1 (12
references, two per dimension-level group). It remains a sensitivity result
and was not retroactively relabeled. A same-rater 30-reference review was
completed and archived separately on 2026-07-26; its interpretation and
limitations are recorded in
`docs/BLIND_VALIDATION_HUMAN_RESULTS_30REF_REVIEW_20260726.md`.

The earlier Excel workbook is retained only as a historical artifact and must
not be used because it exposes section types and implements the superseded
exactly-one-climax rule:
`outputs/human_validation/blind_sample_15_human_section_emotion_annotation.xlsx`.
The CSV below is the machine-readable source template, not the preferred
manual interface:
`data/blind_validation/stage5/blind_sample_15_human_emotion_annotation_template.csv`.

The sheet intentionally contains no model Valence/Arousal predictions. Do not
open the model result tables while rating. Listen to each complete song in
order and fill the following columns for every section.

Use the round `▶` button in each row. While the section is playing it changes
to a pause button; the progress bar can be dragged to seek. The audio has
already been cropped to that row's `start_time_sec` and `end_time_sec`.
Required categorical fields can only be set through the displayed button
choices; the selected value is highlighted in blue.

Current study scope is human-created Chinese popular music only; AIGC comparison
is not part of this experiment.

## Allowed Values

```text
human_valence_level = low | mid | high
human_arousal_level = low | mid | high
human_change_from_prev = up | down | stable
is_climax = 0 | 1
rater_id = stable anonymous identifier, for example rater_01
notes = optional short uncertainty note
```

For the first section of every song, leave `human_change_from_prev` as
`not_applicable`.

## Rating Meaning

- Valence: perceived negative-to-positive emotional quality of the music in
  that section, not whether the listener personally likes it.
- Arousal: perceived calm-to-energetic intensity of the music in that section.
- Previous section: the immediately preceding section in the same song,
  meaning current `section_order - 1`. For example, the previous section of
  section 4 is section 3, not the previous occurrence of the same section type.
- Change: compare only perceived Arousal/energy against that immediately
  preceding section, not positive/negative Valence. Use `up` for a clear
  increase, `down` for a clear decrease, and `stable` when the difference is
  not clearly perceptible.
- Local emotional peak (`is_climax`): mark whether the current section feels
  like a clear emotional peak relative to its surrounding sections. A song may
  contain zero, one, or multiple sections marked `1`. This is not Chorus
  recognition, and not every Chorus must be marked `1`; section types remain
  hidden until human ratings are frozen.

## Frozen Cross-Song Anchors V2

The v2 page includes five human-confirmed audio references for every
dimension-level combination. All 30 anchor songs are outside the frozen
15-song blind-validation sample. Teacher predictions were not used to select
or label them. The original 12 references came from an exported review; the 18
supplemental references were confirmed by the user in chat on 2026-07-25, with
no review CSV exported.

```text
Valence low:  消愁 / 同桌的你 / 富士山下 / 后来 / 嘉宾
Valence mid:  阴天 / 遇见 / 十年 / 江南 / 起风了
Valence high: 爱你 / 稻香 / 一路生花 / 童年 / 红日
Arousal low:  红豆 / 因为爱情 / 月亮代表我的心 / 大鱼 / 我只在乎你
Arousal mid:  小幸运 / 勇气 / 明天，你好 / 四季予你 / 情歌
Arousal high: 逆战 / 最炫民族风 / 奔跑 / 海阔天空 / 倔强
```

Before rating, review references across low/mid/high for the current dimension
(15 clips in total, five per level). Use them as a shared cross-song scale, not
as exact timbre, genre, vocal, or arrangement templates. Valence and Arousal
must be judged independently.
The reference scale applies only to absolute low/mid/high ratings;
`human_change_from_prev` remains a within-song comparison against the
immediately preceding section.

Frozen provenance:

```text
outputs/human_validation/calibration_candidates/frozen_va_anchors_v1.json
outputs/human_validation/calibration_candidates/frozen_va_anchors_v1.csv
outputs/human_validation/calibration_candidates/human_reviews/
outputs/human_validation/calibration_candidates_v2_supplement/frozen_va_anchors_v2_30clips.json
outputs/human_validation/calibration_candidates_v2_supplement/frozen_va_anchors_v2_30clips.csv
outputs/human_validation/calibration_candidates_v2_supplement/human_reviews/
```

## Quality Rules

- Rate from audio perception; lyrics may be heard naturally but should not be
  separately looked up during rating.
- Rate each song in section order. Do not use or look up section-type labels.
- Complete all 179 rows again in the anchored page. Do not copy ratings from
  the unanchored pilot merely to save time.
- Replay an uncertain section once before using `notes`.
- Do not alter song identity, annotation ID, section order, type, or times.
- Prefer two independent raters. If only one rater is available, state that
  limitation in the paper.
- With two raters, preserve separate files and compute inter-rater agreement
  before comparing with pseudo-VA.

After the exported CSV is returned, Stage 5 can calculate ordinal
Valence/Arousal association, change-direction agreement, exact and adjacent
climax hit rates, and rater agreement where available.

Save exported responses to:

```text
data/blind_validation/stage5/human_annotations/
```

Keep one file per rater and annotation version. For a fresh v2 round, use for
example
`blind_sample_15_human_annotations_anchored_v2_30refs_rater_01.csv`.
