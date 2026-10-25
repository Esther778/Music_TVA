# Blind Validation Stage 1 Audit

Report time: 2026-07-20 (Asia/Shanghai)

Research progress: Phase 1 VA teachers are frozen; Phase 2 blind validation
input audit is complete. No 15-song sample has been selected and no Phase 2 VA
inference has started.

## Blind-Safe Input Boundary

Frame-level VA inference may read only the frozen audio file. After frame VA
outputs are frozen, section aggregation may read song ID, section order, start
time, end time, and duration. Section type, lyrics, review history, confidence,
and future human VA/change/climax judgments remain hidden until evaluation.

## Verified Corpus

```text
reviewed Chinese songs: 99
unique section boundaries: 1097
songs with boundaries: 99
local audio files present: 99
missing or invalid boundaries: 0
nonpositive song durations: 0
```

All 99 songs pass the following checks:

- section order is unique and consecutive;
- the first section starts at 0 seconds;
- adjacent sections have no gap or overlap above 0.05 seconds;
- stored section duration matches end minus start;
- the final section end matches song duration within 0.5 seconds.

The previously detected 11-second gap in `倒数 - G.E.M.邓紫棋` was corrected
by the user and passed the rerun audit.

## Database Field Caveats

The `song_review_summary.human_annotation_count` value differs from the actual
`human_annotations_readable` row count for 82 songs, and `is_reviewed` is false
for 66 songs even though all 99 base song statuses are `reviewed`. Therefore:

- use `songs.status = reviewed` for whole-song review status;
- use actual unique rows from `human_annotations_readable` for section counts;
- do not use the summary count or `is_reviewed` flag for sampling eligibility.

The database `source` value is `local_csv` for every song and cannot distinguish
human music from AIGC music. A separate `source_type` metadata field is required
before the final human/AIGC comparison.

No human section-level VA, change-direction, or climax judgment is present in
the blind-safe input. Those judgments must be collected independently after
model outputs are frozen if they are used for validation.

## Decision

Stage 1 passes. The 99-song corpus is eligible for Stage 2 sampling under the
field rules above.
