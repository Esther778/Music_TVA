# GPT analysis data package: top60 quota 100 songs

This is a data-only package. It contains no audio and no intermediate model feature cache.

Use `00_index/PACKAGE_SCOPE.json` first, then `00_index/PACKAGE_CONTENTS.csv`.

Best starting tables:

- `08_analysis_ready/song_analysis_ready.csv`
- `08_analysis_ready/section_analysis_ready.csv`
- `08_analysis_ready/normalized_complete_song_trajectories_101pts.csv`
- `09_derived_tables/va_paired_section_tests.csv`
- `09_derived_tables/va_transition_summary.csv`
- `09_derived_tables/musicological_collapsed_form_sequences_by_song.csv`
- `07_paper_statistics/PAPER_STATISTICS_SUMMARY_ZH.md`
- `09_derived_tables/TOP60_QUOTA_100_VA_MUSIC_FINDINGS_ZH.md`

Interpretation boundary: VA values are teacher-model pseudo-labels for musicological analysis, not direct human emotion ground truth.
Generated at 2026-08-15T12:12:09+08:00.
