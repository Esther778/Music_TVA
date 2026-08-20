# Top60 100-song section label correction, 2026-08-20

## Scope

Two first sections were incorrectly labeled `outro` and were corrected to `intro`:

| Song | Song ID | Human annotation ID | Time | Old | New | Review ID |
|---|---|---|---:|---|---|---|
| 薛之谦《演员》 | `local_1e0b3f9bee7e` | `89665e9b-27e3-4c43-8030-43e5940f1018` | 0-25 s | outro | intro | `ca09cc86-cbf0-4a60-ab35-6ef839fa1cda` |
| 孙燕姿《我怀念的》 | `local_963a6186bd26` | `e9046098-3e28-47f5-877c-bdf361477dd4` | 0-16 s | outro | intro | `51fbf158-cb6d-4837-a57c-fa8415f7de26` |

The canonical `human_annotations` rows and the `human_annotations_readable` view were read back successfully after the update. The machine-generated `auto_sections` rows were not changed.

## Recomputed outputs

- Songs: 100
- Sections: 1046
- Frame-level VA rows: 51,448, unchanged
- Assigned frames: 51,444
- Trailing excluded frames: 4
- Outro: 100 sections across 100 songs
- Intro: 95 sections across 94 songs
- Alignment audit: pass; no overlaps and no section without VA frames

The stage-1 boundary snapshot, stage-3 aligned section features, structural statistics, paper statistics, VA/MIR/MER derived tables, and GPT data package were rebuilt. Teacher inference was not rerun because neither audio nor section boundaries changed.

## Supabase paper-table restoration

The anonymous REST upload could delete the previous dedicated 100-song run but could not recreate it because insert is blocked by RLS. The canonical human annotation tables remain correct, and the original 97-song tables were not touched.

Restore package: `musicva_top60_100_supabase_restore_20260820.zip`.

The package contains ordered SQL batches for restoring only:

- `paper_top60_quota_100_va_runs`
- `paper_top60_quota_100_va_frames`
- `paper_top60_quota_100_section_va`
- `paper_top60_quota_100_song_va`

Expected final counts are 1 run, 51,448 frames, 1,046 sections, and 100 songs.
