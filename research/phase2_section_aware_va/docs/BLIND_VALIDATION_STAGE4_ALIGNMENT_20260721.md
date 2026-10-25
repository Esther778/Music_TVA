# Blind Validation Stage 4 Section Alignment

Report time: 2026-07-21 00:06 (Asia/Shanghai)

Research progress: the frozen Stage 3 frame-level pseudo-VA has been aligned
to the frozen Stage 2 boundaries. Section types and human emotion/climax
answers remain withheld. Stage 5 has not started.

## Input Boundary

Stage 4 used only:

- the frozen 15-song manifest;
- the frozen 179 section boundary intervals and their identifiers;
- the audited 7375 frame-level pseudo-VA rows from Stage 3.

It did not query Supabase or read section types, lyrics, review confidence,
human VA levels, change-direction labels, or climax answers. The frozen
boundary canonical SHA-256 was rechecked as:

```text
4327ad8bef8d4a3c0f9bcba71c4a5fd3ab9743a77c29c2b03ebac8af6d5ca7f1
```

## Alignment Rule

Each 0.5-second frame is represented by its frame-center timestamp `t` and is
assigned to exactly one half-open boundary interval:

```text
section_start_sec <= t < section_end_sec
```

This prevents a frame exactly on a shared boundary from being assigned to both
adjacent sections. Frozen boundaries were checked for continuous order,
positive duration, and complete section numbering before alignment.

## Derived Features

Each section stores:

- duration, frame count, and normalized song position;
- mean, standard deviation, minimum, and maximum Valence/Arousal;
- entry, exit, within-section delta, and linear slope for each dimension;
- peak/trough values and their absolute times;
- mean VA quadrant and distance from the VA origin;
- signed change and Euclidean VA contrast to adjacent sections.

Each song stores:

- global mean and range of Valence/Arousal;
- start-to-end section change;
- maximum adjacent-section contrast and its transition;
- total section-level VA path length;
- the section with maximum mean Valence;
- the section with maximum mean Arousal and its normalized position;
- the complete-curve Arousal peak and normalized time.

`climax_section_order_by_mean_arousal` is an operational model-derived
candidate only. It means "section with the highest mean predicted Arousal" and
must not be reported as a correct human-perceived climax before Stage 5.

## Results And Audit

```text
songs=15
sections=179
frames=7375
assigned frames=7375
unassigned frames=0
overlap frames=0
sections without frames=0
```

Independent consistency checks passed:

- frame counts sum exactly across sections and songs;
- `(song_id, section_order)` and annotation identifiers are unique;
- previous/next section transition values reproduce each other exactly;
- every model-derived climax index exists in that song;
- section type is `withheld_until_stage5` in all 179 rows;
- undefined first/last transition values remain missing rather than being
  falsely encoded as zero;
- all other numeric features are finite.

Observed feature ranges are descriptive diagnostics, not validation accuracy:

| Feature | Minimum | Median | Maximum |
|---|---:|---:|---:|
| Frames per section | 8 | 38 | 110 |
| Within-section Valence std | 0.000533 | 0.020418 | 0.113639 |
| Within-section Arousal std | 0.002660 | 0.029152 | 0.185205 |
| Adjacent-section VA contrast | 0.010318 | 0.107976 | 0.734201 |

## Outputs

```text
data/blind_validation/stage4/blind_sample_15_frame_section_alignment.csv
SHA-256=503c1aa80947ebd7c2d7409f48c255525dd7d6abcf4756d8955c0f55bf445392

data/blind_validation/stage4/blind_sample_15_section_features.csv
SHA-256=26867aebacf17beb475ff05495317816fabd2fb3ded92e8c7ad2cb8edb2884ec

data/blind_validation/stage4/blind_sample_15_song_structure_features.csv
SHA-256=4458666f5742d229e3c13b33cd3028c79a8730721adb9c939f79da93212d6e25

data/blind_validation/stage4/stage4_alignment_audit.csv
SHA-256=78b53555e1fe6473cfccba2f60f39c0e60f5ad71616664342b6efcb2d09dd231
```

Reproducible command:

```bash
.venv/bin/python research/phase2_section_aware_va/scripts/align_blind_stage4_sections.py
```

## Decision

Stage 4 passes. Stop before Stage 5. The next stage may reveal structural
section types and any available human validation fields, join them by the
frozen `human_annotation_id`, and calculate agreement without tuning the
Teacher or changing the frozen 30-second deployment rule.
