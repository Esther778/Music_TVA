# Blind Validation Stage 2 Sample

Report time: 2026-07-20 (Asia/Shanghai)

Research progress: the 15-song blind validation sample is selected and frozen.
No VA inference has started.

## Sampling Protocol

- Population: 99 reviewed Chinese songs that passed Stage 1.
- Seed: `20260720`.
- Duration: five strata with cut points at 217.26, 239.00, 259.18, and
  275.91 seconds.
- Section count: `8-10`, `11-12`, and `13+`.
- Selection: one song from each of the 15 duration-by-section-count cells.
- Constraints: normalized artist and title are unique across the sample.
- Inputs excluded from selection: audio content, section type, lyrics, review
  history, VA predictions, and human VA/change/climax judgments.

## Frozen Sample

| # | Song | Artist | Duration (s) | Sections | Duration stratum | Section stratum |
|---:|---|---|---:|---:|---|---|
| 1 | 春娇与志明 | 街道办GDC、欧阳耀莹 | 204.60 | 9 | D1 | 8-10 |
| 2 | 诀爱 | 詹雯婷 | 189.30 | 12 | D1 | 11-12 |
| 3 | 下山 | 要不要买菜 | 173.45 | 14 | D1 | 13+ |
| 4 | 欧若拉 | 胜屿 | 223.37 | 10 | D2 | 8-10 |
| 5 | 关山酒 | 邓寓君(等什么君) | 234.95 | 11 | D2 | 11-12 |
| 6 | 执迷不悟 | 小乐哥 | 234.66 | 14 | D2 | 13+ |
| 7 | 时间煮雨 | 郁可唯 | 245.69 | 10 | D3 | 8-10 |
| 8 | 荷塘月色 | 凤凰传奇 | 249.73 | 12 | D3 | 11-12 |
| 9 | New Boy | 房东的猫 | 253.00 | 20 | D3 | 13+ |
| 10 | 如愿 | 王菲 | 265.38 | 9 | D4 | 8-10 |
| 11 | 爱情转移 | 陈奕迅 | 260.02 | 12 | D4 | 11-12 |
| 12 | 我曾遇到一束光 | 叶斯淳 | 259.76 | 13 | D4 | 13+ |
| 13 | 凉凉 | 杨宗纬、张碧晨 | 333.33 | 9 | D5 | 8-10 |
| 14 | 恋人 | 李荣浩 | 275.91 | 11 | D5 | 11-12 |
| 15 | 我怀念的 | 孙燕姿 | 289.07 | 13 | D5 | 13+ |

Validation checks:

```text
songs=15
unique song IDs=15
unique artists=15
unique titles=15
covered strata cells=15/15
frozen section boundaries=179
```

## Frozen Artifacts

```text
data/blind_validation/stage2/blind_sample_15.csv
data/blind_validation/stage2/blind_sample_15_boundaries.json
data/blind_validation/stage2/blind_sample_15_protocol.json
```

```text
manifest_sha256=1d6d624c27c6c2fd6317b6b146b97a935846ade4467553cbfb3ef9551d96aff5
boundaries_sha256=4327ad8bef8d4a3c0f9bcba71c4a5fd3ab9743a77c29c2b03ebac8af6d5ca7f1
```

Each manifest row also stores the full audio SHA-256 and per-song boundary
SHA-256. These values must be checked before Stage 3 inference.

## Decision

Stage 2 passes. Do not replace songs or regenerate this sample unless a data
integrity failure is documented. Stage 3 must use audio only and the already
frozen VA teachers; section boundaries are not supplied to frame inference.
