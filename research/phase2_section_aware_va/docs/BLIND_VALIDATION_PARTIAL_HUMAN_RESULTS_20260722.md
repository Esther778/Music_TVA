# Partial Human Validation Preview

Date: 2026-07-22

Status: exploratory preview only. Two of fifteen songs are complete, with one
rater. These results must not be used to tune either frozen VA Teacher or to
make confirmatory paper claims.

Superseded on 2026-07-22 by the complete 15-song report:
`docs/BLIND_VALIDATION_HUMAN_RESULTS_RATER01_20260722.md`.

## Input Audit

```text
response_file=data/blind_validation/stage5/human_annotations/吕靖怡.csv
rows=179
unique human_annotation_id=179
complete songs=2
complete sections=21
partially filled other songs=0
missing joins=0
maximum boundary error=0.0 s
invalid categorical values=0
rater_id=rater_01
```

The two complete songs are `春娇与志明` and `诀爱`.

## Exploratory Metrics

Human low/mid/high levels were mapped to ordinal 0/1/2 only for Spearman rank
correlation. Model section means were generated and frozen before accessing
human emotion labels.

| Song | Sections | Valence rho | Arousal rho | Non-stable change sign | Human peak sections | Frozen model top-A section |
|---|---:|---:|---:|---:|---|---|
| 春娇与志明 | 9 | 0.456 | 0.811 | 4/6 | 4, 7, 9 | 9, exact hit |
| 诀爱 | 12 | 0.681 | 0.775 | 6/9 | 5, 6, 10, 11 | 12, adjacent to 11 |

Mean within-song Spearman:

```text
Valence=0.569
Arousal=0.793
```

Pooled across the 21 sections:

```text
Valence=0.076
Arousal=0.780
```

The gap between within-song and pooled Valence correlation is an early warning
of possible cross-song Valence calibration error. With only two songs it is not
yet possible to distinguish calibration error from rater/song sampling effects.

After emotion ratings were complete, section types were joined by frozen
`human_annotation_id`. All seven human-marked emotional peaks were Chorus
sections. The frozen maximum-mean-Arousal candidate exactly matched a human
peak for `春娇与志明`; for `诀爱`, it selected the following Outro and was
adjacent to the last human-marked Chorus peak.

## Interpretation Boundary

- Arousal section ordering is promising in both completed songs.
- Valence captures some within-song ordering but currently shows an unstable
  between-song scale.
- Change-direction performance is not currently interpretable because its
  original on-screen definition was ambiguous. The two completed songs must be
  reviewed under the clarified rule: compare current-section Arousal/energy
  only with the immediately preceding section in the same song.
- The multiple Chorus peaks support the corrected zero/one/multiple climax
  protocol and contradict the superseded forced-single-climax design.
- RMSE, MAE, CCC, accuracy, confidence intervals, and inferential claims are
  not appropriate against one rater's three-level ordinal labels at n=2 songs.
- Final analysis requires all 15 songs and preferably a second independent
  rater. Model parameters and peak-detection rules must remain frozen.
