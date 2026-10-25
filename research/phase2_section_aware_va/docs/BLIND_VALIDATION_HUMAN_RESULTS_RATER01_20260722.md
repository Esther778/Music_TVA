# Blind Human Validation Results: Rater 01

Report date: 2026-07-22

Research status: Stage 5B is complete for one rater across all 15 frozen songs.
A second independent rater and inter-rater reliability are still pending.

## Provenance And Audit

```text
human response=data/blind_validation/stage5/human_annotations/吕靖怡.csv
human response SHA-256=46a969e198b0986062a1e1966ef77db71deda883f0c0ae9cd27d8545a23a7d61
songs=15/15
sections=179/179
unique human_annotation_id=179
boundary mismatch=0
invalid categorical values=0
rater_id=rater_01
audit=pass
```

The frozen model outputs and section types were joined only after completion of
the blind ratings. No Teacher parameter was changed using these ratings.

## Evaluation Definitions

- Human low/mid/high ratings were mapped to ordinal 0/1/2.
- Spearman correlation was used because human ratings are ordinal.
- Within-song macro metrics calculate one correlation per song and then average
  across songs.
- Pooled metrics combine all 179 sections and therefore include between-song
  calibration effects.
- Change correlation compares human down/stable/up (-1/0/1) with the frozen
  change in model mean Arousal between adjacent sections.
- Peak evaluation uses the already-frozen single maximum-mean-Arousal candidate;
  it was not redefined after seeing human peaks.
- RMSE, MAE, CCC, and categorical accuracy are not reported because the human
  three-level ordinal ratings are not continuous VA ground truth.

## Main Results

| Evaluation | Valence | Arousal |
|---|---:|---:|
| Mean within-song Spearman | 0.345 | 0.503 |
| Bootstrap 95% CI of mean | [0.196, 0.484] | [0.385, 0.606] |
| Pooled section Spearman | 0.419 | 0.589 |
| Within-song-centered pooled Spearman | 0.438 | 0.552 |
| Between-song mean Spearman | 0.400 | 0.732 |

Interpretation: Arousal is consistently stronger than Valence. The frozen
Teachers capture a moderate portion of section-level emotional ordering, with
stronger cross-song Arousal ordering. Valence remains only weak-to-moderate and
must not be presented as precise section-level human ground truth.

## Adjacent-Section Change

```text
pooled Spearman(human change, model Arousal delta)=0.359
mean per-song Spearman=0.371
bootstrap 95% CI=[0.251, 0.487]
non-stable sign matches=84/117=71.8%
```

The model captures the direction of many clear rises and falls, but this is a
supporting trend measure rather than high-accuracy transition classification.

## Perceived Climactic Sections

```text
human positive sections=61/179=34.1%
mean per song=4.07
range per song=2-7
section type after reveal: Chorus=57, Pre-chorus=2, Interlude=2
frozen top-Arousal exact human-peak hit=9/15=60.0%
frozen top-Arousal adjacent hit=13/15=86.7%
```

There are 20 adjacent pairs where both sections were marked positive. The label
therefore represents a perceived climactic/high-intensity region, which may
span multiple sections, not a strict mathematical local maximum. This wording
should be used in the paper.

The two songs without an exact or adjacent hit were `欧若拉` and `New Boy`.
These should receive qualitative error analysis without changing the frozen
Teacher or peak rule.

## Chorus Versus Verse

All 15 songs contained both Chorus and Verse sections.

```text
human Valence Chorus lift=+0.813 ordinal levels; positive in 14/15; p=0.000677
human Arousal Chorus lift=+1.069 ordinal levels; positive in 14/15; p=0.000942
human-model lift correlation: Valence=0.147, Arousal=0.367
```

The human ratings strongly support a general Chorus lift. The frozen model
previously found the same group-level direction, but the low per-song lift
correlations show that it does not reliably reproduce the magnitude of that
lift for each individual song.

## Scientific Conclusion

The current evidence supports using pseudo-VA for exploratory section-aware
structure analysis, especially Arousal progression, directional change, and
coarse climax-region localization. It does not support treating pseudo-VA as
human ground truth or claiming precise section-level Valence prediction.

The result is suitable as a human-validation component only with these limits:

- one rater and 15 songs;
- three-level ordinal judgments;
- no inter-rater reliability yet;
- model and human values are compared by ordering, not absolute regression
  error;
- all inferential claims remain exploratory until a second rater is added.

## Reproducible Outputs

```text
scripts/analyze_blind_human_validation.py
outputs/human_validation/rater_01_evaluation/summary.json
outputs/human_validation/rater_01_evaluation/per_song_metrics.csv
```

Output hashes:

```text
summary.json=bc742cf002f3a145fc4fa6039996fa7b90a8e674ba04050cf5b222c13c8af333
per_song_metrics.csv=35118d3d757ee5e1832ac9b6b88a994f4cd61b9835dbbe8a827f557a05816b8f
```
