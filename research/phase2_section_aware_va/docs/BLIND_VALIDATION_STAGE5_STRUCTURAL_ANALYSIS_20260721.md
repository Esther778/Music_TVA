# Blind Validation Stage 5 Structural Analysis

Report time: 2026-07-21 00:23 (Asia/Shanghai)

Research progress: held-out section types have been revealed and the
structural-label analysis is complete. Direct human emotion validation remains
pending because the existing database contains no human VA, change-direction,
or climax fields.

## Scientific Scope

Music Structure Analysis commonly represents a song as non-overlapping,
functionally labelled sections such as Verse, Chorus, and Bridge. The MIR
literature also emphasizes homogeneity, novelty, repetition, and regularity as
distinct structural principles. This supports analyzing VA distributions and
transitions by section type, but it does not imply that every Chorus must be
the emotional climax.

Relevant method references:

- Nieto et al., *Audio-Based Music Structure Analysis: Current Trends, Open
  Challenges, and Applications*, TISMIR 2020:
  https://transactions.ismir.net/articles/10.5334/tismir.54
- Marmoret et al., *Barwise Music Structure Analysis with the Correlation
  Block-Matching Segmentation Algorithm*, TISMIR 2023:
  https://transactions.ismir.net/articles/10.5334/tismir.167
- Livingstone et al., *Emotional Response to Musical Repetition*, Emotion
  2012: https://pubmed.ncbi.nlm.nih.gov/21707165/

The last study reports that emotional Arousal responses differ more for
contrasting musical material than for repeating material. It motivates the
repeated-section diagnostic, but our 15-song pseudo-VA result is not a direct
replication because the dataset, annotations, and Teacher outputs differ.

## Held-Out Label Integrity

After Stage 3 inference and Stage 4 feature extraction were frozen, a read-only
Supabase query requested only the 179 frozen annotation IDs.

```text
requested IDs=179
fetched rows=179
missing/unexpected IDs=0/0
boundary mismatches=0
missing section types=0
approved sections=179
confidence=1.0 for all rows
held-out label canonical SHA-256=
4bda4c4837ae7d4b65d4c4dedd1ce2acaedfdb0bd36dec465f0d6d6d6d25d555
```

Section counts:

| Type | Sections | Songs |
|---|---:|---:|
| Chorus | 60 | 15 |
| Verse | 49 | 15 |
| Interlude | 21 | 13 |
| Pre-chorus | 16 | 7 |
| Outro | 14 | 13 |
| Intro | 13 | 13 |
| Bridge | 6 | 5 |

The database schema has no human Valence level, Arousal level,
change-from-previous-section, or climax field. Therefore the following results
are structural convergent evidence, not direct VA accuracy.

## Statistical Protocol

- The song, not the section row, is the independent statistical unit.
- Chorus and Verse values are averaged within each song before paired tests.
- Adjacent transitions are averaged within each song before paired tests.
- Two-sided Wilcoxon signed-rank tests and rank-biserial effects are reported.
- Mean-difference confidence intervals use 20,000 song-level bootstrap samples
  with seed `20260721`.
- Tests are exploratory and unadjusted for multiple comparisons. They must not
  be presented as confirmatory hypothesis tests.

## Main Result: Chorus Lift

| Comparison | Dimension | Songs | Mean difference | 95% bootstrap CI | Positive songs | p | Rank-biserial |
|---|---|---:|---:|---|---:|---:|---:|
| Chorus - Verse | Valence | 15 | +0.093922 | [+0.059399, +0.127639] | 13/15 | 0.000305 | 0.950 |
| Chorus - Verse | Arousal | 15 | +0.235362 | [+0.166281, +0.314067] | 15/15 | 0.000061 | 1.000 |

The strongest finding is the within-song Arousal lift: every sampled song has
higher mean predicted Arousal in Chorus than Verse. Predicted Valence is also
higher in most songs, but `欧若拉` (-0.026548) and `恋人` (-0.007420) are
exceptions. This supports using the Teacher for relative section contrast more
strongly than for universal Valence claims.

## Adjacent Transitions

| Transition | Dimension | Songs | Mean change | 95% bootstrap CI | Positive songs | p |
|---|---|---:|---:|---|---:|---:|
| Verse to Chorus | Valence | 9 | +0.070859 | [+0.034699, +0.107822] | 7/9 | 0.019531 |
| Verse to Chorus | Arousal | 9 | +0.119598 | [+0.072633, +0.167534] | 9/9 | 0.003906 |
| Pre-chorus to Chorus | Valence | 6 | +0.044771 | [+0.017418, +0.078959] | 6/6 | 0.031250 |
| Pre-chorus to Chorus | Arousal | 6 | +0.177315 | [+0.113599, +0.237048] | 6/6 | 0.031250 |

These smaller subsets reinforce the Chorus-lift pattern, especially for
Arousal. Their sample sizes are too small for strong generalization.

## Repeated-Section Consistency

The mean pairwise VA distance was `0.220443` for repeated Chorus sections and
`0.205869` for repeated Verse sections. The paired Chorus-minus-Verse
difference was `+0.014574`, with 95% CI `[-0.066518, +0.114426]` and
`p=0.599487`.

There is no evidence here that repeated Chorus sections are more emotionally
consistent than repeated Verse sections. This negative result should be kept:
later choruses may legitimately differ through arrangement, vocal intensity,
or final-chorus expansion, and the deployment feature-domain gap may also add
variation.

## Model-Derived Climax Candidates

Using the predeclared operational rule "section with maximum mean predicted
Arousal":

```text
Chorus=10 songs
Bridge=2
Outro=1
Intro=1
Verse=1
Chorus or Bridge=12/15
candidate midpoint >= 0.70=13/15
median normalized candidate position=0.791571
```

This indicates a late-song, Chorus/Bridge-centered tendency in the model
outputs. It is not a climax hit rate because no human climax answer exists.
`关山酒` selects its Intro and `New Boy` selects a late Verse; these cases
should receive particular attention during human annotation.

## Interpretation

What the 15-song experiment supports:

- complete-song pseudo-VA can be aggregated reproducibly by human-reviewed
  sections;
- the independently revealed section types expose a strong within-song
  Chorus Arousal lift;
- transition and candidate-position features are suitable for subsequent
  exploratory emotion-structure analysis.

What it does not establish:

- absolute VA accuracy on these complete songs;
- human agreement on Valence/Arousal levels or change direction;
- true climax-detection accuracy;
- cross-corpus or AIGC comparisons, which are outside the current paper scope;
- generalization beyond this 15-song Chinese popular-music sample.

## Human Validation Still Required

A model-blind annotation sheet with 179 rows has been generated:

```text
outputs/human_validation/blind_sample_15_human_section_emotion_annotation.xlsx
```

It contains no model predictions and includes validated dropdowns plus a
per-song completion checker. Complete it according to
`docs/BLIND_VALIDATION_HUMAN_ANNOTATION_GUIDE.md`. One rater is usable with an
explicit limitation; two independent raters are preferable. After completion,
the project can calculate ordinal VA association, change-direction agreement,
exact/adjacent climax hit rates, and inter-rater agreement.

## Core Outputs

```text
data/blind_validation/stage5/blind_sample_15_section_features_with_types.csv
data/blind_validation/stage5/blind_sample_15_paired_structural_tests.csv
data/blind_validation/stage5/blind_sample_15_repeated_section_consistency.csv
data/blind_validation/stage5/blind_sample_15_climax_candidates.csv
data/blind_validation/stage5/stage5_analysis_manifest.json
```

Figures:

```text
outputs/plots/blind_validation_stage5/chorus_verse_paired_va.png
outputs/plots/blind_validation_stage5/section_type_va_distribution.png
outputs/plots/blind_validation_stage5/climax_candidate_positions.png
```

## Decision

Stage 5 structural-label analysis passes. Direct human emotion validation is
still in progress and cannot be completed until the model-blind annotation
sheet is filled. Do not tune the Teacher or change the 30-second deployment
rule based on these held-out results.
