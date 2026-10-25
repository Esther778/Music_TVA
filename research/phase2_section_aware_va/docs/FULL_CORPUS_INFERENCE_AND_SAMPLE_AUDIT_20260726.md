# Full-Corpus Inference And Sample Sufficiency Audit

Report time: 2026-07-26 11:36 CST

Research progress: Phase 1 VA Teachers remain frozen. Phase 2 blind human
validation is frozen, and full-corpus pseudo-VA inference, section alignment,
structure extraction, and sample-sufficiency assessment are complete.

## Scope And Protocol

- Frozen input population: 99 reviewed songs and 1,097 continuous section
  boundaries.
- Frozen Teachers: `pmemo_final_clean_valence_teacher_v1` and
  `pmemo_final_clean_arousal_teacher_v1`.
- Features: MERT-v1-95M hidden layer 6 plus PMEmo-style OpenSMILE dynamic
  260-dimensional features.
- Output hop: 0.5 seconds.
- Complete-song deployment: 30-second windows, 15-second stride, Hann
  overlap-add.
- The 15 blind-validation songs were reused only after audio, boundary,
  feature, frame-output, and metadata hashes matched.
- Section types and human emotion answers were not read during Teacher
  inference or boundary alignment.

## Inference Audit

| Item | Result |
| --- | ---: |
| Frozen songs | 99 |
| Reused blind songs | 15 |
| Newly inferred songs | 84 |
| Total 0.5-second VA frames | 49,763 |
| Wall time | 3,043.35 s (50.7 min) |
| Passed songs | 99/99 |
| Failed songs | 0 |
| Curve-collapse warnings | 0 |
| Combined-table reconstruction | pass |

Several MP3 files emitted local decoder warnings. Their decoded duration
differed from the frozen duration by no more than 0.04 seconds and all later
coverage checks passed, so they were retained. Four frames occurred exactly at
the final section endpoint. The documented interval convention is now
`[start,end)` for internal sections and `[start,end]` for the final section.

## Section Alignment

| Item | Result |
| --- | ---: |
| Frames assigned | 49,763/49,763 |
| Sections represented | 1,097/1,097 |
| Unassigned frames | 0 |
| Overlap frames | 0 |
| Sections without frames | 0 |
| Exact final-boundary frames | 4 |

The later Supabase type reveal returned 1,097/1,097 frozen annotation IDs with
zero missing IDs, zero unexpected IDs, zero boundary drift, and zero missing
types. No human VA, change, or climax fields were read.

## Primary Analysis Set

All 99 songs retain predictions for provenance. `ANGEL` and `Easy On Me` are
excluded from the Chinese-language primary analysis, leaving 97 songs and
1,076 sections. This is an analysis-scope exclusion, not a database deletion.

| Section type | Sections | Songs |
| --- | ---: | ---: |
| Verse | 262 | 97 |
| Chorus | 371 | 96 |
| Pre-chorus | 90 | 46 |
| Bridge | 35 | 32 |
| Interlude | 127 | 86 |
| Intro | 94 | 92 |
| Outro | 97 | 94 |

The primary sample contains 75 artists. The maximum is five songs per artist,
the largest artist share is 5.2%, and 62 artists contribute one song.

## Exploratory Structure Results

The independent statistical unit is the song, not the section row.

| Comparison | Dimension | n songs | Mean difference | 95% bootstrap CI |
| --- | --- | ---: | ---: | ---: |
| Chorus - Verse | Valence | 96 | +0.101 | [0.083, 0.119] |
| Chorus - Verse | Arousal | 96 | +0.266 | [0.237, 0.297] |
| Verse to Chorus | Valence | 56 | +0.054 | [0.037, 0.071] |
| Verse to Chorus | Arousal | 56 | +0.139 | [0.115, 0.165] |
| Pre-chorus to Chorus | Valence | 44 | +0.045 | [0.028, 0.062] |
| Pre-chorus to Chorus | Arousal | 44 | +0.151 | [0.126, 0.177] |

All six exploratory Wilcoxon tests remain below 0.05 after Holm correction.
These are pseudo-VA structural patterns, not proof of continuous human VA
ground truth. The predicted maximum-Arousal section is Chorus in 53/97 songs;
it must be called a model-derived climax candidate, not a verified climax.

## Is More Annotation Required?

Current judgment:

- No additional songs are mandatory before reporting the present exploratory
  analysis. The core Chorus-versus-Verse comparison already has 96 independent
  songs and relatively narrow bootstrap intervals.
- External validity remains limited because the 97-song corpus is a convenience
  sample, not a pre-frozen ranking-stratified sample.
- Bridge coverage is the clearest structural gap: 32 songs versus 96-97 songs
  for Chorus/Verse.
- Adding all 170 songs implied by applying 3/2/1 to every top-100 artist is not
  justified by precision gain and does not eliminate platform-ranking bias.

Recommended optional expansion for stronger external validity: **30 songs**.

1. Freeze the dated QQ Music top-100 singer ranking before selection.
2. Randomly select five eligible artists from ranks 1-20, five from 21-50, and
   five from 51-100 with seed `20260726`.
3. Sample 3, 2, and 1 eligible songs per selected artist, respectively, for
   15 + 10 + 5 = 30 songs.
4. Predeclare eligibility: Chinese-language human-created complete songs;
   exclude existing tracks, live/remix/instrumental versions, duplicate
   recordings, and foreign-language tracks.
5. Freeze all exclusions and redraws. Do not tune the Teacher on the added
   songs.

Under stable variance, increasing 97 to 127 songs would reduce standard errors
by approximately 12.6%. If the current Bridge rate holds, about ten added songs
would contain a Bridge. This is a precision projection, not a guaranteed
outcome.

More one-rater songs do not establish inter-rater reliability. If resources
permit only one extra annotation activity, a second independent rater for the
existing frozen 15-song human-validation set has higher validation value than
adding more one-rater VA judgments.

## Reproducible Outputs

```text
data/full_corpus/stage1/full_corpus_99_manifest.csv
data/full_corpus/stage2/full_corpus_99_frame_va.csv
data/full_corpus/stage2/full_corpus_output_audit_summary.json
data/full_corpus/stage3/full_corpus_99_frame_section_alignment.csv
data/full_corpus/stage3/full_corpus_99_section_features.csv
data/full_corpus/stage3/full_corpus_99_song_structure_features.csv
data/full_corpus/stage4/full_corpus_99_frozen_section_types.json
data/full_corpus/stage4/primary_chinese_section_features.csv
data/full_corpus/stage4/primary_chinese_song_features.csv
data/full_corpus/stage4/primary_paired_structural_tests_holm.csv
data/full_corpus/stage4/sample_sufficiency_assessment.json
data/full_corpus/stage4/sample_expansion_scenarios.csv
```

The frozen Teacher has not been retuned using either the 15-song human
validation set or these 99 full-corpus outputs.
