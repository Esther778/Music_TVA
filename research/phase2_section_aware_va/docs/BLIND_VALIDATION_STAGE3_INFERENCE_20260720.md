# Blind Validation Stage 3 Frame-Level VA Inference

Report time: 2026-07-20 23:55 (Asia/Shanghai)

Research progress: audio-only frame-level pseudo-VA inference is complete for
the frozen 15-song sample. Section alignment and human comparison have not
started.

## Frozen Teachers

The canonical final executable Phase 1 models were used without retraining or
parameter tuning:

```text
Valence model = pmemo_final_clean_valence_teacher_v1
Arousal model = pmemo_final_clean_arousal_teacher_v1
Input = MERT-v1-95M hidden-6 + PMEmo-style OpenSMILE dynamic 260d
Architecture = separate Global-Local BiGRU models
Output interval = 0.5 seconds
```

Official complete-clip final metrics on 115 frozen PMEmo clips:

| Dimension | RMSE | MAE | PCC | CCC |
|---|---:|---:|---:|---:|
| Valence | 0.186298 | 0.152855 | 0.802852 | 0.800168 |
| Arousal | 0.188573 | 0.146912 | 0.857441 | 0.854803 |

Official frame-level metrics are V `0.194416/0.158386/0.810613/0.801561`
and A `0.200411/0.157457/0.848749/0.839168` for RMSE/MAE/PCC/CCC.
These PMEmo metrics establish Teacher performance; they are not accuracy
measurements on the 15 complete songs, whose frame-level human VA ground truth
does not exist.

Frozen artifact hashes:

```text
config=cb32a52d4b7c90f3b41aca17f82ab66cde8d33634497d6862bf0b6ab2b8288a1
valence_model=6cddf9ce3f12f341dac5d53b99c108675fdac9c8c17e323164f77388d88918ae
arousal_model=aba66d037ec8f4c30917a67dbb9d97f0dfc2fc781fa4191d4202812051c368bb
final_normalizer=bd86da37db97976e606ab77dfc8a322a84ec13c1ab687037d860496798de54f3
opensmile_norm=d6666011894440a392f0410d0c1f401d1e895bcde17eb56a28ba7bbeeba1fbb3
```

## Blind-Safe Inputs

Inference read only the frozen Stage 2 manifest and local MP3 audio. It did
not read section boundaries, section types, lyrics, review records, or human
VA/change/climax answers. All 15 MP3 SHA-256 values matched the frozen
manifest before inference.

The Stage 2 protocol stores canonical content hashes, while Stage 3 also logs
physical file-byte hashes. The different values are expected:

```text
manifest canonical content hash=1d6d624c27c6c2fd6317b6b146b97a935846ade4467553cbfb3ef9551d96aff5
manifest CSV byte hash=6d5e6e15a69c4804b2a7e73401b412bdb563bb3562a80bc50ceb87f3be1e5dc7
boundaries canonical content hash=4327ad8bef8d4a3c0f9bcba71c4a5fd3ab9743a77c29c2b03ebac8af6d5ca7f1
```

The canonical hashes were recomputed from the current frozen content and
match the Stage 2 protocol exactly. There was no sample replacement.

## Complete-Song Deployment Rule

Directly applying the Global-Local model to the first 204.6-second song
produced near-constant curves because PMEmo training spans are much shorter:

```text
PMEmo labelled span median=21 s, p75=31 s, p95=50 s, max=73 s
direct full-sequence V std=0.001199
direct full-sequence A std=0.004199
```

Before any human answer was read, complete-song inference was therefore frozen
to:

```text
window=30 seconds
stride=15 seconds
blend=Hann weighted overlap-add
```

This keeps each forward pass close to the PMEmo training-length distribution
while covering every complete song. The same rule was used for all 15 songs.
For the first song, windowed V/A standard deviations became `0.086861` and
`0.200925`; this rejects the direct full-sequence collapse without using
section labels.

## Compatibility Checks And Limitations

- Current OpenSMILE exposes legacy PMEmo MFCC descriptors without the
  `pcm_fftMag_` prefix. All 56 affected names are explicitly aliased; missing
  descriptors now cause a hard error rather than silent zero filling.
- `inference_log.csv` intentionally retains the first diagnostic failure caused
  by those legacy names and the rejected direct full-sequence attempt. The
  final per-song metadata, final manifest, and independent audit refer to the
  corrected windowed outputs; final failed-song count is zero.
- A PMEmo song re-extraction check gave MERT hidden-6 correlation `0.999683`
  with the saved training feature. The small numeric difference is mainly due
  to torchaudio versus the original librosa decoding path.
- Raw-MP3 OpenSMILE reproduction is not bit-equivalent to PMEmo's released
  feature CSV. On the checked PMEmo song, correlation was `0.881348`, RMSE
  `0.431454`; an earlier three-song audit gave correlation `0.773-0.881`.
- Therefore the 15-song outputs must be described as externally deployed
  `pseudo-VA`, not human ground truth. The planned section-level human check is
  required before structural claims are made.

## Outputs And Audit

```text
songs=15
frames=7375
frame hop=0.5 s
batch elapsed=1068.346 s
failed songs=0
collapse warnings=0
combined output SHA-256=9fd7d7b758ba707c8e1c8fdde57548eb0eb8225e87b40ae987c1ed39f1582357
```

Across songs, within-song standard deviations were:

| Dimension | Minimum | Median | Maximum |
|---|---:|---:|---:|
| Valence | 0.033668 | 0.114923 | 0.185989 |
| Arousal | 0.045743 | 0.176451 | 0.364871 |

The independent audit checked audio/output hashes, song identity, feature and
frame counts, exact 0.5-second time axes, finite values, `[-1,1]` and `[0,1]`
ranges, scale conversion, curve collapse, and combined-table ordering. All 15
songs passed. The combined table differs from CSV-reloaded per-song values by
at most `2.98e-8`, below the frozen `5e-8` serialization tolerance.

Primary artifacts:

```text
data/blind_validation/stage3/blind_sample_15_frame_va.csv
data/blind_validation/stage3/frame_va/<song_id>_frame_va.csv
data/blind_validation/stage3/features/<song_id>.npz
data/blind_validation/stage3/metadata/<song_id>.json
data/blind_validation/stage3/inference_log.csv
data/blind_validation/stage3/stage3_inference_manifest.json
data/blind_validation/stage3/stage3_output_audit.csv
data/blind_validation/stage3/stage3_output_audit_summary.json
```

Reproducible commands:

```bash
.venv/bin/python research/phase2_section_aware_va/scripts/predict_blind_sample_final_teacher.py
.venv/bin/python research/phase2_section_aware_va/scripts/audit_blind_stage3_outputs.py
```

## Decision

Stage 3 passes. Stop here until the user confirms Stage 4. Stage 4 may reveal
the already frozen section boundaries and aggregate pseudo-VA within sections,
but it must not retrain or tune the Teacher using these 15 songs.
