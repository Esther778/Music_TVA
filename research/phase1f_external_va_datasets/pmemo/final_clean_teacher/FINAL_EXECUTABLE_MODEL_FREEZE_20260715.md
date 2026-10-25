# Final Executable VA Teacher Freeze

Frozen at: 2026-07-15 CST

Status at freeze: final locked labels have not been read. Label-free predictions
cover 115 songs and 5,390 frames.

## Model Identity

```text
Valence model = pmemo_final_clean_valence_teacher_v1
Arousal model = pmemo_final_clean_arousal_teacher_v1
Input = MERT hidden-6 + OpenSMILE dynamic pure-audio features
Frame hop = 0.5 seconds
Architecture = separate global-local BiGRU models
Training pool = 652 PMEmo development songs
Final-fit epochs = Valence 9, Arousal 7
```

## Development OOF

Every outer fold used inner song-level validation for epoch selection, then
refit on the complete outer training pool before held-fold evaluation.

Post-test audit qualification: the split/refit mechanics above are strict, but
the 0.15-weight soft teacher came from repeatedly selected historical
development OOF artifacts. These results are therefore development evidence,
not a fully nested estimate of the complete model-selection process.

| Dimension | Scale | RMSE | MAE | PCC | CCC | Minimum status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Valence | 0.5 s frame | 0.219273 | 0.171219 | 0.735047 | 0.721788 | not pass |
| Valence | complete clip | 0.215695 | 0.169919 | 0.734075 | 0.721576 | not pass |
| Arousal | 0.5 s frame | 0.201995 | 0.155300 | 0.833625 | 0.824880 | RMSE/MAE not pass |
| Arousal | complete clip | 0.204933 | 0.153766 | 0.833100 | 0.823033 | RMSE/MAE not pass |

The clean executable model is materially weaker than the frozen multi-stage
development artifacts. This negative result is retained because it shows that
the historical OOF chain cannot be treated as a single deployable checkpoint.
The prospective test below evaluates this clean executable model, not the
historical multi-stage metric best.

## Frozen Hashes

```text
cb32a52d4b7c90f3b41aca17f82ab66cde8d33634497d6862bf0b6ab2b8288a1  final_clean_teacher_v1.json
6cddf9ce3f12f341dac5d53b99c108675fdac9c8c17e323164f77388d88918ae  final_fit/valence/model.pt
aba66d037ec8f4c30917a67dbb9d97f0dfc2fc781fa4191d4202812051c368bb  final_fit/arousal/model.pt
f2c3b7ae36e9e12ae15d15c9bce6cda532bb09c9af8b8793ea2901937a3b0b9c  final_locked_predictions_LABEL_FREE.csv
65971f15ee892d60c09d778968188f9ed5a1fe18974d6f8c0dd2bc114d8932a2  final_locked_test_index.csv
```

## Final-Test Rule

After the frozen prediction hash is verified, the 115-song labels may be read
once to compute the preregistered complete-clip primary metrics and frame-level
secondary metrics. No post-test tuning, calibration, smoothing, model change,
or replacement is allowed in response to those results.

## Post-Test Audit Update

Audited at: 2026-07-15 22:50 CST.

The label-free prediction was created and hashed before the final labels were
opened. An independent implementation reproduced every saved metric exactly.
No post-test tuning was performed.

Official 115-clip complete-clip RMSE/MAE/PCC/CCC:

```text
Valence = 0.186298 / 0.152855 / 0.802852 / 0.800168
Arousal = 0.188573 / 0.146912 / 0.857441 / 0.854803
```

Official 115-clip frame-level RMSE/MAE/PCC/CCC:

```text
Valence = 0.194416 / 0.158386 / 0.810613 / 0.801561
Arousal = 0.200411 / 0.157457 / 0.848749 / 0.839168
```

Important limitation: SHA-256 comparison found three exact audio duplicates
across development and final sets (`90/92`, `316/317`, `759/758`). The official
115 result remains the preregistered result; a 112-clip post-hoc sensitivity
analysis is reported separately and is not a new blind test. PMEmo's median
labelled span is 20.5 seconds, so this evaluation does not by itself prove
multi-minute complete-song generalization.

Full details: `FINAL_VA_TEACHER_RIGOR_AUDIT_20260715.md`.
