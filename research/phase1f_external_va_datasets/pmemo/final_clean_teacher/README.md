# Final Clean VA Teacher

Created: 2026-07-15 CST

Purpose: materialize a reproducible, pure-audio VA teacher before the one-time
prospective PMEmo final test.

Current status: the one-time test has been completed. Read
`FINAL_VA_TEACHER_RIGOR_AUDIT_20260715.md` before using or reporting the model.

This directory is independent from the historical multi-stage OOF artifacts.
Those artifacts remain frozen as development metric evidence. The clean model
uses their OOF curves only as a weak distillation target; true development
labels remain the primary supervision.

## Fixed Protocol

- Development pool: 652 PMEmo clips.
- Prospective final test: 115 `musicId`-disjoint clips. A later SHA-256 audit
  found three exact audio duplicates across development and final sets.
- Input: MERT hidden-6 frame embedding plus OpenSMILE dynamic frame features.
- Sampling: one prediction every 0.5 seconds.
- Models: separate Valence and Arousal global-local BiGRU regressors.
- Epoch selection: inner song-level validation inside each outer development
  fold.
- OOF report: after inner epoch selection, refit on the complete outer training
  pool and evaluate the held song fold, which is never used for gradient
  updates.
- Final fit: all 652 development clips for the median selected epoch count.
- Final prediction: the inference path reads `X` and `time_sec` only; it never
  accesses the final NPZ `Y` array.
- Final test labels may be opened only after the prediction CSV and manifest
  hashes are fixed.

## Files

- `configs/final_clean_teacher_v1.json`: frozen model and loss settings.
- `scripts/train_final_clean_teacher.py`: OOF training, final fit, and
  label-free final prediction.
- `outputs/`: generated checkpoints, predictions, histories, and summaries.
- `FINAL_VA_TEACHER_RIGOR_AUDIT_20260715.md`: final metrics, independent
  recomputation, duplicate audit, deployment boundary, and paper claim limits.

## Claim Boundary

The clean executable model and the historical development metric best are not
assumed to be numerically identical. Their metrics must be reported in separate
rows. Only the clean executable model is eligible for prospective final-test
reporting and arbitrary-song inference.

“Arbitrary-song inference” here means that the pure-audio model can technically
accept compatible features from a new recording. The reported final-test
metrics use PMEmo official features and mostly short labelled excerpts; they do
not guarantee the same accuracy on raw, multi-minute songs.
