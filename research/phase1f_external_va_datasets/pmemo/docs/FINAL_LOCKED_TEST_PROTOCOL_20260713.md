# Final Locked Test Protocol

Report time: 2026-07-13 CST

Current research status:
- We are still in Stage 1 VA Teacher optimization.
- Arousal and Valence models are not yet frozen.
- No more VA-labelled data are available from the user.
- Therefore a final test subset must be protected before any further training.

## Core Decision

The previous `pmemo_blind_20260709` holdout/test has already been inspected
during model development. It can remain a historical reference, but it must not
be described as a pristine final blind test.

A new prospective final test was locked:

```text
data/splits/pmemo_final_locked_20260713/
```

This is the strictest possible protocol from this point onward given the finite
labelled data.

## Split Counts

```text
total PMEmo songs = 767
final_locked_test = 115 songs
development_pool = 652 songs
old inspected blind_test retained in development_pool = 116 songs
```

The final locked test:
- is song-level disjoint from the development pool;
- does not overlap the old inspected blind-test songs;
- is not allowed in any future training or validation index.

## Files To Use

Future development folds:

```text
data/splits/pmemo_final_locked_20260713/devfolds_3fold_20260713/fold_01/sequence_index.csv
data/splits/pmemo_final_locked_20260713/devfolds_3fold_20260713/fold_02/sequence_index.csv
data/splits/pmemo_final_locked_20260713/devfolds_3fold_20260713/fold_03/sequence_index.csv
```

Final test index, do not read before model freeze:

```text
data/splits/pmemo_final_locked_20260713/final_locked_test_index.csv
```

Deprecated old folds:

```text
data/splits/pmemo_trainval_devfolds_20260709/
```

These old folds contain the newly locked final-test songs and must not be used
for new training.

## Guard Command

Run this before future training:

```bash
.venv/bin/python research/phase1f_external_va_datasets/pmemo/scripts/guard_pmemo_final_locked_split.py \
  research/phase1f_external_va_datasets/pmemo/data/splits/pmemo_final_locked_20260713/devfolds_3fold_20260713/fold_01/sequence_index.csv \
  research/phase1f_external_va_datasets/pmemo/data/splits/pmemo_final_locked_20260713/devfolds_3fold_20260713/fold_02/sequence_index.csv \
  research/phase1f_external_va_datasets/pmemo/data/splits/pmemo_final_locked_20260713/devfolds_3fold_20260713/fold_03/sequence_index.csv
```

Expected result:

```text
final_overlap_count = 0 for every checked index
violations = []
```

## Non-Negotiable Rules

1. Do not train on `final_locked_test`.
2. Do not validate on `final_locked_test`.
3. Do not use `final_locked_test` for model selection, hyperparameter tuning,
   post-processing selection, normalization fitting, calibration fitting, or
   ensemble-weight selection.
4. Do not compute final-test metrics until both Arousal and Valence teacher
   pipelines are fully frozen.
5. After final-test metrics are computed, do not change the model based on
   those results.

## Reporting Language

Before model freeze:
- Use “development-fold validation performance”.
- Do not call dev-fold metrics “final test performance”.

After model freeze:
- Use “prospective locked final-test performance”.
- Also disclose the limitation that the split was locked after earlier
  exploration because no new labelled data were available.

## Practical Next Step

All future optimization should restart from the new development folds:

```text
pmemo_final_locked_20260713/devfolds_3fold_20260713
```

The previous best model results remain useful as historical guidance, but they
were obtained under the older development protocol and must be rechecked under
the new locked protocol before being treated as current.
