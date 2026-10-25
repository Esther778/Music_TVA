# Phase 1F External VA Datasets

This stage adds external dynamic Valence-Arousal datasets to strengthen the
Phase 1 Teacher model.

## Purpose

Phase 1E showed that DEAM-only complete-song generalization is unstable,
especially for Valence. Phase 1F is a data-strategy stage:

```text
DEAM remains the main reference dataset.
PMEmo is introduced as an external dynamic VA dataset candidate.
No mixed-dataset training should start before dataset auditing is complete.
```

## Current Substages

```text
pmemo/       PMEmo dataset audit, normalization, and future fusion experiments
```

## Required Order

1. Put the PMEmo raw package under `pmemo/data/raw/`.
2. Run the PMEmo audit script.
3. Confirm audio coverage, VA label range, label time step, and song IDs.
4. Build a normalized PMEmo manifest.
5. Only then create DEAM + PMEmo training experiments.
