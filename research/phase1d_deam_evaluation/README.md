# Phase 1D: DEAM VA Evaluation And Diagnostics

Status: active.

## Scope

This phase evaluates the VA Teacher on DEAM only. It does not perform lyrics,
section, or lyric-music alignment analysis.

Evaluation levels:

```text
frame
fixed windows: 2 s, 5 s, 10 s, 15 s
clip
```

All metrics remain on the original DEAM `[-1,+1]` scale:

```text
Valence/Arousal RMSE
Valence/Arousal MAE
Valence/Arousal Pearson r
Valence/Arousal CCC
mean RMSE/MAE/Pearson/CCC
```

## Rules

- Song-level split only.
- Dynamic labels begin at 15 seconds with a 0.5-second hop.
- No 0-1 label conversion for headline metrics.
- Smoothing parameters are selected on validation only.
- Linear calibration is fitted on validation and applied once to test.
- Label-shift scanning is diagnostic. A selected shift must later be applied
  consistently during retraining, not only to the test labels.
- DEAM fixed windows are not sections or lyric sentences.

## Entry Point

```bash
.venv/bin/python \
  research/phase1d_deam_evaluation/scripts/evaluate_mert_bilstm.py
```

Generated outputs are under `outputs/` and are gitignored. Human-readable
results are recorded in `docs/EVALUATION_RESULTS.md`.
