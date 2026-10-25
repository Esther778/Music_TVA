# Phase 1E: Teacher Optimization

Status: validation optimization active; test remains locked.

## Objective

Improve the DEAM song-level VA Teacher after completing the evaluation and
feature ablations.

Current evidence:

- MERT is strongest for Valence.
- Librosa handcrafted descriptors are strongest for Arousal.
- Global label shift is not the main error source.
- Smoothing gives only a small gain.
- Affine calibration does not transfer to test.

## Current Best Validation Pipeline

```text
MERT layer 6 + MERT final layer + handcrafted descriptors
-> dimension-specific three-way gated BiLSTM candidates
-> validation-only nonnegative dimension ensemble
-> optional Gaussian smoothing within each song
```

Recommended smoothing is sigma 2 frames:

| Metric | Validation |
| --- | ---: |
| RMSE | 0.184297 |
| MAE | 0.141234 |
| Mean PCC | 0.779024 |
| Mean CCC | 0.757454 |

Arousal meets the requested thresholds. Valence remains the limiting
dimension. See `docs/OPTIMIZATION_REPORT_20260622.md` for the full comparison.

Latest robust Valence evidence uses five-fold OOF evaluation on the original
DEAM training songs:

```text
global-local temporal RMSE=0.185384
cross-fitted extended stack RMSE=0.181819
cross-fitted extended stack MAE=0.139021
cross-fitted extended stack PCC=0.679455
cross-fitted extended stack CCC=0.640288
```

See `docs/GLOBAL_LOCAL_TEMPORAL_CV_REPORT.md`. These values are not Test
results and do not replace the locked final evaluation.

The latest duration-regime audit found that 39 long tracks account for
16.9% of frames and dominate the remaining error. Separate short/long
layer-6 Ridge models under strict five-fold OOF evaluation currently provide
the lowest Valence RMSE:

```text
clip-type direct Ridge RMSE=0.176925
MAE=0.138954
PCC=0.703481
CCC=0.684054
```

True raw-audio MERT LoRA was also tested with the historical 60-second
context. It worsened fold-0 RMSE from 0.208149 to 0.211506 and was rejected.

The next experiment order is now based on a review of current MIR papers and
official GitHub implementations:

1. frozen MusicFM and MuQ representation screening;
2. PDMER-inspired local/global dual-scale attention on cached embeddings;
3. PMEmo dynamic pretraining;
4. short-to-long domain generalization;
5. DEAM multi-annotator uncertainty modeling.

See `docs/MIR_FRONTIER_RESEARCH_20260623.md`.

## Full-Song Evaluation

The next Teacher experiments use a frozen complete-song protocol:

```text
DEAM inventory=1,744 short clips + 58 complete songs
PDMER-compatible test=short clips train / all complete songs test
primary evaluation=nested five-fold complete-song validation and test
```

The 58 complete songs are IDs 2001-2058; song 2000 is a 45-second excerpt.
See `docs/FULL_SONG_EVALUATION_PROTOCOL.md` and
`data/full_song_protocol/protocol_audit.json`.

The first dual-scale complete-song Fold-0 pilot has completed and was
rejected:

```text
Valence RMSE/MAE/PCC/CCC=0.286758/0.198130/0.546796/0.446766
Arousal RMSE/MAE/PCC/CCC=0.190826/0.158357/0.757155/0.726246
outer test used=no
```

The result confirms that dual-scale attention alone does not overcome the
short-clip-to-complete-song distribution shift.

Cached layer-6 adapter, metric-learning, music-auxiliary, and joint
adapter-plus-auxiliary experiments are now complete. They did not provide a
large, stable standalone gain. The next priority is true parameter-efficient
MERT adaptation from raw audio, followed by multi-scale temporal context and,
only if needed, bounded dynamic annotation alignment. See
`docs/LITERATURE_GUIDED_NEXT_EXPERIMENTS.md`.

## Evaluation Rule

- Select using validation RMSE only.
- Report dimension-specific RMSE, MAE, PCC, and CCC.
- Do not inspect test for every architecture iteration.
- Run multiple seeds only after a single-seed candidate improves validation.
- Do not unlock test until the final architecture, ensemble weights, and
  smoothing rule are frozen.
