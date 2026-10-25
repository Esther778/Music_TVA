# Valence-Only Residual Continuation

Report time: 2026-07-15 CST

Research stage: Phase 1 VA Teacher development. Arousal was frozen and no
Arousal model, parameter, or prediction was changed in this continuation.
All Valence experiments used only the 652-song locked development pool. The
115-song final locked test remained excluded.

## Starting Point

Robust Valence development best:

```text
model_id = pmemo_valence_locked_global_shift_amp_guarded_20260715
RMSE / MAE / PCC / CCC =
0.181284 / 0.142308 / 0.824000 / 0.807112
minimum status = MAE, PCC, CCC pass; RMSE misses by 0.001284
```

Hardest fold-03 reference:

```text
RMSE / MAE / PCC / CCC =
0.193754 / 0.149194 / 0.818762 / 0.799365
```

## Low-Rank Temporal Residual Gate

Added:
- `scripts/train_pmemo_valence_locked_mert_lowrank_residual.py`

Inputs and design:
- Current robust Valence curve as an immutable skip connection.
- Frozen MERT hidden-4/6/8 frame embeddings.
- EffNet mood/theme audio tags and prediction statistics.
- Zero-initialized frame and song residual heads.
- Song-level split, stratified inner epoch selection, and held fold-03 gate.

Initial gate:

```text
frame correction cap = 0.010
song correction cap = 0.015
learning rate = 1e-4
selected epoch = 15
fold-03 RMSE / MAE / PCC / CCC =
0.197734 / 0.152182 / 0.811467 / 0.792872
decision = reject
```

## Multi-Scale Conservative Gate

Reason:
- The robust base was separately evaluated at multiple temporal resolutions.
- Frame-level RMSE was `0.181284`, 10-second-window RMSE was `0.180098`, and
  clip-level RMSE was `0.178526`.
- This indicates that aggregate song emotion is usable, while short-time
  trajectory error is the remaining bottleneck.

Changes:
- Added 2/5/10-second multi-scale mean losses.
- Reduced frame cap to `0.003` and song cap to `0.005`.
- Reduced learning rate to `5e-5`.
- Retained zero initialization and correction L2 regularization.

Result:

```text
selected epoch = 17
fold-03 RMSE / MAE / PCC / CCC =
0.194659 / 0.149761 / 0.817052 / 0.797974
decision = reject
```

The conservative model was closer to the base than the initial gate but still
worsened RMSE, PCC, and CCC. It was not expanded to all three folds.

## Multi-Level Reference

Metrics below use the unchanged robust Valence best:

| Level | RMSE | MAE | PCC | CCC |
| --- | ---: | ---: | ---: | ---: |
| 0.5 s frame | 0.181284 | 0.142308 | 0.824000 | 0.807112 |
| 2 s window | 0.180927 | 0.142055 | 0.824553 | 0.807704 |
| 5 s window | 0.180344 | 0.141640 | 0.825125 | 0.808235 |
| 10 s window | 0.180098 | 0.141445 | 0.824660 | 0.807817 |
| 15 s window | 0.180371 | 0.141505 | 0.826649 | 0.809200 |
| clip/song | 0.178526 | 0.140722 | 0.821365 | 0.805417 |

The clip-level result passes all four minimum criteria. This does not change
the fact that the 0.5-second frame-level RMSE remains above `0.18`.

## Decision

- Keep Arousal frozen.
- Keep the Valence robust best unchanged.
- Reject both frozen-feature temporal residual gates.
- Do not tune another cap, learning rate, or residual head after observing
  fold-03 outcomes; that would increase development-fold overfitting.
- The remaining technically distinct route is parameter-efficient adaptation
  of MERT upper layers. The current local runtime has neither CUDA nor MPS, so
  this should be run only with suitable accelerator resources.
- Final locked test remains untouched.
