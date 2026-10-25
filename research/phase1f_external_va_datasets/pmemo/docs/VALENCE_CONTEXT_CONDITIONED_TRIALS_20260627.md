# Valence Context-Conditioned Trials - 2026-06-27

Report time: 2026-06-27

Current research progress: Phase 1F is optimizing the Valence teacher. Arousal
already has a threshold-passing candidate; Valence remains below the target
thresholds and is the active bottleneck.

## Purpose

Previous diagnostics showed that Valence errors are strongly affected by
song-level affective bias. This round tested whether adding an explicit
song-level context branch could improve frame-level Valence prediction.

Base feature family:

```text
MERT hidden_6 768 + Librosa harmonic 36
```

Base frame model family:

```text
Valence-only BiLSTM with extreme-weighted frame loss,
frame CCC loss, and clip mean loss
```

New idea:

```text
Compute per-song context from masked mean/std/min/max of the full input
sequence, then inject it into frame-level prediction by FiLM or concatenation.
```

## Thresholds Used For Reporting

| Metric | Minimum pass | Good | Excellent | Direction |
| --- | ---: | ---: | ---: | --- |
| RMSE | 0.180000 | 0.150000 | 0.100000 | lower is better |
| MAE | 0.150000 | 0.120000 | 0.080000 | lower is better |
| PCC | 0.800000 | 0.850000 | 0.900000 | higher is better |
| CCC | 0.800000 | 0.850000 | 0.900000 | higher is better |

## Historical Best Before This Round

| Item | Value |
| --- | --- |
| model_id | `pmemo_valence_mert_hidden6_harmonic_extreme_weighted_v1` |
| trial_id | `h6_harm_h80_ext125_clip010_ccc005` |
| feature | MERT hidden_6 768 + Librosa harmonic 36 |
| model | Valence-only BiLSTM, projection 320, hidden_dim 80, dropout 0.30 |
| loss | extreme-weighted frame MSE + 0.05 frame CCC + 0.10 clip mean MSE |

Historical best test metrics:

| Metric | Historical best | Minimum pass | Gap to pass | Good | Gap to good | Excellent | Gap to excellent |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RMSE | 0.205673 | 0.180000 | +0.025673 | 0.150000 | +0.055673 | 0.100000 | +0.105673 |
| MAE | 0.164567 | 0.150000 | +0.014567 | 0.120000 | +0.044567 | 0.080000 | +0.084567 |
| PCC | 0.783400 | 0.800000 | +0.016600 | 0.850000 | +0.066600 | 0.900000 | +0.116600 |
| CCC | 0.767304 | 0.800000 | +0.032696 | 0.850000 | +0.082696 | 0.900000 | +0.132696 |

## Current Experiment Results

| Trial | Context method | Best epoch | Val RMSE | Val PCC | Val CCC | Test RMSE | Test MAE | Test PCC | Test CCC | Decision |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `h6_harm_ctx96_film_ext125_clip010_std003` | FiLM + std loss | 16 | 0.219076 | 0.814867 | 0.805687 | 0.212169 | 0.170714 | 0.766003 | 0.753634 | not promoted |
| `h6_harm_ctx96_film_ext125_clip010_ccc005` | FiLM | 16 | 0.219491 | 0.814675 | 0.805860 | 0.212338 | 0.170747 | 0.765542 | 0.753926 | not promoted |
| `h6_harm_ctx96_film_ext125_clip010_ctxmse005` | FiLM + context mean MSE | 14 | 0.222206 | 0.816757 | 0.801654 | 0.216605 | 0.176744 | 0.747223 | 0.742236 | not promoted |
| `h6_harm_ctx64_concat_ext125_clip010_ccc005` | context concat | 16 | 0.218607 | 0.820914 | 0.815634 | 0.222233 | 0.180388 | 0.740241 | 0.734249 | not promoted |

Best current-experiment frame-level result:

```text
trial = h6_harm_ctx96_film_ext125_clip010_std003
test RMSE/MAE/PCC/CCC = 0.212169/0.170714/0.766003/0.753634
```

Comparison against historical best:

| Metric | Current experiment best | Historical best | Delta vs historical best | Minimum pass | Current gap to pass | Good | Current gap to good | Excellent | Current gap to excellent |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RMSE | 0.212169 | 0.205673 | +0.006496 | 0.180000 | +0.032169 | 0.150000 | +0.062169 | 0.100000 | +0.112169 |
| MAE | 0.170714 | 0.164567 | +0.006147 | 0.150000 | +0.020714 | 0.120000 | +0.050714 | 0.080000 | +0.090714 |
| PCC | 0.766003 | 0.783400 | -0.017398 | 0.800000 | +0.033997 | 0.850000 | +0.083997 | 0.900000 | +0.133997 |
| CCC | 0.753634 | 0.767304 | -0.013671 | 0.800000 | +0.046366 | 0.850000 | +0.096366 | 0.900000 | +0.146366 |

Decision:

```text
Do not promote any context-conditioned trial.
The historical best remains h6_harm_h80_ext125_clip010_ccc005.
```

## Important Diagnostic Signal

The `context_mse005` trial learned song-level Valence mean reasonably well:

```text
test context RMSE/PCC/CCC = 0.217595/0.769932/0.705797
```

But its frame-level result was weaker:

```text
test frame RMSE/MAE/PCC/CCC = 0.216605/0.176744/0.747223/0.742236
```

Interpretation:

```text
Song-level Valence bias is learnable, but directly injecting a high-dimensional
context branch into frame prediction currently hurts generalization.
```

## Conclusion

This round confirms that the Valence bottleneck is not solved by simply adding
global context to the existing BiLSTM. The model can partially infer song-level
Valence, but the frame-level teacher becomes less stable.

The next experiment should use the song-level signal more conservatively:

```text
Train a two-stage residual/calibrated model:
1. keep the historical best frame-level model frozen or reproduced;
2. train a lightweight validation-selected song-level bias correction model;
3. apply only a bounded correction to frame-level predictions;
4. select correction strength on validation, then test once.
```

This is less invasive than context-conditioned retraining and directly targets
the observed test mean underprediction without damaging the local frame
trajectory as much.

Artifacts:

```text
script = scripts/train_pmemo_mert_hidden6_harmonic_context_conditioned.py
outputs = outputs/mert_hidden6_harmonic_context_conditioned/
comparison = outputs/mert_hidden6_harmonic_context_conditioned/comparison.csv
```
