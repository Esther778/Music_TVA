# Valence Optimization Round

Report time: 2026-06-28

Current research progress: Phase 1 VA Teacher optimization. Arousal has a
pass-level complete-song predictor. Valence has a deployable complete-song
predictor but is still below the minimum metric target.

## Current Best Before This Round

```text
model_id = pmemo_valence_prediction_only_rescale_ccc_preserving_20260628
test RMSE/MAE/PCC/CCC = 0.195316/0.151793/0.797766/0.784834
```

Gap to thresholds:

| Metric | Current | Minimum pass | Gap to pass | Good | Gap to good | Excellent | Gap to excellent |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RMSE | 0.195316 | 0.180000 | +0.015316 | 0.150000 | +0.045316 | 0.100000 | +0.095316 |
| MAE | 0.151793 | 0.150000 | +0.001793 | 0.120000 | +0.031793 | 0.080000 | +0.071793 |
| PCC | 0.797766 | 0.800000 | +0.002234 | 0.850000 | +0.052234 | 0.900000 | +0.102234 |
| CCC | 0.784834 | 0.800000 | +0.015166 | 0.850000 | +0.065166 | 0.900000 | +0.115166 |

## Trial 1: Strict Predicted-Arousal Residual

Script:

```text
scripts/train_pmemo_valence_oof_arousal_residual.py
```

Output:

```text
outputs/valence_oof_arousal_residual_20260628/
```

Protocol:

```text
For each Valence split:
1. Train an Arousal predictor using train songs only.
2. Predict Arousal for train/val/test songs.
3. Train a Valence residual corrector using train songs only.
4. Select residual strength/model/cap by validation.
5. Report held-out test metrics.
```

Result:

| Model | RMSE | MAE | PCC | CCC | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| Current best | 0.195316 | 0.151793 | 0.797766 | 0.784834 | keep |
| Predicted-Arousal residual | 0.195679 | 0.151946 | 0.797234 | 0.784940 | reject |

Interpretation:

```text
CCC improves only +0.000106, while RMSE/MAE/PCC get worse. The strictly
trained Arousal auxiliary predictor is not strong enough to provide reliable
Valence information under this protocol.
```

## Trial 2: MERT hidden_6 + All Librosa Features

Script:

```text
scripts/train_pmemo_valence_multisplit_full_librosa.py
```

Output:

```text
outputs/valence_multisplit_full_librosa_20260628/
```

Feature:

```text
MERT hidden_6 768 + Librosa 118
groups = mfcc, spectral, harmonic, energy, tempo
```

Result:

| Model | RMSE | MAE | PCC | CCC | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| Current best | 0.195316 | 0.151793 | 0.797766 | 0.784834 | keep |
| MERT + all Librosa | 0.216849 | 0.170641 | 0.761971 | 0.752211 | reject |

Interpretation:

```text
Adding all Librosa features increases input dimensionality and appears to
increase overfitting, especially on split_02 and split_03. More features do not
automatically improve Valence.
```

## Trial 3: Current Best + Full-Librosa Ensemble

Output:

```text
outputs/valence_current_full_librosa_ensemble_20260628/
```

Result:

| Model | RMSE | MAE | PCC | CCC | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| Current best | 0.195316 | 0.151793 | 0.797766 | 0.784834 | keep |
| Current + full-Librosa ensemble | 0.195857 | 0.152724 | 0.797930 | 0.785162 | reject |

Interpretation:

```text
The full-Librosa model has slight correlation complementarity, but the ensemble
increases RMSE/MAE. It is not a better Teacher candidate.
```

## Trial 4: MERT hidden_6 + Harmonic + Energy + Tempo

Script:

```text
scripts/train_pmemo_valence_multisplit_full_librosa.py \
  --out-dir outputs/valence_multisplit_harm_energy_tempo_20260628 \
  --trial-id h6_harm_energy_tempo_h80_ext125_clip010_ccc005 \
  --feature-groups harmonic energy tempo
```

Result:

| Model | RMSE | MAE | PCC | CCC | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| Current best | 0.195316 | 0.151793 | 0.797766 | 0.784834 | keep |
| MERT + harmonic + energy + tempo | 0.207346 | 0.163667 | 0.773293 | 0.761681 | reject |

Interpretation:

```text
Adding energy/tempo to harmonic features does not solve the Valence bottleneck
and still degrades test metrics.
```

## Conclusion

No model from this round replaces the current Valence best.

Current Valence best remains:

```text
pmemo_valence_prediction_only_rescale_ccc_preserving_20260628
test RMSE/MAE/PCC/CCC = 0.195316/0.151793/0.797766/0.784834
```

This round rules out three tempting directions:

```text
1. Strict predicted-Arousal residual correction is too weak with the current
   auxiliary Arousal predictor.
2. Adding all Librosa features causes overfitting rather than improvement.
3. Adding energy/tempo to harmonic features is not enough to improve Valence.
```

## Next Recommendation

The remaining Valence gap is small for MAE/PCC but still meaningful for RMSE
and CCC. The next useful work should not be blind feature addition. Better next
steps:

1. Inspect worst songs in split_02 and split_03 manually to determine whether
   label noise, genre bias, or polarity ambiguity dominates.
2. Build a stronger song-level Valence polarity estimator with external
   metadata/audio pretraining only if it can be validated cleanly.
3. Consider accepting the current Valence model as a working Teacher for Phase
   2 while reporting the Valence limitation honestly, because Arousal already
   passes and Valence is close but not pass-level.
