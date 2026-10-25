# PMEmo Valence Song-Global Auxiliary Head Trials

Report time: 2026-06-26 Asia/Shanghai

Current research progress: Valence remains below the current minimum Teacher
threshold. The active best before this round is:

```text
model_id = pmemo_valence_mert_hidden6_harmonic_extreme_weighted_v1
trial_id = h6_harm_h80_ext125_clip010_ccc005
test RMSE/MAE/PCC/CCC = 0.205673/0.164567/0.783400/0.767304
```

## Purpose

Previous diagnostics showed that many Valence errors are whole-song mean
failures. This round adds a lightweight song-global auxiliary head on top of
the hidden_6 MERT + harmonic BiLSTM:

```text
frame path = BiLSTM sequence output -> frame Valence head
global path = masked mean pooling over BiLSTM sequence output -> song mean Valence head
loss = extreme-weighted frame MSE + frame CCC loss + global mean loss
```

The goal is to help the model estimate each song's overall Valence level while
preserving frame-level prediction quality.

## Script

```text
research/phase1f_external_va_datasets/pmemo/scripts/train_pmemo_mert_hidden6_harmonic_global_aux.py
```

## Results

| Trial | Best Epoch | Frame RMSE | Frame MAE | Frame PCC | Frame CCC | Global RMSE | Global PCC | Global CCC | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| h6_harm_h80_globalmse005_ext125 | 25 | 0.213870 | 0.169718 | 0.756001 | 0.746612 | 0.197235 | 0.795511 | 0.765543 | weaker frame metrics |
| h6_harm_h80_globalmse010_ext125 | 36 | 0.209428 | 0.169266 | 0.765446 | 0.753392 | 0.195682 | 0.798443 | 0.775573 | best global PCC/CCC, weaker frame metrics |
| h6_harm_h80_globalmse020_ext125 | 27 | 0.215268 | 0.176247 | 0.748848 | 0.733662 | 0.203868 | 0.785358 | 0.742439 | weaker |
| h6_harm_h80_globalmse010_globalccc005 | 15 | 0.223662 | 0.182096 | 0.743651 | 0.735296 | 0.212365 | 0.774102 | 0.743874 | weaker |
| h6_harm_h80_globalmse010_blend010 | 36 | 0.208418 | 0.168563 | 0.767188 | 0.753432 | 0.195682 | 0.798443 | 0.775573 | best frame RMSE in this round, still weaker than current best |

## Comparison With Current Best

Current best:

```text
test RMSE/MAE/PCC/CCC = 0.205673/0.164567/0.783400/0.767304
```

Best global-head frame-level trial:

```text
trial_id = h6_harm_h80_globalmse010_blend010
test RMSE/MAE/PCC/CCC = 0.208418/0.168563/0.767188/0.753432
```

Delta relative to current best:

```text
RMSE +0.002745
MAE  +0.003996
PCC  -0.016213
CCC  -0.013872
```

## Decision

Do not promote any global-head trial. The song-global auxiliary head learns a
useful song-level signal, but the tested integration harms frame-level test
generalization.

## Interpretation

The result is still informative:

```text
The global head reaches song-level test PCC around 0.798, close to the 0.80
minimum target. However, directly sharing this auxiliary objective with the
frame predictor pulls frame-level predictions away from the current best.
```

This suggests the next step should not simply increase global-head weight.
Better options are:

1. Use global prediction only as a separate diagnostic or calibration signal.
2. Try song-balanced frame loss so long songs do not dominate frame-level MSE.
3. Test validation-selected post-hoc correction from global prediction to frame
   prediction, but only if selected strictly on validation.
