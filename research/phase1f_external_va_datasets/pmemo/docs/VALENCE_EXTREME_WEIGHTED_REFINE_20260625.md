# PMEmo Valence Extreme-Weighted Refine Trials

Report time: 2026-06-25 Asia/Shanghai

Current research progress: Valence remains below the current minimum Teacher
threshold. The active best before this round is:

```text
model_id = pmemo_valence_mert_hidden6_harmonic_extreme_weighted_v1
trial_id = h6_harm_h80_ext125_clip010_ccc005
test RMSE/MAE/PCC/CCC = 0.205673/0.164567/0.783400/0.767304
```

## Purpose

This round tests whether the useful extreme-weighted objective can be improved
by small changes around the current best:

```text
feature = MERT hidden_6 768 + Librosa harmonic 36
model = Valence-only BiLSTM, projection 320, hidden_dim 80, dropout 0.30
base loss = extreme-weighted frame MSE + frame CCC loss + clip mean MSE
tested changes = extreme_weight, clip_mse_weight, frame_ccc_weight
```

## Results

| Trial | Best Epoch | Test RMSE | Test MAE | Test PCC | Test CCC | Note |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| h6_harm_h80_ext100_clip008_ccc005 | 19 | 0.205316 | 0.164293 | 0.783217 | 0.766108 | best RMSE/MAE in refine, lower PCC/CCC |
| h6_harm_h80_ext125_clip008_ccc005 | 19 | 0.205610 | 0.164448 | 0.783400 | 0.767131 | close to current best, no full improvement |
| h6_harm_h80_ext150_clip008_ccc005 | 19 | 0.206032 | 0.164734 | 0.783389 | 0.767376 | CCC slightly higher, RMSE/MAE weaker |
| h6_harm_h80_ext150_clip010_ccc008 | 19 | 0.205522 | 0.164833 | 0.783310 | 0.768086 | best CCC in refine, MAE/PCC weaker |
| h6_harm_h80_ext150_clip012_ccc005 | 19 | 0.206008 | 0.164849 | 0.783536 | 0.767674 | PCC/CCC slightly higher, RMSE/MAE weaker |
| h6_harm_h80_ext175_clip008_ccc005 | 19 | 0.206012 | 0.164535 | 0.783667 | 0.767594 | best PCC in refine, RMSE weaker |

## Decision

Do not promote any refine trial as the new current best. None improves all four
primary metrics over `h6_harm_h80_ext125_clip010_ccc005`.

The current best remains:

```text
model_id = pmemo_valence_mert_hidden6_harmonic_extreme_weighted_v1
trial_id = h6_harm_h80_ext125_clip010_ccc005
test RMSE/MAE/PCC/CCC = 0.205673/0.164567/0.783400/0.767304
```

## Interpretation

The extreme-weight/clip-weight direction has reached a local plateau. Changing
`extreme_weight` from 1.0 to 1.75 and `clip_mse_weight` from 0.08 to 0.12 only
shifts tradeoffs between RMSE/MAE and PCC/CCC.

## Threshold Gap Of Current Best

| Metric | Current | Minimum Pass | Gap To Pass |
| --- | ---: | ---: | ---: |
| RMSE | 0.205673 | 0.180000 | +0.025673 |
| MAE | 0.164567 | 0.150000 | +0.014567 |
| PCC | 0.783400 | 0.800000 | +0.016600 |
| CCC | 0.767304 | 0.800000 | +0.032696 |

## Next Step

Stop simple scalar loss-weight tuning for now. The next meaningful experiment
should add a lightweight song-global auxiliary head so the model can better
estimate each song's overall Valence mean, which is the main observed error
pattern.
