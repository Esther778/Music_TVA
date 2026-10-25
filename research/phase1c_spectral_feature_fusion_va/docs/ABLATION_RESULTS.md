# Pure-Audio Feature Ablation Results

Last updated: 2026-06-21

## Protocol

```text
dataset=DEAM averaged dynamic VA
label range=[-1,+1]
label start=15.0 seconds
split=song-level 1441/180/181
seed=2026 for new Phase 1C models
selection=validation RMSE only
historical emotion input=none
```

All rows are pure-audio systems. Test metrics are reported after selecting the
checkpoint epoch by validation RMSE.

## Overall Comparison

| Model | Validation RMSE | Test RMSE | Test MAE | Mean Pearson r | Test mean CCC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Training-mean baseline | - | 0.249551 | 0.208344 | 0.000000 | 0.000000 |
| OpenL3 + MLP | 0.220933 | 0.208569 | 0.163968 | 0.539056 | 0.501581 |
| **MERT + BiLSTM** | **0.219934** | 0.202452 | 0.160708 | **0.576296** | **0.538263** |
| Handcrafted + BiLSTM | 0.246835 | 0.204089 | 0.160564 | 0.528830 | 0.455992 |
| Log-Mel DAMFF-inspired | 0.235476 | 0.213163 | 0.168448 | 0.520700 | 0.455621 |
| Log-Mel + handcrafted fusion | 0.239780 | **0.201222** | **0.155877** | 0.553726 | 0.486217 |

MERT+BiLSTM remains the selected model because it has the lowest validation
RMSE. The fusion model's lower test RMSE cannot be used to select it after
observing the test set.

## Valence Metrics

| Model | RMSE | MAE | PCC | CCC |
| --- | ---: | ---: | ---: | ---: |
| Training-mean baseline | 0.235845 | 0.196828 | 0.000000 | 0.000000 |
| OpenL3 + MLP | 0.220053 | 0.171007 | 0.422190 | 0.385054 |
| **MERT + BiLSTM** | **0.205797** | **0.163710** | **0.511522** | **0.464444** |
| Handcrafted + BiLSTM | 0.218433 | 0.169896 | 0.377841 | 0.271342 |
| Log-Mel DAMFF-inspired | 0.216116 | 0.166683 | 0.412244 | 0.323860 |
| Log-Mel + handcrafted fusion | 0.215536 | 0.164720 | 0.411554 | 0.314980 |

MERT is clearly strongest for Valence. Neither log-Mel nor the current
handcrafted fusion closes the Valence gap.

## Arousal Metrics

| Model | RMSE | MAE | PCC | CCC |
| --- | ---: | ---: | ---: | ---: |
| Training-mean baseline | 0.262543 | 0.219859 | 0.000000 | 0.000000 |
| OpenL3 + MLP | 0.196414 | 0.156928 | 0.655921 | 0.618109 |
| MERT + BiLSTM | 0.199050 | 0.157707 | 0.641070 | 0.612083 |
| Handcrafted + BiLSTM | 0.188658 | 0.151232 | 0.679819 | 0.640643 |
| Log-Mel DAMFF-inspired | 0.210168 | 0.170214 | 0.629156 | 0.587381 |
| **Log-Mel + handcrafted fusion** | **0.185808** | **0.147034** | **0.695898** | **0.657454** |

Handcrafted descriptors provide strong Arousal information. Fusion improves
all four Arousal metrics relative to the current MERT baseline.

## Interpretation

1. The current DAMFF-inspired log-Mel baseline is useful as an independent
   comparison, but it does not outperform MERT.
2. Handcrafted audio features are weak for Valence but strong for Arousal.
3. Log-Mel and handcrafted fusion improves Arousal, but its validation RMSE is
   worse than MERT because Valence remains difficult.
4. The next feature-fusion experiment should use:

```text
MERT representation + handcrafted descriptors
shared temporal encoder + separate Valence/Arousal heads
```

5. A dimension-aware design is justified:
   MERT should carry most Valence information, while handcrafted energy,
   rhythm, and spectral features should supplement Arousal.

## Method Boundaries

- `DAMFF-inspired` means the local architecture borrows multi-scale
  convolution, channel/spatial attention, and BiLSTM concepts. It is not
  presented as an exact reproduction.
- The user-supplied DAMFF values are retained only as literature context:
  DEAM2015 Arousal `0.2401`, Valence `0.3403`; MER1101 Arousal `0.1494`,
  Valence `0.2595`.
- ENMI's reported `0.0440/0.0352` is not treated as a comparable pure-audio
  result because its rolling-forecast input may contain historical emotion
  features.
- All new Phase 1C models currently have one seed. Multi-seed verification
  should be reserved for a candidate that improves validation RMSE.

Machine-readable table:

```text
outputs/ablation/comparison.csv
outputs/ablation/comparison.json
```
