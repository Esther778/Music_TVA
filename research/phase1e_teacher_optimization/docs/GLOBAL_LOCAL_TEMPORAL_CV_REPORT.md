# Global-Local Temporal Cross-Validation Report

Date: 2026-06-22

## Research Question

Can DEAM dynamic Valence prediction be improved by explicitly separating:

1. between-song mean emotion;
2. within-song emotional trajectory?

This distinction is essential for the later section-aware complete-song
analysis. A model that predicts only which songs are globally positive or
negative is not sufficient for analyzing section progression.

## Protocol

- DEAM original training songs: 1,441.
- Five-fold stratified song-level cross-validation.
- Legacy validation used: no.
- Test used: no.
- Feature: frozen MERT-v1-95M layer 6.
- One fixed model configuration.
- Fold checkpoints selected by held-fold frame RMSE.
- Reported metrics include frame micro, song mean, within-song centered,
  first difference, 5-second window, and 10-second window.

## Model

```text
MERT layer6 sequence
├── attention pooling -> song mean Valence
└── centered sequence -> BiLSTM -> zero-mean local deviation

prediction = global mean + learned scale * local deviation
```

The loss separately constrains frame prediction, song mean, local deviation,
local PCC/CCC, first differences, and predicted variance.

## Five-Fold Result

| Metric | Mean | Standard deviation |
| --- | ---: | ---: |
| Frame RMSE | 0.185384 | 0.023274 |
| Frame MAE | 0.143655 | 0.015601 |
| Frame PCC | 0.666099 | 0.055700 |
| Frame CCC | 0.624040 | 0.060640 |
| Within-song PCC | 0.142098 | 0.066894 |
| Song-mean RMSE | 0.162613 | 0.008949 |
| Song-mean PCC | 0.726275 | 0.030015 |
| Song-mean CCC | 0.693067 | 0.029637 |
| 10-second RMSE | 0.183008 | 0.024106 |
| 10-second PCC | 0.673219 | 0.058014 |

Compared with global-local Ridge:

```text
mean RMSE reduction=0.005490
folds with lower RMSE=5/5
mean PCC improvement=0.015864
within-song PCC improvement=0.020494
```

The RMSE acceptance criterion is satisfied. Within-song PCC improves in only
three of five folds, so dynamic generalization is not yet stable.

## Cross-Fitted OOF Stack

The final OOF experiment combines direct layer6 Ridge, global-local Ridge,
and the temporal model. Meta weights for each fold are fitted only on the
other four folds.

| Metric | Result |
| --- | ---: |
| Valence RMSE | **0.183046** |
| Valence MAE | **0.139923** |
| Valence PCC | **0.673847** |
| Valence CCC | **0.631831** |
| Within-song PCC | 0.133477 |
| Song-mean PCC | 0.732268 |
| Song-mean CCC | 0.703598 |
| 10-second RMSE | 0.180933 |
| 10-second PCC | 0.679874 |
| 10-second CCC | 0.636939 |

Cross-fitted affine calibration was rejected because all primary metrics
except within-song RMSE worsened.

## Threshold Assessment

| Valence criterion | Minimum | Current OOF | Status |
| --- | ---: | ---: | --- |
| RMSE | <= 0.16 | 0.183046 | Not reached |
| MAE reference | <= 0.15 | 0.139923 | Reached |
| PCC | >= 0.80 | 0.673847 | Not reached |
| CCC | >= 0.80 | 0.631831 | Not reached |

These are training-set OOF estimates, not Test results.

## Conclusion

The global-local temporal decomposition is the first architecture in this
phase to pass a predeclared five-fold RMSE improvement criterion. It is a
valid architectural improvement, but not a qualifying final Valence model.

The remaining error is representation-limited and dynamic-generalization
limited:

- song-mean prediction is substantially stronger than within-song dynamics;
- first-difference PCC remains about 0.04;
- fold variance remains high.

The next high-value experiment is training-only representation adaptation:

1. retain the global-local head;
2. add a small adapter to MERT layer 6 or top encoder blocks;
3. freeze most MERT parameters;
4. compare using the same five folds and acceptance rule.

MERT-330M should remain deferred until a low-cost adapter demonstrates that
Valence benefits from representation adaptation.
