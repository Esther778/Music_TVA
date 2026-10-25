# Five-Fold MERT Representation Report

Date: 2026-06-22

## Protocol

- Dataset: 1,441 songs from the original DEAM training split.
- Split: stratified five-fold song-level cross-validation.
- Stratification: song-level mean Valence quintiles.
- Fold size: 288-289 songs.
- Legacy validation used: no.
- Test used: no.
- Model: song-balanced Ridge regression, alpha 100.

## Representation Screening

| Representation | RMSE | MAE | PCC | CCC |
| --- | ---: | ---: | ---: | ---: |
| MERT final | 0.215333 | 0.167478 | 0.511855 | 0.461936 |
| 25% layer6 + 75% final | 0.205420 | 0.158858 | 0.570203 | 0.528005 |
| 50% layer6 + 50% final | 0.198452 | 0.152759 | 0.607996 | 0.571240 |
| 75% layer6 + 25% final | 0.195065 | 0.149203 | 0.625832 | 0.592134 |
| MERT layer6 | **0.193463** | **0.147204** | **0.634325** | **0.602373** |

Layer 6 beats the final layer in every fold. Fixed mixing does not beat pure
layer 6.

## Dynamic Error Decomposition

Pure layer-6 OOF prediction:

| Level | RMSE | PCC | CCC |
| --- | ---: | ---: | ---: |
| Song mean | 0.159158 | 0.734419 | 0.703161 |
| Within-song centered | 0.089303 | 0.041133 | 0.040586 |
| First difference | 0.070568 | 0.004340 | 0.001429 |

The apparent frame-level correlation is dominated by between-song
differences. The linear model does not recover local trajectory changes.

## Global-Local Decomposition

A second Ridge system separately predicts song mean and within-song
deviation.

| Metric group | RMSE | PCC | CCC |
| --- | ---: | ---: | ---: |
| Frame micro | 0.190874 | 0.650235 | 0.624113 |
| Song mean | 0.172436 | 0.693359 | 0.680748 |
| Within-song centered | 0.058173 | 0.121603 | 0.054934 |

It lowers mean RMSE by 0.002588 and improves within-song PCC by 0.080470,
but only three of five folds improve. It does not pass the acceptance rule.

## Decision

1. Keep MERT layer 6 as the preferred frozen representation.
2. Do not extract all layer-3/9 features yet.
3. Do not adopt the Ridge global-local model as Teacher.
4. The next model should use a global song-context branch plus a temporal
   local-deviation branch, trained and evaluated under the same five folds.
5. The primary research metric must include within-song centered PCC/CCC,
   not only frame-micro metrics.
