# Valence Improvement Plan

Date: 2026-06-22

## Current Evidence

Current recommended strict song-level validation result:

| Dimension | RMSE | MAE | PCC | CCC |
| --- | ---: | ---: | ---: | ---: |
| Valence | 0.202493 | 0.153771 | 0.724139 | 0.692638 |
| Arousal | 0.164094 | 0.128698 | 0.833909 | 0.822270 |
| Combined | 0.184297 | 0.141234 | 0.779024 | 0.757454 |

Valence prediction distribution:

```text
true mean/std=0.0252/0.2901
pred mean/std=0.0550/0.2200
true range=-0.7683..0.6320
pred range=-0.5710..0.5820
```

The predicted Valence standard deviation is about 24% smaller than the
target standard deviation. Mean bias and range compression explain part of
the CCC gap, but not the full PCC gap.

## Evaluation Governance

The existing validation split has already been used for repeated model
selection. It is frozen for stage confirmation and must not be used for
additional parameter grids.

New experiments will use a song-level shadow validation split created from
the existing training songs. The split should preserve distributions of
song-level mean Valence, mean Arousal, duration, and short/long clip type.

The existing validation split will be evaluated only after a candidate and
all post-processing parameters are frozen on shadow validation. Test remains
locked. Random seeds remain deferred until a single-seed candidate improves
the frozen validation result.

## Experiment 1: Shadow-Validation Post-Processing

For Valence only:

1. compare RMSE-optimal, CCC-optimal, and balanced ensemble objectives;
2. search temporal shift and Gaussian smoothing on shadow validation;
3. fit positive affine calibration on shadow validation;
4. report raw and calibrated distribution statistics.

Calibration can improve CCC by correcting mean and variance, but it cannot
improve Pearson correlation under a positive affine transform.

## Experiment 2: Multi-Scale Trend and Residual Valence Model

Use the established layer-6/final-layer MERT and handcrafted inputs.

```text
shared encoder
-> trend head: 5-second and 10-second Valence targets
-> residual head: raw Valence minus trend
-> final prediction: trend + learned residual scale
```

Training objective:

```text
frame robust regression
+ frame CCC/Pearson
+ 5-second window CCC
+ 10-second window CCC
+ low-weight pairwise ranking
+ low-weight variance matching
```

The trend head targets long-term PCC/CCC. The residual head preserves
frame-level RMSE. This differs from the rejected target-smoothing trial,
which replaced the raw target instead of jointly modeling both components.

Status: completed and rejected. It improved shadow RMSE by only 0.000133,
did not improve shadow PCC/CCC, and transferred poorly to frozen validation.

## Experiment 3: Learned MERT Layer Mix

If Experiment 2 improves shadow validation, extract selected intermediate
MERT layers for the required split and learn a constrained scalar mix.
Start with layers 3, 6, 9, and final. Do not expand to MERT-330M until the
95M scalar-mix experiment demonstrates that representation diversity helps
Valence.

Before this experiment, replace single shadow-validation selection with
song-level cross-validation or OOF evaluation because the frozen validation
Valence distribution differs materially from the training distribution.

## Deferred Work

- MERT-330M frozen features: expensive; run only after scalar-mix evidence.
- LoRA/adapter fine-tuning: high cost and overfitting risk on DEAM; later.
- OOF stacking: methodologically strongest ensemble option but requires
  retraining every base model across folds; use for final paper evidence,
  not the next quick trial.
- Additional tuning on the current validation split: prohibited.

## Stop Conditions

Stop an experiment family when:

- shadow-validation Valence RMSE does not improve by at least 0.003;
- PCC and CCC do not improve without material RMSE degradation;
- the candidate receives effectively zero weight in a shadow-validation
  ensemble;
- train-shadow generalization gap increases materially.

Only a frozen candidate that improves the existing validation result should
proceed to three random seeds. Test evaluation occurs once after the final
pipeline is fixed.
