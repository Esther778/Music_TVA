# Model Metric Reporting Standard

Date: 2026-06-24

For all future VA model reports, include both raw metric values and a threshold
gap table. The user should be able to immediately see how far the current model
is from passable, good, and excellent performance.

## Canonical Scale

Unless explicitly stated otherwise, report RMSE and MAE on the canonical
DEAM-style `[-1, 1]` Valence-Arousal scale.

If a dataset uses `[0, 1]`, also mention that RMSE and MAE are roughly half of
the `[-1, 1]` values after linear scale conversion. PCC and CCC do not change
under linear rescaling.

## Thresholds

| Level | RMSE | MAE | Pearson r | CCC |
| --- | ---: | ---: | ---: | ---: |
| Minimum pass | <= 0.18 | <= 0.15 | >= 0.80 | >= 0.80 |
| Good | <= 0.15 | <= 0.12 | >= 0.85 | >= 0.85 |
| Excellent | <= 0.10 | <= 0.08 | >= 0.90 | >= 0.90 |

These thresholds are project targets. They are stricter than many older MER
baselines and should not be claimed as universal published standards.

## Required Report Format

Every model result should include:

```text
model name
dataset
split protocol
label scale
feature type
checkpoint path
Valence metrics if applicable
Arousal metrics if applicable
mean metrics if applicable
```

Every key model report should include:

| Metric | Current | Minimum pass | Gap to pass | Good | Gap to good | Excellent | Gap to excellent | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |

For metrics where lower is better, gap means:

```text
current - threshold
```

For metrics where higher is better, gap means:

```text
threshold - current
```

Negative or zero gap means the threshold is met.

## Interpretation Rule

Do not use only averaged VA metrics when one dimension is weak. Always report
Valence and Arousal separately when both are predicted.

If one dimension has already reached the target and the other has not, state
that explicitly and focus optimization on the weaker dimension.
