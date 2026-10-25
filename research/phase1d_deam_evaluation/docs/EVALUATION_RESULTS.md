# DEAM Multi-Level Evaluation Results

Last updated: 2026-06-21

## Evaluation Definition

```text
model=MERT+BiLSTM baseline
label scale=[-1,+1]
label start=15.0 seconds
frame hop=0.5 seconds
split=song-level 1441/180/181
```

Fixed windows are non-overlapping complete windows within each song. They are
not music sections or lyric sentences.

## Test Results By Level

| Level | Window | RMSE | MAE | Mean PCC | Mean CCC |
| --- | ---: | ---: | ---: | ---: | ---: |
| frame | 0.5 s | 0.202452 | 0.160708 | 0.576296 | 0.538263 |
| fixed window | 2 s | 0.200898 | 0.159458 | 0.581807 | 0.541806 |
| fixed window | 5 s | 0.198859 | 0.157768 | 0.588486 | 0.546193 |
| fixed window | 10 s | 0.196395 | 0.155410 | 0.595831 | 0.550504 |
| fixed window | 15 s | 0.194549 | 0.153845 | 0.601463 | 0.554439 |
| clip | full clip | 0.187807 | 0.149477 | 0.643315 | 0.601326 |

Longer aggregation improves stability, but PCC and CCC remain below the
graduation target.

## Label Shift Diagnostic

Validation RMSE is lowest at `0 seconds`:

```text
shift=0s validation RMSE=0.219934
shift=+1s validation RMSE=0.220794
shift=-1s validation RMSE=0.220929
```

Conclusion: there is no evidence that a global integer-second label shift
should replace the current alignment. Test differences around `+1s` are too
small and were not used for selection.

## Prediction Smoothing

Validation selected:

```text
Gaussian smoothing sigma=2 frames
```

Frame-level effect:

| Split | Variant | RMSE | MAE | Mean PCC | Mean CCC |
| --- | --- | ---: | ---: | ---: | ---: |
| validation | raw | 0.219934 | 0.171226 | 0.674311 | 0.627697 |
| validation | smoothed | 0.218235 | 0.169795 | 0.680004 | 0.630933 |
| test | raw | 0.202452 | 0.160708 | 0.576296 | 0.538263 |
| test | smoothed | 0.200521 | 0.159138 | 0.584035 | 0.542662 |

Smoothing is useful but insufficient to meet the graduation target.

## Affine Calibration

Validation-fitted parameters:

```text
Valence scale=0.951748 bias=-0.060971
Arousal scale=0.952277 bias=-0.003931
```

Calibration worsened test frame RMSE and mean CCC:

```text
smoothed test RMSE=0.200521, mean CCC=0.542662
calibrated test RMSE=0.201922, mean CCC=0.527475
```

Decision: do not use this affine calibration in the Teacher.

## Diagnosis

- Evaluation and song-level reconstruction are correct.
- The current global time alignment is already best among tested integer shifts.
- Smoothing produces only a small gain.
- Calibration does not transfer reliably.
- The main remaining bottleneck is feature representation, especially Valence.

Next model experiment:

```text
MERT + handcrafted feature fusion
dimension-specific gates
separate Valence and Arousal BiLSTM branches
```
