from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

from search_validation_ensemble import MERT_ROOT, ROOT, metrics


def main() -> None:
    prediction = np.load(ROOT / "outputs/all_validation_ensemble_pred.npy")
    target = np.load(MERT_ROOT / "data/datasets/mert_va/frame/Y_val.npy")
    index = pd.read_csv(MERT_ROOT / "data/datasets/mert_va/sequence_index.csv")
    validation = index[index["split"] == "val"].sort_values("song_id")
    lengths = validation["frames"].astype(int).tolist()
    if sum(lengths) != len(target):
        raise ValueError(f"Validation lengths sum to {sum(lengths)}, expected {len(target)}.")

    slices = []
    start = 0
    for length in lengths:
        slices.append(slice(start, start + length))
        start += length
    sigmas = (
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        8.0,
        10.0,
        12.0,
        16.0,
        20.0,
        24.0,
    )
    rows = []
    predictions = {}
    for sigma in sigmas:
        smoothed = prediction.copy()
        if sigma > 0:
            for song_slice in slices:
                smoothed[song_slice] = gaussian_filter1d(
                    prediction[song_slice],
                    sigma=sigma,
                    axis=0,
                    mode="nearest",
                )
        predictions[sigma] = smoothed
        rows.append({"sigma_frames": sigma, **metrics(smoothed, target)})
    best = min(rows, key=lambda row: row["rmse"])
    best_prediction = predictions[best["sigma_frames"]]
    result = {
        "split": "validation",
        "test_used": False,
        "frame_hop_sec": 0.5,
        "selection_metric": "frame-level RMSE",
        "smoothing_scope": "within each song only",
        "best": best,
        "all_trials": rows,
    }
    (ROOT / "outputs/ensemble_smoothing_search.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    np.save(
        ROOT / "outputs/all_validation_ensemble_smoothed_pred.npy",
        best_prediction.astype(np.float32),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
