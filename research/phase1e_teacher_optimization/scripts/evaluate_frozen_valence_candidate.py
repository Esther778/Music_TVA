from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import gaussian_filter1d
from torch.utils.data import DataLoader

from train_gated_fusion import MERT_ROOT, ROOT
from train_shadow_valence import (
    BaselineRegressor,
    ShadowDataset,
    TrendResidualRegressor,
    collate,
    evaluate,
)
from train_valence_harmonic_context import scalar_metrics


def main() -> None:
    config = json.loads(
        (ROOT / "configs/shadow_valence_experiment.json").read_text(encoding="utf-8")
    )
    norm_file = np.load(ROOT / "outputs/full_candidate_feature_norm.npz")
    norms = tuple(
        norm_file[key]
        for key in (
            "layer6_mean", "layer6_std", "final_mean", "final_std",
            "hand_mean", "hand_std",
        )
    )
    index = pd.read_csv(MERT_ROOT / "data/datasets/mert_va/sequence_index.csv")
    rows = index[index["split"] == "val"].copy()
    rows["shadow_split"] = "frozen_val"
    dataset = ShadowDataset(rows, "frozen_val", norms)
    loader = DataLoader(dataset, batch_size=24, shuffle=False, collate_fn=collate)
    first = dataset[0]
    dimensions = (first[0].shape[1], first[1].shape[1], first[2].shape[1])
    device = torch.device("cpu")
    baseline = BaselineRegressor(dimensions, config).to(device)
    trend = TrendResidualRegressor(dimensions, config).to(device)
    baseline.load_state_dict(
        torch.load(
            ROOT / "outputs/full_valence_baseline/best.pt",
            map_location=device,
            weights_only=True,
        )
    )
    trend.load_state_dict(
        torch.load(
            ROOT / "outputs/full_valence_trend/best.pt",
            map_location=device,
            weights_only=True,
        )
    )
    baseline_metrics, baseline_prediction, target, _ = evaluate(
        baseline, loader, device
    )
    trend_metrics, trend_prediction, trend_target, _ = evaluate(
        trend, loader, device
    )
    if not np.allclose(target, trend_target, atol=1e-7):
        raise ValueError("Frozen validation targets differ between models.")
    combined = 0.5 * baseline_prediction.reshape(-1) + 0.5 * trend_prediction.reshape(-1)
    smoothed = combined.copy()
    start = 0
    for length in rows.sort_values("song_id")["frames"].astype(int):
        stop = start + length
        smoothed[start:stop] = gaussian_filter1d(
            combined[start:stop], sigma=8.0, mode="nearest"
        )
        start = stop
    result = {
        "split": "frozen_validation",
        "selection_source": "shadow_val only",
        "test_used": False,
        "candidate": {
            "baseline_weight": 0.5,
            "trend_weight": 0.5,
            "shift_frames": 0,
            "sigma_frames": 8.0,
            "calibration": False,
        },
        "baseline_metrics": baseline_metrics,
        "trend_metrics": trend_metrics,
        "combined_raw_metrics": scalar_metrics(combined, target),
        "combined_smoothed_metrics": scalar_metrics(smoothed, target),
        "distribution": {
            "target_mean": float(target.mean()),
            "prediction_mean": float(smoothed.mean()),
            "target_std": float(target.std()),
            "prediction_std": float(smoothed.std()),
        },
    }
    (ROOT / "outputs/frozen_valence_candidate.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    np.save(ROOT / "outputs/frozen_valence_candidate_pred.npy", smoothed)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
