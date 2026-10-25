from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize

from search_validation_ensemble import MERT_ROOT, ROOT, metrics


def optimize(predictions: np.ndarray, target: np.ndarray):
    def objective(weights):
        prediction = np.tensordot(weights, predictions, axes=(0, 0))
        return float(np.sqrt(np.mean((prediction - target) ** 2)))

    count = len(predictions)
    result = minimize(
        objective,
        np.full(count, 1.0 / count),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * count,
        constraints={"type": "eq", "fun": lambda weights: weights.sum() - 1.0},
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    if not result.success:
        raise RuntimeError(result.message)
    weights = np.maximum(result.x, 0)
    weights /= weights.sum()
    return weights


def main() -> None:
    target = np.load(MERT_ROOT / "data/datasets/mert_va/frame/Y_val.npy")
    two_dimensional = {
        "mert_baseline": MERT_ROOT
        / "outputs/predictions/bilstm/bilstm_baseline_val_pred.npy",
        "gated_mse": ROOT / "outputs/mert_handcrafted_dual_gate_mse/val_pred.npy",
        "gated_composite": ROOT
        / "outputs/mert_handcrafted_dual_gate_composite/val_pred.npy",
        "layer6_gated": ROOT
        / "outputs/mert_layer6_handcrafted_dual_gate_mse/val_pred.npy",
        "layer6_valence_weighted": ROOT
        / "outputs/mert_layer6_handcrafted_valence_weighted/val_pred.npy",
        "multilayer_fusion": ROOT
        / "outputs/mert_layer6_final_handcrafted_valence_fusion/val_pred.npy",
        "multilayer_regularized": ROOT
        / "outputs/mert_layer6_final_handcrafted_valence_fusion_regularized/val_pred.npy",
        "target_smoothed": ROOT
        / "outputs/mert_multilayer_valence_target_smoothed/val_pred.npy",
    }
    names = list(two_dimensional)
    predictions = np.stack([np.load(path) for path in two_dimensional.values()])
    valence_names = names + ["harmonic_context"]
    valence_predictions = np.concatenate(
        [
            predictions[:, :, 0],
            np.load(
                ROOT / "outputs/valence_harmonic_context_tcn_bilstm/val_pred.npy"
            ).reshape(1, -1),
        ],
        axis=0,
    )
    valence_weights = optimize(valence_predictions, target[:, 0])
    arousal_weights = optimize(predictions[:, :, 1], target[:, 1])
    combined = np.empty_like(target)
    combined[:, 0] = np.tensordot(
        valence_weights, valence_predictions, axes=(0, 0)
    )
    combined[:, 1] = np.tensordot(
        arousal_weights, predictions[:, :, 1], axes=(0, 0)
    )

    index = pd.read_csv(MERT_ROOT / "data/datasets/mert_va/sequence_index.csv")
    lengths = (
        index[index["split"] == "val"]
        .sort_values("song_id")["frames"]
        .astype(int)
        .tolist()
    )
    smoothed = combined.copy()
    start = 0
    for length in lengths:
        stop = start + length
        smoothed[start:stop] = gaussian_filter1d(
            combined[start:stop], sigma=2.0, axis=0, mode="nearest"
        )
        start = stop
    result = {
        "split": "validation",
        "test_used": False,
        "selection_metric": "frame-level dimension RMSE",
        "weights": {
            "valence": dict(zip(valence_names, valence_weights.tolist())),
            "arousal": dict(zip(names, arousal_weights.tolist())),
        },
        "raw_metrics": metrics(combined, target),
        "recommended_song_bounded_smoothing": {
            "sigma_frames": 2.0,
            "metrics": metrics(smoothed, target),
        },
    }
    (ROOT / "outputs/final_validation_ensemble.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    np.save(ROOT / "outputs/final_validation_ensemble_pred.npy", combined)
    np.save(ROOT / "outputs/final_validation_ensemble_smoothed_pred.npy", smoothed)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
