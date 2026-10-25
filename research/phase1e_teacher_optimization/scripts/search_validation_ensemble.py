from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parents[1]
MERT_ROOT = PROJECT_ROOT / "research" / "phase1b_bert_deam_va_training"


def metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    error = prediction - target
    pred_centered = prediction - prediction.mean(axis=0, keepdims=True)
    target_centered = target - target.mean(axis=0, keepdims=True)
    pcc = np.sum(pred_centered * target_centered, axis=0) / np.maximum(
        np.sqrt(
            np.sum(pred_centered**2, axis=0)
            * np.sum(target_centered**2, axis=0)
        ),
        1e-8,
    )
    pred_mean = prediction.mean(axis=0)
    target_mean = target.mean(axis=0)
    covariance = ((prediction - pred_mean) * (target - target_mean)).mean(axis=0)
    ccc = 2 * covariance / np.maximum(
        prediction.var(axis=0)
        + target.var(axis=0)
        + (pred_mean - target_mean) ** 2,
        1e-8,
    )
    return {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "rmse_valence": float(np.sqrt(np.mean(error[:, 0] ** 2))),
        "rmse_arousal": float(np.sqrt(np.mean(error[:, 1] ** 2))),
        "pcc_valence": float(pcc[0]),
        "pcc_arousal": float(pcc[1]),
        "mean_pcc": float(np.mean(pcc)),
        "ccc_valence": float(ccc[0]),
        "ccc_arousal": float(ccc[1]),
        "mean_ccc": float(np.mean(ccc)),
    }


def main() -> None:
    target = np.load(
        MERT_ROOT / "data" / "datasets" / "mert_va" / "frame" / "Y_val.npy"
    )
    names = ["mert_baseline", "gated_mse", "gated_composite"]
    predictions = [
        np.load(
            MERT_ROOT
            / "outputs"
            / "predictions"
            / "bilstm"
            / "bilstm_baseline_val_pred.npy"
        ),
        np.load(
            ROOT
            / "outputs"
            / "mert_handcrafted_dual_gate_mse"
            / "val_pred.npy"
        ),
        np.load(
            ROOT
            / "outputs"
            / "mert_handcrafted_dual_gate_composite"
            / "val_pred.npy"
        ),
    ]
    candidates = []
    for first_index in range(21):
        for second_index in range(21 - first_index):
            weights = [
                first_index / 20,
                second_index / 20,
                1.0 - first_index / 20 - second_index / 20,
            ]
            prediction = sum(
                weights[index] * predictions[index] for index in range(3)
            )
            candidates.append(
                {
                    "weights": dict(zip(names, weights)),
                    **metrics(prediction, target),
                }
            )
    best_shared = min(candidates, key=lambda row: row["rmse"])

    dimension_weights = {}
    dimension_prediction = np.empty_like(target)
    for dimension, dimension_name in enumerate(("valence", "arousal")):
        best = None
        for candidate in candidates:
            weights = candidate["weights"]
            prediction = sum(
                weights[names[index]] * predictions[index][:, dimension]
                for index in range(3)
            )
            rmse = float(
                np.sqrt(np.mean((prediction - target[:, dimension]) ** 2))
            )
            if best is None or rmse < best["rmse"]:
                best = {"weights": weights, "rmse": rmse}
        dimension_weights[dimension_name] = best
        dimension_prediction[:, dimension] = sum(
            best["weights"][names[index]] * predictions[index][:, dimension]
            for index in range(3)
        )

    result = {
        "split": "validation",
        "grid_step": 0.05,
        "test_used": False,
        "best_shared_weights": best_shared,
        "best_dimension_specific": {
            "weights": dimension_weights,
            "metrics": metrics(dimension_prediction, target),
        },
    }
    output = ROOT / "outputs" / "validation_ensemble_search.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
