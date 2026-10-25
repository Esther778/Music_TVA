from __future__ import annotations

import json

import numpy as np
from scipy.optimize import minimize

from search_validation_ensemble import MERT_ROOT, ROOT, metrics


def simplex_rmse_weights(predictions: np.ndarray, target: np.ndarray):
    count = predictions.shape[0]

    def objective(weights):
        prediction = np.tensordot(weights, predictions, axes=(0, 0))
        return float(np.sqrt(np.mean((prediction - target) ** 2)))

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
    return weights, objective(weights)


def main() -> None:
    target = np.load(MERT_ROOT / "data/datasets/mert_va/frame/Y_val.npy")
    sources = {
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
    }
    names = list(sources)
    predictions = np.stack([np.load(path) for path in sources.values()])
    if predictions.shape[1:] != target.shape:
        raise ValueError(
            f"Prediction shape {predictions.shape[1:]} != target {target.shape}."
        )

    shared_weights, _ = simplex_rmse_weights(predictions, target)
    shared_prediction = np.tensordot(shared_weights, predictions, axes=(0, 0))
    dimension_prediction = np.empty_like(target)
    dimension_weights = {}
    for dimension, dimension_name in enumerate(("valence", "arousal")):
        weights, rmse = simplex_rmse_weights(
            predictions[:, :, dimension], target[:, dimension]
        )
        dimension_weights[dimension_name] = {
            "weights": dict(zip(names, weights.tolist())),
            "rmse": rmse,
        }
        dimension_prediction[:, dimension] = np.tensordot(
            weights, predictions[:, :, dimension], axes=(0, 0)
        )
    result = {
        "split": "validation",
        "method": "nonnegative SLSQP, weights sum to one",
        "selection_metric": "frame-level RMSE",
        "test_used": False,
        "best_shared": {
            "weights": dict(zip(names, shared_weights.tolist())),
            "metrics": metrics(shared_prediction, target),
        },
        "best_dimension_specific": {
            "weights": dimension_weights,
            "metrics": metrics(dimension_prediction, target),
        },
    }
    output = ROOT / "outputs/all_validation_ensemble_search.json"
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    np.save(
        ROOT / "outputs/all_validation_ensemble_pred.npy",
        dimension_prediction.astype(np.float32),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
