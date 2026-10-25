from __future__ import annotations

import json

import numpy as np

from search_validation_ensemble import MERT_ROOT, ROOT, metrics


def main() -> None:
    target = np.load(
        MERT_ROOT / "data/datasets/mert_va/frame/Y_val.npy"
    )
    names = ["mert_baseline", "gated_mse", "gated_composite", "layer6_gated"]
    predictions = [
        np.load(
            MERT_ROOT
            / "outputs/predictions/bilstm/bilstm_baseline_val_pred.npy"
        ),
        np.load(ROOT / "outputs/mert_handcrafted_dual_gate_mse/val_pred.npy"),
        np.load(
            ROOT / "outputs/mert_handcrafted_dual_gate_composite/val_pred.npy"
        ),
        np.load(
            ROOT / "outputs/mert_layer6_handcrafted_dual_gate_mse/val_pred.npy"
        ),
    ]
    if any(prediction.shape != target.shape for prediction in predictions):
        raise ValueError("Validation prediction shapes do not match Y_val.")
    candidates = []
    for a in range(21):
        for b in range(21 - a):
            for c in range(21 - a - b):
                weights = np.array([a, b, c, 20 - a - b - c]) / 20
                prediction = sum(w * p for w, p in zip(weights, predictions))
                candidates.append(
                    {
                        "weights": dict(zip(names, weights.tolist())),
                        **metrics(prediction, target),
                    }
                )
    best_shared = min(candidates, key=lambda row: row["rmse"])
    dimension_prediction = np.empty_like(target)
    dimension_weights = {}
    for dimension, name in enumerate(("valence", "arousal")):
        best = None
        for row in candidates:
            weights = row["weights"]
            prediction = sum(
                weights[model] * values[:, dimension]
                for model, values in zip(names, predictions)
            )
            rmse = float(
                np.sqrt(np.mean((prediction - target[:, dimension]) ** 2))
            )
            if best is None or rmse < best["rmse"]:
                best = {"weights": weights, "rmse": rmse}
        dimension_weights[name] = best
        dimension_prediction[:, dimension] = sum(
            best["weights"][model] * values[:, dimension]
            for model, values in zip(names, predictions)
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
    output = ROOT / "outputs/layer6_validation_ensemble_search.json"
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
