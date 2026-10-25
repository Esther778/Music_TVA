from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


SEEDS = [2026, 3407, 614]
HIDDEN = [512, 256]
DROPOUT = 0.10
METRICS = [
    "val_rmse",
    "val_mae",
    "val_rmse_valence",
    "val_rmse_arousal",
    "val_pearson_valence",
    "val_pearson_arousal",
    "val_ccc_valence",
    "val_ccc_arousal",
    "val_mean_ccc",
    "test_rmse",
    "test_mae",
    "test_rmse_valence",
    "test_rmse_arousal",
    "test_pearson_valence",
    "test_pearson_arousal",
    "test_ccc_valence",
    "test_ccc_arousal",
    "test_mean_ccc",
]


class MLPRegressor(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, HIDDEN[0]),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN[0], HIDDEN[1]),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN[1], 2),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


def metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    error = pred - target
    pred_centered = pred - pred.mean(axis=0, keepdims=True)
    target_centered = target - target.mean(axis=0, keepdims=True)
    pearson = np.sum(pred_centered * target_centered, axis=0) / np.maximum(
        np.sqrt(
            np.sum(pred_centered**2, axis=0)
            * np.sum(target_centered**2, axis=0)
        ),
        1e-8,
    )
    pred_mean = pred.mean(axis=0)
    target_mean = target.mean(axis=0)
    covariance = ((pred - pred_mean) * (target - target_mean)).mean(axis=0)
    ccc = (2.0 * covariance) / np.maximum(
        pred.var(axis=0)
        + target.var(axis=0)
        + (pred_mean - target_mean) ** 2,
        1e-8,
    )
    return {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "rmse_valence": float(np.sqrt(np.mean(error[:, 0] ** 2))),
        "rmse_arousal": float(np.sqrt(np.mean(error[:, 1] ** 2))),
        "mae_valence": float(np.mean(np.abs(error[:, 0]))),
        "mae_arousal": float(np.mean(np.abs(error[:, 1]))),
        "pearson_valence": float(pearson[0]),
        "pearson_arousal": float(pearson[1]),
        "ccc_valence": float(ccc[0]),
        "ccc_arousal": float(ccc[1]),
        "mean_ccc": float(np.mean(ccc)),
    }


@torch.no_grad()
def predict(model: nn.Module, features: np.ndarray) -> np.ndarray:
    outputs = []
    for start in range(0, len(features), 2048):
        batch = torch.from_numpy(features[start : start + 2048]).float()
        outputs.append(model(batch).numpy())
    return np.concatenate(outputs, axis=0)


def main() -> None:
    baseline_root = Path(__file__).resolve().parents[1]
    result_dir = (
        baseline_root / "outputs" / "results" / "mlp_song_level_aligned15"
    )
    dataset_dir = (
        baseline_root / "artifacts" / "datasets" / "mlp_song_level_aligned15"
    )
    rows = []
    for seed in SEEDS:
        path = (
            result_dir
            / f"openl3_mlp_song_level_aligned15_seed{seed}_summary.json"
        )
        with path.open("r", encoding="utf-8") as handle:
            rows.append(json.load(handle))

    aggregate = {
        "experiment_id": "openl3_mlp_song_level_aligned15_3seed_v2",
        "seeds": SEEDS,
        "selection_rule": "Checkpoint epoch selected by validation RMSE only.",
        "rows": [
            {
                "seed": row["seed"],
                "best_epoch": row["best_epoch"],
                "checkpoint": row["checkpoint"],
                **{metric: row[metric] for metric in METRICS},
            }
            for row in rows
        ],
    }
    for metric in METRICS:
        values = [float(row[metric]) for row in rows]
        aggregate[f"{metric}_mean"] = statistics.mean(values)
        aggregate[f"{metric}_std"] = statistics.stdev(values)

    norm = np.load(dataset_dir / "feature_norm.npz")
    ensemble = {}
    for split in ("val", "test"):
        features = np.load(dataset_dir / f"X_{split}.npy").astype(np.float32)
        labels = np.load(dataset_dir / f"Y_{split}.npy").astype(np.float32)
        features = (features - norm["mean"]) / norm["std"]
        member_predictions = []
        for row in rows:
            model = MLPRegressor(features.shape[1])
            model.load_state_dict(
                torch.load(row["checkpoint"], map_location="cpu", weights_only=True)
            )
            model.eval()
            member_predictions.append(predict(model, features))
        ensemble[split] = metrics(np.mean(member_predictions, axis=0), labels)
    aggregate["ensemble"] = {
        "aggregation": "arithmetic_mean_of_three_seed_predictions",
        **ensemble,
    }

    with (result_dir / "song_level_3seed_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(aggregate, handle, ensure_ascii=False, indent=2)
    with (result_dir / "song_level_3seed_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "mean", "std"])
        for metric in METRICS:
            writer.writerow(
                [metric, aggregate[f"{metric}_mean"], aggregate[f"{metric}_std"]]
            )
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
