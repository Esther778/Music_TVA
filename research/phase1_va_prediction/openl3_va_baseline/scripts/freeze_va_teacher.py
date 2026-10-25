from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


BASE_TRIAL = "mlp_tune_w512_d010_lr2e4_e120"
SEEDS = [2026, 3407, 614]
HIDDEN = [512, 256]
DROPOUT = 0.10


class MLPRegressor(nn.Module):
    def __init__(self, input_dim: int = 512):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
def predict(model: nn.Module, features: np.ndarray, batch_size: int = 2048) -> np.ndarray:
    outputs = []
    for start in range(0, len(features), batch_size):
        batch = torch.from_numpy(features[start : start + batch_size]).float()
        outputs.append(model(batch).numpy())
    return np.concatenate(outputs, axis=0)


def relative_to_root(path: Path, root: Path) -> str:
    return str(path.resolve().relative_to(root.resolve()))


def main() -> None:
    baseline_root = Path(__file__).resolve().parents[1]
    project_root = baseline_root.parents[2]
    dataset_dir = baseline_root / "artifacts" / "datasets" / "mlp"
    result_dir = baseline_root / "outputs" / "results" / "mlp_tuning"
    model_dir = baseline_root / "outputs" / "models" / "mlp_tuning"
    norm_path = baseline_root / "outputs" / "models" / "mlp" / "mlp_input_norm.npz"
    config_path = baseline_root / "configs" / "frozen_teacher_v1.json"
    audit_path = result_dir / "frozen_teacher_v1_audit.json"

    x_train = np.load(dataset_dir / "X_train.npy")
    x_val = np.load(dataset_dir / "X_val.npy")
    y_val = np.load(dataset_dir / "Y_val.npy")
    x_test = np.load(dataset_dir / "X_test.npy")
    y_test = np.load(dataset_dir / "Y_test.npy")
    norm = np.load(norm_path)
    expected_mean = x_train.mean(axis=0, keepdims=True)
    expected_std = x_train.std(axis=0, keepdims=True) + 1e-8
    if not np.array_equal(norm["mean"], expected_mean):
        raise ValueError("Saved input mean does not match the current training dataset.")
    if not np.array_equal(norm["std"], expected_std):
        raise ValueError("Saved input std does not match the current training dataset.")

    x_val = (x_val - norm["mean"]) / norm["std"]
    x_test = (x_test - norm["mean"]) / norm["std"]

    candidates = []
    val_predictions = []
    test_predictions = []
    for seed in SEEDS:
        checkpoint = model_dir / f"{BASE_TRIAL}_seed{seed}_best.pt"
        summary_path = result_dir / f"{BASE_TRIAL}_seed{seed}_summary.json"
        with summary_path.open("r", encoding="utf-8") as handle:
            training_summary = json.load(handle)

        model = MLPRegressor(input_dim=x_val.shape[1])
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        model.eval()
        val_pred = predict(model, x_val)
        test_pred = predict(model, x_test)
        val_metrics = metrics(val_pred, y_val)
        test_metrics = metrics(test_pred, y_test)
        if abs(val_metrics["rmse"] - float(training_summary["val_rmse"])) > 1e-6:
            raise ValueError(f"Validation metric mismatch for seed {seed}.")
        if abs(test_metrics["rmse"] - float(training_summary["test_rmse"])) > 1e-6:
            raise ValueError(f"Test metric mismatch for seed {seed}.")

        candidates.append(
            {
                "seed": seed,
                "best_epoch": int(training_summary["best_epoch"]),
                "checkpoint": relative_to_root(checkpoint, project_root),
                "checkpoint_sha256": sha256(checkpoint),
                "val": val_metrics,
                "test": test_metrics,
            }
        )
        val_predictions.append(val_pred)
        test_predictions.append(test_pred)

    best_single = min(candidates, key=lambda row: row["val"]["rmse"])
    ensemble_val = metrics(np.mean(val_predictions, axis=0), y_val)
    ensemble_test = metrics(np.mean(test_predictions, axis=0), y_test)
    use_ensemble = ensemble_val["rmse"] <= best_single["val"]["rmse"]
    selected_mode = "mean_ensemble" if use_ensemble else "single_checkpoint"

    teacher = {
        "teacher_model_id": "openl3_mlp_w512_v1",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen_for_phase2_pseudo_va",
        "selection_rule": (
            "Select architecture and deployment mode by validation RMSE only; "
            "test metrics are final reporting metrics."
        ),
        "selected_mode": selected_mode,
        "feature_backbone": {
            "name": "OpenL3",
            "input_repr": "mel256",
            "content_type": "music",
            "embedding_size": 512,
            "hop_size_sec": 0.5,
            "sample_rate": 44100,
        },
        "head": {
            "family": "MLP",
            "hidden": HIDDEN,
            "dropout": DROPOUT,
            "output_order": ["valence", "arousal"],
            "parameter_count": 394498,
        },
        "normalization": {
            "path": relative_to_root(norm_path, project_root),
            "sha256": sha256(norm_path),
            "source": "OpenL3 MLP training split statistics",
        },
        "ensemble": {
            "aggregation": "arithmetic_mean_of_frame_predictions",
            "members": candidates,
            "val": ensemble_val,
            "test": ensemble_test,
        },
        "best_single_by_validation": best_single,
        "evaluation_protocol": {
            "dataset": "DEAM dynamic VA annotations",
            "split": "frame-level random 80/10/10 split, seed 42",
            "critical_limitation": (
                "Frames from the same song may occur in train, validation, and test. "
                "Reported metrics are not a song-level generalization estimate and "
                "must not be directly compared with the current song-level MERT split."
            ),
        },
        "phase2_usage": {
            "purpose": "Generate 0.5-second pseudo-VA trajectories for reviewed complete songs.",
            "claim_boundary": "Pseudo-VA is a model-derived estimate, not human ground truth.",
        },
    }

    config_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(teacher, handle, ensure_ascii=False, indent=2)
    with audit_path.open("w", encoding="utf-8") as handle:
        json.dump(teacher, handle, ensure_ascii=False, indent=2)

    print(json.dumps(teacher, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
