#!/usr/bin/env python3
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        return resolve_device()
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if device.type == "mps" and not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
        raise RuntimeError("MPS was requested but is not available.")
    return device


def rmse_np(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - target) ** 2)))


def ccc_np(pred: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    values = []
    for idx in range(2):
        x = pred[:, idx]
        y = target[:, idx]
        if len(x) < 2:
            values.append(0.0)
            continue
        mean_x = float(np.mean(x))
        mean_y = float(np.mean(y))
        var_x = float(np.var(x))
        var_y = float(np.var(y))
        cov_xy = float(np.mean((x - mean_x) * (y - mean_y)))
        denominator = var_x + var_y + (mean_x - mean_y) ** 2
        values.append(float(2 * cov_xy / denominator) if denominator > 1e-12 else 0.0)
    return values[0], values[1]


def pearson_np(pred: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    values = []
    for idx in range(2):
        x = pred[:, idx]
        y = target[:, idx]
        if len(x) < 2 or np.std(x) < 1e-8 or np.std(y) < 1e-8:
            values.append(0.0)
        else:
            values.append(float(np.corrcoef(x, y)[0, 1]))
    return values[0], values[1]


def metrics_np(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    if not np.isfinite(pred).all():
        raise ValueError("Prediction array contains NaN or Inf values.")
    if not np.isfinite(target).all():
        raise ValueError("Target array contains NaN or Inf values.")
    error = pred - target
    squared_error = error ** 2
    absolute_error = np.abs(error)
    mse = float(np.mean(squared_error))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(absolute_error))
    valence_rmse = float(np.sqrt(np.mean(squared_error[:, 0])))
    arousal_rmse = float(np.sqrt(np.mean(squared_error[:, 1])))
    valence_mae = float(np.mean(absolute_error[:, 0]))
    arousal_mae = float(np.mean(absolute_error[:, 1]))
    rv, ra = pearson_np(pred, target)
    cv, ca = ccc_np(pred, target)
    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "rmse_valence": valence_rmse,
        "rmse_arousal": arousal_rmse,
        "mae_valence": valence_mae,
        "mae_arousal": arousal_mae,
        "pearson_valence": rv,
        "pearson_arousal": ra,
        "ccc_valence": cv,
        "ccc_arousal": ca,
        "mean_ccc": float((cv + ca) / 2),
    }


def count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def assert_finite_array(name: str, array: np.ndarray) -> None:
    if np.isfinite(array).all():
        return
    total = int(array.size)
    invalid = int(total - np.isfinite(array).sum())
    raise ValueError(f"{name} contains {invalid}/{total} NaN or Inf values.")


def prefixed_metrics(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}
