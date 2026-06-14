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
    mse = float(np.mean((pred - target) ** 2))
    rmse = float(np.sqrt(mse))
    rv, ra = pearson_np(pred, target)
    return {
        "mse": mse,
        "rmse": rmse,
        "pearson_valence": rv,
        "pearson_arousal": ra,
    }


def count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def assert_finite_array(name: str, array: np.ndarray) -> None:
    if np.isfinite(array).all():
        return
    total = int(array.size)
    invalid = int(total - np.isfinite(array).sum())
    raise ValueError(f"{name} contains {invalid}/{total} NaN or Inf values.")
