#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from training_utils import ROOT, assert_finite_array, count_parameters, load_json, metrics_np, prefixed_metrics, select_device, set_seed


class MLPRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden: list[int], dropout: float):
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for dim in hidden:
            layers.extend([nn.Linear(prev, dim), nn.ReLU(), nn.Dropout(dropout)])
            prev = dim
        layers.append(nn.Linear(prev, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def weighted_mse(pred: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return torch.mean(((pred - target) ** 2) * weights)


def load_frame_data(data_dir: Path) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray, np.ndarray]:
    norm = np.load(data_dir / "feature_norm.npz")
    mean = norm["mean"]
    std = norm["std"]
    X, Y = {}, {}
    for split in ["train", "val", "test"]:
        X[split] = ((np.load(data_dir / f"X_{split}.npy") - mean) / std).astype(np.float32)
        Y[split] = np.load(data_dir / f"Y_{split}.npy").astype(np.float32)
        assert_finite_array(f"X_{split}", X[split])
        assert_finite_array(f"Y_{split}", Y[split])
    return X, Y, mean, std


@torch.no_grad()
def evaluate(model: nn.Module, X: np.ndarray, Y: np.ndarray, device: torch.device, batch_size: int) -> tuple[float, dict[str, float], np.ndarray]:
    model.eval()
    preds = []
    loader = DataLoader(TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(Y).float()), batch_size=batch_size, shuffle=False)
    total_loss = 0.0
    total = 0
    criterion = nn.MSELoss(reduction="sum")
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        pred = model(xb)
        loss = criterion(pred, yb)
        total_loss += float(loss.detach().cpu())
        total += len(xb)
        preds.append(pred.detach().cpu().numpy())
    pred_np = np.concatenate(preds, axis=0)
    return total_loss / max(total, 1), metrics_np(pred_np, Y), pred_np


def run_trial(trial: dict[str, Any], X: dict[str, np.ndarray], Y: dict[str, np.ndarray], device: torch.device, out_root: Path, seed: int, patience: int) -> dict[str, Any]:
    set_seed(seed)
    trial_id = trial["trial_id"]
    model_dir = out_root / "models" / "mlp"
    metric_dir = out_root / "metrics" / "mlp"
    pred_dir = out_root / "predictions" / "mlp"
    model_dir.mkdir(parents=True, exist_ok=True)
    metric_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    model = MLPRegressor(X["train"].shape[1], list(trial["hidden"]), float(trial["dropout"])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(trial["lr"]), weight_decay=float(trial["weight_decay"]))
    loss_weights = torch.tensor(trial.get("loss_weights", [1.0, 1.0]), dtype=torch.float32, device=device)
    grad_clip = float(trial.get("grad_clip", 0.0))
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X["train"]).float(), torch.from_numpy(Y["train"]).float()),
        batch_size=int(trial["batch_size"]),
        shuffle=True,
    )

    history = []
    best_val_rmse = float("inf")
    best_epoch = 0
    bad_epochs = 0
    best_path = model_dir / f"{trial_id}_best.pt"

    for epoch in range(1, int(trial["epochs"]) + 1):
        model.train()
        train_loss = 0.0
        total = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = weighted_mse(pred, yb, loss_weights)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            train_loss += float(loss.detach().cpu()) * len(xb)
            total += len(xb)
        _, val_metrics, _ = evaluate(model, X["val"], Y["val"], device, int(trial["batch_size"]))
        row = {
            "trial_id": trial_id,
            "epoch": epoch,
            "train_mse": train_loss / max(total, 1),
            "val_rmse": val_metrics["rmse"],
            "val_pearson_valence": val_metrics["pearson_valence"],
            "val_pearson_arousal": val_metrics["pearson_arousal"],
        }
        history.append(row)
        print(
            f"{trial_id} epoch={epoch:03d} train_mse={row['train_mse']:.5f} "
            f"val_rmse={row['val_rmse']:.5f} val_rv={row['val_pearson_valence']:.3f} val_ra={row['val_pearson_arousal']:.3f}"
        )
        if not np.isfinite(val_metrics["rmse"]):
            raise ValueError(f"{trial_id} produced non-finite validation RMSE at epoch {epoch}.")
        if val_metrics["rmse"] < best_val_rmse:
            best_val_rmse = val_metrics["rmse"]
            best_epoch = epoch
            bad_epochs = 0
            torch.save({"model_state": model.state_dict(), "trial": trial, "input_dim": X["train"].shape[1]}, best_path)
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    if best_epoch == 0 or not best_path.is_file():
        raise RuntimeError(f"{trial_id} did not produce a valid checkpoint.")

    with (metric_dir / f"{trial_id}_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    split_metrics: dict[str, dict[str, float]] = {}
    for split in ["train", "val", "test"]:
        _, m, pred = evaluate(model, X[split], Y[split], device, int(trial["batch_size"]))
        split_metrics[split] = m
        np.save(pred_dir / f"{trial_id}_{split}_pred.npy", pred.astype(np.float32))

    summary = {
        "model_family": "mlp",
        "trial_id": trial_id,
        "best_epoch": best_epoch,
        "best_val_rmse": best_val_rmse,
        "parameters": count_parameters(model),
        "config_json": json.dumps(trial, ensure_ascii=False),
        "checkpoint": str(best_path),
    }
    for split in ["train", "val", "test"]:
        summary.update(prefixed_metrics(split, split_metrics[split]))
    summary["train_val_rmse_gap"] = summary["val_rmse"] - summary["train_rmse"]
    with (metric_dir / f"{trial_id}_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MERT+DEAM MLP VA heads.")
    parser.add_argument("--plan", default=str(ROOT / "configs" / "stage3_training_plan.json"))
    parser.add_argument("--data-dir", default=str(ROOT / "data" / "datasets" / "mert_va" / "frame"))
    parser.add_argument("--trial", default="", help="Optional single trial_id.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"], help="Training device.")
    args = parser.parse_args()

    plan = load_json(Path(args.plan))
    set_seed(int(plan["seed"]))
    device = select_device(args.device)
    print(f"device={device}")
    X, Y, _, _ = load_frame_data(Path(args.data_dir))
    trials = plan["mlp_trials"]
    if args.trial:
        trials = [trial for trial in trials if trial["trial_id"] == args.trial]
        if not trials:
            raise ValueError(f"No MLP trial matched --trial={args.trial}")
    summaries = [run_trial(trial, X, Y, device, ROOT / "outputs", int(plan["seed"]), int(plan["early_stopping_patience"])) for trial in trials]

    out_path = ROOT / "outputs" / "metrics" / "mlp" / "mlp_trial_summary.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
