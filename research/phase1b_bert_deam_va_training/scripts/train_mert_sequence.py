#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from training_utils import ROOT, assert_finite_array, count_parameters, load_json, metrics_np, select_device, set_seed


class SequenceDataset(Dataset):
    def __init__(self, index: pd.DataFrame, split: str, mean: np.ndarray, std: np.ndarray):
        self.rows = index[index["split"] == split].sort_values("song_id").to_dict("records")
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        row = self.rows[idx]
        data = np.load(row["sequence_path"])
        X = ((data["X"].astype(np.float32) - self.mean) / self.std).astype(np.float32)
        Y = data["Y"].astype(np.float32)
        assert_finite_array(f"X_song_{row['song_id']}", X)
        assert_finite_array(f"Y_song_{row['song_id']}", Y)
        return torch.from_numpy(X), torch.from_numpy(Y), int(row["song_id"])


def collate(batch):
    lengths = torch.tensor([item[0].shape[0] for item in batch], dtype=torch.long)
    max_len = int(lengths.max())
    feat_dim = batch[0][0].shape[1]
    X = torch.zeros((len(batch), max_len, feat_dim), dtype=torch.float32)
    Y = torch.zeros((len(batch), max_len, 2), dtype=torch.float32)
    mask = torch.zeros((len(batch), max_len), dtype=torch.bool)
    song_ids = []
    for i, (x, y, sid) in enumerate(batch):
        length = x.shape[0]
        X[i, :length] = x
        Y[i, :length] = y
        mask[i, :length] = True
        song_ids.append(sid)
    return X, Y, mask, song_ids


class BiLSTMRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True, dropout=lstm_dropout)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_dim * 2, 2)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        y, _ = self.lstm(x)
        return self.out(self.drop(y))


class TransformerRegressor(nn.Module):
    def __init__(self, input_dim: int, d_model: int, nhead: int, num_layers: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.in_proj = nn.Linear(input_dim, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.out = nn.Linear(d_model, 2)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.in_proj(x)
        pad_mask = ~mask
        y = self.encoder(x, src_key_padding_mask=pad_mask)
        return self.out(y)


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    diff = (pred - target) ** 2
    diff = diff[mask]
    return diff.mean()


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, dict[str, float], np.ndarray, np.ndarray]:
    model.eval()
    losses = []
    preds, targets = [], []
    for X, Y, mask, _ in loader:
        X = X.to(device)
        Y = Y.to(device)
        mask = mask.to(device)
        pred = model(X, mask)
        losses.append(float(masked_mse(pred, Y, mask).detach().cpu()))
        preds.append(pred[mask].detach().cpu().numpy())
        targets.append(Y[mask].detach().cpu().numpy())
    pred_np = np.concatenate(preds, axis=0)
    target_np = np.concatenate(targets, axis=0)
    return float(np.mean(losses)), metrics_np(pred_np, target_np), pred_np, target_np


def make_model(model_family: str, input_dim: int, trial: dict[str, Any]) -> nn.Module:
    if model_family == "bilstm":
        return BiLSTMRegressor(input_dim, int(trial["hidden_dim"]), int(trial["num_layers"]), float(trial["dropout"]))
    if model_family == "transformer":
        return TransformerRegressor(
            input_dim,
            int(trial["d_model"]),
            int(trial["nhead"]),
            int(trial["num_layers"]),
            int(trial["dim_feedforward"]),
            float(trial["dropout"]),
        )
    raise ValueError(model_family)


def run_trial(model_family: str, trial: dict[str, Any], datasets: dict[str, SequenceDataset], device: torch.device, input_dim: int, seed: int, patience: int) -> dict[str, Any]:
    set_seed(seed)
    trial_id = trial["trial_id"]
    model_dir = ROOT / "outputs" / "models" / model_family
    metric_dir = ROOT / "outputs" / "metrics" / model_family
    pred_dir = ROOT / "outputs" / "predictions" / model_family
    model_dir.mkdir(parents=True, exist_ok=True)
    metric_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    model = make_model(model_family, input_dim, trial).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(trial["lr"]), weight_decay=float(trial["weight_decay"]))
    loaders = {
        split: DataLoader(datasets[split], batch_size=int(trial["batch_size"]), shuffle=(split == "train"), collate_fn=collate)
        for split in ["train", "val", "test"]
    }
    history = []
    best_val_rmse = float("inf")
    best_epoch = 0
    bad_epochs = 0
    best_path = model_dir / f"{trial_id}_best.pt"

    for epoch in range(1, int(trial["epochs"]) + 1):
        model.train()
        losses = []
        for X, Y, mask, _ in loaders["train"]:
            X = X.to(device)
            Y = Y.to(device)
            mask = mask.to(device)
            optimizer.zero_grad()
            pred = model(X, mask)
            loss = masked_mse(pred, Y, mask)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        _, val_metrics, _, _ = evaluate(model, loaders["val"], device)
        row = {
            "trial_id": trial_id,
            "epoch": epoch,
            "train_mse": float(np.mean(losses)),
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
            torch.save({"model_state": model.state_dict(), "trial": trial, "input_dim": input_dim, "model_family": model_family}, best_path)
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
        _, m, pred, _ = evaluate(model, loaders[split], device)
        split_metrics[split] = m
        np.save(pred_dir / f"{trial_id}_{split}_pred.npy", pred.astype(np.float32))

    summary = {
        "model_family": model_family,
        "trial_id": trial_id,
        "best_epoch": best_epoch,
        "best_val_rmse": best_val_rmse,
        "parameters": count_parameters(model),
        "train_rmse": split_metrics["train"]["rmse"],
        "val_rmse": split_metrics["val"]["rmse"],
        "test_rmse": split_metrics["test"]["rmse"],
        "test_pearson_valence": split_metrics["test"]["pearson_valence"],
        "test_pearson_arousal": split_metrics["test"]["pearson_arousal"],
        "config_json": json.dumps(trial, ensure_ascii=False),
        "checkpoint": str(best_path),
    }
    with (metric_dir / f"{trial_id}_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MERT+DEAM sequence VA heads.")
    parser.add_argument("--model-family", choices=["bilstm", "transformer"], required=True)
    parser.add_argument("--plan", default=str(ROOT / "configs" / "stage3_training_plan.json"))
    parser.add_argument("--dataset-dir", default=str(ROOT / "data" / "datasets" / "mert_va"))
    parser.add_argument("--trial", default="")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"], help="Training device.")
    args = parser.parse_args()

    plan = load_json(Path(args.plan))
    set_seed(int(plan["seed"]))
    device = select_device(args.device)
    print(f"device={device}")
    dataset_dir = Path(args.dataset_dir)
    index = pd.read_csv(dataset_dir / "sequence_index.csv")
    norm = np.load(dataset_dir / "frame" / "feature_norm.npz")
    mean = norm["mean"].squeeze(0)
    std = norm["std"].squeeze(0)
    input_dim = int(mean.shape[0])
    datasets = {split: SequenceDataset(index, split, mean, std) for split in ["train", "val", "test"]}
    trials = plan[f"{args.model_family}_trials"]
    if args.trial:
        trials = [trial for trial in trials if trial["trial_id"] == args.trial]
        if not trials:
            raise ValueError(f"No {args.model_family} trial matched --trial={args.trial}")
    summaries = [
        run_trial(args.model_family, trial, datasets, device, input_dim, int(plan["seed"]), int(plan["early_stopping_patience"]))
        for trial in trials
    ]
    out_path = ROOT / "outputs" / "metrics" / args.model_family / f"{args.model_family}_trial_summary.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
