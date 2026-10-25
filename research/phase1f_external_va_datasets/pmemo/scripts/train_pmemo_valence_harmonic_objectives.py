#!/usr/bin/env python3
"""Train Valence harmonic models with alternative objectives.

The current best Valence model uses OpenSMILE + Librosa harmonic features with
plain frame-level MSE. This script keeps the same feature family and split, and
tests whether label smoothing, clip-level auxiliary loss, or Huber/CCC losses
improve held-out PMEmo Valence metrics.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from train_pmemo_valence_selected_fusion import (
    GOOD_TARGETS,
    EXCELLENT_TARGETS,
    PASS_TARGETS,
    ROOT,
    ccc_loss,
    metrics,
    selected_indices,
    target_gap,
    threshold_rows,
)


DEFAULT_OPENSMILE_DATASET = ROOT / "outputs" / "opensmile_baseline" / "dataset"
DEFAULT_LIBROSA_DATASET = ROOT / "outputs" / "arousal_fullsong_librosa" / "dataset"
DEFAULT_OUT = ROOT / "outputs" / "valence_harmonic_objectives"

HISTORICAL_BEST = {
    "model_id": "pmemo_valence_opensmile_harmonic_bilstm_v1",
    "trial_id": "v_sel_harmonic_h96_l1",
    "rmse": 0.23809608817100525,
    "mae": 0.19014787673950195,
    "pcc": 0.6698117858424377,
    "ccc": 0.6217963793040013,
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def moving_average_1d(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(values) <= 1:
        return values.astype(np.float32)
    local_window = min(window, len(values))
    kernel = np.ones(local_window, dtype=np.float32) / float(local_window)
    return np.convolve(values, kernel, mode="same").astype(np.float32)


def build_harmonic_norm(librosa_dataset: Path, opensmile_index: pd.DataFrame, out_dir: Path, force: bool) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    norm_path = out_dir / "harmonic_norm.npz"
    summary_path = out_dir / "dataset_summary.json"
    feature_idx = selected_indices(["harmonic"])
    if norm_path.exists() and summary_path.exists() and not force:
        return json.loads(summary_path.read_text(encoding="utf-8"))
    xs = []
    for row in opensmile_index[opensmile_index["split"] == "train"].to_dict("records"):
        sid = int(row["musicId"])
        xs.append(np.load(librosa_dataset / "sequence" / f"{sid}.npz")["X"].astype(np.float32)[:, feature_idx])
    concat = np.concatenate(xs, axis=0)
    mean = concat.mean(axis=0).astype(np.float32)
    std = (concat.std(axis=0) + 1e-8).astype(np.float32)
    np.savez(norm_path, mean=mean, std=std, feature_idx=np.asarray(feature_idx, dtype=np.int32))
    summary = {
        "opensmile_dim": 260,
        "harmonic_dim": int(len(feature_idx)),
        "fusion_dim": int(260 + len(feature_idx)),
        "songs": int(len(opensmile_index)),
        "splits": {
            split: {
                "songs": int((opensmile_index["split"] == split).sum()),
                "frames": int(opensmile_index[opensmile_index["split"] == split]["frames"].sum()),
            }
            for split in ["train", "val", "test"]
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


class HarmonicDataset(Dataset):
    def __init__(self, opensmile_dataset: Path, librosa_dataset: Path, split: str, mean: np.ndarray, std: np.ndarray, feature_idx: np.ndarray):
        index = pd.read_csv(opensmile_dataset / "sequence_index.csv")
        self.rows = index[index["split"] == split].sort_values("musicId").to_dict("records")
        self.librosa_dataset = librosa_dataset
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)
        self.feature_idx = feature_idx.astype(np.int64)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        sid = int(row["musicId"])
        os_data = np.load(row["path"])
        lib_data = np.load(self.librosa_dataset / "sequence" / f"{sid}.npz")
        x_os = os_data["X"].astype(np.float32)
        x_harm = lib_data["X"].astype(np.float32)[:, self.feature_idx]
        x_harm = ((x_harm - self.mean) / self.std).astype(np.float32)
        y = os_data["Y"].astype(np.float32)[:, 0]
        time_sec = os_data["time_sec"].astype(np.float32)
        n = min(len(x_os), len(x_harm), len(y), len(time_sec))
        x = np.concatenate([x_os[:n], x_harm[:n]], axis=1).astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(y[:n]), torch.full((n,), sid, dtype=torch.int32), torch.from_numpy(time_sec[:n])


def collate(batch):
    lengths = [x.shape[0] for x, _, _, _ in batch]
    max_len = max(lengths)
    feat_dim = batch[0][0].shape[1]
    x_out = torch.zeros(len(batch), max_len, feat_dim)
    y_out = torch.zeros(len(batch), max_len)
    y_smooth_out = torch.zeros(len(batch), max_len)
    song_out = torch.zeros(len(batch), max_len, dtype=torch.int32)
    time_out = torch.zeros(len(batch), max_len)
    mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
    for i, (x, y, song, time_sec) in enumerate(batch):
        x_out[i, : x.shape[0]] = x
        y_out[i, : y.shape[0]] = y
        y_smooth_out[i, : y.shape[0]] = torch.from_numpy(moving_average_1d(y.numpy(), 5))
        song_out[i, : song.shape[0]] = song
        time_out[i, : time_sec.shape[0]] = time_sec
        mask[i, : x.shape[0]] = True
    return x_out, y_out, y_smooth_out, song_out, time_out, mask


class ValenceBiLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden: int, layers: int, dropout: float, head_hidden: int):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, num_layers=layers, batch_first=True, bidirectional=True, dropout=dropout if layers > 1 else 0.0)
        self.out = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden * 2, head_hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(head_hidden, 1))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        y, _ = self.lstm(x)
        return self.out(y).squeeze(-1)


def delta_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if pred.shape[1] < 2:
        return pred.new_tensor(0.0)
    valid = mask[:, 1:] & mask[:, :-1]
    if valid.sum() == 0:
        return pred.new_tensor(0.0)
    pred_delta = pred[:, 1:] - pred[:, :-1]
    target_delta = target[:, 1:] - target[:, :-1]
    return ((pred_delta[valid] - target_delta[valid]) ** 2).mean()


def clip_mean_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    lengths = mask.sum(dim=1).clamp_min(1).float()
    pred_mean = (pred * mask.float()).sum(dim=1) / lengths
    target_mean = (target * mask.float()).sum(dim=1) / lengths
    return ((pred_mean - target_mean) ** 2).mean()


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, Any]:
    model.eval()
    preds, targets, songs, times = [], [], [], []
    for x, y, _, song, time_sec, mask in loader:
        pred = model(x.to(device), mask.to(device)).detach().cpu()
        preds.append(pred[mask].numpy())
        targets.append(y[mask].numpy())
        songs.append(song[mask].numpy())
        times.append(time_sec[mask].numpy())
    pred_np = np.concatenate(preds)
    target_np = np.concatenate(targets)
    return {"metrics": metrics(pred_np, target_np), "pred": pred_np, "target": target_np, "song_id": np.concatenate(songs), "time_sec": np.concatenate(times)}


@dataclass
class TrialConfig:
    trial_id: str
    hidden_dim: int
    layers: int
    dropout: float
    head_hidden: int
    learning_rate: float
    weight_decay: float
    batch_size: int
    loss_type: str
    smooth_mix: float
    ccc_weight: float
    clip_weight: float
    delta_weight: float
    patience: int = 14
    epochs: int = 120


TRIALS = [
    TrialConfig("v_harm_smoothmix025_mse_delta002", 96, 1, 0.15, 64, 8e-4, 1e-4, 32, "mse", 0.25, 0.0, 0.0, 0.02),
    TrialConfig("v_harm_smoothmix050_mse_clip020", 96, 1, 0.15, 64, 8e-4, 1e-4, 32, "mse", 0.50, 0.0, 0.20, 0.0),
    TrialConfig("v_harm_huber_ccc005_clip010", 96, 1, 0.20, 64, 7e-4, 1e-4, 32, "huber", 0.0, 0.05, 0.10, 0.0),
    TrialConfig("v_harm_mse_ccc010_clip020_delta002", 96, 1, 0.18, 64, 7e-4, 1e-4, 32, "mse", 0.0, 0.10, 0.20, 0.02),
]


def compute_loss(config: TrialConfig, pred: torch.Tensor, y: torch.Tensor, y_smooth: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    train_target = (1.0 - config.smooth_mix) * y + config.smooth_mix * y_smooth
    pred_valid = pred[mask]
    target_valid = train_target[mask]
    if config.loss_type == "huber":
        loss = nn.functional.smooth_l1_loss(pred_valid, target_valid, beta=0.08)
    elif config.loss_type == "mse":
        loss = ((pred_valid - target_valid) ** 2).mean()
    else:
        raise ValueError(config.loss_type)
    if config.ccc_weight:
        loss = loss + config.ccc_weight * ccc_loss(pred_valid, y[mask])
    if config.clip_weight:
        loss = loss + config.clip_weight * clip_mean_loss(pred, y, mask)
    if config.delta_weight:
        loss = loss + config.delta_weight * delta_loss(pred, y, mask)
    return loss


def metric_diff(current: dict[str, float], previous: dict[str, float]) -> dict[str, float]:
    return {
        "rmse_delta_vs_historical": float(current["rmse"] - previous["rmse"]),
        "mae_delta_vs_historical": float(current["mae"] - previous["mae"]),
        "pcc_delta_vs_historical": float(current["pcc"] - previous["pcc"]),
        "ccc_delta_vs_historical": float(current["ccc"] - previous["ccc"]),
    }


def run_trial(config: TrialConfig, opensmile_dataset: Path, librosa_dataset: Path, out_dir: Path, device: torch.device, seed: int) -> dict[str, Any]:
    set_seed(seed)
    trial_dir = out_dir / config.trial_id
    trial_dir.mkdir(parents=True, exist_ok=True)
    norm = np.load(out_dir / "dataset" / "harmonic_norm.npz")
    loaders = {
        split: DataLoader(
            HarmonicDataset(opensmile_dataset, librosa_dataset, split, norm["mean"], norm["std"], norm["feature_idx"]),
            batch_size=config.batch_size,
            shuffle=(split == "train"),
            collate_fn=collate,
        )
        for split in ["train", "val", "test"]
    }
    sample_x, _, _, _ = loaders["train"].dataset[0]
    model = ValenceBiLSTM(int(sample_x.shape[1]), config.hidden_dim, config.layers, config.dropout, config.head_hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    ckpt = trial_dir / "best.pt"
    history = []
    best = float("inf")
    best_epoch = 0
    bad = 0
    for epoch in range(1, config.epochs + 1):
        model.train()
        losses = []
        for x, y, y_smooth, _, _, mask in loaders["train"]:
            x = x.to(device)
            y = y.to(device)
            y_smooth = y_smooth.to(device)
            mask = mask.to(device)
            optimizer.zero_grad()
            pred = model(x, mask)
            loss = compute_loss(config, pred, y, y_smooth, mask)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        val_eval = evaluate(model, loaders["val"], device)
        val_m = val_eval["metrics"]
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), **{f"val_{k}": v for k, v in val_m.items()}}
        history.append(row)
        print(f"{config.trial_id} epoch={epoch:03d} train_loss={row['train_loss']:.5f} val_rmse={val_m['rmse']:.5f} val_pcc={val_m['pcc']:.3f} val_ccc={val_m['ccc']:.3f}", flush=True)
        if val_m["rmse"] < best:
            best = val_m["rmse"]
            best_epoch = epoch
            bad = 0
            torch.save({"model_state": model.state_dict(), "seed": seed, "trial_config": asdict(config), "input_dim": int(sample_x.shape[1])}, ckpt)
        else:
            bad += 1
            if bad >= config.patience:
                break
    pd.DataFrame(history).to_csv(trial_dir / "history.csv", index=False)
    model.load_state_dict(torch.load(ckpt, map_location=device)["model_state"])
    split_evals = {split: evaluate(model, loaders[split], device) for split in ["train", "val", "test"]}
    for split, data in split_evals.items():
        pd.DataFrame(
            {
                "musicId": data["song_id"],
                "time_sec": data["time_sec"],
                "target_valence": data["target"],
                "pred_valence": data["pred"],
            }
        ).to_csv(trial_dir / f"{split}_predictions.csv", index=False)
    test_m = split_evals["test"]["metrics"]
    summary = {
        "trial_id": config.trial_id,
        "seed": seed,
        "feature_type": "pmemo_opensmile_260_plus_librosa_harmonic_36",
        "label_scale": "minus1_1",
        "target": "valence",
        "best_epoch": best_epoch,
        "checkpoint": str(ckpt),
        "config": asdict(config),
        **{f"{split}_{k}": v for split, e in split_evals.items() for k, v in e["metrics"].items()},
        "historical_best": HISTORICAL_BEST,
        "delta_vs_historical_best": metric_diff(test_m, HISTORICAL_BEST),
        "test_gap_rows": threshold_rows(test_m),
    }
    (trial_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--opensmile-dataset", type=Path, default=DEFAULT_OPENSMILE_DATASET)
    parser.add_argument("--librosa-dataset", type=Path, default=DEFAULT_LIBROSA_DATASET)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force-norm", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    index = pd.read_csv(args.opensmile_dataset / "sequence_index.csv")
    dataset_summary = build_harmonic_norm(args.librosa_dataset, index, args.out_dir / "dataset", args.force_norm)
    print(json.dumps({"dataset": dataset_summary, "historical_best": HISTORICAL_BEST}, ensure_ascii=False, indent=2), flush=True)
    device = resolve_device(args.device)
    summaries = [run_trial(trial, args.opensmile_dataset, args.librosa_dataset, args.out_dir, device, args.seed) for trial in TRIALS]
    rows = []
    for s in summaries:
        rows.append(
            {
                "trial_id": s["trial_id"],
                "best_epoch": s["best_epoch"],
                "test_rmse": s["test_rmse"],
                "test_mae": s["test_mae"],
                "test_pcc": s["test_pcc"],
                "test_ccc": s["test_ccc"],
                **s["delta_vs_historical_best"],
            }
        )
    pd.DataFrame(rows).to_csv(args.out_dir / "comparison.csv", index=False)
    (args.out_dir / "comparison.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(rows, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
