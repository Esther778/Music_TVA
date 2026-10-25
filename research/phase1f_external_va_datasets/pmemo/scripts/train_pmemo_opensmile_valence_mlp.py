#!/usr/bin/env python3
"""Train PMEmo OpenSMILE Valence-only MLP baselines."""

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


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "outputs" / "opensmile_baseline" / "dataset"
DEFAULT_OUT = ROOT / "outputs" / "opensmile_valence_mlp"


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


def pearson(pred: np.ndarray, target: np.ndarray) -> float:
    if len(pred) < 2 or np.std(pred) < 1e-8 or np.std(target) < 1e-8:
        return 0.0
    return float(np.corrcoef(pred, target)[0, 1])


def ccc(pred: np.ndarray, target: np.ndarray) -> float:
    mean_x = float(np.mean(pred))
    mean_y = float(np.mean(target))
    var_x = float(np.var(pred))
    var_y = float(np.var(target))
    cov = float(np.mean((pred - mean_x) * (target - mean_y)))
    denom = var_x + var_y + (mean_x - mean_y) ** 2
    return float(2 * cov / denom) if denom > 1e-12 else 0.0


def metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    err = pred - target
    return {
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "pcc": pearson(pred, target),
        "ccc": ccc(pred, target),
    }


class FrameDataset(Dataset):
    def __init__(self, data_dir: Path, split: str):
        self.x = np.load(data_dir / f"X_{split}.npy").astype(np.float32)
        self.y = np.load(data_dir / f"Y_{split}.npy").astype(np.float32)[:, 0]
        self.song = np.load(data_dir / f"song_id_{split}.npy").astype(np.int32)
        self.time_sec = np.load(data_dir / f"time_sec_{split}.npy").astype(np.float32)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        return (
            torch.from_numpy(self.x[idx]),
            torch.tensor(self.y[idx], dtype=torch.float32),
            int(self.song[idx]),
            float(self.time_sec[idx]),
        )


class ValenceMLP(nn.Module):
    def __init__(self, input_dim: int, hidden: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, Any]:
    model.eval()
    preds, targets, songs, times = [], [], [], []
    for x, y, song, time_sec in loader:
        pred = model(x.to(device)).detach().cpu().numpy()
        preds.append(pred)
        targets.append(y.numpy())
        songs.append(song.numpy())
        times.append(time_sec.numpy())
    pred_np = np.concatenate(preds)
    target_np = np.concatenate(targets)
    return {
        "metrics": metrics(pred_np, target_np),
        "pred": pred_np,
        "target": target_np,
        "song_id": np.concatenate(songs),
        "time_sec": np.concatenate(times),
    }


@dataclass
class TrialConfig:
    trial_id: str
    hidden: int
    dropout: float
    learning_rate: float
    weight_decay: float
    batch_size: int
    patience: int = 12
    epochs: int = 120


TRIALS = [
    TrialConfig("v_mlp_h384_do025", 384, 0.25, 1e-3, 1e-4, 512),
    TrialConfig("v_mlp_h256_do020", 256, 0.20, 8e-4, 1e-4, 512),
    TrialConfig("v_mlp_h512_do035", 512, 0.35, 6e-4, 2e-4, 512),
]


def run_trial(config: TrialConfig, data_dir: Path, out_dir: Path, device: torch.device, seed: int) -> dict[str, Any]:
    set_seed(seed)
    trial_dir = out_dir / config.trial_id
    trial_dir.mkdir(parents=True, exist_ok=True)
    loaders = {
        split: DataLoader(FrameDataset(data_dir, split), batch_size=config.batch_size, shuffle=(split == "train"))
        for split in ["train", "val", "test"]
    }
    input_dim = int(loaders["train"].dataset.x.shape[1])
    model = ValenceMLP(input_dim, config.hidden, config.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    best = float("inf")
    best_epoch = 0
    bad = 0
    history = []
    ckpt = trial_dir / "best.pt"
    for epoch in range(1, config.epochs + 1):
        model.train()
        losses = []
        for x, y, _, _ in loaders["train"]:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = ((pred - y) ** 2).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        val = evaluate(model, loaders["val"], device)["metrics"]
        row = {"epoch": epoch, "train_mse": float(np.mean(losses)), **{f"val_{k}": v for k, v in val.items()}}
        history.append(row)
        print(
            f"{config.trial_id} epoch={epoch:03d} train_mse={row['train_mse']:.5f} "
            f"val_rmse={val['rmse']:.5f} val_pcc={val['pcc']:.3f} val_ccc={val['ccc']:.3f}",
            flush=True,
        )
        if val["rmse"] < best:
            best = val["rmse"]
            best_epoch = epoch
            bad = 0
            torch.save({"model_state": model.state_dict(), "input_dim": input_dim, "config": asdict(config), "seed": seed}, ckpt)
        else:
            bad += 1
            if bad >= config.patience:
                break
    model.load_state_dict(torch.load(ckpt, map_location=device)["model_state"])
    split_evals = {split: evaluate(model, loaders[split], device) for split in ["train", "val", "test"]}
    pd.DataFrame(history).to_csv(trial_dir / "history.csv", index=False)
    for split, data in split_evals.items():
        pd.DataFrame(
            {
                "musicId": data["song_id"],
                "time_sec": data["time_sec"],
                "target_valence": data["target"],
                "pred_valence": data["pred"],
            }
        ).to_csv(trial_dir / f"{split}_predictions.csv", index=False)
    summary = {
        "trial_id": config.trial_id,
        "target": "valence",
        "feature_type": "pmemo_official_opensmile_dynamic_features",
        "label_scale": "minus1_1",
        "best_epoch": best_epoch,
        "checkpoint": str(ckpt),
        "config": asdict(config),
        **{f"{split}_{k}": v for split, e in split_evals.items() for k, v in e["metrics"].items()},
    }
    (trial_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    summaries = [run_trial(trial, args.data_dir, args.out_dir, device, args.seed) for trial in TRIALS]
    rows = [
        {
            "trial_id": s["trial_id"],
            "best_epoch": s["best_epoch"],
            "test_rmse": s["test_rmse"],
            "test_mae": s["test_mae"],
            "test_pcc": s["test_pcc"],
            "test_ccc": s["test_ccc"],
        }
        for s in summaries
    ]
    pd.DataFrame(rows).to_csv(args.out_dir / "comparison.csv", index=False)
    (args.out_dir / "comparison.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(rows, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
