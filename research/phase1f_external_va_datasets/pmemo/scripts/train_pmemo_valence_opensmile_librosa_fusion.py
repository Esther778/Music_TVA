#!/usr/bin/env python3
"""Train PMEmo Valence-only models with OpenSMILE + librosa feature fusion."""

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
DEFAULT_OPENSMILE_DATASET = ROOT / "outputs" / "opensmile_baseline" / "dataset"
DEFAULT_LIBROSA_DATASET = ROOT / "outputs" / "arousal_fullsong_librosa" / "dataset"
DEFAULT_OUT = ROOT / "outputs" / "valence_opensmile_librosa_fusion"

PASS_TARGETS = {"rmse": 0.18, "mae": 0.15, "pcc": 0.80, "ccc": 0.80}
GOOD_TARGETS = {"rmse": 0.15, "mae": 0.12, "pcc": 0.85, "ccc": 0.85}
EXCELLENT_TARGETS = {"rmse": 0.10, "mae": 0.08, "pcc": 0.90, "ccc": 0.90}


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


def ccc_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_mean = pred.mean()
    target_mean = target.mean()
    pred_var = pred.var(unbiased=False)
    target_var = target.var(unbiased=False)
    cov = ((pred - pred_mean) * (target - target_mean)).mean()
    denom = pred_var + target_var + (pred_mean - target_mean) ** 2 + 1e-8
    return 1.0 - (2.0 * cov / denom)


def smoothness_loss(pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if pred.shape[1] < 2:
        return pred.new_tensor(0.0)
    valid = mask[:, 1:] & mask[:, :-1]
    if valid.sum() == 0:
        return pred.new_tensor(0.0)
    diff = pred[:, 1:] - pred[:, :-1]
    return (diff[valid] ** 2).mean()


def target_gap(value: float, metric_name: str, target: float) -> float:
    return float(value - target) if metric_name in {"rmse", "mae"} else float(target - value)


def threshold_rows(m: dict[str, float]) -> list[dict[str, Any]]:
    rows = []
    for key in ["rmse", "mae", "pcc", "ccc"]:
        pass_gap = target_gap(float(m[key]), key, PASS_TARGETS[key])
        rows.append(
            {
                "metric": key,
                "current": float(m[key]),
                "minimum_pass": PASS_TARGETS[key],
                "gap_to_pass": pass_gap,
                "good": GOOD_TARGETS[key],
                "gap_to_good": target_gap(float(m[key]), key, GOOD_TARGETS[key]),
                "excellent": EXCELLENT_TARGETS[key],
                "gap_to_excellent": target_gap(float(m[key]), key, EXCELLENT_TARGETS[key]),
                "status": "pass" if pass_gap <= 0 else "not_pass",
            }
        )
    return rows


def build_feature_norm(librosa_dataset: Path, opensmile_index: pd.DataFrame, out_dir: Path, force: bool) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    norm_path = out_dir / "librosa_norm.npz"
    summary_path = out_dir / "fusion_dataset_summary.json"
    if norm_path.exists() and summary_path.exists() and not force:
        return json.loads(summary_path.read_text(encoding="utf-8"))
    train_rows = opensmile_index[opensmile_index["split"] == "train"].to_dict("records")
    xs = []
    for row in train_rows:
        sid = int(row["musicId"])
        lib_path = librosa_dataset / "sequence" / f"{sid}.npz"
        if not lib_path.exists():
            raise FileNotFoundError(lib_path)
        xs.append(np.load(lib_path)["X"].astype(np.float32))
    concat = np.concatenate(xs, axis=0)
    mean = concat.mean(axis=0).astype(np.float32)
    std = (concat.std(axis=0) + 1e-8).astype(np.float32)
    np.savez(norm_path, mean=mean, std=std)
    summary = {
        "opensmile_dim": 260,
        "librosa_dim": int(concat.shape[1]),
        "fusion_dim": int(260 + concat.shape[1]),
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


class FusionSequenceDataset(Dataset):
    def __init__(self, opensmile_dataset: Path, librosa_dataset: Path, split: str, lib_mean: np.ndarray, lib_std: np.ndarray):
        index = pd.read_csv(opensmile_dataset / "sequence_index.csv")
        self.rows = index[index["split"] == split].sort_values("musicId").to_dict("records")
        self.librosa_dataset = librosa_dataset
        self.lib_mean = lib_mean.astype(np.float32)
        self.lib_std = lib_std.astype(np.float32)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        sid = int(row["musicId"])
        os_data = np.load(row["path"])
        lib_data = np.load(self.librosa_dataset / "sequence" / f"{sid}.npz")
        x_os = os_data["X"].astype(np.float32)
        x_lib = ((lib_data["X"].astype(np.float32) - self.lib_mean) / self.lib_std).astype(np.float32)
        y = os_data["Y"].astype(np.float32)[:, 0]
        time_sec = os_data["time_sec"].astype(np.float32)
        n = min(len(x_os), len(x_lib), len(y))
        x = np.concatenate([x_os[:n], x_lib[:n]], axis=1).astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(y[:n]), torch.full((n,), sid, dtype=torch.int32), torch.from_numpy(time_sec[:n])


def collate(batch):
    lengths = [x.shape[0] for x, _, _, _ in batch]
    max_len = max(lengths)
    feat_dim = batch[0][0].shape[1]
    x_out = torch.zeros(len(batch), max_len, feat_dim)
    y_out = torch.zeros(len(batch), max_len)
    song_out = torch.zeros(len(batch), max_len, dtype=torch.int32)
    time_out = torch.zeros(len(batch), max_len)
    mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
    for i, (x, y, song, time_sec) in enumerate(batch):
        x_out[i, : x.shape[0]] = x
        y_out[i, : y.shape[0]] = y
        song_out[i, : song.shape[0]] = song
        time_out[i, : time_sec.shape[0]] = time_sec
        mask[i, : x.shape[0]] = True
    return x_out, y_out, song_out, time_out, mask


class FusionBiLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden: int, layers: int, dropout: float, head_hidden: int):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden,
            num_layers=layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.out = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        y, _ = self.lstm(x)
        return self.out(y).squeeze(-1)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, Any]:
    model.eval()
    preds, targets, songs, times = [], [], [], []
    for x, y, song, time_sec, mask in loader:
        x = x.to(device)
        mask_device = mask.to(device)
        pred = model(x, mask_device).detach().cpu()
        preds.append(pred[mask].numpy())
        targets.append(y[mask].numpy())
        songs.append(song[mask].numpy())
        times.append(time_sec[mask].numpy())
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
    hidden_dim: int
    layers: int
    dropout: float
    head_hidden: int
    learning_rate: float
    weight_decay: float
    batch_size: int
    mse_weight: float
    ccc_weight: float
    smooth_weight: float
    patience: int = 14
    epochs: int = 120


TRIALS = [
    TrialConfig("v_fusion_h96_l1_do015_mse", 96, 1, 0.15, 64, 8e-4, 1e-4, 32, 1.0, 0.0, 0.0),
    TrialConfig("v_fusion_h112_l1_do020_mse", 112, 1, 0.20, 80, 7e-4, 1e-4, 28, 1.0, 0.0, 0.0),
    TrialConfig("v_fusion_h128_l2_do030_mse_ccc", 128, 2, 0.30, 96, 6e-4, 1e-4, 24, 1.0, 0.15, 0.0),
    TrialConfig("v_fusion_h160_l2_do035_mse_ccc_smooth", 160, 2, 0.35, 128, 5e-4, 2e-4, 20, 1.0, 0.20, 0.02),
]


def run_trial(
    config: TrialConfig,
    opensmile_dataset: Path,
    librosa_dataset: Path,
    out_dir: Path,
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    set_seed(seed)
    trial_dir = out_dir / config.trial_id
    trial_dir.mkdir(parents=True, exist_ok=True)
    norm = np.load(out_dir / "dataset" / "librosa_norm.npz")
    lib_mean, lib_std = norm["mean"], norm["std"]
    loaders = {
        split: DataLoader(
            FusionSequenceDataset(opensmile_dataset, librosa_dataset, split, lib_mean, lib_std),
            batch_size=config.batch_size,
            shuffle=(split == "train"),
            collate_fn=collate,
        )
        for split in ["train", "val", "test"]
    }
    sample_x, _, _, _ = loaders["train"].dataset[0]
    input_dim = int(sample_x.shape[1])
    model = FusionBiLSTM(input_dim, config.hidden_dim, config.layers, config.dropout, config.head_hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    ckpt = trial_dir / "best.pt"
    history = []
    best = float("inf")
    best_epoch = 0
    bad = 0
    for epoch in range(1, config.epochs + 1):
        model.train()
        losses = []
        for x, y, _, _, mask in loaders["train"]:
            x = x.to(device)
            y = y.to(device)
            mask = mask.to(device)
            optimizer.zero_grad()
            pred = model(x, mask)
            pred_valid = pred[mask]
            y_valid = y[mask]
            loss = config.mse_weight * ((pred_valid - y_valid) ** 2).mean()
            if config.ccc_weight:
                loss = loss + config.ccc_weight * ccc_loss(pred_valid, y_valid)
            if config.smooth_weight:
                loss = loss + config.smooth_weight * smoothness_loss(pred, mask)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        val_eval = evaluate(model, loaders["val"], device)
        val_m = val_eval["metrics"]
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), **{f"val_{k}": v for k, v in val_m.items()}}
        history.append(row)
        print(
            f"{config.trial_id} epoch={epoch:03d} train_loss={row['train_loss']:.5f} "
            f"val_rmse={val_m['rmse']:.5f} val_pcc={val_m['pcc']:.3f} val_ccc={val_m['ccc']:.3f}",
            flush=True,
        )
        if val_m["rmse"] < best:
            best = val_m["rmse"]
            best_epoch = epoch
            bad = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "input_dim": input_dim,
                    "seed": seed,
                    "trial_config": asdict(config),
                },
                ckpt,
            )
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
    summary = {
        "trial_id": config.trial_id,
        "seed": seed,
        "feature_type": "pmemo_opensmile_260_plus_librosa_118",
        "label_scale": "minus1_1",
        "target": "valence",
        "best_epoch": best_epoch,
        "checkpoint": str(ckpt),
        "config": asdict(config),
        **{f"{split}_{k}": v for split, e in split_evals.items() for k, v in e["metrics"].items()},
        "test_gap_rows": threshold_rows(split_evals["test"]["metrics"]),
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
    dataset_summary = build_feature_norm(args.librosa_dataset, index, args.out_dir / "dataset", args.force_norm)
    print(json.dumps(dataset_summary, ensure_ascii=False, indent=2), flush=True)
    device = resolve_device(args.device)
    summaries = [run_trial(trial, args.opensmile_dataset, args.librosa_dataset, args.out_dir, device, args.seed) for trial in TRIALS]
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
