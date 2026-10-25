#!/usr/bin/env python3
"""Train PMEmo OpenSMILE Arousal-only BiLSTM trials.

This script uses the released PMEmo OpenSMILE dynamic features. It is not yet
an arbitrary-song extractor, but it tells us whether the current strongest
Arousal feature family can meet the project metric thresholds when the model
and selection rule are aligned with Arousal only.
"""

from __future__ import annotations

import argparse
import csv
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
DEFAULT_OUT = ROOT / "outputs" / "opensmile_arousal_trials"


PASS_TARGETS = {
    "rmse": 0.18,
    "mae": 0.15,
    "pcc": 0.80,
    "ccc": 0.80,
}
GOOD_TARGETS = {
    "rmse": 0.15,
    "mae": 0.12,
    "pcc": 0.85,
    "ccc": 0.85,
}
EXCELLENT_TARGETS = {
    "rmse": 0.10,
    "mae": 0.08,
    "pcc": 0.90,
    "ccc": 0.90,
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


def moving_average_by_song(pred: np.ndarray, song_ids: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return pred.copy()
    out = pred.copy()
    for sid in np.unique(song_ids):
        idx = np.where(song_ids == sid)[0]
        local_window = min(window, len(idx))
        if local_window <= 1:
            continue
        kernel = np.ones(local_window, dtype=np.float32) / float(local_window)
        out[idx] = np.convolve(pred[idx], kernel, mode="same")
    return out


def fit_linear_calibration(pred: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    x = pred.astype(np.float64)
    y = target.astype(np.float64)
    var = float(np.var(x))
    if var < 1e-12:
        return 1.0, 0.0
    a = float(np.mean((x - x.mean()) * (y - y.mean())) / var)
    b = float(y.mean() - a * x.mean())
    return a, b


def target_gap(value: float, metric_name: str, target: float) -> float:
    if metric_name in {"rmse", "mae"}:
        return float(value - target)
    return float(target - value)


def threshold_rows(m: dict[str, float]) -> list[dict[str, Any]]:
    rows = []
    for key in ["rmse", "mae", "pcc", "ccc"]:
        value = float(m[key])
        pass_gap = target_gap(value, key, PASS_TARGETS[key])
        good_gap = target_gap(value, key, GOOD_TARGETS[key])
        excellent_gap = target_gap(value, key, EXCELLENT_TARGETS[key])
        rows.append(
            {
                "metric": key,
                "current": value,
                "minimum_pass": PASS_TARGETS[key],
                "gap_to_pass": pass_gap,
                "good": GOOD_TARGETS[key],
                "gap_to_good": good_gap,
                "excellent": EXCELLENT_TARGETS[key],
                "gap_to_excellent": excellent_gap,
                "status": "pass" if pass_gap <= 0 else "not_pass",
            }
        )
    return rows


class SequenceDataset(Dataset):
    def __init__(self, index: pd.DataFrame, split: str):
        self.rows = index[index["split"] == split].sort_values("musicId").to_dict("records")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        data = np.load(self.rows[idx]["path"])
        x = data["X"].astype(np.float32)
        y = data["Y"].astype(np.float32)[:, 1]
        song = np.full(len(y), int(data["music_id"]), dtype=np.int32)
        time_sec = data["time_sec"].astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(song), torch.from_numpy(time_sec)


def collate(batch):
    lengths = [x.shape[0] for x, _, _, _ in batch]
    max_len = max(lengths)
    feat_dim = batch[0][0].shape[1]
    x_out = torch.zeros(len(batch), max_len, feat_dim, dtype=torch.float32)
    y_out = torch.zeros(len(batch), max_len, dtype=torch.float32)
    song_out = torch.zeros(len(batch), max_len, dtype=torch.int32)
    time_out = torch.zeros(len(batch), max_len, dtype=torch.float32)
    mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
    for i, (x, y, song, time_sec) in enumerate(batch):
        x_out[i, : x.shape[0]] = x
        y_out[i, : y.shape[0]] = y
        song_out[i, : song.shape[0]] = song
        time_out[i, : time_sec.shape[0]] = time_sec
        mask[i, : x.shape[0]] = True
    return x_out, y_out, song_out, time_out, mask


class ArousalBiLSTM(nn.Module):
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
    song_np = np.concatenate(songs)
    time_np = np.concatenate(times)
    return {
        "metrics": metrics(pred_np, target_np),
        "pred": pred_np,
        "target": target_np,
        "song_id": song_np,
        "time_sec": time_np,
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


DEFAULT_TRIALS = [
    TrialConfig("a_only_h128_l2_do025_mse", 128, 2, 0.25, 96, 8e-4, 1e-4, 24, 1.0, 0.0, 0.0),
    TrialConfig("a_only_h160_l2_do030_mse_ccc", 160, 2, 0.30, 128, 6e-4, 1e-4, 20, 1.0, 0.15, 0.0),
    TrialConfig("a_only_h192_l2_do035_mse_ccc_smooth", 192, 2, 0.35, 128, 5e-4, 2e-4, 16, 1.0, 0.20, 0.02),
    TrialConfig("a_only_h96_l1_do015_mse", 96, 1, 0.15, 64, 8e-4, 1e-4, 32, 1.0, 0.0, 0.0),
    TrialConfig("a_only_h96_l1_do010_lr6e4_mse", 96, 1, 0.10, 64, 6e-4, 1e-4, 32, 1.0, 0.0, 0.0),
    TrialConfig("a_only_h112_l1_do015_lr7e4_mse", 112, 1, 0.15, 80, 7e-4, 1e-4, 28, 1.0, 0.0, 0.0),
    TrialConfig("a_only_h80_l1_do010_lr8e4_mse", 80, 1, 0.10, 64, 8e-4, 1e-4, 36, 1.0, 0.0, 0.0),
]


def run_trial(
    config: TrialConfig,
    data_dir: Path,
    out_dir: Path,
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    set_seed(seed)
    trial_dir = out_dir / config.trial_id
    trial_dir.mkdir(parents=True, exist_ok=True)
    index = pd.read_csv(data_dir / "sequence_index.csv")
    loaders = {
        split: DataLoader(
            SequenceDataset(index, split),
            batch_size=config.batch_size,
            shuffle=(split == "train"),
            collate_fn=collate,
        )
        for split in ["train", "val", "test"]
    }
    input_dim = int(np.load(index.iloc[0]["path"])["X"].shape[1])
    model = ArousalBiLSTM(input_dim, config.hidden_dim, config.layers, config.dropout, config.head_hidden).to(device)
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

    val_pred = split_evals["val"]["pred"]
    val_target = split_evals["val"]["target"]
    val_song = split_evals["val"]["song_id"]
    test_pred = split_evals["test"]["pred"]
    test_target = split_evals["test"]["target"]
    test_song = split_evals["test"]["song_id"]

    best_post = {"postprocess": "raw", "val": split_evals["val"]["metrics"], "test": split_evals["test"]["metrics"]}
    post_rows = []
    for window in [1, 3, 5, 7, 9, 11]:
        sm_val = moving_average_by_song(val_pred, val_song, window)
        sm_test = moving_average_by_song(test_pred, test_song, window)
        a, b = fit_linear_calibration(sm_val, val_target)
        variants = [
            (f"smooth_w{window}", sm_val, sm_test),
            (f"smooth_w{window}_linear_cal", a * sm_val + b, a * sm_test + b),
        ]
        for name, pp_val, pp_test in variants:
            val_m = metrics(pp_val, val_target)
            test_m = metrics(pp_test, test_target)
            row = {"postprocess": name, **{f"val_{k}": v for k, v in val_m.items()}, **{f"test_{k}": v for k, v in test_m.items()}}
            post_rows.append(row)
            if val_m["rmse"] < best_post["val"]["rmse"]:
                best_post = {"postprocess": name, "val": val_m, "test": test_m, "calibration_a": a, "calibration_b": b}

    pd.DataFrame(post_rows).to_csv(trial_dir / "postprocess_search.csv", index=False)
    for split, data in split_evals.items():
        pd.DataFrame(
            {
                "musicId": data["song_id"],
                "time_sec": data["time_sec"],
                "target_arousal": data["target"],
                "pred_arousal": data["pred"],
            }
        ).to_csv(trial_dir / f"{split}_predictions.csv", index=False)

    summary = {
        "trial_id": config.trial_id,
        "seed": seed,
        "feature_type": "pmemo_official_opensmile_dynamic_features",
        "label_scale": "minus1_1",
        "target": "arousal",
        "best_epoch": best_epoch,
        "checkpoint": str(ckpt),
        "config": asdict(config),
        **{f"{split}_{k}": v for split, e in split_evals.items() for k, v in e["metrics"].items()},
        "best_postprocess": best_post,
        "test_gap_rows_raw": threshold_rows(split_evals["test"]["metrics"]),
        "test_gap_rows_best_postprocess": threshold_rows(best_post["test"]),
    }
    (trial_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trial", choices=[t.trial_id for t in DEFAULT_TRIALS] + ["all"], default="all")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    trials = DEFAULT_TRIALS if args.trial == "all" else [t for t in DEFAULT_TRIALS if t.trial_id == args.trial]
    summaries = []
    for trial in trials:
        summaries.append(run_trial(trial, args.data_dir, args.out_dir, device, args.seed))
    rows = []
    for s in summaries:
        raw = {k.replace("test_", ""): v for k, v in s.items() if k.startswith("test_") and isinstance(v, float)}
        best = s["best_postprocess"]["test"]
        rows.append(
            {
                "trial_id": s["trial_id"],
                "best_epoch": s["best_epoch"],
                "raw_rmse": raw["rmse"],
                "raw_mae": raw["mae"],
                "raw_pcc": raw["pcc"],
                "raw_ccc": raw["ccc"],
                "best_postprocess": s["best_postprocess"]["postprocess"],
                "post_rmse": best["rmse"],
                "post_mae": best["mae"],
                "post_pcc": best["pcc"],
                "post_ccc": best["ccc"],
            }
        )
    pd.DataFrame(rows).to_csv(args.out_dir / "comparison.csv", index=False)
    (args.out_dir / "comparison.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(rows, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
