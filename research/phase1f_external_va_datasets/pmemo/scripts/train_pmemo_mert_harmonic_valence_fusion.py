#!/usr/bin/env python3
"""Train PMEmo Valence baselines on MERT + harmonic Librosa fusion."""

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

from train_pmemo_valence_selected_fusion import selected_indices
from train_pmemo_mert_valence_baseline import (
    DEFAULT_INDEX,
    DEFAULT_LABEL_DIR,
    DEFAULT_MERT_DIR,
    ROOT,
    ccc_loss,
    metrics,
    resolve_device,
    threshold_rows,
)


DEFAULT_LIBROSA_DATASET = ROOT / "outputs" / "arousal_fullsong_librosa" / "dataset"
DEFAULT_MERT_NORM = ROOT / "outputs" / "mert_valence_baseline" / "dataset" / "mert_norm.npz"
DEFAULT_OUT = ROOT / "outputs" / "mert_harmonic_valence_fusion"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_harmonic_norm(index: pd.DataFrame, librosa_dataset: Path, feature_idx: list[int], out_dir: Path, force: bool) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    norm_path = out_dir / "harmonic_norm.npz"
    summary_path = out_dir / "dataset_summary.json"
    if norm_path.exists() and summary_path.exists() and not force:
        return json.loads(summary_path.read_text(encoding="utf-8"))
    xs = []
    for row in index[index["split"] == "train"].to_dict("records"):
        sid = int(row["musicId"])
        xs.append(np.load(librosa_dataset / "sequence" / f"{sid}.npz")["X"].astype(np.float32)[:, feature_idx])
    concat = np.concatenate(xs, axis=0)
    mean = concat.mean(axis=0).astype(np.float32)
    std = (concat.std(axis=0) + 1e-8).astype(np.float32)
    np.savez(norm_path, mean=mean, std=std, feature_idx=np.asarray(feature_idx, dtype=np.int32))
    summary = {"mert_dim": 768, "harmonic_dim": len(feature_idx), "fusion_dim": 768 + len(feature_idx)}
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


class FusionDataset(Dataset):
    def __init__(
        self,
        index: pd.DataFrame,
        split: str,
        mert_dir: Path,
        label_dir: Path,
        librosa_dataset: Path,
        mert_mean: np.ndarray,
        mert_std: np.ndarray,
        harm_mean: np.ndarray,
        harm_std: np.ndarray,
        feature_idx: np.ndarray,
    ):
        self.rows = index[index["split"] == split].sort_values("musicId").to_dict("records")
        self.mert_dir = mert_dir
        self.label_dir = label_dir
        self.librosa_dataset = librosa_dataset
        self.mert_mean = mert_mean.astype(np.float32)
        self.mert_std = mert_std.astype(np.float32)
        self.harm_mean = harm_mean.astype(np.float32)
        self.harm_std = harm_std.astype(np.float32)
        self.feature_idx = feature_idx.astype(np.int64)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        sid = int(self.rows[idx]["musicId"])
        mert = np.load(self.mert_dir / f"{sid}.npz")
        label = np.load(self.label_dir / f"{sid}.npz")
        lib = np.load(self.librosa_dataset / "sequence" / f"{sid}.npz")
        x_mert = ((mert["embedding"].astype(np.float32) - self.mert_mean) / self.mert_std).astype(np.float32)
        x_harm = lib["X"].astype(np.float32)[:, self.feature_idx]
        x_harm = ((x_harm - self.harm_mean) / self.harm_std).astype(np.float32)
        y = label["Y"].astype(np.float32)[:, 0]
        time_sec = mert["time_sec"].astype(np.float32)
        n = min(len(x_mert), len(x_harm), len(y), len(time_sec))
        x = np.concatenate([x_mert[:n], x_harm[:n]], axis=1).astype(np.float32)
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
    def __init__(self, input_dim: int, proj_dim: int, hidden: int, dropout: float):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, proj_dim), nn.GELU(), nn.Dropout(dropout))
        self.lstm = nn.LSTM(proj_dim, hidden, num_layers=1, batch_first=True, bidirectional=True)
        self.out = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden * 2, 1))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        y, _ = self.lstm(self.proj(x))
        return self.out(y).squeeze(-1)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, Any]:
    model.eval()
    preds, targets, songs, times = [], [], [], []
    for x, y, song, time_sec, mask in loader:
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
    proj_dim: int
    hidden_dim: int
    dropout: float
    learning_rate: float
    weight_decay: float
    batch_size: int
    ccc_weight: float
    patience: int = 10
    epochs: int = 80


TRIALS = [
    TrialConfig("mert_harm_h64_proj256_do030_mse", 256, 64, 0.30, 5e-4, 2e-4, 24, 0.0),
    TrialConfig("mert_harm_h96_proj256_do035_mse_ccc", 256, 96, 0.35, 4e-4, 2e-4, 20, 0.10),
]


def run_trial(config: TrialConfig, index: pd.DataFrame, args: argparse.Namespace, device: torch.device, seed: int) -> dict[str, Any]:
    set_seed(seed)
    trial_dir = args.out_dir / config.trial_id
    trial_dir.mkdir(parents=True, exist_ok=True)
    mert_norm = np.load(args.mert_norm)
    harm_norm = np.load(args.out_dir / "dataset" / "harmonic_norm.npz")
    loaders = {
        split: DataLoader(
            FusionDataset(index, split, args.mert_dir, args.label_dir, args.librosa_dataset, mert_norm["mean"], mert_norm["std"], harm_norm["mean"], harm_norm["std"], harm_norm["feature_idx"]),
            batch_size=config.batch_size,
            shuffle=(split == "train"),
            collate_fn=collate,
        )
        for split in ["train", "val", "test"]
    }
    sample_x, _, _, _ = loaders["train"].dataset[0]
    model = FusionBiLSTM(int(sample_x.shape[1]), config.proj_dim, config.hidden_dim, config.dropout).to(device)
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
            loss = ((pred_valid - y_valid) ** 2).mean()
            if config.ccc_weight:
                loss = loss + config.ccc_weight * ccc_loss(pred_valid, y_valid)
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
        pd.DataFrame({"musicId": data["song_id"], "time_sec": data["time_sec"], "target_valence": data["target"], "pred_valence": data["pred"]}).to_csv(trial_dir / f"{split}_predictions.csv", index=False)
    summary = {
        "trial_id": config.trial_id,
        "seed": seed,
        "feature_type": "pmemo_mert_768_plus_librosa_harmonic_36",
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
    parser.add_argument("--mert-dir", type=Path, default=DEFAULT_MERT_DIR)
    parser.add_argument("--label-dir", type=Path, default=DEFAULT_LABEL_DIR)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--librosa-dataset", type=Path, default=DEFAULT_LIBROSA_DATASET)
    parser.add_argument("--mert-norm", type=Path, default=DEFAULT_MERT_NORM)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force-norm", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    index = pd.read_csv(args.index)
    feature_idx = selected_indices(["harmonic"])
    dataset_summary = build_harmonic_norm(index, args.librosa_dataset, feature_idx, args.out_dir / "dataset", args.force_norm)
    print(json.dumps(dataset_summary, ensure_ascii=False, indent=2), flush=True)
    device = resolve_device(args.device)
    summaries = [run_trial(trial, index, args, device, args.seed) for trial in TRIALS]
    rows = [{"trial_id": s["trial_id"], "best_epoch": s["best_epoch"], "test_rmse": s["test_rmse"], "test_mae": s["test_mae"], "test_pcc": s["test_pcc"], "test_ccc": s["test_ccc"]} for s in summaries]
    pd.DataFrame(rows).to_csv(args.out_dir / "comparison.csv", index=False)
    (args.out_dir / "comparison.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(rows, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
