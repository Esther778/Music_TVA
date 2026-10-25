#!/usr/bin/env python3
"""Train PMEmo OpenSMILE VA baselines on canonical [-1, 1] labels."""

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
DEFAULT_EXTRACTED = ROOT / "data" / "raw" / "PMEmo_2019_updated" / "extracted" / "PMEmo2019"
DEFAULT_MANIFEST = ROOT / "data" / "processed" / "pmemo2019_dynamic_frame_manifest_normalized.csv"
DEFAULT_OUT = ROOT / "outputs" / "opensmile_baseline"


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


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.std(x) < 1e-8 or np.std(y) < 1e-8:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def ccc(x: np.ndarray, y: np.ndarray) -> float:
    mean_x = float(np.mean(x))
    mean_y = float(np.mean(y))
    var_x = float(np.var(x))
    var_y = float(np.var(y))
    cov = float(np.mean((x - mean_x) * (y - mean_y)))
    denom = var_x + var_y + (mean_x - mean_y) ** 2
    return float(2 * cov / denom) if denom > 1e-12 else 0.0


def metrics(pred: np.ndarray, target: np.ndarray, prefix: str = "") -> dict[str, float]:
    err = pred - target
    out = {
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "valence_rmse": float(np.sqrt(np.mean(err[:, 0] ** 2))),
        "arousal_rmse": float(np.sqrt(np.mean(err[:, 1] ** 2))),
        "valence_mae": float(np.mean(np.abs(err[:, 0]))),
        "arousal_mae": float(np.mean(np.abs(err[:, 1]))),
        "valence_pcc": pearson(pred[:, 0], target[:, 0]),
        "arousal_pcc": pearson(pred[:, 1], target[:, 1]),
        "valence_ccc": ccc(pred[:, 0], target[:, 0]),
        "arousal_ccc": ccc(pred[:, 1], target[:, 1]),
    }
    out["mean_pcc"] = float((out["valence_pcc"] + out["arousal_pcc"]) / 2)
    out["mean_ccc"] = float((out["valence_ccc"] + out["arousal_ccc"]) / 2)
    return {f"{prefix}{k}": v for k, v in out.items()}


def split_song_ids(song_ids: list[int], seed: int) -> dict[int, str]:
    rng = np.random.default_rng(seed)
    ids = np.asarray(sorted(song_ids), dtype=np.int64)
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(round(n * 0.8))
    n_val = int(round(n * 0.1))
    split = {}
    for sid in ids[:n_train]:
        split[int(sid)] = "train"
    for sid in ids[n_train : n_train + n_val]:
        split[int(sid)] = "val"
    for sid in ids[n_train + n_val :]:
        split[int(sid)] = "test"
    return split


def prepare_arrays(extracted_dir: Path, manifest_path: Path, out_dir: Path, seed: int) -> dict[str, Any]:
    manifest = pd.read_csv(manifest_path)
    features = pd.read_csv(extracted_dir / "features" / "dynamic_features.csv")
    feature_cols = [c for c in features.columns if c not in {"musicId", "frameTime"}]
    labels = manifest[
        ["musicId", "time_sec", "valence_deam_scale", "arousal_deam_scale"]
    ].rename(columns={"time_sec": "frameTime"})
    merged = features.merge(labels, on=["musicId", "frameTime"], how="inner")
    merged = merged.replace([np.inf, -np.inf], np.nan).dropna(
        subset=feature_cols + ["valence_deam_scale", "arousal_deam_scale"]
    )
    split_map = split_song_ids(sorted(merged["musicId"].unique().astype(int).tolist()), seed)
    merged["split"] = merged["musicId"].astype(int).map(split_map)

    out_dir.mkdir(parents=True, exist_ok=True)
    merged[["musicId", "frameTime", "split", "valence_deam_scale", "arousal_deam_scale"]].to_csv(
        out_dir / "pmemo_opensmile_frame_index.csv", index=False
    )

    arrays: dict[str, Any] = {"feature_cols": feature_cols, "splits": {}}
    train = merged[merged["split"] == "train"]
    mean = train[feature_cols].mean(axis=0).to_numpy(dtype=np.float32)
    std = train[feature_cols].std(axis=0).replace(0, 1.0).to_numpy(dtype=np.float32)
    std = std + 1e-8
    np.savez(out_dir / "feature_norm.npz", mean=mean, std=std, feature_cols=np.asarray(feature_cols, dtype=object))

    for split in ["train", "val", "test"]:
        part = merged[merged["split"] == split].sort_values(["musicId", "frameTime"])
        X = part[feature_cols].to_numpy(dtype=np.float32)
        Y = part[["valence_deam_scale", "arousal_deam_scale"]].to_numpy(dtype=np.float32)
        song = part["musicId"].to_numpy(dtype=np.int32)
        time = part["frameTime"].to_numpy(dtype=np.float32)
        Xn = ((X - mean) / std).astype(np.float32)
        np.save(out_dir / f"X_{split}.npy", Xn)
        np.save(out_dir / f"Y_{split}.npy", Y)
        np.save(out_dir / f"song_id_{split}.npy", song)
        np.save(out_dir / f"time_sec_{split}.npy", time)
        arrays["splits"][split] = {
            "frames": int(len(part)),
            "songs": int(part["musicId"].nunique()),
        }

    seq_dir = out_dir / "sequence"
    seq_dir.mkdir(exist_ok=True)
    seq_rows = []
    for split in ["train", "val", "test"]:
        part = merged[merged["split"] == split].sort_values(["musicId", "frameTime"])
        for music_id, group in part.groupby("musicId", sort=True):
            X = ((group[feature_cols].to_numpy(dtype=np.float32) - mean) / std).astype(np.float32)
            Y = group[["valence_deam_scale", "arousal_deam_scale"]].to_numpy(dtype=np.float32)
            T = group["frameTime"].to_numpy(dtype=np.float32)
            path = seq_dir / f"{int(music_id)}.npz"
            np.savez_compressed(path, X=X, Y=Y, time_sec=T, music_id=int(music_id), split=split)
            seq_rows.append({"musicId": int(music_id), "split": split, "frames": len(group), "path": str(path)})
    pd.DataFrame(seq_rows).to_csv(out_dir / "sequence_index.csv", index=False)
    arrays["feature_dim"] = len(feature_cols)
    arrays["frame_rows"] = int(len(merged))
    arrays["song_rows"] = int(merged["musicId"].nunique())
    (out_dir / "dataset_summary.json").write_text(json.dumps(arrays, ensure_ascii=False, indent=2), encoding="utf-8")
    return arrays


class FrameDataset(Dataset):
    def __init__(self, data_dir: Path, split: str):
        self.X = np.load(data_dir / f"X_{split}.npy").astype(np.float32)
        self.Y = np.load(data_dir / f"Y_{split}.npy").astype(np.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.from_numpy(self.X[idx]), torch.from_numpy(self.Y[idx])


class SequenceDataset(Dataset):
    def __init__(self, index: pd.DataFrame, split: str):
        self.rows = index[index["split"] == split].sort_values("musicId").to_dict("records")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        data = np.load(self.rows[idx]["path"])
        return torch.from_numpy(data["X"].astype(np.float32)), torch.from_numpy(data["Y"].astype(np.float32))


def collate_seq(batch):
    lengths = [x.shape[0] for x, _ in batch]
    max_len = max(lengths)
    feat_dim = batch[0][0].shape[1]
    X = torch.zeros(len(batch), max_len, feat_dim, dtype=torch.float32)
    Y = torch.zeros(len(batch), max_len, 2, dtype=torch.float32)
    mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
    for i, (x, y) in enumerate(batch):
        X[i, : x.shape[0]] = x
        Y[i, : y.shape[0]] = y
        mask[i, : x.shape[0]] = True
    return X, Y, mask


class MLP(nn.Module):
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
            nn.Linear(hidden // 2, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BiLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden: int, layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden,
            num_layers=layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.out = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden * 2, 2))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        y, _ = self.lstm(x)
        return self.out(y)


@torch.no_grad()
def eval_mlp(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, Any]:
    model.eval()
    preds, targets = [], []
    for X, Y in loader:
        pred = model(X.to(device)).detach().cpu().numpy()
        preds.append(pred)
        targets.append(Y.numpy())
    pred = np.concatenate(preds, axis=0)
    target = np.concatenate(targets, axis=0)
    return {"metrics": metrics(pred, target), "pred": pred, "target": target}


@torch.no_grad()
def eval_seq(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, Any]:
    model.eval()
    preds, targets = [], []
    for X, Y, mask in loader:
        X = X.to(device)
        mask = mask.to(device)
        pred = model(X, mask).detach().cpu()
        preds.append(pred[mask.cpu()].numpy())
        targets.append(Y[mask.cpu()].numpy())
    pred = np.concatenate(preds, axis=0)
    target = np.concatenate(targets, axis=0)
    return {"metrics": metrics(pred, target), "pred": pred, "target": target}


@dataclass
class TrialResult:
    trial_id: str
    model_type: str
    best_epoch: int
    val_rmse: float
    test_rmse: float
    test_mae: float
    test_mean_pcc: float
    test_mean_ccc: float
    checkpoint: str
    summary_path: str


def train_mlp(data_dir: Path, out_dir: Path, input_dim: int, device: torch.device, seed: int) -> TrialResult:
    set_seed(seed)
    loaders = {
        split: DataLoader(FrameDataset(data_dir, split), batch_size=512, shuffle=(split == "train"))
        for split in ["train", "val", "test"]
    }
    model = MLP(input_dim, hidden=384, dropout=0.25).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    best = float("inf")
    best_epoch = 0
    ckpt = out_dir / "mlp_best.pt"
    history = []
    patience = 12
    bad = 0
    for epoch in range(1, 121):
        model.train()
        losses = []
        for X, Y in loaders["train"]:
            X = X.to(device)
            Y = Y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(X), Y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        val = eval_mlp(model, loaders["val"], device)["metrics"]
        history.append({"epoch": epoch, "train_mse": float(np.mean(losses)), **{f"val_{k}": v for k, v in val.items()}})
        print(f"mlp epoch={epoch:03d} train_mse={np.mean(losses):.5f} val_rmse={val['rmse']:.5f} val_ccc={val['mean_ccc']:.3f}")
        if val["rmse"] < best:
            best = val["rmse"]
            best_epoch = epoch
            bad = 0
            torch.save({"model_state": model.state_dict(), "input_dim": input_dim, "seed": seed}, ckpt)
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(torch.load(ckpt, map_location=device)["model_state"])
    split_metrics = {split: eval_mlp(model, loaders[split], device)["metrics"] for split in ["train", "val", "test"]}
    summary = {"trial_id": "pmemo_opensmile_mlp", "model_type": "mlp", "best_epoch": best_epoch, "scale": "minus1_1", **{f"{s}_{k}": v for s, m in split_metrics.items() for k, v in m.items()}}
    pd.DataFrame(history).to_csv(out_dir / "mlp_history.csv", index=False)
    summary_path = out_dir / "mlp_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return TrialResult("pmemo_opensmile_mlp", "mlp", best_epoch, summary["val_rmse"], summary["test_rmse"], summary["test_mae"], summary["test_mean_pcc"], summary["test_mean_ccc"], str(ckpt), str(summary_path))


def train_bilstm(data_dir: Path, out_dir: Path, input_dim: int, device: torch.device, seed: int) -> TrialResult:
    set_seed(seed)
    index = pd.read_csv(data_dir / "sequence_index.csv")
    loaders = {
        split: DataLoader(SequenceDataset(index, split), batch_size=24, shuffle=(split == "train"), collate_fn=collate_seq)
        for split in ["train", "val", "test"]
    }
    model = BiLSTM(input_dim, hidden=128, layers=2, dropout=0.25).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    best = float("inf")
    best_epoch = 0
    ckpt = out_dir / "bilstm_best.pt"
    history = []
    patience = 12
    bad = 0
    for epoch in range(1, 121):
        model.train()
        losses = []
        for X, Y, mask in loaders["train"]:
            X = X.to(device)
            Y = Y.to(device)
            mask = mask.to(device)
            opt.zero_grad()
            pred = model(X, mask)
            loss = ((pred - Y) ** 2)[mask].mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        val = eval_seq(model, loaders["val"], device)["metrics"]
        history.append({"epoch": epoch, "train_mse": float(np.mean(losses)), **{f"val_{k}": v for k, v in val.items()}})
        print(f"bilstm epoch={epoch:03d} train_mse={np.mean(losses):.5f} val_rmse={val['rmse']:.5f} val_ccc={val['mean_ccc']:.3f}")
        if val["rmse"] < best:
            best = val["rmse"]
            best_epoch = epoch
            bad = 0
            torch.save({"model_state": model.state_dict(), "input_dim": input_dim, "seed": seed}, ckpt)
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(torch.load(ckpt, map_location=device)["model_state"])
    split_metrics = {split: eval_seq(model, loaders[split], device)["metrics"] for split in ["train", "val", "test"]}
    summary = {"trial_id": "pmemo_opensmile_bilstm", "model_type": "bilstm", "best_epoch": best_epoch, "scale": "minus1_1", **{f"{s}_{k}": v for s, m in split_metrics.items() for k, v in m.items()}}
    pd.DataFrame(history).to_csv(out_dir / "bilstm_history.csv", index=False)
    summary_path = out_dir / "bilstm_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return TrialResult("pmemo_opensmile_bilstm", "bilstm", best_epoch, summary["val_rmse"], summary["test_rmse"], summary["test_mae"], summary["test_mean_pcc"], summary["test_mean_ccc"], str(ckpt), str(summary_path))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extracted-dir", type=Path, default=DEFAULT_EXTRACTED)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--model", choices=["mlp", "bilstm", "both"], default="both")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    arrays = prepare_arrays(args.extracted_dir, args.manifest, args.out_dir / "dataset", args.seed)
    device = resolve_device(args.device)
    print(f"device={device} feature_dim={arrays['feature_dim']} splits={arrays['splits']}")
    results: list[TrialResult] = []
    if args.model in {"mlp", "both"}:
        results.append(train_mlp(args.out_dir / "dataset", args.out_dir, arrays["feature_dim"], device, args.seed))
    if args.model in {"bilstm", "both"}:
        results.append(train_bilstm(args.out_dir / "dataset", args.out_dir, arrays["feature_dim"], device, args.seed))
    rows = [asdict(r) for r in results]
    pd.DataFrame(rows).to_csv(args.out_dir / "comparison.csv", index=False)
    (args.out_dir / "comparison.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
