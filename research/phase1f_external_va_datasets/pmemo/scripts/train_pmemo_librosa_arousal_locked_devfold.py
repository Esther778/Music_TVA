#!/usr/bin/env python3
"""Train Librosa Arousal BiLSTM on a final-locked development fold."""

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

from train_pmemo_librosa_arousal import ArousalBiLSTM, collate, metrics, resolve_device
from train_pmemo_opensmile_arousal_trials import ROOT, moving_average_by_song, threshold_rows


DEFAULT_LIBROSA_DATASET = ROOT / "outputs" / "arousal_fullsong_librosa" / "dataset"


@dataclass
class Config:
    trial_id: str = "librosa_h128_l2_do025_lr8e4"
    hidden_dim: int = 128
    layers: int = 2
    dropout: float = 0.25
    learning_rate: float = 8e-4
    weight_decay: float = 1e-4
    batch_size: int = 24
    epochs: int = 90
    patience: int = 12


class LibrosaFoldDataset(Dataset):
    def __init__(self, rows: pd.DataFrame, mean: np.ndarray, std: np.ndarray):
        self.rows = rows.sort_values("musicId").to_dict("records")
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        data = np.load(row["path"])
        x = ((data["X"].astype(np.float32) - self.mean) / self.std).astype(np.float32)
        y = data["y"].astype(np.float32)
        time_sec = data["time_sec"].astype(np.float32)
        n = min(len(x), len(y), len(time_sec))
        return torch.from_numpy(x[:n]), torch.from_numpy(y[:n])


class LibrosaEvalDataset(Dataset):
    def __init__(self, rows: pd.DataFrame, mean: np.ndarray, std: np.ndarray):
        self.rows = rows.sort_values("musicId").to_dict("records")
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        data = np.load(row["path"])
        x = ((data["X"].astype(np.float32) - self.mean) / self.std).astype(np.float32)
        y = data["y"].astype(np.float32)
        time_sec = data["time_sec"].astype(np.float32)
        n = min(len(x), len(y), len(time_sec))
        song = np.full(n, int(row["musicId"]), dtype=np.int32)
        return torch.from_numpy(x[:n]), torch.from_numpy(y[:n]), torch.from_numpy(song), torch.from_numpy(time_sec[:n])


def collate_eval(batch):
    lengths = [x.shape[0] for x, _, _, _ in batch]
    max_len = max(lengths)
    feat_dim = batch[0][0].shape[1]
    x = torch.zeros(len(batch), max_len, feat_dim)
    y = torch.zeros(len(batch), max_len)
    song = torch.zeros(len(batch), max_len, dtype=torch.int32)
    time_sec = torch.zeros(len(batch), max_len)
    mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
    for i, (xi, yi, si, ti) in enumerate(batch):
        x[i, : xi.shape[0]] = xi
        y[i, : yi.shape[0]] = yi
        song[i, : xi.shape[0]] = si
        time_sec[i, : xi.shape[0]] = ti
        mask[i, : xi.shape[0]] = True
    return x, y, song, time_sec, mask


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_index(locked_index: Path, librosa_dataset: Path, out_dir: Path) -> pd.DataFrame:
    locked = pd.read_csv(locked_index)
    rows = []
    for row in locked.to_dict("records"):
        sid = int(row["musicId"])
        path = librosa_dataset / "sequence" / f"{sid}.npz"
        if not path.exists():
            continue
        data = np.load(path)
        rows.append({"musicId": sid, "split": row["split"], "frames": int(len(data["y"])), "path": str(path)})
    index = pd.DataFrame(rows)
    index.to_csv(out_dir / "sequence_index.csv", index=False)
    return index


def compute_norm(index: pd.DataFrame, out_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    xs = [np.load(row["path"])["X"].astype(np.float32) for row in index[index["split"] == "train"].to_dict("records")]
    concat = np.concatenate(xs, axis=0)
    mean = concat.mean(axis=0).astype(np.float32)
    std = (concat.std(axis=0) + 1e-8).astype(np.float32)
    np.savez(out_dir / "feature_norm.npz", mean=mean, std=std)
    return mean, std


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


def postprocess_search(val_eval: dict[str, Any]) -> tuple[dict[str, Any], pd.DataFrame]:
    pred = val_eval["pred"]
    target = val_eval["target"]
    song = val_eval["song_id"]
    rows = []
    best_m = metrics(pred, target)
    best = {"postprocess": "raw", "metrics": best_m, "score": best_m["rmse"]}
    for window in [1, 3, 7, 11, 15, 21, 31, 51, 71, 91]:
        smoothed = moving_average_by_song(pred, song, window)
        for blend in [0.25, 0.5, 0.75, 1.0]:
            pp = (1.0 - blend) * pred + blend * smoothed
            m = metrics(pp, target)
            row = {"postprocess": f"smooth_w{window}_b{blend:g}", "window": window, "blend": blend, **m, "score": m["rmse"]}
            rows.append(row)
            if m["rmse"] < best["score"]:
                best = {"postprocess": row["postprocess"], "metrics": m, "score": m["rmse"], "window": window, "blend": blend}
    return best, pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--librosa-dataset", type=Path, default=DEFAULT_LIBROSA_DATASET)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    config = Config()
    set_seed(args.seed)
    device = resolve_device(args.device)
    index = build_index(args.index, args.librosa_dataset, args.out_dir)
    mean, std = compute_norm(index, args.out_dir)
    loaders = {
        "train": DataLoader(LibrosaFoldDataset(index[index["split"] == "train"], mean, std), batch_size=config.batch_size, shuffle=True, collate_fn=collate),
        "val_train": DataLoader(LibrosaEvalDataset(index[index["split"] == "train"], mean, std), batch_size=config.batch_size, shuffle=False, collate_fn=collate_eval),
        "val": DataLoader(LibrosaEvalDataset(index[index["split"] == "val"], mean, std), batch_size=config.batch_size, shuffle=False, collate_fn=collate_eval),
    }
    input_dim = int(mean.shape[0])
    model = ArousalBiLSTM(input_dim, config.hidden_dim, config.layers, config.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    ckpt = args.out_dir / "best.pt"
    best_rmse = float("inf")
    best_epoch = 0
    bad = 0
    history = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        losses = []
        for x, y, mask in loaders["train"]:
            x = x.to(device); y = y.to(device); mask = mask.to(device)
            opt.zero_grad()
            pred = model(x, mask)
            loss = ((pred - y) ** 2)[mask].mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        val_eval = evaluate(model, loaders["val"], device)
        m = val_eval["metrics"]
        history.append({"epoch": epoch, "train_mse": float(np.mean(losses)), **{f"val_{k}": v for k, v in m.items()}})
        print(f"{config.trial_id} epoch={epoch:03d} train_mse={np.mean(losses):.5f} val_rmse={m['rmse']:.5f} val_pcc={m['pcc']:.3f} val_ccc={m['ccc']:.3f}", flush=True)
        if m["rmse"] < best_rmse:
            best_rmse = m["rmse"]
            best_epoch = epoch
            bad = 0
            torch.save({"model_state": model.state_dict(), "config": asdict(config), "input_dim": input_dim}, ckpt)
        else:
            bad += 1
            if bad >= config.patience:
                break
    pd.DataFrame(history).to_csv(args.out_dir / "history.csv", index=False)
    model.load_state_dict(torch.load(ckpt, map_location=device)["model_state"])
    evals = {"train": evaluate(model, loaders["val_train"], device), "val": evaluate(model, loaders["val"], device)}
    for split, data in evals.items():
        pd.DataFrame({"musicId": data["song_id"], "time_sec": data["time_sec"], "target_arousal": data["target"], "pred_arousal": data["pred"]}).to_csv(args.out_dir / f"{split}_predictions.csv", index=False)
    best_pp, pp_rows = postprocess_search(evals["val"])
    pp_rows.to_csv(args.out_dir / "val_postprocess_search.csv", index=False)
    summary = {
        "created_at": "2026-07-13",
        "purpose": "Librosa Arousal locked development fold trial; final test excluded",
        "best_epoch": best_epoch,
        "config": asdict(config),
        **{f"{split}_{k}": v for split, e in evals.items() for k, v in e["metrics"].items()},
        "best_val_postprocess": best_pp,
        "val_gap_rows_best_postprocess": threshold_rows(best_pp["metrics"]),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
