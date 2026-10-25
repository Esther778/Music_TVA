#!/usr/bin/env python3
"""Valence model with audio-derived Arousal-teacher auxiliary features."""

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

from search_pmemo_valence_dev_val_postprocess import fit_calibration, moving_average_by_song
from train_pmemo_mert_harmonic_valence_fusion import FusionBiLSTM, build_harmonic_norm, collate, selected_indices
from train_pmemo_mert_hidden6_harmonic_extreme_weighted import clip_mean_loss, weighted_frame_mse
from train_pmemo_mert_valence_baseline import ROOT, ccc_loss, metrics, resolve_device, threshold_rows
from train_pmemo_valence_multisplit_best import build_mert_norm


DEFAULT_MERT_DIR = ROOT / "outputs" / "mert_pmemo_hidden6" / "features"
DEFAULT_LABEL_DIR = ROOT / "outputs" / "opensmile_baseline" / "dataset" / "sequence"
DEFAULT_LIBROSA_DATASET = ROOT / "outputs" / "arousal_fullsong_librosa" / "dataset"


@dataclass
class Config:
    trial_id: str = "v_harm_arousal_teacher_h80_do035_lr3e4"
    proj_dim: int = 320
    hidden_dim: int = 80
    dropout: float = 0.35
    learning_rate: float = 3e-4
    weight_decay: float = 4e-4
    batch_size: int = 20
    frame_ccc_weight: float = 0.05
    clip_mse_weight: float = 0.08
    extreme_weight: float = 1.15
    epochs: int = 80
    patience: int = 10


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def arousal_paths(fold: str) -> dict[str, Path]:
    return {
        "h96": ROOT / "outputs" / "arousal_locked_devfolds_h96_20260713" / fold / "a_dev_h96_l1_do015_lr8e4_mse",
        "mert_smile": ROOT / "outputs" / "arousal_locked_mert_opensmile_fusion_20260713" / fold,
    }


def pred_col(df: pd.DataFrame) -> str:
    for col in ["pred_arousal", "pred", "prediction"]:
        if col in df.columns:
            return col
    raise ValueError(f"No prediction column in {list(df.columns)}")


def load_arousal_teacher(fold: str, split: str) -> pd.DataFrame:
    frames = []
    for name, base in arousal_paths(fold).items():
        path = base / f"{split}_predictions.csv"
        df = pd.read_csv(path)
        frames.append(df[["musicId", "time_sec", pred_col(df)]].rename(columns={pred_col(df): f"arousal_{name}"}))
    out = frames[0]
    for frame in frames[1:]:
        out = out.merge(frame, on=["musicId", "time_sec"], how="inner")
    out["arousal_teacher"] = 0.5 * out["arousal_h96"].to_numpy(dtype=np.float64) + 0.5 * out["arousal_mert_smile"].to_numpy(dtype=np.float64)
    pieces = []
    for _, g in out.sort_values(["musicId", "time_sec"]).groupby("musicId", sort=False):
        gg = g.copy()
        a = gg["arousal_teacher"].to_numpy(dtype=np.float32)
        gg["arousal_teacher_delta"] = np.diff(a, prepend=a[0])
        gg["arousal_teacher_mean"] = float(a.mean())
        gg["arousal_teacher_std"] = float(a.std())
        pieces.append(gg)
    return pd.concat(pieces, ignore_index=True)


def build_arousal_norm(train_teacher: pd.DataFrame, out_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    cols = ["arousal_teacher", "arousal_teacher_delta", "arousal_teacher_mean", "arousal_teacher_std"]
    x = train_teacher[cols].to_numpy(dtype=np.float32)
    mean = x.mean(axis=0).astype(np.float32)
    std = (x.std(axis=0) + 1e-8).astype(np.float32)
    np.savez(out_dir / "arousal_teacher_norm.npz", mean=mean, std=std, columns=np.array(cols))
    return mean, std


class ValenceArousalTeacherDataset(Dataset):
    def __init__(
        self,
        index: pd.DataFrame,
        split: str,
        fold: str,
        mert_dir: Path,
        label_dir: Path,
        librosa_dataset: Path,
        mert_mean: np.ndarray,
        mert_std: np.ndarray,
        harm_mean: np.ndarray,
        harm_std: np.ndarray,
        feature_idx: np.ndarray,
        arousal_mean: np.ndarray,
        arousal_std: np.ndarray,
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
        self.arousal_mean = arousal_mean.astype(np.float32)
        self.arousal_std = arousal_std.astype(np.float32)
        self.teacher = load_arousal_teacher(fold, split)

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
        time_sec = label["time_sec"].astype(np.float32)
        t = self.teacher[self.teacher["musicId"].astype(int) == sid].sort_values("time_sec")
        x_a = t[["arousal_teacher", "arousal_teacher_delta", "arousal_teacher_mean", "arousal_teacher_std"]].to_numpy(dtype=np.float32)
        x_a = ((x_a - self.arousal_mean) / self.arousal_std).astype(np.float32)
        n = min(len(x_mert), len(x_harm), len(x_a), len(y), len(time_sec))
        x = np.concatenate([x_mert[:n], x_harm[:n], x_a[:n]], axis=1).astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(y[:n]), torch.full((n,), sid, dtype=torch.int32), torch.from_numpy(time_sec[:n])


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
    best = {"postprocess": "raw", "metrics": val_eval["metrics"], "score": val_eval["metrics"]["rmse"]}
    for window in [1, 11, 21, 31, 41]:
        smoothed = moving_average_by_song(pred, song, window)
        for blend in [0.25, 0.50, 0.75, 1.0]:
            base = (1.0 - blend) * pred + blend * smoothed
            for method in ["identity", "linear", "mean_std"]:
                calibrated, params = fit_calibration(base, target, method)
                m = metrics(calibrated, target)
                score = m["rmse"] + 0.03 * max(0.0, 0.80 - m["pcc"]) + 0.05 * max(0.0, 0.80 - m["ccc"])
                row = {"postprocess": f"w{window}_b{blend:g}_{method}", "window": window, "blend": blend, "method": method, "params": json.dumps(params), **m, "score": score}
                rows.append(row)
                if score < best["score"]:
                    best = {"postprocess": row["postprocess"], "metrics": m, "score": score, **row}
    return best, pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", choices=["fold_01", "fold_02", "fold_03"], required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--mert-dir", type=Path, default=DEFAULT_MERT_DIR)
    parser.add_argument("--label-dir", type=Path, default=DEFAULT_LABEL_DIR)
    parser.add_argument("--librosa-dataset", type=Path, default=DEFAULT_LIBROSA_DATASET)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    config = Config()
    set_seed(args.seed)
    index = pd.read_csv(args.index)
    build_mert_norm(index, args.mert_dir, args.out_dir / "dataset")
    build_harmonic_norm(index, args.librosa_dataset, selected_indices(["harmonic"]), args.out_dir / "dataset", force=True)
    train_teacher = load_arousal_teacher(args.fold, "train")
    arousal_mean, arousal_std = build_arousal_norm(train_teacher, args.out_dir / "dataset")
    mert_norm = np.load(args.out_dir / "dataset" / "mert_norm.npz")
    harm_norm = np.load(args.out_dir / "dataset" / "harmonic_norm.npz")
    loaders = {
        split: DataLoader(
            ValenceArousalTeacherDataset(
                index,
                split,
                args.fold,
                args.mert_dir,
                args.label_dir,
                args.librosa_dataset,
                mert_norm["mean"],
                mert_norm["std"],
                harm_norm["mean"],
                harm_norm["std"],
                harm_norm["feature_idx"],
                arousal_mean,
                arousal_std,
            ),
            batch_size=config.batch_size,
            shuffle=(split == "train"),
            collate_fn=collate,
        )
        for split in ["train", "val"]
    }
    sample_x, _, _, _ = loaders["train"].dataset[0]
    device = resolve_device(args.device)
    model = FusionBiLSTM(int(sample_x.shape[1]), config.proj_dim, config.hidden_dim, config.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    ckpt = args.out_dir / "best.pt"
    best_score = float("inf")
    best_epoch = 0
    bad = 0
    history = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        losses = []
        for x, y, _, _, mask in loaders["train"]:
            x = x.to(device); y = y.to(device); mask = mask.to(device)
            opt.zero_grad()
            pred = model(x, mask)
            pred_valid = pred[mask]
            y_valid = y[mask]
            loss = weighted_frame_mse(pred_valid, y_valid, config.extreme_weight)
            loss = loss + config.frame_ccc_weight * ccc_loss(pred_valid, y_valid)
            loss = loss + clip_mean_loss(pred, y, mask, config.clip_mse_weight)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        val_eval = evaluate(model, loaders["val"], device)
        m = val_eval["metrics"]
        score = m["rmse"] + 0.03 * max(0.0, 0.80 - m["pcc"]) + 0.05 * max(0.0, 0.80 - m["ccc"])
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), **{f"val_{k}": v for k, v in m.items()}, "score": score})
        print(f"{config.trial_id} epoch={epoch:03d} train_loss={float(np.mean(losses)):.5f} val_rmse={m['rmse']:.5f} val_mae={m['mae']:.5f} val_pcc={m['pcc']:.3f} val_ccc={m['ccc']:.3f}", flush=True)
        if score < best_score:
            best_score = score
            best_epoch = epoch
            bad = 0
            torch.save({"model_state": model.state_dict(), "input_dim": int(sample_x.shape[1]), "config": asdict(config)}, ckpt)
        else:
            bad += 1
            if bad >= config.patience:
                break
    pd.DataFrame(history).to_csv(args.out_dir / "history.csv", index=False)
    model.load_state_dict(torch.load(ckpt, map_location=device)["model_state"])
    evals = {split: evaluate(model, loaders[split], device) for split in ["train", "val"]}
    for split, data in evals.items():
        pd.DataFrame({"musicId": data["song_id"], "time_sec": data["time_sec"], "target_valence": data["target"], "pred_valence": data["pred"]}).to_csv(args.out_dir / f"{split}_predictions.csv", index=False)
    best_pp, pp_rows = postprocess_search(evals["val"])
    pp_rows.to_csv(args.out_dir / "val_postprocess_search.csv", index=False)
    summary = {
        "created_at": "2026-07-13",
        "purpose": "Valence model with audio-derived Arousal teacher auxiliary features; final locked test excluded",
        "selection_boundary": "Development-fold validation only; not final locked test performance",
        "fold": args.fold,
        "trial_id": config.trial_id,
        "config": asdict(config),
        "best_epoch": best_epoch,
        "val_metrics": evals["val"]["metrics"],
        "best_val_postprocess": best_pp,
        "gap_rows_raw": threshold_rows(evals["val"]["metrics"]),
        "gap_rows_best_postprocess": threshold_rows(best_pp["metrics"]),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
