#!/usr/bin/env python3
"""Train MERT + full Librosa Valence model on locked development folds.

The full Librosa feature set includes MFCC, spectral descriptors, chroma,
tonnetz, RMS, onset, and tempogram pools. This is intended as a Valence-focused
audio-only feature upgrade while keeping arbitrary-song inference feasible.
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

from train_pmemo_mert_opensmile_arousal_fusion_locked_devfold import DEFAULT_MERT_DIR, build_mert_norm
from train_pmemo_mert_valence_baseline import ROOT, ccc_loss, metrics, resolve_device, threshold_rows
from train_pmemo_valence_selected_fusion import selected_indices
from train_pmemo_valence_dev_only_trials import postprocess_search, selection_score
from train_pmemo_valence_locked_transformer_smile import clip_mean_loss
from train_pmemo_opensmile_arousal_trials import collate, smoothness_loss


DEFAULT_INDEX = ROOT / "data" / "splits" / "pmemo_final_locked_20260713" / "devfolds_3fold_20260713" / "fold_01" / "sequence_index.csv"
DEFAULT_LABEL_DIR = ROOT / "outputs" / "opensmile_baseline" / "dataset" / "sequence"
DEFAULT_LIBROSA_DATASET = ROOT / "outputs" / "arousal_fullsong_librosa" / "dataset" / "sequence"
DEFAULT_OUT = ROOT / "outputs" / "valence_locked_mert_full_librosa_20260713" / "fold_01"


@dataclass
class Config:
    trial_id: str = "v_mert_full_librosa_p384_h96_do035_lr25e4"
    proj_dim: int = 384
    hidden_dim: int = 96
    dropout: float = 0.35
    learning_rate: float = 2.5e-4
    weight_decay: float = 4e-4
    batch_size: int = 18
    ccc_weight: float = 0.04
    smooth_weight: float = 0.002
    clip_mean_weight: float = 0.08
    use_delta: bool = True
    feature_groups: tuple[str, ...] = ("all",)
    epochs: int = 90
    patience: int = 12


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_feature_idx(feature_groups: tuple[str, ...]) -> np.ndarray | None:
    if "all" in feature_groups:
        return None
    return np.asarray(selected_indices(list(feature_groups)), dtype=np.int64)


def build_librosa_norm(index: pd.DataFrame, librosa_dir: Path, out_dir: Path, use_delta: bool, feature_idx: np.ndarray | None) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    xs = []
    for row in index[index["split"] == "train"].to_dict("records"):
        sid = int(row["musicId"])
        x = np.load(librosa_dir / f"{sid}.npz")["X"].astype(np.float32)
        if feature_idx is not None:
            x = x[:, feature_idx]
        if use_delta:
            dx = np.vstack([np.zeros((1, x.shape[1]), dtype=np.float32), np.diff(x, axis=0)])
            x = np.concatenate([x, dx], axis=1)
        xs.append(x)
    concat = np.concatenate(xs, axis=0)
    mean = concat.mean(axis=0).astype(np.float32)
    std = (concat.std(axis=0) + 1e-8).astype(np.float32)
    np.savez(out_dir / "librosa_norm.npz", mean=mean, std=std)
    summary = {
        "librosa_dim": int(concat.shape[1]),
        "use_delta": bool(use_delta),
        "feature_idx": None if feature_idx is None else feature_idx.astype(int).tolist(),
        "train_frames": int(concat.shape[0]),
    }
    (out_dir / "librosa_norm_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


class MertFullLibrosaDataset(Dataset):
    def __init__(
        self,
        index: pd.DataFrame,
        split: str,
        mert_dir: Path,
        label_dir: Path,
        librosa_dir: Path,
        mert_mean: np.ndarray,
        mert_std: np.ndarray,
        lib_mean: np.ndarray,
        lib_std: np.ndarray,
        use_delta: bool,
        feature_idx: np.ndarray | None,
    ):
        self.rows = index[index["split"] == split].sort_values("musicId").to_dict("records")
        self.mert_dir = mert_dir
        self.label_dir = label_dir
        self.librosa_dir = librosa_dir
        self.mert_mean = mert_mean.astype(np.float32)
        self.mert_std = mert_std.astype(np.float32)
        self.lib_mean = lib_mean.astype(np.float32)
        self.lib_std = lib_std.astype(np.float32)
        self.use_delta = use_delta
        self.feature_idx = feature_idx

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        sid = int(self.rows[idx]["musicId"])
        mert = np.load(self.mert_dir / f"{sid}.npz")
        label = np.load(self.label_dir / f"{sid}.npz")
        lib = np.load(self.librosa_dir / f"{sid}.npz")
        x_mert = ((mert["embedding"].astype(np.float32) - self.mert_mean) / self.mert_std).astype(np.float32)
        x_lib = lib["X"].astype(np.float32)
        if self.feature_idx is not None:
            x_lib = x_lib[:, self.feature_idx]
        if self.use_delta:
            dx = np.vstack([np.zeros((1, x_lib.shape[1]), dtype=np.float32), np.diff(x_lib, axis=0)])
            x_lib = np.concatenate([x_lib, dx], axis=1)
        x_lib = ((x_lib - self.lib_mean) / self.lib_std).astype(np.float32)
        y = label["Y"].astype(np.float32)[:, 0]
        time_sec = mert["time_sec"].astype(np.float32)
        n = min(len(x_mert), len(x_lib), len(y), len(time_sec))
        x = np.concatenate([x_mert[:n], x_lib[:n]], axis=1).astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(y[:n]), torch.full((n,), sid, dtype=torch.int32), torch.from_numpy(time_sec[:n])


class FusionBiLSTM(nn.Module):
    def __init__(self, input_dim: int, cfg: Config):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, cfg.proj_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )
        self.lstm = nn.LSTM(cfg.proj_dim, cfg.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim * 2, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        y, _ = self.lstm(self.proj(x))
        return self.head(y).squeeze(-1)


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
    return {
        "metrics": metrics(pred_np, target_np),
        "pred": pred_np,
        "target": target_np,
        "song_id": np.concatenate(songs),
        "time_sec": np.concatenate(times),
    }


def run(args: argparse.Namespace, cfg: Config) -> dict[str, Any]:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    index = pd.read_csv(args.index)
    index = index[index["split"].isin(["train", "val"])].copy()
    mert_mean, mert_std = build_mert_norm(index, "train", args.mert_dir, args.out_dir)
    feature_idx = resolve_feature_idx(cfg.feature_groups)
    lib_summary = build_librosa_norm(index, args.librosa_dir, args.out_dir, cfg.use_delta, feature_idx)
    lib_norm = np.load(args.out_dir / "librosa_norm.npz")
    loaders = {
        split: DataLoader(
            MertFullLibrosaDataset(
                index,
                split,
                args.mert_dir,
                args.label_dir,
                args.librosa_dir,
                mert_mean,
                mert_std,
                lib_norm["mean"],
                lib_norm["std"],
                cfg.use_delta,
                feature_idx,
            ),
            batch_size=cfg.batch_size,
            shuffle=(split == "train"),
            collate_fn=collate,
        )
        for split in ["train", "val"]
    }
    sample_x, _, _, _ = loaders["train"].dataset[0]
    device = resolve_device(args.device)
    model = FusionBiLSTM(int(sample_x.shape[1]), cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    ckpt = args.out_dir / "best.pt"
    best_score = float("inf")
    best_epoch = 0
    bad = 0
    history = []
    for epoch in range(1, cfg.epochs + 1):
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
            loss = loss + cfg.ccc_weight * ccc_loss(pred_valid, y_valid)
            loss = loss + cfg.smooth_weight * smoothness_loss(pred, mask)
            loss = loss + clip_mean_loss(pred, y, mask, cfg.clip_mean_weight)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        val_eval = evaluate(model, loaders["val"], device)
        m = val_eval["metrics"]
        score = selection_score(m)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), **{f"val_{k}": v for k, v in m.items()}, "score": score}
        history.append(row)
        print(
            f"{cfg.trial_id} epoch={epoch:03d} train_loss={row['train_loss']:.5f} "
            f"val_rmse={m['rmse']:.5f} val_mae={m['mae']:.5f} val_pcc={m['pcc']:.3f} val_ccc={m['ccc']:.3f}",
            flush=True,
        )
        if score < best_score:
            best_score = score
            best_epoch = epoch
            bad = 0
            torch.save({"model_state": model.state_dict(), "input_dim": int(sample_x.shape[1]), "config": asdict(cfg)}, ckpt)
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    pd.DataFrame(history).to_csv(args.out_dir / "history.csv", index=False)
    model.load_state_dict(torch.load(ckpt, map_location=device)["model_state"])
    evals = {split: evaluate(model, loaders[split], device) for split in ["train", "val"]}
    for split, data in evals.items():
        pd.DataFrame(
            {
                "musicId": data["song_id"],
                "time_sec": data["time_sec"],
                "target_valence": data["target"],
                "pred_valence": data["pred"],
            }
        ).to_csv(args.out_dir / f"{split}_predictions.csv", index=False)
    best_pp, pp_rows = postprocess_search(evals["val"])
    pp_rows.to_csv(args.out_dir / "val_postprocess_search.csv", index=False)
    summary = {
        "created_at": "2026-07-13",
        "purpose": "MERT hidden6 + full Librosa Valence on locked devfold; final locked test excluded",
        "selection_boundary": "Development-fold validation only; not final locked test performance",
        "trial_id": cfg.trial_id,
        "best_epoch": best_epoch,
        "config": asdict(cfg),
        "librosa_summary": lib_summary,
        "val_metrics": evals["val"]["metrics"],
        "best_val_postprocess": best_pp,
        "gap_rows_raw": threshold_rows(evals["val"]["metrics"]),
        "gap_rows_best_postprocess": threshold_rows(best_pp["metrics"]),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--mert-dir", type=Path, default=DEFAULT_MERT_DIR)
    parser.add_argument("--label-dir", type=Path, default=DEFAULT_LABEL_DIR)
    parser.add_argument("--librosa-dir", type=Path, default=DEFAULT_LIBROSA_DATASET)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--feature-groups", default="all", help="Comma-separated Librosa groups: all,mfcc,spectral,harmonic,energy,tempo")
    parser.add_argument("--no-delta", action="store_true")
    args = parser.parse_args()
    cfg = Config(
        trial_id=(
            "v_mert_librosa_"
            + args.feature_groups.replace(",", "_")
            + ("_nodelta" if args.no_delta else "_delta")
        ),
        use_delta=not args.no_delta,
        feature_groups=tuple(x.strip() for x in args.feature_groups.split(",") if x.strip()),
    )
    run(args, cfg)


if __name__ == "__main__":
    main()
