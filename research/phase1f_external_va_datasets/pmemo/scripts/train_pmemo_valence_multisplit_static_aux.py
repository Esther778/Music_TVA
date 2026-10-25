#!/usr/bin/env python3
"""Multi-split Valence model with PMEmo static Valence auxiliary supervision.

The previous global/polarity auxiliary trials used the dynamic frame mean as
the song-level target. This experiment uses PMEmo static_annotations.csv as a
separate global emotion target, while preserving the same held-out multisplit
protocol used by the current Valence candidate.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from train_pmemo_mert_harmonic_valence_fusion import (
    DEFAULT_LABEL_DIR,
    DEFAULT_LIBROSA_DATASET,
    FusionDataset,
    build_harmonic_norm,
    collate,
    selected_indices,
)
from train_pmemo_mert_hidden6_harmonic_extreme_weighted import clip_mean_loss, clip_std_loss, weighted_frame_mse
from train_pmemo_mert_valence_baseline import ROOT, ccc_loss, metrics, resolve_device, threshold_rows
from train_pmemo_valence_multisplit_best import DEFAULT_MERT_DIR, aggregate_metrics, build_mert_norm


BASE_SPLIT_DIR = ROOT / "outputs" / "valence_multisplit_best_20260627"
STATIC_ANNOTATIONS = ROOT / "data" / "raw" / "PMEmo_2019_updated" / "extracted" / "PMEmo2019" / "annotations" / "static_annotations.csv"
DEFAULT_OUT = ROOT / "outputs" / "valence_multisplit_static_aux_20260628"

CURRENT_BEST = {
    "model_id": "pmemo_valence_prediction_only_rescale_ccc_preserving_20260628",
    "rmse": 0.19531630337282654,
    "mae": 0.15179290177231075,
    "pcc": 0.7977655929469952,
    "ccc": 0.7848342582143512,
}

PASS = {"rmse": 0.18, "mae": 0.15, "pcc": 0.80, "ccc": 0.80}


@dataclass
class StaticAuxConfig:
    trial_id: str = "bilstm_static_aux_h80_do030_sw001_cons000"
    proj_dim: int = 320
    hidden_dim: int = 80
    dropout: float = 0.30
    learning_rate: float = 4e-4
    weight_decay: float = 2e-4
    batch_size: int = 20
    frame_ccc_weight: float = 0.05
    clip_mse_weight: float = 0.10
    clip_std_weight: float = 0.0
    static_head_weight: float = 0.01
    static_frame_consistency_weight: float = 0.0
    extreme_weight: float = 1.25
    blend_static_weight: float = 0.0
    patience: int = 8
    epochs: int = 60


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def selection_score(m: dict[str, float], static_m: dict[str, float]) -> float:
    _ = static_m
    return m["rmse"] + 0.05 * max(0.0, 0.80 - m["ccc"]) + 0.03 * max(0.0, 0.80 - m["pcc"]) + 0.01 * m["mae"]


def load_static_valence(path: Path) -> dict[int, float]:
    df = pd.read_csv(path)
    return {int(r["musicId"]): float(r["Valence(mean)"]) * 2.0 - 1.0 for _, r in df.iterrows()}


def batch_static_targets(song: torch.Tensor, static_map: dict[int, float], device: torch.device) -> torch.Tensor:
    ids = song[:, 0].detach().cpu().numpy().astype(int).tolist()
    values = [static_map[int(sid)] for sid in ids]
    return torch.tensor(values, dtype=torch.float32, device=device)


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_f = mask.float()
    return (x * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp_min(1.0)


class StaticAuxBiLSTM(nn.Module):
    def __init__(self, input_dim: int, config: StaticAuxConfig):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, config.proj_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.lstm = nn.LSTM(config.proj_dim, config.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        hidden2 = config.hidden_dim * 2
        self.frame_head = nn.Sequential(nn.Dropout(config.dropout), nn.Linear(hidden2, 1))
        self.static_head = nn.Sequential(
            nn.LayerNorm(hidden2 * 3),
            nn.Linear(hidden2 * 3, hidden2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden2, 1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h, _ = self.lstm(self.proj(x))
        frame = self.frame_head(h).squeeze(-1)
        mask_f = mask.float().unsqueeze(-1)
        count = mask_f.sum(dim=1).clamp_min(1.0)
        mean = (h * mask_f).sum(dim=1) / count
        var = (((h - mean.unsqueeze(1)) ** 2) * mask_f).sum(dim=1) / count
        std = torch.sqrt(var + 1e-6)
        max_pool = h.masked_fill(~mask.unsqueeze(-1), -1e9).max(dim=1).values
        static = self.static_head(torch.cat([mean, std, max_pool], dim=1)).squeeze(-1)
        return frame, static


def blended_prediction(frame_pred: torch.Tensor, static_pred: torch.Tensor, blend_weight: float) -> torch.Tensor:
    if not blend_weight:
        return frame_pred
    return (1.0 - float(blend_weight)) * frame_pred + float(blend_weight) * static_pred.unsqueeze(1)


def song_level_metrics(song_ids: np.ndarray, target: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    df = pd.DataFrame({"musicId": song_ids, "target": target, "pred": pred})
    grouped = df.groupby("musicId", sort=False).mean(numeric_only=True)
    return metrics(grouped["pred"].to_numpy(dtype=np.float64), grouped["target"].to_numpy(dtype=np.float64))


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, static_map: dict[int, float], config: StaticAuxConfig) -> dict[str, Any]:
    model.eval()
    preds, raw_preds, targets, songs, times = [], [], [], [], []
    static_preds, static_targets, repeated_static_preds = [], [], []
    for x, y, song, time_sec, mask in loader:
        x = x.to(device)
        mask_dev = mask.to(device)
        frame_pred, static_pred = model(x, mask_dev)
        pred = blended_prediction(frame_pred, static_pred, config.blend_static_weight).detach().cpu()
        raw_pred = frame_pred.detach().cpu()
        static_cpu = static_pred.detach().cpu()
        static_target = batch_static_targets(song, static_map, device=torch.device("cpu"))
        repeated = torch.zeros_like(raw_pred)
        for i in range(raw_pred.shape[0]):
            repeated[i, mask[i]] = static_cpu[i]
        preds.append(pred[mask].numpy())
        raw_preds.append(raw_pred[mask].numpy())
        targets.append(y[mask].numpy())
        songs.append(song[mask].numpy())
        times.append(time_sec[mask].numpy())
        static_preds.append(static_cpu.numpy())
        static_targets.append(static_target.numpy())
        repeated_static_preds.append(repeated[mask].numpy())
    pred_np = np.concatenate(preds)
    raw_pred_np = np.concatenate(raw_preds)
    target_np = np.concatenate(targets)
    song_np = np.concatenate(songs)
    static_pred_np = np.concatenate(static_preds)
    static_target_np = np.concatenate(static_targets)
    repeated_static_pred_np = np.concatenate(repeated_static_preds)
    return {
        "metrics": metrics(pred_np, target_np),
        "raw_metrics": metrics(raw_pred_np, target_np),
        "song_mean_metrics": song_level_metrics(song_np, target_np, pred_np),
        "static_head_metrics": metrics(static_pred_np, static_target_np),
        "pred": pred_np,
        "raw_pred": raw_pred_np,
        "target": target_np,
        "song_id": song_np,
        "time_sec": np.concatenate(times),
        "static_head_pred": repeated_static_pred_np,
    }


def train_one_split(split_dir_in: Path, split_id: int, seed: int, args: argparse.Namespace, device: torch.device, config: StaticAuxConfig, static_map: dict[int, float]) -> dict[str, Any]:
    set_seed(seed)
    split_dir = args.out_dir / config.trial_id / f"split_{split_id:02d}_seed_{seed}"
    split_dir.mkdir(parents=True, exist_ok=True)
    index = pd.read_csv(split_dir_in / "sequence_index.csv")
    index.to_csv(split_dir / "sequence_index.csv", index=False)
    build_mert_norm(index, args.mert_dir, split_dir / "dataset")
    build_harmonic_norm(index, args.librosa_dataset, selected_indices(["harmonic"]), split_dir / "dataset", force=True)
    mert_norm = np.load(split_dir / "dataset" / "mert_norm.npz")
    harm_norm = np.load(split_dir / "dataset" / "harmonic_norm.npz")
    loaders = {
        split: DataLoader(
            FusionDataset(
                index,
                split,
                args.mert_dir,
                args.label_dir,
                args.librosa_dataset,
                mert_norm["mean"],
                mert_norm["std"],
                harm_norm["mean"],
                harm_norm["std"],
                harm_norm["feature_idx"],
            ),
            batch_size=config.batch_size,
            shuffle=(split == "train"),
            collate_fn=collate,
        )
        for split in ["train", "val", "test"]
    }
    sample_x, _, _, _ = loaders["train"].dataset[0]
    model = StaticAuxBiLSTM(int(sample_x.shape[1]), config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    ckpt = split_dir / "best.pt"
    history = []
    best_score = float("inf")
    best_epoch = 0
    bad = 0
    for epoch in range(1, config.epochs + 1):
        model.train()
        losses = []
        for x, y, song, _, mask in loaders["train"]:
            x = x.to(device)
            y = y.to(device)
            mask = mask.to(device)
            static_target = batch_static_targets(song, static_map, device)
            optimizer.zero_grad()
            frame_pred, static_pred = model(x, mask)
            pred_valid = frame_pred[mask]
            y_valid = y[mask]
            frame_mean = masked_mean(frame_pred, mask)
            loss = weighted_frame_mse(pred_valid, y_valid, config.extreme_weight)
            loss = loss + config.frame_ccc_weight * ccc_loss(pred_valid, y_valid)
            loss = loss + clip_mean_loss(frame_pred, y, mask, config.clip_mse_weight)
            loss = loss + clip_std_loss(frame_pred, y, mask, config.clip_std_weight)
            loss = loss + config.static_head_weight * ((static_pred - static_target) ** 2).mean()
            loss = loss + config.static_frame_consistency_weight * ((frame_mean - static_pred.detach()) ** 2).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        val_eval = evaluate(model, loaders["val"], device, static_map, config)
        val_m = val_eval["metrics"]
        val_static = val_eval["static_head_metrics"]
        score = selection_score(val_m, val_static)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            **{f"val_{k}": v for k, v in val_m.items()},
            **{f"val_static_{k}": v for k, v in val_static.items()},
            "val_select_score": score,
        }
        history.append(row)
        print(
            f"static_aux split={split_id:02d} seed={seed} epoch={epoch:03d} loss={row['train_loss']:.5f} "
            f"val_rmse={val_m['rmse']:.5f} val_pcc={val_m['pcc']:.3f} val_ccc={val_m['ccc']:.3f} "
            f"static_rmse={val_static['rmse']:.5f}",
            flush=True,
        )
        if score < best_score:
            best_score = score
            best_epoch = epoch
            bad = 0
            torch.save({"model_state": model.state_dict(), "seed": seed, "config": asdict(config), "input_dim": int(sample_x.shape[1])}, ckpt)
        else:
            bad += 1
            if bad >= config.patience:
                break
    pd.DataFrame(history).to_csv(split_dir / "history.csv", index=False)
    model.load_state_dict(torch.load(ckpt, map_location=device)["model_state"])
    split_evals = {split: evaluate(model, loaders[split], device, static_map, config) for split in ["train", "val", "test"]}
    for split, data in split_evals.items():
        pd.DataFrame(
            {
                "musicId": data["song_id"],
                "time_sec": data["time_sec"],
                "target_valence": data["target"],
                "pred_valence": data["pred"],
                "raw_pred_valence": data["raw_pred"],
                "static_head_pred": data["static_head_pred"],
            }
        ).to_csv(split_dir / f"{split}_predictions.csv", index=False)
    summary = {
        "split_id": split_id,
        "seed": seed,
        "best_epoch": best_epoch,
        "model_family": "MERT hidden_6 + harmonic BiLSTM with PMEmo static Valence auxiliary head",
        "config": asdict(config),
        **{f"{split}_{k}": v for split, e in split_evals.items() for k, v in e["metrics"].items()},
        **{f"{split}_raw_{k}": v for split, e in split_evals.items() for k, v in e["raw_metrics"].items()},
        **{f"{split}_song_mean_{k}": v for split, e in split_evals.items() for k, v in e["song_mean_metrics"].items()},
        **{f"{split}_static_head_{k}": v for split, e in split_evals.items() for k, v in e["static_head_metrics"].items()},
        "test_gap_rows": threshold_rows(split_evals["test"]["metrics"]),
    }
    (split_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def split_seed(split_dir: Path) -> tuple[int, int]:
    parts = split_dir.name.split("_")
    return int(parts[1]), int(parts[3])


def aggregate(rows: list[dict[str, Any]], out_dir: Path, config: StaticAuxConfig) -> dict[str, Any]:
    trial_dir = out_dir / config.trial_id
    pd.DataFrame(rows).to_csv(trial_dir / "comparison.csv", index=False)
    data: dict[str, Any] = {
        "created_at": "2026-06-28",
        "current_best_reference": CURRENT_BEST,
        "static_annotation_source": str(STATIC_ANNOTATIONS),
        "config": asdict(config),
        "num_splits": len(rows),
        "aggregate": {**aggregate_metrics(rows, "val"), **aggregate_metrics(rows, "test")},
        "per_split": rows,
    }
    test_mean = {k: data["aggregate"][f"test_{k}_mean"] for k in ["rmse", "mae", "pcc", "ccc"]}
    data["mean_test_gap_rows"] = threshold_rows(test_mean)
    data["mean_test_delta_vs_current_best"] = {
        k: float(test_mean[k] - CURRENT_BEST[k]) for k in ["rmse", "mae", "pcc", "ccc"]
    }
    (trial_dir / "summary.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(data, ensure_ascii=False, indent=2), flush=True)
    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mert-dir", type=Path, default=DEFAULT_MERT_DIR)
    parser.add_argument("--label-dir", type=Path, default=DEFAULT_LABEL_DIR)
    parser.add_argument("--librosa-dataset", type=Path, default=DEFAULT_LIBROSA_DATASET)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    config = StaticAuxConfig()
    trial_dir = args.out_dir / config.trial_id
    if args.aggregate_only:
        rows = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(trial_dir.glob("split_*/summary.json"))]
        aggregate(rows, args.out_dir, config)
        return
    static_map = load_static_valence(STATIC_ANNOTATIONS)
    device = resolve_device(args.device)
    train_args = SimpleNamespace(out_dir=args.out_dir, mert_dir=args.mert_dir, label_dir=args.label_dir, librosa_dataset=args.librosa_dataset)
    rows = []
    for split_dir in sorted(BASE_SPLIT_DIR.glob("split_*")):
        split_id, seed = split_seed(split_dir)
        rows.append(train_one_split(split_dir, split_id, seed, train_args, device, config, static_map))
    aggregate(rows, args.out_dir, config)


if __name__ == "__main__":
    main()
