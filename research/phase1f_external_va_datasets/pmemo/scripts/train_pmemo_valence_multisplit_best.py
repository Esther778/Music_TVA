#!/usr/bin/env python3
"""Multi-split validation for the current best PMEmo Valence model family."""

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
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader

from train_pmemo_mert_harmonic_valence_fusion import (
    DEFAULT_INDEX,
    DEFAULT_LABEL_DIR,
    DEFAULT_LIBROSA_DATASET,
    FusionBiLSTM,
    FusionDataset,
    build_harmonic_norm,
    collate,
    evaluate,
    selected_indices,
)
from train_pmemo_mert_hidden6_harmonic_extreme_weighted import (
    clip_mean_loss,
    clip_std_loss,
    weighted_frame_mse,
)
from train_pmemo_mert_valence_baseline import ROOT, ccc_loss, metrics, resolve_device, threshold_rows


DEFAULT_MERT_DIR = ROOT / "outputs" / "mert_pmemo_hidden6" / "features"
DEFAULT_OUT = ROOT / "outputs" / "valence_multisplit_best_20260627"

HISTORICAL_SINGLE_SPLIT_BEST = {
    "model_id": "pmemo_valence_mert_hidden6_harmonic_extreme_weighted_v1",
    "trial_id": "h6_harm_h80_ext125_clip010_ccc005",
    "rmse": 0.205672949552536,
    "mae": 0.16456688940525055,
    "pcc": 0.7834003513415063,
    "ccc": 0.7673044285835935,
}

PASS = {"rmse": 0.18, "mae": 0.15, "pcc": 0.80, "ccc": 0.80}


@dataclass
class BestConfig:
    trial_id: str = "h6_harm_h80_ext125_clip010_ccc005"
    proj_dim: int = 320
    hidden_dim: int = 80
    dropout: float = 0.30
    learning_rate: float = 4e-4
    weight_decay: float = 2e-4
    batch_size: int = 20
    frame_ccc_weight: float = 0.05
    clip_mse_weight: float = 0.10
    extreme_weight: float = 1.25
    clip_std_weight: float = 0.0
    patience: int = 8
    epochs: int = 60


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def target_gap(value: float, metric_name: str, target: float) -> float:
    return float(value - target) if metric_name in {"rmse", "mae"} else float(target - value)


def aggregate_metrics(rows: list[dict[str, Any]], prefix: str) -> dict[str, float]:
    out = {}
    for key in ["rmse", "mae", "pcc", "ccc"]:
        values = np.asarray([float(row[f"{prefix}_{key}"]) for row in rows], dtype=np.float64)
        out[f"{prefix}_{key}_mean"] = float(values.mean())
        out[f"{prefix}_{key}_std"] = float(values.std(ddof=0))
        out[f"{prefix}_{key}_min"] = float(values.min())
        out[f"{prefix}_{key}_max"] = float(values.max())
    return out


def read_song_target_means(index: pd.DataFrame, label_dir: Path) -> pd.DataFrame:
    rows = []
    for row in index.to_dict("records"):
        sid = int(row["musicId"])
        y = np.load(label_dir / f"{sid}.npz")["Y"].astype(np.float32)[:, 0]
        rows.append({"musicId": sid, "frames": int(len(y)), "target_mean": float(np.mean(y))})
    return pd.DataFrame(rows)


def stratify_bins(values: pd.Series, bins: int = 8) -> np.ndarray:
    ranked = values.rank(method="first")
    q = min(bins, max(2, int(len(values) // 20)))
    return pd.qcut(ranked, q=q, labels=False, duplicates="drop").astype(int).to_numpy()


def make_split_index(song_stats: pd.DataFrame, seed: int) -> pd.DataFrame:
    y_bins = stratify_bins(song_stats["target_mean"])
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.10, random_state=seed)
    train_val_idx, test_idx = next(splitter.split(song_stats, y_bins))
    train_val = song_stats.iloc[train_val_idx].reset_index(drop=True)
    y_bins_tv = stratify_bins(train_val["target_mean"])
    val_splitter = StratifiedShuffleSplit(n_splits=1, test_size=1.0 / 9.0, random_state=seed + 1000)
    train_idx_rel, val_idx_rel = next(val_splitter.split(train_val, y_bins_tv))
    out = song_stats.loc[:, ["musicId", "frames"]].copy()
    out["split"] = "train"
    out.loc[out["musicId"].isin(train_val.iloc[val_idx_rel]["musicId"]), "split"] = "val"
    out.loc[out["musicId"].isin(song_stats.iloc[test_idx]["musicId"]), "split"] = "test"
    out["path"] = ""
    return out.loc[:, ["musicId", "split", "frames", "path"]].sort_values("musicId").reset_index(drop=True)


def build_mert_norm(index: pd.DataFrame, mert_dir: Path, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    norm_path = out_dir / "mert_norm.npz"
    summary_path = out_dir / "mert_norm_summary.json"
    xs = []
    for row in index[index["split"] == "train"].to_dict("records"):
        sid = int(row["musicId"])
        xs.append(np.load(mert_dir / f"{sid}.npz")["embedding"].astype(np.float32))
    concat = np.concatenate(xs, axis=0)
    mean = concat.mean(axis=0).astype(np.float32)
    std = (concat.std(axis=0) + 1e-8).astype(np.float32)
    np.savez(norm_path, mean=mean, std=std)
    summary = {"feature_dim": int(concat.shape[1]), "train_frames": int(concat.shape[0]), "path": str(norm_path)}
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def selection_score(m: dict[str, float]) -> float:
    return m["rmse"] + 0.05 * max(0.0, 0.80 - m["ccc"]) + 0.03 * max(0.0, 0.80 - m["pcc"]) + 0.01 * m["mae"]


def train_one_split(
    split_id: int,
    seed: int,
    index: pd.DataFrame,
    args: argparse.Namespace,
    device: torch.device,
    config: BestConfig,
) -> dict[str, Any]:
    set_seed(seed)
    split_dir = args.out_dir / f"split_{split_id:02d}_seed_{seed}"
    split_dir.mkdir(parents=True, exist_ok=True)
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
    model = FusionBiLSTM(int(sample_x.shape[1]), config.proj_dim, config.hidden_dim, config.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    ckpt = split_dir / "best.pt"
    history = []
    best_score = float("inf")
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
            loss = weighted_frame_mse(pred_valid, y_valid, config.extreme_weight)
            loss = loss + float(config.frame_ccc_weight) * ccc_loss(pred_valid, y_valid)
            loss = loss + clip_mean_loss(pred, y, mask, config.clip_mse_weight)
            loss = loss + clip_std_loss(pred, y, mask, config.clip_std_weight)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        val_eval = evaluate(model, loaders["val"], device)
        val_m = val_eval["metrics"]
        score = selection_score(val_m)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            **{f"val_{k}": v for k, v in val_m.items()},
            "val_select_score": score,
        }
        history.append(row)
        print(
            f"split={split_id:02d} seed={seed} epoch={epoch:03d} train_loss={row['train_loss']:.5f} "
            f"val_rmse={val_m['rmse']:.5f} val_pcc={val_m['pcc']:.3f} val_ccc={val_m['ccc']:.3f}",
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
    split_evals = {split: evaluate(model, loaders[split], device) for split in ["train", "val", "test"]}
    for split, data in split_evals.items():
        pd.DataFrame(
            {
                "musicId": data["song_id"],
                "time_sec": data["time_sec"],
                "target_valence": data["target"],
                "pred_valence": data["pred"],
            }
        ).to_csv(split_dir / f"{split}_predictions.csv", index=False)
    summary = {
        "split_id": split_id,
        "seed": seed,
        "best_epoch": best_epoch,
        "config": asdict(config),
        "split_song_counts": {split: int((index["split"] == split).sum()) for split in ["train", "val", "test"]},
        "split_frame_counts": {split: int(index[index["split"] == split]["frames"].sum()) for split in ["train", "val", "test"]},
        **{f"{split}_{k}": v for split, e in split_evals.items() for k, v in e["metrics"].items()},
        "test_gap_rows": threshold_rows(split_evals["test"]["metrics"]),
    }
    (split_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def summarize(rows: list[dict[str, Any]], out_dir: Path) -> dict[str, Any]:
    comparison = pd.DataFrame(rows)
    comparison.to_csv(out_dir / "comparison.csv", index=False)
    aggregate = {
        "created_at": "2026-06-27",
        "selection_rule": "each split selects checkpoint by validation selection_score; test is held out per split",
        "historical_single_split_best": HISTORICAL_SINGLE_SPLIT_BEST,
        "num_splits": len(rows),
        "aggregate": {
            **aggregate_metrics(rows, "val"),
            **aggregate_metrics(rows, "test"),
        },
        "per_split": rows,
    }
    test_mean = {k: aggregate["aggregate"][f"test_{k}_mean"] for k in ["rmse", "mae", "pcc", "ccc"]}
    aggregate["mean_test_gap_rows"] = threshold_rows(test_mean)
    aggregate["mean_test_delta_vs_historical_single_split"] = {
        "rmse": float(test_mean["rmse"] - HISTORICAL_SINGLE_SPLIT_BEST["rmse"]),
        "mae": float(test_mean["mae"] - HISTORICAL_SINGLE_SPLIT_BEST["mae"]),
        "pcc": float(test_mean["pcc"] - HISTORICAL_SINGLE_SPLIT_BEST["pcc"]),
        "ccc": float(test_mean["ccc"] - HISTORICAL_SINGLE_SPLIT_BEST["ccc"]),
    }
    (out_dir / "summary.json").write_text(json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(aggregate, ensure_ascii=False, indent=2), flush=True)
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--label-dir", type=Path, default=DEFAULT_LABEL_DIR)
    parser.add_argument("--mert-dir", type=Path, default=DEFAULT_MERT_DIR)
    parser.add_argument("--librosa-dataset", type=Path, default=DEFAULT_LIBROSA_DATASET)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seeds", nargs="*", type=int, default=[101, 202, 303])
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.aggregate_only:
        rows = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(args.out_dir.glob("split_*/summary.json"))]
        summarize(rows, args.out_dir)
        return
    base_index = pd.read_csv(args.index)
    song_stats = read_song_target_means(base_index, args.label_dir)
    song_stats.to_csv(args.out_dir / "song_target_means.csv", index=False)
    device = resolve_device(args.device)
    config = BestConfig()
    rows = []
    for split_id, seed in enumerate(args.seeds, start=1):
        split_index = make_split_index(song_stats, seed)
        rows.append(train_one_split(split_id, seed, split_index, args, device, config))
    summarize(rows, args.out_dir)


if __name__ == "__main__":
    main()
