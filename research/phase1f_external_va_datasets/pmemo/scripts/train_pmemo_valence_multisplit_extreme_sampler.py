#!/usr/bin/env python3
"""Extreme-song-aware PMEmo Valence BiLSTM trials.

The split-02 audit showed that many failures come from strongly positive or
negative songs being predicted as near-neutral. This script keeps the current
deployable feature/model family and only changes the training distribution:
training songs with more extreme song-level Valence means are sampled more
often. Validation and test splits are unchanged.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from train_pmemo_mert_harmonic_valence_fusion import (
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
from train_pmemo_mert_valence_baseline import ROOT, ccc_loss, resolve_device, threshold_rows
from train_pmemo_valence_multisplit_best import (
    DEFAULT_MERT_DIR,
    BestConfig,
    aggregate_metrics,
    build_mert_norm,
    selection_score,
    set_seed,
)


BASE_SPLIT_DIR = ROOT / "outputs" / "valence_multisplit_best_20260627"
DEFAULT_OUT = ROOT / "outputs" / "valence_multisplit_extreme_sampler_20260628"

CURRENT_BEST = {
    "model_id": "pmemo_valence_frame_calibrator_song_bias_20260628",
    "rmse": 0.19748015854593182,
    "mae": 0.15352887995061668,
    "pcc": 0.7932350928356282,
    "ccc": 0.7800520380365018,
}


@dataclass
class ExtremeSamplerConfig(BestConfig):
    song_extreme_threshold: float = 0.35
    song_sample_strength: float = 2.0
    song_sample_cap: float = 3.0


TRIALS = [
    ExtremeSamplerConfig(
        trial_id="bilstm_extsong_sampler_h80_s05_clip010_ccc005",
        proj_dim=320,
        hidden_dim=80,
        dropout=0.30,
        learning_rate=4e-4,
        weight_decay=2e-4,
        batch_size=20,
        frame_ccc_weight=0.05,
        clip_mse_weight=0.10,
        extreme_weight=1.25,
        clip_std_weight=0.0,
        patience=8,
        epochs=60,
        song_extreme_threshold=0.45,
        song_sample_strength=0.5,
        song_sample_cap=1.5,
    ),
    ExtremeSamplerConfig(
        trial_id="bilstm_extsong_sampler_h80_s2_clip010_ccc005",
        proj_dim=320,
        hidden_dim=80,
        dropout=0.30,
        learning_rate=4e-4,
        weight_decay=2e-4,
        batch_size=20,
        frame_ccc_weight=0.05,
        clip_mse_weight=0.10,
        extreme_weight=1.25,
        clip_std_weight=0.0,
        patience=8,
        epochs=60,
        song_extreme_threshold=0.35,
        song_sample_strength=2.0,
        song_sample_cap=3.0,
    ),
    ExtremeSamplerConfig(
        trial_id="bilstm_extsong_sampler_h80_s3_clip015_ccc008",
        proj_dim=320,
        hidden_dim=80,
        dropout=0.35,
        learning_rate=3e-4,
        weight_decay=3e-4,
        batch_size=20,
        frame_ccc_weight=0.08,
        clip_mse_weight=0.15,
        extreme_weight=1.35,
        clip_std_weight=0.01,
        patience=8,
        epochs=60,
        song_extreme_threshold=0.30,
        song_sample_strength=3.0,
        song_sample_cap=4.0,
    ),
]


def split_seed(split_dir: Path) -> tuple[int, int]:
    parts = split_dir.name.split("_")
    return int(parts[1]), int(parts[3])


def train_song_mean_map(index: pd.DataFrame, label_dir: Path) -> dict[int, float]:
    means = {}
    for row in index[index["split"] == "train"].to_dict("records"):
        sid = int(row["musicId"])
        y = np.load(label_dir / f"{sid}.npz")["Y"].astype(np.float32)[:, 0]
        means[sid] = float(np.mean(y))
    return means


def make_extreme_sampler(dataset: FusionDataset, song_means: dict[int, float], config: ExtremeSamplerConfig) -> WeightedRandomSampler:
    weights = []
    for row in dataset.rows:
        sid = int(row["musicId"])
        extremity = abs(float(song_means[sid]))
        scaled = min(extremity / max(config.song_extreme_threshold, 1e-6), config.song_sample_cap)
        weights.append(1.0 + config.song_sample_strength * scaled)
    weights_t = torch.as_tensor(weights, dtype=torch.double)
    return WeightedRandomSampler(weights_t, num_samples=len(weights), replacement=True)


def train_one_split_extreme_sampler(
    split_id: int,
    seed: int,
    index: pd.DataFrame,
    args: argparse.Namespace,
    device: torch.device,
    config: ExtremeSamplerConfig,
) -> dict[str, Any]:
    set_seed(seed)
    split_dir = args.out_dir / config.trial_id / f"split_{split_id:02d}_seed_{seed}"
    split_dir.mkdir(parents=True, exist_ok=True)
    index.to_csv(split_dir / "sequence_index.csv", index=False)
    build_mert_norm(index, args.mert_dir, split_dir / "dataset")
    build_harmonic_norm(index, args.librosa_dataset, selected_indices(["harmonic"]), split_dir / "dataset", force=True)
    mert_norm = np.load(split_dir / "dataset" / "mert_norm.npz")
    harm_norm = np.load(split_dir / "dataset" / "harmonic_norm.npz")

    datasets = {
        split: FusionDataset(
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
        )
        for split in ["train", "val", "test"]
    }
    song_means = train_song_mean_map(index, args.label_dir)
    sampler = make_extreme_sampler(datasets["train"], song_means, config)
    loaders = {
        "train": DataLoader(datasets["train"], batch_size=config.batch_size, sampler=sampler, collate_fn=collate),
        "val": DataLoader(datasets["val"], batch_size=config.batch_size, shuffle=False, collate_fn=collate),
        "test": DataLoader(datasets["test"], batch_size=config.batch_size, shuffle=False, collate_fn=collate),
    }

    sample_x, _, _, _ = datasets["train"][0]
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
            f"{config.trial_id} split={split_id:02d} epoch={epoch:03d} "
            f"train_loss={row['train_loss']:.5f} val_rmse={val_m['rmse']:.5f} "
            f"val_pcc={val_m['pcc']:.3f} val_ccc={val_m['ccc']:.3f}",
            flush=True,
        )
        if score < best_score:
            best_score = score
            best_epoch = epoch
            bad = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "seed": seed,
                    "config": asdict(config),
                    "input_dim": int(sample_x.shape[1]),
                    "song_sample_strategy": "extreme_song_weighted_sampler",
                },
                ckpt,
            )
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


def summarize_trial(rows: list[dict[str, Any]], out_dir: Path, config: ExtremeSamplerConfig) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / "comparison.csv", index=False)
    aggregate = {
        "created_at": "2026-06-28",
        "model_family": "MERT hidden_6 + Librosa harmonic Valence-only BiLSTM with extreme-song sampler",
        "fixed_split_source": str(BASE_SPLIT_DIR),
        "config": asdict(config),
        "current_best_reference": CURRENT_BEST,
        "num_splits": len(rows),
        "aggregate": {
            **aggregate_metrics(rows, "val"),
            **aggregate_metrics(rows, "test"),
        },
        "per_split": rows,
    }
    test_mean = {k: aggregate["aggregate"][f"test_{k}_mean"] for k in ["rmse", "mae", "pcc", "ccc"]}
    aggregate["mean_test_gap_rows"] = threshold_rows(test_mean)
    aggregate["mean_test_delta_vs_current_best"] = {
        "rmse": float(test_mean["rmse"] - CURRENT_BEST["rmse"]),
        "mae": float(test_mean["mae"] - CURRENT_BEST["mae"]),
        "pcc": float(test_mean["pcc"] - CURRENT_BEST["pcc"]),
        "ccc": float(test_mean["ccc"] - CURRENT_BEST["ccc"]),
    }
    (out_dir / "summary.json").write_text(json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(aggregate, ensure_ascii=False, indent=2), flush=True)
    return aggregate


def summarize_all(summaries: list[dict[str, Any]], out_dir: Path) -> None:
    rows = []
    for s in summaries:
        row = {
            "trial_id": s["config"]["trial_id"],
            **{f"test_{m}_mean": s["aggregate"][f"test_{m}_mean"] for m in ["rmse", "mae", "pcc", "ccc"]},
            **{f"test_{m}_std": s["aggregate"][f"test_{m}_std"] for m in ["rmse", "mae", "pcc", "ccc"]},
            **{f"delta_vs_current_best_{m}": s["mean_test_delta_vs_current_best"][m] for m in ["rmse", "mae", "pcc", "ccc"]},
        }
        rows.append(row)
    pd.DataFrame(rows).sort_values(["test_rmse_mean", "test_ccc_mean"], ascending=[True, False]).to_csv(
        out_dir / "trial_comparison.csv", index=False
    )
    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "created_at": "2026-06-28",
                "current_best_reference": CURRENT_BEST,
                "trial_count": len(summaries),
                "trials": rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mert-dir", type=Path, default=DEFAULT_MERT_DIR)
    parser.add_argument("--label-dir", type=Path, default=DEFAULT_LABEL_DIR)
    parser.add_argument("--librosa-dataset", type=Path, default=DEFAULT_LIBROSA_DATASET)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--trial-id", choices=[t.trial_id for t in TRIALS], default=None)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    configs = [cfg for cfg in TRIALS if args.trial_id in {None, cfg.trial_id}]
    summaries = []
    for config in configs:
        trial_out = args.out_dir / config.trial_id
        train_args = SimpleNamespace(
            out_dir=args.out_dir,
            mert_dir=args.mert_dir,
            label_dir=args.label_dir,
            librosa_dataset=args.librosa_dataset,
        )
        rows = []
        for split_dir in sorted(BASE_SPLIT_DIR.glob("split_*")):
            split_id, seed = split_seed(split_dir)
            index = pd.read_csv(split_dir / "sequence_index.csv")
            rows.append(train_one_split_extreme_sampler(split_id, seed, index, train_args, device, config))
        summaries.append(summarize_trial(rows, trial_out, config))
    summarize_all(summaries, args.out_dir)


if __name__ == "__main__":
    main()
