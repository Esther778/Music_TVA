#!/usr/bin/env python3
"""Valence-only development trials that do not read the sealed blind test set."""

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
from torch.utils.data import DataLoader

from search_pmemo_valence_dev_val_postprocess import fit_calibration, moving_average_by_song
from train_pmemo_mert_harmonic_valence_fusion import (
    FusionBiLSTM,
    FusionDataset,
    build_harmonic_norm,
    collate,
    evaluate,
    selected_indices,
)
from train_pmemo_mert_valence_baseline import ROOT, ccc_loss, metrics, resolve_device, threshold_rows
from train_pmemo_valence_multisplit_best import build_mert_norm
from train_pmemo_mert_hidden6_harmonic_extreme_weighted import (
    clip_mean_loss,
    clip_std_loss,
    weighted_frame_mse,
)


DEFAULT_INDEX = ROOT / "data" / "splits" / "pmemo_blind_20260709" / "pmemo_blind_trainvaltest_index.csv"
DEFAULT_MERT_DIR = ROOT / "outputs" / "mert_pmemo_hidden6" / "features"
DEFAULT_LABEL_DIR = ROOT / "outputs" / "opensmile_baseline" / "dataset" / "sequence"
DEFAULT_LIBROSA_DATASET = ROOT / "outputs" / "arousal_fullsong_librosa" / "dataset"
DEFAULT_OUT = ROOT / "outputs" / "valence_dev_only_trials_20260709"


@dataclass
class TrialConfig:
    trial_id: str
    proj_dim: int
    hidden_dim: int
    dropout: float
    learning_rate: float
    weight_decay: float
    batch_size: int
    frame_ccc_weight: float
    clip_mse_weight: float
    extreme_weight: float
    clip_std_weight: float = 0.0
    patience: int = 10
    epochs: int = 80


TRIALS = [
    TrialConfig("v_dev_best_like_h80_do030_lr4e4", 320, 80, 0.30, 4e-4, 2e-4, 20, 0.05, 0.10, 1.25),
    TrialConfig("v_dev_conservative_h80_do035_lr25e4", 320, 80, 0.35, 2.5e-4, 4e-4, 20, 0.04, 0.08, 1.15),
    TrialConfig("v_dev_wider_h96_do035_lr25e4", 384, 96, 0.35, 2.5e-4, 4e-4, 18, 0.04, 0.08, 1.15),
    TrialConfig("v_dev_lowccc_h96_do030_lr3e4", 384, 96, 0.30, 3e-4, 3e-4, 18, 0.02, 0.06, 1.10),
    TrialConfig("v_dev_small_h64_do045_lr2e4_wd1e3", 256, 64, 0.45, 2e-4, 1e-3, 24, 0.04, 0.08, 1.00),
    TrialConfig("v_dev_tiny_h48_do040_lr25e4_wd8e4", 192, 48, 0.40, 2.5e-4, 8e-4, 28, 0.03, 0.06, 0.90),
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def selection_score(m: dict[str, float]) -> float:
    return (
        m["rmse"]
        + 0.015 * max(0.0, m["mae"] - 0.15)
        + 0.03 * max(0.0, 0.80 - m["ccc"])
        + 0.02 * max(0.0, 0.80 - m["pcc"])
    )


def postprocess_search(val_eval: dict[str, Any]) -> tuple[dict[str, Any], pd.DataFrame]:
    pred = val_eval["pred"]
    target = val_eval["target"]
    song = val_eval["song_id"]
    rows = []
    best: dict[str, Any] = {"postprocess": "raw", "metrics": val_eval["metrics"], "score": selection_score(val_eval["metrics"])}
    for window in [1, 3, 5, 7, 9, 11, 15, 21, 31, 41]:
        smoothed = moving_average_by_song(pred, song, window)
        for smooth_blend in [0.25, 0.50, 0.75, 1.0]:
            base = (1.0 - smooth_blend) * pred + smooth_blend * smoothed
            for method in ["identity", "bias", "linear", "mean_std"]:
                calibrated, params = fit_calibration(base, target, method)
                for cal_blend in [0.25, 0.50, 0.75, 1.0]:
                    pp = (1.0 - cal_blend) * base + cal_blend * calibrated
                    m = metrics(pp, target)
                    row = {
                        "postprocess": f"w{window}_sb{smooth_blend:g}_{method}_cb{cal_blend:g}",
                        "window": window,
                        "smooth_blend": smooth_blend,
                        "method": method,
                        "cal_blend": cal_blend,
                        "params": json.dumps(params, ensure_ascii=False),
                        **m,
                    }
                    row["score"] = selection_score(m)
                    rows.append(row)
                    if row["score"] < best["score"]:
                        best = {"postprocess": row["postprocess"], "metrics": m, "score": row["score"], **row}
    return best, pd.DataFrame(rows)


def build_loaders(
    index: pd.DataFrame,
    args: argparse.Namespace,
    trial_dir: Path,
    config: TrialConfig,
) -> dict[str, DataLoader]:
    build_mert_norm(index, args.mert_dir, trial_dir / "dataset")
    build_harmonic_norm(index, args.librosa_dataset, selected_indices(["harmonic"]), trial_dir / "dataset", force=True)
    mert_norm = np.load(trial_dir / "dataset" / "mert_norm.npz")
    harm_norm = np.load(trial_dir / "dataset" / "harmonic_norm.npz")
    return {
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
        for split in ["train", "val"]
    }


def run_trial(config: TrialConfig, index: pd.DataFrame, args: argparse.Namespace, device: torch.device, seed: int) -> dict[str, Any]:
    set_seed(seed)
    trial_dir = args.out_dir / config.trial_id
    trial_dir.mkdir(parents=True, exist_ok=True)
    index.to_csv(trial_dir / "sequence_index_train_val_only.csv", index=False)
    loaders = build_loaders(index, args, trial_dir, config)
    sample_x, _, _, _ = loaders["train"].dataset[0]
    model = FusionBiLSTM(int(sample_x.shape[1]), config.proj_dim, config.hidden_dim, config.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    ckpt = trial_dir / "best.pt"
    best_score = float("inf")
    best_epoch = 0
    bad = 0
    history = []
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
            loss = loss + config.frame_ccc_weight * ccc_loss(pred_valid, y_valid)
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
            f"{config.trial_id} epoch={epoch:03d} train_loss={row['train_loss']:.5f} "
            f"val_rmse={val_m['rmse']:.5f} val_mae={val_m['mae']:.5f} "
            f"val_pcc={val_m['pcc']:.3f} val_ccc={val_m['ccc']:.3f}",
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

    pd.DataFrame(history).to_csv(trial_dir / "history.csv", index=False)
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
        ).to_csv(trial_dir / f"{split}_predictions.csv", index=False)
    best_pp, pp_rows = postprocess_search(evals["val"])
    pp_rows.to_csv(trial_dir / "val_postprocess_search.csv", index=False)
    summary = {
        "created_at": "2026-07-09",
        "purpose": "Valence train/val optimization without reading sealed blind_test",
        "trial_id": config.trial_id,
        "seed": seed,
        "best_epoch": best_epoch,
        "config": asdict(config),
        "split_song_counts": {split: int((index["split"] == split).sum()) for split in ["train", "val"]},
        "split_frame_counts": {split: int(index[index["split"] == split]["frames"].sum()) for split in ["train", "val"]},
        **{f"{split}_{k}": v for split, e in evals.items() for k, v in e["metrics"].items()},
        "best_val_postprocess": best_pp,
        "val_gap_rows_raw": threshold_rows(evals["val"]["metrics"]),
        "val_gap_rows_best_postprocess": threshold_rows(best_pp["metrics"]),
        "selection_boundary": "Development only. Sealed blind_test is not read.",
    }
    (trial_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--mert-dir", type=Path, default=DEFAULT_MERT_DIR)
    parser.add_argument("--label-dir", type=Path, default=DEFAULT_LABEL_DIR)
    parser.add_argument("--librosa-dataset", type=Path, default=DEFAULT_LIBROSA_DATASET)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--trial", default="all")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    index = pd.read_csv(args.index)
    index = index[index["split"].isin(["train", "val"])].copy()
    if set(index["split"].unique()) != {"train", "val"}:
        raise ValueError("Expected train/val rows after filtering out blind_test.")
    index.to_csv(args.out_dir / "dev_only_sequence_index.csv", index=False)
    trials = TRIALS if args.trial == "all" else [trial for trial in TRIALS if trial.trial_id == args.trial]
    if not trials:
        raise ValueError(f"Unknown trial: {args.trial}")
    device = resolve_device(args.device)
    rows = [run_trial(trial, index, args, device, args.seed) for trial in trials]
    comparison = []
    for row in rows:
        pp = row["best_val_postprocess"]
        ppm = pp["metrics"]
        comparison.append(
            {
                "trial_id": row["trial_id"],
                "best_epoch": row["best_epoch"],
                "val_rmse": row["val_rmse"],
                "val_mae": row["val_mae"],
                "val_pcc": row["val_pcc"],
                "val_ccc": row["val_ccc"],
                "best_val_postprocess": pp["postprocess"],
                "post_val_rmse": ppm["rmse"],
                "post_val_mae": ppm["mae"],
                "post_val_pcc": ppm["pcc"],
                "post_val_ccc": ppm["ccc"],
            }
        )
    pd.DataFrame(comparison).to_csv(args.out_dir / "comparison.csv", index=False)
    summary = {
        "created_at": "2026-07-09",
        "purpose": "Valence train/val optimization without reading sealed blind_test",
        "selection_boundary": "Development only. Do not claim blind/test performance.",
        "comparison": comparison,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
