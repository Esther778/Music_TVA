#!/usr/bin/env python3
"""Arousal-only development trials that do not read the sealed blind test set."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from train_pmemo_opensmile_arousal_trials import (
    ArousalBiLSTM,
    ROOT,
    SequenceDataset,
    ccc_loss,
    collate,
    evaluate,
    metrics,
    moving_average_by_song,
    resolve_device,
    set_seed,
    smoothness_loss,
    threshold_rows,
)


DEFAULT_DATASET = ROOT / "outputs" / "opensmile_arousal_blind_holdout_20260709" / "dataset"
DEFAULT_OUT = ROOT / "outputs" / "opensmile_arousal_dev_only_trials_20260709"


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
    mse_weight: float = 1.0
    ccc_weight: float = 0.0
    smooth_weight: float = 0.0
    patience: int = 12
    epochs: int = 90


TRIALS = [
    TrialConfig("a_dev_h96_l1_do015_lr8e4_mse", 96, 1, 0.15, 64, 8e-4, 1e-4, 32),
    TrialConfig("a_dev_h112_l1_do015_lr7e4_mse", 112, 1, 0.15, 80, 7e-4, 1e-4, 28),
    TrialConfig("a_dev_h128_l1_do020_lr6e4_mse", 128, 1, 0.20, 96, 6e-4, 1.5e-4, 24),
    TrialConfig("a_dev_h128_l2_do025_lr6e4_mse", 128, 2, 0.25, 96, 6e-4, 1e-4, 24),
    TrialConfig("a_dev_h160_l2_do030_lr5e4_ccc010", 160, 2, 0.30, 128, 5e-4, 1e-4, 18, ccc_weight=0.10),
    TrialConfig("a_dev_h192_l2_do035_lr4e4_ccc012_smooth005", 192, 2, 0.35, 128, 4e-4, 2e-4, 16, ccc_weight=0.12, smooth_weight=0.005),
    TrialConfig("a_dev_h96_l1_do020_lr6e4_mse_ccc005", 96, 1, 0.20, 64, 6e-4, 1e-4, 32, ccc_weight=0.05),
    TrialConfig("a_dev_h112_l1_do020_lr6e4_mse_smooth001", 112, 1, 0.20, 80, 6e-4, 1e-4, 28, smooth_weight=0.01),
    TrialConfig("a_dev_h64_l1_do035_lr4e4_wd8e4_smooth003", 64, 1, 0.35, 48, 4e-4, 8e-4, 36, smooth_weight=0.03),
    TrialConfig("a_dev_h80_l1_do025_lr5e4_wd4e4_smooth002", 80, 1, 0.25, 56, 5e-4, 4e-4, 32, smooth_weight=0.02),
]


def selection_score(m: dict[str, float]) -> float:
    return m["rmse"] + 0.02 * max(0.0, m["mae"] - 0.15) + 0.01 * max(0.0, 0.80 - m["ccc"])


def postprocess_search(val_eval: dict[str, Any]) -> tuple[dict[str, Any], pd.DataFrame]:
    pred = val_eval["pred"]
    target = val_eval["target"]
    song = val_eval["song_id"]
    rows = []
    best = {"postprocess": "raw", "metrics": val_eval["metrics"], "score": selection_score(val_eval["metrics"])}
    for window in [1, 3, 5, 7, 9, 11, 15]:
        sm = moving_average_by_song(pred, song, window)
        for blend in [0.25, 0.50, 0.75, 1.0]:
            pp = (1.0 - blend) * pred + blend * sm
            m = metrics(pp, target)
            score = selection_score(m)
            row = {"postprocess": f"smooth_w{window}_b{blend:g}", "window": window, "blend": blend, **m, "score": score}
            rows.append(row)
            if score < best["score"]:
                best = {"postprocess": row["postprocess"], "metrics": m, "score": score, "window": window, "blend": blend}
    return best, pd.DataFrame(rows)


def run_trial(config: TrialConfig, data_dir: Path, out_dir: Path, device: torch.device, seed: int) -> dict[str, Any]:
    set_seed(seed)
    trial_dir = out_dir / config.trial_id
    trial_dir.mkdir(parents=True, exist_ok=True)
    index = pd.read_csv(data_dir / "sequence_index.csv")
    index = index[index["split"].isin(["train", "val"])].copy()
    loaders = {
        split: DataLoader(
            SequenceDataset(index, split),
            batch_size=config.batch_size,
            shuffle=(split == "train"),
            collate_fn=collate,
        )
        for split in ["train", "val"]
    }
    input_dim = int(np.load(index.iloc[0]["path"])["X"].shape[1])
    model = ArousalBiLSTM(input_dim, config.hidden_dim, config.layers, config.dropout, config.head_hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    ckpt = trial_dir / "best.pt"
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
        score = selection_score(val_m)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), **{f"val_{k}": v for k, v in val_m.items()}, "val_score": score}
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
            torch.save({"model_state": model.state_dict(), "input_dim": input_dim, "config": asdict(config), "seed": seed}, ckpt)
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
                "target_arousal": data["target"],
                "pred_arousal": data["pred"],
            }
        ).to_csv(trial_dir / f"{split}_predictions.csv", index=False)
    best_pp, pp_rows = postprocess_search(evals["val"])
    pp_rows.to_csv(trial_dir / "val_postprocess_search.csv", index=False)
    summary = {
        "created_at": "2026-07-09",
        "trial_id": config.trial_id,
        "purpose": "Arousal dev-only trial; sealed blind_test is not read",
        "best_epoch": best_epoch,
        "config": asdict(config),
        **{f"{split}_{k}": v for split, e in evals.items() for k, v in e["metrics"].items()},
        "best_val_postprocess": best_pp,
        "val_gap_rows_raw": threshold_rows(evals["val"]["metrics"]),
        "val_gap_rows_best_postprocess": threshold_rows(best_pp["metrics"]),
    }
    (trial_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--trial", default="all")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    trials = TRIALS if args.trial == "all" else [t for t in TRIALS if t.trial_id == args.trial]
    if not trials:
        raise ValueError(f"Unknown trial: {args.trial}")
    rows = [run_trial(trial, args.data_dir, args.out_dir, device, args.seed) for trial in trials]
    comparison = []
    for row in rows:
        pp = row["best_val_postprocess"]
        comparison.append(
            {
                "trial_id": row["trial_id"],
                "best_epoch": row["best_epoch"],
                "val_rmse": row["val_rmse"],
                "val_mae": row["val_mae"],
                "val_pcc": row["val_pcc"],
                "val_ccc": row["val_ccc"],
                "best_val_postprocess": pp["postprocess"],
                "post_val_rmse": pp["metrics"]["rmse"],
                "post_val_mae": pp["metrics"]["mae"],
                "post_val_pcc": pp["metrics"]["pcc"],
                "post_val_ccc": pp["metrics"]["ccc"],
            }
        )
    pd.DataFrame(comparison).to_csv(args.out_dir / "comparison.csv", index=False)
    summary = {
        "created_at": "2026-07-09",
        "purpose": "Arousal train/val optimization without reading sealed blind_test",
        "selection_boundary": "Use this output for development only. Do not claim blind performance.",
        "comparison": comparison,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
