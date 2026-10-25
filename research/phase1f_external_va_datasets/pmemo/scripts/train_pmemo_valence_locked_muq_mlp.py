#!/usr/bin/env python3
"""Train/val-only MuQ + harmonic frame MLP screen for Valence."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from train_pmemo_mert_valence_baseline import FrameMLP, ROOT, ccc_loss, resolve_device
from train_pmemo_mert_hidden6_harmonic_extreme_weighted import clip_mean_loss, weighted_frame_mse
from train_pmemo_valence_dev_only_trials import (
    build_loaders,
    postprocess_search,
    selection_score,
    set_seed,
)
from train_pmemo_mert_harmonic_valence_fusion import evaluate


DEFAULT_INDEX = ROOT / "data/splits/pmemo_final_locked_20260713/devfolds_3fold_20260713/fold_03/sequence_index.csv"
DEFAULT_FEATURES = ROOT / "outputs/muq_pmemo_locked_development/layer_06/features"
DEFAULT_OUT = ROOT / "outputs/valence_locked_muq_layer06_mlp_fold03_20260713"
DEFAULT_LABELS = ROOT / "outputs/opensmile_baseline/dataset/sequence"
DEFAULT_LIBROSA = ROOT / "outputs/arousal_fullsong_librosa/dataset"


@dataclass
class Config:
    trial_id: str
    hidden_dim: int
    dropout: float
    learning_rate: float
    weight_decay: float
    batch_size: int
    frame_ccc_weight: float = 0.04
    clip_mse_weight: float = 0.08
    extreme_weight: float = 1.15
    patience: int = 10
    epochs: int = 80


TRIALS = [
    Config("muq_mlp_h256_do030_lr5e4", 256, 0.30, 5e-4, 2e-4, 32),
    Config("muq_mlp_h512_do035_lr3e4", 512, 0.35, 3e-4, 4e-4, 24),
]


def run(config: Config, index: pd.DataFrame, args: argparse.Namespace, device: torch.device) -> dict:
    set_seed(args.seed)
    trial_dir = args.out_dir / config.trial_id
    trial_dir.mkdir(parents=True, exist_ok=True)
    loaders = build_loaders(index, args, trial_dir, config)
    sample_x, _, _, _ = loaders["train"].dataset[0]
    model = FrameMLP(int(sample_x.shape[1]), config.hidden_dim, config.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    checkpoint = trial_dir / "best.pt"
    best_score, best_epoch, bad = float("inf"), 0, 0
    history = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        losses = []
        for x, y, _, _, mask in loaders["train"]:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            optimizer.zero_grad()
            pred = model(x, mask)
            loss = weighted_frame_mse(pred[mask], y[mask], config.extreme_weight)
            loss = loss + config.frame_ccc_weight * ccc_loss(pred[mask], y[mask])
            loss = loss + clip_mean_loss(pred, y, mask, config.clip_mse_weight)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        val = evaluate(model, loaders["val"], device)
        score = selection_score(val["metrics"])
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), **{f"val_{k}": v for k, v in val["metrics"].items()}, "score": score}
        history.append(row)
        print(f"{config.trial_id} epoch={epoch:03d} rmse={row['val_rmse']:.5f} mae={row['val_mae']:.5f} pcc={row['val_pcc']:.3f} ccc={row['val_ccc']:.3f}", flush=True)
        if score < best_score:
            best_score, best_epoch, bad = score, epoch, 0
            torch.save({"model_state": model.state_dict(), "config": config.__dict__, "input_dim": int(sample_x.shape[1])}, checkpoint)
        else:
            bad += 1
            if bad >= config.patience:
                break
    pd.DataFrame(history).to_csv(trial_dir / "history.csv", index=False)
    model.load_state_dict(torch.load(checkpoint, map_location=device)["model_state"])
    val = evaluate(model, loaders["val"], device)
    pd.DataFrame({"musicId": val["song_id"], "time_sec": val["time_sec"], "target_valence": val["target"], "pred_valence": val["pred"]}).to_csv(trial_dir / "val_predictions.csv", index=False)
    post, rows = postprocess_search(val)
    rows.to_csv(trial_dir / "val_postprocess_search.csv", index=False)
    summary = {"trial_id": config.trial_id, "best_epoch": best_epoch, "config": config.__dict__, "raw": val["metrics"], "post": post, "boundary": "locked development train/val only; final test excluded"}
    (trial_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--mert-dir", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--label-dir", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--librosa-dataset", type=Path, default=DEFAULT_LIBROSA)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--trial", default="all")
    args = parser.parse_args()
    index = pd.read_csv(args.index)
    index = index[index["split"].isin(["train", "val"])].copy()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    trials = TRIALS if args.trial == "all" else [config for config in TRIALS if config.trial_id == args.trial]
    if not trials:
        raise ValueError(f"Unknown trial: {args.trial}")
    results = [run(config, index, args, device) for config in trials]
    (args.out_dir / "comparison.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
