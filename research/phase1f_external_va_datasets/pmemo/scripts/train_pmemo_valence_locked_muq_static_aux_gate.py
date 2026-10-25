#!/usr/bin/env python3
"""One-fold MuQ weak-static-aux gate under the final-locked dev protocol."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from train_pmemo_mert_harmonic_valence_fusion import (
    DEFAULT_LABEL_DIR, DEFAULT_LIBROSA_DATASET, FusionDataset,
    build_harmonic_norm, collate, selected_indices,
)
from train_pmemo_mert_hidden6_harmonic_extreme_weighted import (
    clip_mean_loss, weighted_frame_mse,
)
from train_pmemo_mert_valence_baseline import ROOT, ccc_loss, resolve_device, threshold_rows
from train_pmemo_valence_multisplit_best import build_mert_norm
from train_pmemo_valence_multisplit_static_aux import (
    STATIC_ANNOTATIONS, StaticAuxBiLSTM, StaticAuxConfig,
    batch_static_targets, evaluate, load_static_valence,
)
from train_pmemo_valence_dev_only_trials import postprocess_search, selection_score


DEFAULT_INDEX = ROOT / "data/splits/pmemo_final_locked_20260713/devfolds_3fold_20260713/fold_03/sequence_index.csv"
DEFAULT_FEATURES = ROOT / "outputs/muq_pmemo_locked_development/layer_06/features"
DEFAULT_OUT = ROOT / "outputs/valence_locked_muq_static_aux_fold03_sw0005_20260714"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--label-dir", type=Path, default=DEFAULT_LABEL_DIR)
    parser.add_argument("--librosa-dataset", type=Path, default=DEFAULT_LIBROSA_DATASET)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--static-weight", type=float, default=0.0005)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    index = pd.read_csv(args.index)
    index = index[index["split"].isin(["train", "val"])].copy()
    index.to_csv(args.out_dir / "sequence_index.csv", index=False)
    build_mert_norm(index, args.features, args.out_dir / "dataset")
    build_harmonic_norm(index, args.librosa_dataset, selected_indices(["harmonic"]), args.out_dir / "dataset", force=True)
    feature_norm = np.load(args.out_dir / "dataset/mert_norm.npz")
    harmonic_norm = np.load(args.out_dir / "dataset/harmonic_norm.npz")
    config = StaticAuxConfig(
        trial_id="muq_h6_h96_static_aux_sw0005", proj_dim=384, hidden_dim=96,
        dropout=0.35, learning_rate=2.5e-4, weight_decay=4e-4,
        batch_size=18, frame_ccc_weight=0.04, clip_mse_weight=0.08,
        extreme_weight=1.15, static_head_weight=args.static_weight,
        static_frame_consistency_weight=0.0, blend_static_weight=0.0,
        patience=10, epochs=80,
    )
    loaders = {
        split: DataLoader(
            FusionDataset(
                index, split, args.features, args.label_dir, args.librosa_dataset,
                feature_norm["mean"], feature_norm["std"],
                harmonic_norm["mean"], harmonic_norm["std"], harmonic_norm["feature_idx"],
            ),
            batch_size=config.batch_size, shuffle=(split == "train"), collate_fn=collate,
        )
        for split in ["train", "val"]
    }
    sample_x, _, _, _ = loaders["train"].dataset[0]
    device = resolve_device(args.device)
    model = StaticAuxBiLSTM(int(sample_x.shape[1]), config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    static_map = load_static_valence(STATIC_ANNOTATIONS)
    checkpoint = args.out_dir / "best.pt"
    best_score, best_epoch, bad = float("inf"), 0, 0
    history = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        losses = []
        for x, y, song, _, mask in loaders["train"]:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            static_target = batch_static_targets(song, static_map, device)
            optimizer.zero_grad()
            frame_pred, static_pred = model(x, mask)
            loss = weighted_frame_mse(frame_pred[mask], y[mask], config.extreme_weight)
            loss = loss + config.frame_ccc_weight * ccc_loss(frame_pred[mask], y[mask])
            loss = loss + clip_mean_loss(frame_pred, y, mask, config.clip_mse_weight)
            loss = loss + config.static_head_weight * nn.functional.mse_loss(static_pred, static_target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        val_eval = evaluate(model, loaders["val"], device, static_map, config)
        score = selection_score(val_eval["metrics"])
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), **val_eval["metrics"], "score": score}
        history.append(row)
        print(f"epoch={epoch:03d} val_rmse={row['rmse']:.6f} val_pcc={row['pcc']:.4f} val_ccc={row['ccc']:.4f}", flush=True)
        if score < best_score:
            best_score, best_epoch, bad = score, epoch, 0
            torch.save(model.state_dict(), checkpoint)
        else:
            bad += 1
            if bad >= config.patience:
                break

    pd.DataFrame(history).to_csv(args.out_dir / "history.csv", index=False)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    val_eval = evaluate(model, loaders["val"], device, static_map, config)
    best_post, post_rows = postprocess_search(val_eval)
    post_rows.to_csv(args.out_dir / "val_postprocess_search.csv", index=False)
    pd.DataFrame({
        "musicId": val_eval["song_id"], "time_sec": val_eval["time_sec"],
        "target_valence": val_eval["target"], "pred_valence": val_eval["pred"],
    }).to_csv(args.out_dir / "val_predictions.csv", index=False)
    summary = {
        "created_at": "2026-07-14", "best_epoch": best_epoch,
        "selection_boundary": "fold_03 train/val only; final locked test excluded",
        "config": asdict(config), "raw": val_eval["metrics"],
        "post": best_post, "post_gap_rows": threshold_rows(best_post["metrics"]),
        "reference_muq_fold03_post": {"rmse": 0.210439, "mae": 0.160516, "pcc": 0.781060, "ccc": 0.757806},
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
