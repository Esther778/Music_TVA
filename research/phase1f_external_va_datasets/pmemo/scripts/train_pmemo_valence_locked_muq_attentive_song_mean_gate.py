#!/usr/bin/env python3
"""Gate an attentive MuQ song-level Valence model on locked development data."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from train_pmemo_mert_harmonic_valence_fusion import (
    FusionDataset, build_harmonic_norm, collate, selected_indices,
)
from train_pmemo_mert_valence_baseline import ROOT, ccc_loss, metrics, resolve_device, threshold_rows
from train_pmemo_valence_multisplit_best import build_mert_norm


DEFAULT_INDEX = ROOT / "data/splits/pmemo_final_locked_20260713/devfolds_3fold_20260713/fold_03/sequence_index.csv"
DEFAULT_FEATURES = ROOT / "outputs/muq_pmemo_locked_development/layer_06/features"
DEFAULT_LABELS = ROOT / "outputs/opensmile_baseline/dataset/sequence"
DEFAULT_LIBROSA = ROOT / "outputs/arousal_fullsong_librosa/dataset"
DEFAULT_BASE = ROOT / "outputs/valence_locked_muq_arousal_stacker_fine_20260714/fold_03_val_predictions.csv"
DEFAULT_OUT = ROOT / "outputs/valence_locked_muq_attentive_song_mean_fold03_20260714"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class AttentiveSongMean(nn.Module):
    def __init__(self, input_dim: int, proj_dim: int = 192, hidden_dim: int = 64, dropout: float = 0.35):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(input_dim), nn.Linear(input_dim, proj_dim),
            nn.GELU(), nn.Dropout(dropout),
        )
        self.gru = nn.GRU(proj_dim, hidden_dim, batch_first=True, bidirectional=True)
        width = hidden_dim * 2
        self.attn = nn.Sequential(nn.Linear(width, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))
        self.head = nn.Sequential(
            nn.LayerNorm(width * 3), nn.Dropout(dropout),
            nn.Linear(width * 3, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        z, _ = self.gru(self.proj(x))
        score = self.attn(z).squeeze(-1).masked_fill(~mask, -1e4)
        weight = torch.softmax(score, dim=1).unsqueeze(-1)
        attn_mean = (z * weight).sum(dim=1)
        attn_var = ((z - attn_mean.unsqueeze(1)).square() * weight).sum(dim=1).clamp_min(1e-6)
        mask_f = mask.float().unsqueeze(-1)
        plain_mean = (z * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp_min(1.0)
        pooled = torch.cat([attn_mean, torch.sqrt(attn_var), plain_mean], dim=-1)
        return self.head(pooled).squeeze(-1)


def song_targets(y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_f = mask.float()
    return (y * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp_min(1.0)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, Any]:
    model.eval()
    preds, targets, ids = [], [], []
    for x, y, song, _, mask in loader:
        pred = model(x.to(device), mask.to(device)).cpu().numpy()
        target = song_targets(y, mask).numpy()
        song_id = song[:, 0].numpy()
        preds.append(pred); targets.append(target); ids.append(song_id)
    pred = np.concatenate(preds); target = np.concatenate(targets); song_id = np.concatenate(ids)
    return {"pred": pred, "target": target, "song_id": song_id, "metrics": metrics(pred, target)}


def corrected_frame_metrics(base_path: Path, song_eval: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    base = pd.read_csv(base_path)
    mapping = dict(zip(song_eval["song_id"].astype(int), song_eval["pred"]))
    base["attentive_song_pred"] = base["musicId"].astype(int).map(mapping)
    if base["attentive_song_pred"].isna().any():
        raise ValueError("Missing attentive song predictions for base rows")
    base_mean = base.groupby("musicId")["pred_valence"].transform("mean")
    rows, cache = [], {}
    for strength in [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.70, 1.0]:
        for cap in [0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.06, 0.08, 0.12, 0.20]:
            shift = np.clip(strength * (base["attentive_song_pred"] - base_mean), -cap, cap)
            pred = np.clip(base["pred_valence"] + shift, -1.0, 1.0).to_numpy()
            row = {"strength": strength, "cap": cap, **metrics(pred, base["target_valence"].to_numpy())}
            row["guarded"] = row["mae"] <= 0.1532243 and row["pcc"] >= 0.8054875 and row["ccc"] >= 0.7884480
            rows.append(row); cache[(strength, cap)] = pred
    table = pd.DataFrame(rows).sort_values(["rmse", "mae", "ccc"], ascending=[True, True, False])
    eligible = table[table["guarded"]]
    selected = (eligible if not eligible.empty else table).iloc[0].to_dict()
    pred = cache[(float(selected["strength"]), float(selected["cap"]))]
    out = base[["musicId", "time_sec", "target_valence", "pred_valence", "attentive_song_pred"]].copy()
    out = out.rename(columns={"pred_valence": "base_pred_valence"})
    out["pred_valence"] = pred
    return out, selected, table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--librosa", type=Path, default=DEFAULT_LIBROSA)
    parser.add_argument("--base-predictions", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    set_seed(args.seed); args.out_dir.mkdir(parents=True, exist_ok=True)
    index = pd.read_csv(args.index)
    index = index[index["split"].isin(["train", "val"])].copy()
    index.to_csv(args.out_dir / "sequence_index.csv", index=False)
    build_mert_norm(index, args.features, args.out_dir / "dataset")
    build_harmonic_norm(index, args.librosa, selected_indices(["harmonic"]), args.out_dir / "dataset", force=True)
    fn = np.load(args.out_dir / "dataset/mert_norm.npz")
    hn = np.load(args.out_dir / "dataset/harmonic_norm.npz")
    loaders = {
        split: DataLoader(
            FusionDataset(index, split, args.features, args.labels, args.librosa,
                          fn["mean"], fn["std"], hn["mean"], hn["std"], hn["feature_idx"]),
            batch_size=24, shuffle=(split == "train"), collate_fn=collate,
        ) for split in ["train", "val"]
    }
    sample, _, _, _ = loaders["train"].dataset[0]
    device = resolve_device(args.device)
    model = AttentiveSongMean(int(sample.shape[1])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-3)
    checkpoint = args.out_dir / "best.pt"
    best_score, best_epoch, bad, history = float("inf"), 0, 0, []
    for epoch in range(1, 81):
        model.train(); losses = []
        for x, y, _, _, mask in loaders["train"]:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            optimizer.zero_grad(); pred = model(x, mask); target = song_targets(y, mask)
            loss = nn.functional.mse_loss(pred, target) + 0.03 * ccc_loss(pred, target)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            losses.append(float(loss.detach().cpu()))
        val = evaluate(model, loaders["val"], device); vm = val["metrics"]
        score = vm["rmse"] + 0.02 * max(0.0, 0.80 - vm["pcc"]) + 0.03 * max(0.0, 0.80 - vm["ccc"])
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), **{f"val_{k}": v for k, v in vm.items()}, "score": score})
        print(f"epoch={epoch:03d} song_rmse={vm['rmse']:.6f} song_pcc={vm['pcc']:.4f} song_ccc={vm['ccc']:.4f}", flush=True)
        if score < best_score:
            best_score, best_epoch, bad = score, epoch, 0
            torch.save({"model_state": model.state_dict(), "input_dim": int(sample.shape[1]), "seed": args.seed}, checkpoint)
        else:
            bad += 1
            if bad >= 12: break
    pd.DataFrame(history).to_csv(args.out_dir / "history.csv", index=False)
    model.load_state_dict(torch.load(checkpoint, map_location=device)["model_state"])
    train_eval = evaluate(model, loaders["train"], device); val_eval = evaluate(model, loaders["val"], device)
    pd.DataFrame({"musicId": val_eval["song_id"], "target_song_valence": val_eval["target"], "pred_song_valence": val_eval["pred"]}).to_csv(args.out_dir / "val_song_predictions.csv", index=False)
    corrected, selected, candidates = corrected_frame_metrics(args.base_predictions, val_eval)
    corrected.to_csv(args.out_dir / "val_frame_predictions.csv", index=False)
    candidates.to_csv(args.out_dir / "correction_candidates.csv", index=False)
    summary = {
        "created_at": "2026-07-14",
        "selection_boundary": "fold_03 locked development train/val only; final locked test excluded",
        "model": "MuQ hidden6 + harmonic + BiGRU attentive statistics pooling",
        "best_epoch": best_epoch, "train_song_metrics": train_eval["metrics"],
        "val_song_metrics": val_eval["metrics"], "selected_frame_correction": selected,
        "fold03_base": {"rmse": 0.1992603871, "mae": 0.1532242629, "pcc": 0.8064875170, "ccc": 0.7894479664},
        "gap_rows": threshold_rows({k: float(selected[k]) for k in ["rmse", "mae", "pcc", "ccc"]}),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
