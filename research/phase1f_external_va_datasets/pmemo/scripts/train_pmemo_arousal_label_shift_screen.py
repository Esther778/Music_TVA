#!/usr/bin/env python3
"""Arousal label-shift training screen on development folds only."""

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
from torch.utils.data import DataLoader, Dataset

from train_pmemo_opensmile_arousal_trials import (
    ArousalBiLSTM,
    ROOT,
    ccc_loss,
    collate,
    evaluate,
    metrics,
    moving_average_by_song,
    resolve_device,
    set_seed,
    threshold_rows,
)


@dataclass
class Config:
    trial_id: str = "a_shift_h96_do015_lr8e4"
    hidden_dim: int = 96
    layers: int = 1
    dropout: float = 0.15
    head_hidden: int = 64
    learning_rate: float = 8e-4
    weight_decay: float = 1e-4
    batch_size: int = 32
    ccc_weight: float = 0.0
    epochs: int = 70
    patience: int = 10


class ShiftTrainDataset(Dataset):
    def __init__(self, index: pd.DataFrame, split: str, shift_frames: int, train_shift: bool):
        self.rows = index[index["split"] == split].sort_values("musicId").to_dict("records")
        self.shift_frames = int(shift_frames)
        self.train_shift = bool(train_shift)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        data = np.load(self.rows[idx]["path"])
        x = data["X"].astype(np.float32)
        y_orig = data["Y"].astype(np.float32)[:, 1]
        time_sec = data["time_sec"].astype(np.float32)
        n = min(len(x), len(y_orig), len(time_sec))
        x = x[:n]
        y_orig = y_orig[:n]
        time_sec = time_sec[:n]
        if self.train_shift and self.shift_frames != 0:
            # Positive shift means training frame t against future label t+shift.
            y = np.empty_like(y_orig)
            if self.shift_frames > 0:
                s = min(self.shift_frames, n - 1)
                y[:-s] = y_orig[s:]
                y[-s:] = y_orig[-1]
            else:
                s = min(-self.shift_frames, n - 1)
                y[s:] = y_orig[:-s]
                y[:s] = y_orig[0]
        else:
            y = y_orig
        song = np.full(n, int(data["music_id"]), dtype=np.int32)
        return torch.from_numpy(x), torch.from_numpy(y.astype(np.float32)), torch.from_numpy(song), torch.from_numpy(time_sec)


def selection_score(m: dict[str, float]) -> float:
    return m["rmse"] + 0.02 * max(0.0, m["mae"] - 0.15) + 0.01 * max(0.0, 0.80 - m["ccc"])


def postprocess_search(val_eval: dict[str, Any]) -> tuple[dict[str, Any], pd.DataFrame]:
    pred = val_eval["pred"]
    target = val_eval["target"]
    song = val_eval["song_id"]
    rows = []
    best = {"postprocess": "raw", "metrics": val_eval["metrics"], "score": selection_score(val_eval["metrics"])}
    for window in [1, 3, 5, 7, 9, 11, 15, 21]:
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


def run_one(data_dir: Path, out_dir: Path, shift_frames: int, seed: int, device: torch.device) -> dict[str, Any]:
    config = Config(trial_id=f"a_shift{shift_frames:+d}_h96_do015_lr8e4")
    set_seed(seed)
    trial_dir = out_dir / config.trial_id
    trial_dir.mkdir(parents=True, exist_ok=True)
    index = pd.read_csv(data_dir / "sequence_index.csv")
    loaders = {
        "train": DataLoader(ShiftTrainDataset(index, "train", shift_frames, True), batch_size=config.batch_size, shuffle=True, collate_fn=collate),
        "val": DataLoader(ShiftTrainDataset(index, "val", 0, False), batch_size=config.batch_size, shuffle=False, collate_fn=collate),
    }
    input_dim = int(np.load(index.iloc[0]["path"])["X"].shape[1])
    model = ArousalBiLSTM(input_dim, config.hidden_dim, config.layers, config.dropout, config.head_hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    ckpt = trial_dir / "best.pt"
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
            loss = ((pred[mask] - y[mask]) ** 2).mean()
            if config.ccc_weight:
                loss = loss + config.ccc_weight * ccc_loss(pred[mask], y[mask])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        val_eval = evaluate(model, loaders["val"], device)
        m = val_eval["metrics"]
        score = selection_score(m)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), **{f"val_{k}": v for k, v in m.items()}, "score": score})
        print(f"{config.trial_id} epoch={epoch:03d} val_rmse={m['rmse']:.5f} val_mae={m['mae']:.5f} val_pcc={m['pcc']:.3f} val_ccc={m['ccc']:.3f}", flush=True)
        if score < best_score:
            best_score = score
            best_epoch = epoch
            bad = 0
            torch.save({"model_state": model.state_dict(), "input_dim": input_dim, "config": asdict(config), "shift_frames": shift_frames}, ckpt)
        else:
            bad += 1
            if bad >= config.patience:
                break
    pd.DataFrame(history).to_csv(trial_dir / "history.csv", index=False)
    model.load_state_dict(torch.load(ckpt, map_location=device)["model_state"])
    evals = {split: evaluate(model, loaders[split], device) for split in ["train", "val"]}
    for split, data in evals.items():
        pd.DataFrame({"musicId": data["song_id"], "time_sec": data["time_sec"], "target_arousal": data["target"], "pred_arousal": data["pred"]}).to_csv(trial_dir / f"{split}_predictions.csv", index=False)
    best_pp, pp_rows = postprocess_search(evals["val"])
    pp_rows.to_csv(trial_dir / "val_postprocess_search.csv", index=False)
    summary = {
        "created_at": "2026-07-13",
        "purpose": "Arousal label-shift training screen; final locked test excluded",
        "shift_frames": shift_frames,
        "best_epoch": best_epoch,
        "config": asdict(config),
        **{f"{split}_{k}": v for split, e in evals.items() for k, v in e["metrics"].items()},
        "best_val_postprocess": best_pp,
        "val_gap_rows_best_postprocess": threshold_rows(best_pp["metrics"]),
    }
    (trial_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--shifts", nargs="+", type=int, default=[-8, -4, 4, 8])
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    rows = [run_one(args.data_dir, args.out_dir, s, args.seed, device) for s in args.shifts]
    comparison = []
    for row in rows:
        m = row["best_val_postprocess"]["metrics"]
        comparison.append({"shift_frames": row["shift_frames"], "best_epoch": row["best_epoch"], **{f"post_val_{k}": v for k, v in m.items()}})
    pd.DataFrame(comparison).to_csv(args.out_dir / "comparison.csv", index=False)
    print(json.dumps({"comparison": comparison}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
