#!/usr/bin/env python3
"""Train MERT hidden6 + OpenSMILE Valence fusion on locked development folds."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from train_pmemo_mert_opensmile_arousal_fusion_locked_devfold import (
    DEFAULT_MERT_DIR,
    DEFAULT_OPENSMILE_DATASET,
    MertOpenSmileArousalDataset,
    build_mert_norm,
    postprocess_search,
    set_seed,
)
from train_pmemo_mert_opensmile_valence_fusion import FusionBiLSTM
from train_pmemo_opensmile_arousal_trials import ROOT, ccc_loss, collate, resolve_device, smoothness_loss
from train_pmemo_mert_valence_baseline import metrics, threshold_rows


DEFAULT_INDEX = ROOT / "data" / "splits" / "pmemo_final_locked_20260713" / "devfolds_3fold_20260713" / "fold_01" / "sequence_index.csv"
DEFAULT_OUT = ROOT / "outputs" / "valence_locked_mert_opensmile_fusion_20260713" / "fold_01"


@dataclass
class Config:
    trial_id: str = "v_mert_smile_h96_proj384_do035_ccc004_smooth003"
    proj_dim: int = 384
    hidden_dim: int = 96
    dropout: float = 0.35
    learning_rate: float = 3e-4
    weight_decay: float = 3e-4
    batch_size: int = 18
    ccc_weight: float = 0.04
    smooth_weight: float = 0.003
    epochs: int = 80
    patience: int = 10


class MertOpenSmileValenceDataset(MertOpenSmileArousalDataset):
    def __getitem__(self, idx: int):
        x, _, song, time_sec = super().__getitem__(idx)
        sid = int(self.rows[idx]["musicId"])
        smile = np.load(self.opensmile_dataset / f"{sid}.npz")
        y = smile["Y"].astype(np.float32)[:, 0]
        n = min(len(x), len(y), len(time_sec))
        return x[:n], torch.from_numpy(y[:n]), song[:n], time_sec[:n]


@torch.no_grad()
def evaluate(model, loader, device):
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--mert-dir", type=Path, default=DEFAULT_MERT_DIR)
    parser.add_argument("--opensmile-dataset", type=Path, default=DEFAULT_OPENSMILE_DATASET)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    config = Config()
    set_seed(args.seed)
    index = pd.read_csv(args.index)
    mert_mean, mert_std = build_mert_norm(index, "train", args.mert_dir, args.out_dir)
    loaders = {
        split: DataLoader(
            MertOpenSmileValenceDataset(index, split, args.mert_dir, args.opensmile_dataset, mert_mean, mert_std),
            batch_size=config.batch_size,
            shuffle=(split == "train"),
            collate_fn=collate,
        )
        for split in ["train", "val"]
    }
    sample_x, _, _, _ = loaders["train"].dataset[0]
    device = resolve_device(args.device)
    model = FusionBiLSTM(int(sample_x.shape[1]), config.proj_dim, config.hidden_dim, config.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    ckpt = args.out_dir / "best.pt"
    best_score = float("inf")
    best_epoch = 0
    bad = 0
    history = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        losses = []
        for x, y, _, _, mask in loaders["train"]:
            x = x.to(device); y = y.to(device); mask = mask.to(device)
            optimizer.zero_grad()
            pred = model(x, mask)
            pred_valid = pred[mask]
            y_valid = y[mask]
            loss = ((pred_valid - y_valid) ** 2).mean()
            loss = loss + config.ccc_weight * ccc_loss(pred_valid, y_valid)
            loss = loss + config.smooth_weight * smoothness_loss(pred, mask)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        val_eval = evaluate(model, loaders["val"], device)
        m = val_eval["metrics"]
        score = m["rmse"] + 0.03 * max(0.0, 0.80 - m["ccc"]) + 0.02 * max(0.0, 0.80 - m["pcc"])
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), **{f"val_{k}": v for k, v in m.items()}, "score": score})
        print(
            f"{config.trial_id} epoch={epoch:03d} train_loss={float(np.mean(losses)):.5f} "
            f"val_rmse={m['rmse']:.5f} val_mae={m['mae']:.5f} val_pcc={m['pcc']:.3f} val_ccc={m['ccc']:.3f}",
            flush=True,
        )
        if score < best_score:
            best_score = score
            best_epoch = epoch
            bad = 0
            torch.save({"model_state": model.state_dict(), "input_dim": int(sample_x.shape[1]), "config": asdict(config)}, ckpt)
        else:
            bad += 1
            if bad >= config.patience:
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
        "purpose": "MERT hidden6 + OpenSMILE Valence fusion on locked development fold; final test excluded",
        "selection_boundary": "Development-fold validation only; not final locked test performance",
        "trial_id": config.trial_id,
        "best_epoch": best_epoch,
        "config": asdict(config),
        "val_metrics": evals["val"]["metrics"],
        "best_val_postprocess": best_pp,
        "gap_rows_raw": threshold_rows(evals["val"]["metrics"]),
        "gap_rows_best_postprocess": threshold_rows(best_pp["metrics"]),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
