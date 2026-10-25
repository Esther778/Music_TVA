from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from train_gated_fusion import (
    FEATURE_ROOT,
    GatedDualBranchRegressor,
    MERT_ROOT,
    ROOT,
    collate,
    evaluate,
    metrics,
    set_seed,
    song_balanced_loss,
)


LAYER6_ROOT = ROOT / "data" / "mert_layer6"


def time_indices(source: np.ndarray, target: np.ndarray, song_id: int) -> np.ndarray:
    indices = []
    for time_sec in target:
        matches = np.flatnonzero(np.isclose(source, time_sec, atol=1e-4))
        if len(matches) != 1:
            raise ValueError(
                f"Song {song_id}: expected one feature at {time_sec:.3f}s, "
                f"found {len(matches)}."
            )
        indices.append(int(matches[0]))
    return np.asarray(indices, dtype=np.int64)


def load_aligned(song_id: int, sequence_path: str | Path):
    with np.load(sequence_path) as sequence:
        target_times = sequence["time_sec"].astype(np.float32)
        target = sequence["Y"].astype(np.float32)
    with np.load(LAYER6_ROOT / f"{song_id}.npz") as layer6:
        if not np.array_equal(layer6["layers"].astype(int), np.array([6])):
            raise ValueError(f"Song {song_id}: layer metadata is not [6].")
        layer_times = layer6["time_sec"].astype(np.float32)
        mert = layer6["embeddings"][:, 0].astype(np.float32)
    mert = mert[time_indices(layer_times, target_times, song_id)]
    with np.load(FEATURE_ROOT / f"{song_id}.npz") as handcrafted_file:
        hand_times = handcrafted_file["target_time_sec"].astype(np.float32)
        handcrafted = handcrafted_file["handcrafted"].astype(np.float32)
        hand_target = handcrafted_file["labels"].astype(np.float32)
    hand_indices = time_indices(hand_times, target_times, song_id)
    handcrafted = handcrafted[hand_indices]
    hand_target = hand_target[hand_indices]
    if not np.allclose(target, hand_target, atol=1e-6):
        raise ValueError(f"Song {song_id}: handcrafted and sequence labels differ.")
    if not (
        np.isfinite(mert).all()
        and np.isfinite(handcrafted).all()
        and np.isfinite(target).all()
    ):
        raise ValueError(f"Song {song_id}: aligned arrays contain NaN or Inf.")
    return mert, handcrafted, target


def build_norm(index: pd.DataFrame, output: Path):
    if output.is_file():
        norm = np.load(output)
        return (
            norm["mert_mean"],
            norm["mert_std"],
            norm["hand_mean"],
            norm["hand_std"],
        )
    count = 0
    mert_sum = mert_square = hand_sum = hand_square = None
    for row in index[index["split"] == "train"].itertuples():
        mert, hand, _ = load_aligned(int(row.song_id), row.sequence_path)
        if mert_sum is None:
            mert_sum = mert.astype(np.float64).sum(axis=0)
            mert_square = (mert.astype(np.float64) ** 2).sum(axis=0)
            hand_sum = hand.astype(np.float64).sum(axis=0)
            hand_square = (hand.astype(np.float64) ** 2).sum(axis=0)
        else:
            mert_sum += mert.sum(axis=0)
            mert_square += (mert.astype(np.float64) ** 2).sum(axis=0)
            hand_sum += hand.sum(axis=0)
            hand_square += (hand.astype(np.float64) ** 2).sum(axis=0)
        count += len(mert)
    mert_mean = (mert_sum / count).astype(np.float32)
    hand_mean = (hand_sum / count).astype(np.float32)
    mert_std = np.sqrt(
        np.maximum(mert_square / count - mert_mean.astype(np.float64) ** 2, 1e-8)
    ).astype(np.float32)
    hand_std = np.sqrt(
        np.maximum(hand_square / count - hand_mean.astype(np.float64) ** 2, 1e-8)
    ).astype(np.float32)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        mert_mean=mert_mean,
        mert_std=mert_std,
        hand_mean=hand_mean,
        hand_std=hand_std,
        frame_count=count,
    )
    return mert_mean, mert_std, hand_mean, hand_std


class Layer6FusionDataset(Dataset):
    def __init__(self, index: pd.DataFrame, split: str, norms):
        self.rows = (
            index[index["split"] == split].sort_values("song_id").to_dict("records")
        )
        self.mert_mean, self.mert_std, self.hand_mean, self.hand_std = norms

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, item):
        row = self.rows[item]
        song_id = int(row["song_id"])
        mert, hand, target = load_aligned(song_id, row["sequence_path"])
        mert = (mert - self.mert_mean) / self.mert_std
        hand = (hand - self.hand_mean) / self.hand_std
        return (
            torch.from_numpy(mert),
            torch.from_numpy(hand),
            torch.from_numpy(target),
            song_id,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/layer6_gated_fusion.json"
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    set_seed(int(config["seed"]))
    device = torch.device(args.device)
    index = pd.read_csv(
        MERT_ROOT / "data/datasets/mert_va/sequence_index.csv"
    )
    index = index[index["split"].isin(["train", "val"])].copy()
    output_dir = ROOT / "outputs" / config["trial_id"]
    output_dir.mkdir(parents=True, exist_ok=True)
    norms = build_norm(index, output_dir / "feature_norm.npz")
    datasets = {
        split: Layer6FusionDataset(index, split, norms)
        for split in ("train", "val")
    }
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=int(config["batch_size"]),
            shuffle=split == "train",
            collate_fn=collate,
        )
        for split, dataset in datasets.items()
    }
    first = datasets["train"][0]
    model = GatedDualBranchRegressor(
        first[0].shape[1], first[1].shape[1], config
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    checkpoint = output_dir / "best.pt"
    history = []
    best_rmse = float("inf")
    best_epoch = bad_epochs = 0
    print(
        json.dumps(
            {
                "trial_id": config["trial_id"],
                "device": str(device),
                "songs": {key: len(value) for key, value in datasets.items()},
                "parameters": sum(p.numel() for p in model.parameters()),
                "test_locked": True,
            },
            indent=2,
        )
    )
    for epoch in range(1, int(config["epochs"]) + 1):
        model.train()
        losses = []
        for mert, hand, target, mask, _, _ in loaders["train"]:
            mert, hand = mert.to(device), hand.to(device)
            target, mask = target.to(device), mask.to(device)
            optimizer.zero_grad()
            prediction = model(mert, hand)
            loss = song_balanced_loss(prediction, target, mask, config["loss"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
            optimizer.step()
            losses.append(float(loss.detach()))
        val_metrics, _ = evaluate(model, loaders["val"], device)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        history.append(row)
        print(
            f"epoch={epoch:03d} train_loss={row['train_loss']:.6f} "
            f"val_rmse={val_metrics['rmse']:.6f} "
            f"val_pcc={val_metrics['mean_pcc']:.6f} "
            f"val_ccc={val_metrics['mean_ccc']:.6f}"
        )
        if val_metrics["rmse"] < best_rmse:
            best_rmse, best_epoch, bad_epochs = val_metrics["rmse"], epoch, 0
            torch.save(model.state_dict(), checkpoint)
        else:
            bad_epochs += 1
            if bad_epochs >= int(config["patience"]):
                break
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    train_metrics, train_prediction = evaluate(model, loaders["train"], device)
    val_metrics, val_prediction = evaluate(model, loaders["val"], device)
    summary = {
        "trial_id": config["trial_id"],
        "seed": int(config["seed"]),
        "best_epoch": best_epoch,
        "checkpoint": str(checkpoint),
        "test_evaluated": False,
        "feature_layer": 6,
        "config": config,
        **{f"train_{key}": value for key, value in train_metrics.items()},
        **{f"val_{key}": value for key, value in val_metrics.items()},
    }
    np.save(output_dir / "train_pred.npy", train_prediction.astype(np.float32))
    np.save(output_dir / "val_pred.npy", val_prediction.astype(np.float32))
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (output_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
