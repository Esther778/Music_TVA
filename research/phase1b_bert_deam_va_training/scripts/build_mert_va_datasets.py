#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parents[1]
DEAM_DYNAMIC_DIR = (
    PROJECT_ROOT
    / "research"
    / "phase1_va_prediction"
    / "openl3_va_baseline"
    / "data"
    / "DEAM"
    / "annotations"
    / "annotations averaged per song"
    / "dynamic (per second annotations)"
)


def load_label_maps() -> tuple[dict[int, pd.Series], dict[int, pd.Series]]:
    valence = pd.read_csv(DEAM_DYNAMIC_DIR / "valence.csv").set_index("song_id")
    arousal = pd.read_csv(DEAM_DYNAMIC_DIR / "arousal.csv").set_index("song_id")
    return valence.to_dict("index"), arousal.to_dict("index")


def labels_for_times(song_id: int, times: np.ndarray, valence: dict[int, dict[str, Any]], arousal: dict[int, dict[str, Any]]) -> np.ndarray:
    v_row = valence[song_id]
    a_row = arousal[song_id]
    labels = []
    for time_sec in times:
        ms = int(round(float(time_sec) * 1000))
        column = f"sample_{ms}ms"
        labels.append([float(v_row[column]), float(a_row[column])])
    return np.asarray(labels, dtype=np.float32)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build MERT+DEAM VA datasets for MLP and sequence models.")
    parser.add_argument("--manifest", default=str(ROOT / "data" / "manifests" / "split_manifest.csv"))
    parser.add_argument("--features-dir", default=str(ROOT / "data" / "features_mert"))
    parser.add_argument("--out-dir", default=str(ROOT / "data" / "datasets" / "mert_va"))
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    features_dir = Path(args.features_dir)
    out_dir = Path(args.out_dir)
    frame_dir = out_dir / "frame"
    seq_dir = out_dir / "sequence"
    frame_dir.mkdir(parents=True, exist_ok=True)
    seq_dir.mkdir(parents=True, exist_ok=True)

    valence, arousal = load_label_maps()
    frame_parts = {split: {"X": [], "Y": [], "song_id": [], "time_sec": []} for split in ["train", "val", "test"]}
    sequence_rows: list[dict[str, Any]] = []
    dropped_by_split = {split: 0 for split in ["train", "val", "test"]}

    for row in manifest.sort_values("song_id").to_dict("records"):
        song_id = int(row["song_id"])
        split = str(row["split"])
        feature_path = features_dir / f"{song_id}.npz"
        if not feature_path.is_file():
            raise FileNotFoundError(feature_path)
        data = np.load(feature_path)
        times = data["time_sec"].astype(np.float32)
        X = data["embedding"].astype(np.float32)
        Y = labels_for_times(song_id, times, valence, arousal)
        if len(X) != len(Y):
            raise ValueError(f"Length mismatch song_id={song_id}: X={len(X)} Y={len(Y)}")
        finite_mask = np.isfinite(X).all(axis=1) & np.isfinite(Y).all(axis=1)
        dropped_invalid = int(len(finite_mask) - finite_mask.sum())
        if dropped_invalid:
            X = X[finite_mask]
            Y = Y[finite_mask]
            times = times[finite_mask]
            dropped_by_split[split] += dropped_invalid
        if len(X) == 0:
            raise ValueError(f"All frames were invalid after finite-value filtering: song_id={song_id}")

        np.savez_compressed(seq_dir / f"{song_id}.npz", song_id=song_id, split=split, time_sec=times, X=X, Y=Y)
        sequence_rows.append(
            {
                "song_id": song_id,
                "split": split,
                "frames": len(X),
                "dropped_invalid_frames": dropped_invalid,
                "feature_dim": X.shape[1],
                "sequence_path": str(seq_dir / f"{song_id}.npz"),
            }
        )
        frame_parts[split]["X"].append(X)
        frame_parts[split]["Y"].append(Y)
        frame_parts[split]["song_id"].append(np.full(len(X), song_id, dtype=np.int32))
        frame_parts[split]["time_sec"].append(times)

    summary: dict[str, Any] = {"splits": {}}
    for split, parts in frame_parts.items():
        X = np.concatenate(parts["X"], axis=0).astype(np.float32)
        Y = np.concatenate(parts["Y"], axis=0).astype(np.float32)
        song_id = np.concatenate(parts["song_id"], axis=0).astype(np.int32)
        time_sec = np.concatenate(parts["time_sec"], axis=0).astype(np.float32)
        np.save(frame_dir / f"X_{split}.npy", X)
        np.save(frame_dir / f"Y_{split}.npy", Y)
        np.save(frame_dir / f"song_id_{split}.npy", song_id)
        np.save(frame_dir / f"time_sec_{split}.npy", time_sec)
        summary["splits"][split] = {
            "frames": int(len(X)),
            "songs": int(sum(1 for row in sequence_rows if row["split"] == split)),
            "feature_dim": int(X.shape[1]),
            "dropped_invalid_frames": int(dropped_by_split[split]),
        }

    X_train = np.load(frame_dir / "X_train.npy")
    mean = X_train.mean(axis=0, keepdims=True).astype(np.float32)
    std = (X_train.std(axis=0, keepdims=True) + 1e-8).astype(np.float32)
    np.savez(frame_dir / "feature_norm.npz", mean=mean, std=std)
    summary["feature_norm"] = {"mean_shape": list(mean.shape), "std_shape": list(std.shape)}

    write_csv(
        out_dir / "sequence_index.csv",
        sequence_rows,
        ["song_id", "split", "frames", "dropped_invalid_frames", "feature_dim", "sequence_path"],
    )
    with (out_dir / "dataset_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
