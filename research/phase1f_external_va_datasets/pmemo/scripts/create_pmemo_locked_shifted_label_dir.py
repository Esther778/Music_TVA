#!/usr/bin/env python3
"""Create train-only shifted PMEmo label sequence files for a locked dev fold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from train_pmemo_mert_valence_baseline import ROOT


DEFAULT_LABEL_DIR = ROOT / "outputs" / "opensmile_baseline" / "dataset" / "sequence"


def shift_1d(values: np.ndarray, frames: int) -> np.ndarray:
    if frames == 0 or len(values) <= 1:
        return values.copy()
    out = np.empty_like(values)
    if frames > 0:
        k = min(frames, len(values) - 1)
        out[:-k] = values[k:]
        out[-k:] = values[-1]
    else:
        k = min(-frames, len(values) - 1)
        out[:k] = values[0]
        out[k:] = values[:-k]
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--source-label-dir", type=Path, default=DEFAULT_LABEL_DIR)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--shift-frames", type=int, required=True)
    parser.add_argument("--target", choices=["valence", "arousal"], default="valence")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    index = pd.read_csv(args.index)
    train_ids = set(index.loc[index["split"] == "train", "musicId"].astype(int).tolist())
    target_idx = 0 if args.target == "valence" else 1
    count = 0
    for src in sorted(args.source_label_dir.glob("*.npz")):
        sid = int(src.stem)
        data = np.load(src)
        arrays = {key: data[key] for key in data.files}
        y = arrays["Y"].astype(np.float32).copy()
        if sid in train_ids:
            y[:, target_idx] = shift_1d(y[:, target_idx], args.shift_frames)
            count += 1
        arrays["Y"] = y
        np.savez(args.out_dir / src.name, **arrays)
    meta = {
        "created_at": "2026-07-13",
        "source_label_dir": str(args.source_label_dir),
        "index": str(args.index),
        "target": args.target,
        "shift_frames": int(args.shift_frames),
        "shift_seconds": float(args.shift_frames * 0.5),
        "train_only_shift": True,
        "shifted_train_song_count": count,
        "validation_labels_original": True,
    }
    (args.out_dir / "_SHIFT_META.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
