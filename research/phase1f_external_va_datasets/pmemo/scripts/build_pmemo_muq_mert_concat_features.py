#!/usr/bin/env python3
"""Build aligned MuQ-6 + MERT-6 feature files for locked development songs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INDEX = (
    ROOT / "data" / "splits" / "pmemo_final_locked_20260713"
    / "devfolds_3fold_20260713" / "fold_01" / "sequence_index.csv"
)
DEFAULT_MUQ = ROOT / "outputs" / "muq_pmemo_locked_development" / "layer_06" / "features"
DEFAULT_MERT = ROOT / "outputs" / "mert_pmemo_hidden6" / "features"
DEFAULT_OUT = ROOT / "outputs" / "muq6_mert6_concat_locked_development" / "features"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--muq-dir", type=Path, default=DEFAULT_MUQ)
    parser.add_argument("--mert-dir", type=Path, default=DEFAULT_MERT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    index = pd.read_csv(args.index)
    if not set(index["split"].unique()).issubset({"train", "val"}):
        raise ValueError("Development train/val index required.")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for row in index.itertuples():
        sid = int(row.musicId)
        out = args.out_dir / f"{sid}.npz"
        if out.exists() and not args.force:
            data = np.load(out)
            rows.append({"musicId": sid, "status": "skipped", "frames": len(data["time_sec"])})
            continue
        muq = np.load(args.muq_dir / f"{sid}.npz")
        mert = np.load(args.mert_dir / f"{sid}.npz")
        n = min(len(muq["time_sec"]), len(mert["time_sec"]))
        muq_time = muq["time_sec"][:n].astype(np.float32)
        mert_time = mert["time_sec"][:n].astype(np.float32)
        if not np.allclose(muq_time, mert_time):
            raise ValueError(f"Time mismatch for musicId={sid}")
        embedding = np.concatenate(
            [muq["embedding"][:n].astype(np.float32), mert["embedding"][:n].astype(np.float32)],
            axis=1,
        )
        np.savez_compressed(
            out,
            musicId=sid,
            split=str(row.split),
            time_sec=muq_time,
            embedding=embedding,
            feature_type="MuQ hidden_6 1024 + MERT-v1-95M hidden_6 768",
        )
        rows.append({"musicId": sid, "status": "ok", "frames": n})
    summary = {
        "songs": len(rows),
        "ok": sum(item["status"] == "ok" for item in rows),
        "skipped": sum(item["status"] == "skipped" for item in rows),
        "feature_dim": 1792,
        "selection_boundary": "Locked development songs only; final locked test excluded.",
    }
    (args.out_dir.parent / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
