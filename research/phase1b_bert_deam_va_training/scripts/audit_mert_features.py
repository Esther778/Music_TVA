#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit extracted MERT feature files against the DEAM split manifest.")
    parser.add_argument("--manifest", default=str(ROOT / "data" / "manifests" / "split_manifest.csv"))
    parser.add_argument("--features-dir", default=str(ROOT / "data" / "features_mert"))
    parser.add_argument("--output", default=str(ROOT / "outputs" / "mert_feature_inventory.csv"))
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    features_dir = Path(args.features_dir)
    rows = []
    for row in manifest.to_dict("records"):
        song_id = str(row["song_id"])
        feature_path = features_dir / f"{song_id}.npz"
        exists = feature_path.is_file()
        frames = ""
        dim = ""
        if exists:
            data = np.load(feature_path)
            embedding = data["embedding"]
            frames = embedding.shape[0]
            dim = embedding.shape[1]
        rows.append(
            {
                "song_id": song_id,
                "split": row["split"],
                "feature_exists": str(exists).lower(),
                "feature_path": str(feature_path),
                "frames": frames,
                "dim": dim,
            }
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["song_id", "split", "feature_exists", "feature_path", "frames", "dim"])
        writer.writeheader()
        writer.writerows(rows)

    total = len(rows)
    existing = sum(1 for row in rows if row["feature_exists"] == "true")
    print(f"features={existing}/{total} output={output}")


if __name__ == "__main__":
    main()

