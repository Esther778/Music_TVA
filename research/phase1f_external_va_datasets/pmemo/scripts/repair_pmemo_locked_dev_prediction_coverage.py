#!/usr/bin/env python3
"""Restore missing locked-dev edge frames with within-song edge hold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from train_pmemo_mert_valence_baseline import ROOT, metrics, threshold_rows


FOLDS = ["fold_01", "fold_02", "fold_03"]
DEFAULT_SPLITS = ROOT / "data/splits/pmemo_final_locked_20260713/devfolds_3fold_20260713"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimension", choices=["valence", "arousal"], required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLITS)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    target_col, pred_col = f"target_{args.dimension}", f"pred_{args.dimension}"
    target_index = 0 if args.dimension == "valence" else 1
    per_fold, repaired_rows = [], []

    for fold in FOLDS:
        source = pd.read_csv(args.input_dir / f"{fold}_val_predictions.csv")
        source["time_sec"] = source["time_sec"].round(4)
        expected_parts = []
        index = pd.read_csv(args.split_dir / fold / "sequence_index.csv").query("split == 'val'")
        for row in index.itertuples():
            sequence = np.load(row.path)
            expected_parts.append(pd.DataFrame({
                "musicId": int(row.musicId),
                "time_sec": np.round(sequence["time_sec"].astype(float), 4),
                target_col: sequence["Y"][:, target_index].astype(float),
            }))
        expected = pd.concat(expected_parts, ignore_index=True)
        output = expected.merge(
            source[["musicId", "time_sec", pred_col]],
            on=["musicId", "time_sec"],
            how="left",
            validate="one_to_one",
        )
        missing = output[pred_col].isna()
        repaired_rows.extend([
            {"fold": fold, "musicId": int(row.musicId), "time_sec": float(row.time_sec)}
            for row in output.loc[missing, ["musicId", "time_sec"]].itertuples(index=False)
        ])
        output["coverage_repaired"] = missing.astype(int)
        output[pred_col] = output.groupby("musicId", sort=False)[pred_col].transform(lambda values: values.ffill().bfill())
        if output[pred_col].isna().any():
            raise ValueError(f"Unrepairable song in {fold}")
        output.to_csv(args.out_dir / f"{fold}_val_predictions.csv", index=False)
        per_fold.append({"fold": fold, "frames": len(output), **metrics(output[pred_col], output[target_col])})

    macro = {
        key: float(np.mean([row[key] for row in per_fold]))
        for key in ["rmse", "mae", "pcc", "ccc"]
    }
    pd.DataFrame(repaired_rows).to_csv(args.out_dir / "repaired_frames.csv", index=False)
    summary = {
        "created_at": "2026-07-15",
        "dimension": args.dimension,
        "input_dir": str(args.input_dir),
        "split_dir": str(args.split_dir),
        "policy": "within-song forward fill then backward fill; only missing edge frames",
        "repaired_frame_count": len(repaired_rows),
        "total_frames": int(sum(row["frames"] for row in per_fold)),
        "per_fold": per_fold,
        "macro": macro,
        "gap_rows": threshold_rows(macro),
        "final_locked_test": "excluded",
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
