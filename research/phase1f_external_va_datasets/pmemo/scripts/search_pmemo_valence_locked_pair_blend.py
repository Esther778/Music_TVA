#!/usr/bin/env python3
"""Search a reproducible two-candidate blend on locked PMEmo dev folds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from train_pmemo_mert_valence_baseline import ROOT, metrics, threshold_rows


FOLDS = ["fold_01", "fold_02", "fold_03"]


def macro(rows: list[dict[str, float]]) -> dict[str, float]:
    return {key: float(np.mean([row[key] for row in rows])) for key in ["rmse", "mae", "pcc", "ccc"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-dir", type=Path, required=True)
    parser.add_argument("--secondary-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    primary = {f: pd.read_csv(args.primary_dir / f"{f}_val_predictions.csv") for f in FOLDS}
    secondary = {f: pd.read_csv(args.secondary_dir / f"{f}_val_predictions.csv") for f in FOLDS}
    candidates = []
    cache: dict[float, dict[str, np.ndarray]] = {}
    for weight in np.linspace(0.0, 1.0, 201):
        per_fold = []
        preds = {}
        for fold in FOLDS:
            left, right = primary[fold], secondary[fold]
            if not np.array_equal(left["musicId"].to_numpy(), right["musicId"].to_numpy()):
                raise ValueError(f"Row mismatch in {fold}")
            pred = (1.0 - weight) * left["pred_valence"].to_numpy() + weight * right["pred_valence"].to_numpy()
            preds[fold] = pred
            per_fold.append(metrics(pred, left["target_valence"].to_numpy()))
        result = macro(per_fold)
        candidates.append({"secondary_weight": float(weight), **result})
        cache[float(weight)] = preds
    table = pd.DataFrame(candidates).sort_values(["rmse", "mae", "ccc"], ascending=[True, True, False])
    table.to_csv(args.out_dir / "candidates.csv", index=False)
    selected = table.iloc[0].to_dict()
    preds = cache[float(selected["secondary_weight"])]
    per_fold_rows = []
    for fold in FOLDS:
        out = primary[fold].copy().rename(columns={"pred_valence": "primary_pred_valence"})
        out["pred_valence"] = preds[fold]
        out.to_csv(args.out_dir / f"{fold}_val_predictions.csv", index=False)
        per_fold_rows.append({"fold": fold, **metrics(out["pred_valence"], out["target_valence"])})
    selected_macro = macro(per_fold_rows)
    summary = {
        "created_at": "2026-07-14",
        "selection_boundary": "Locked development OOF blend search only; final locked test excluded",
        "primary_dir": str(args.primary_dir), "secondary_dir": str(args.secondary_dir),
        "selected": selected, "macro": selected_macro, "per_fold": per_fold_rows,
        "gap_rows": threshold_rows(selected_macro),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
