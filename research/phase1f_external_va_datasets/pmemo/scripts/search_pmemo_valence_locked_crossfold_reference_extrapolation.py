#!/usr/bin/env python3
"""Cross-fold reference extrapolation for the locked Valence development folds.

The correction weight for each held fold is fitted on the other two folds only.
The final locked test set is never loaded by this script.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from train_pmemo_mert_valence_baseline import ROOT, metrics, threshold_rows


FOLDS = ["fold_01", "fold_02", "fold_03"]
DEFAULT_BASE = ROOT / "outputs/valence_locked_ordinal_residual_pls_fine3_20260714"
DEFAULT_REFERENCE = ROOT / "outputs/valence_locked_muq_song_mean_correction_20260713"
DEFAULT_OUT = ROOT / "outputs/valence_locked_crossfold_reference_extrapolation_20260714"


def macro(rows: list[dict[str, float]]) -> dict[str, float]:
    return {key: float(np.mean([row[key] for row in rows])) for key in ["rmse", "mae", "pcc", "ccc"]}


def load_aligned(base_dir: Path, reference_dir: Path, fold: str) -> pd.DataFrame:
    base = pd.read_csv(base_dir / f"{fold}_val_predictions.csv")
    reference = pd.read_csv(reference_dir / f"{fold}_val_predictions.csv")
    keys = [column for column in ["musicId", "time_sec", "frame_index"] if column in base and column in reference]
    if not keys:
        raise ValueError(f"No alignment keys found for {fold}")
    keep = [*keys, "pred_valence"]
    aligned = base.merge(reference[keep], on=keys, how="inner", validate="one_to_one", suffixes=("_base", "_reference"))
    if len(aligned) != len(base):
        raise ValueError(f"Incomplete alignment for {fold}: {len(aligned)} != {len(base)}")
    return aligned


def fit_weight(frames: list[pd.DataFrame], lower: float, upper: float) -> float:
    target = np.concatenate([frame["target_valence"].to_numpy(dtype=np.float64) for frame in frames])
    base = np.concatenate([frame["pred_valence_base"].to_numpy(dtype=np.float64) for frame in frames])
    reference = np.concatenate([frame["pred_valence_reference"].to_numpy(dtype=np.float64) for frame in frames])
    direction = reference - base
    denominator = float(np.dot(direction, direction))
    raw = 0.0 if denominator <= 1e-12 else float(np.dot(target - base, direction) / denominator)
    return float(np.clip(raw, lower, upper))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--weight-lower", type=float, default=-0.5)
    parser.add_argument("--weight-upper", type=float, default=0.1)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    frames = {fold: load_aligned(args.base_dir, args.reference_dir, fold) for fold in FOLDS}
    base_rows = [metrics(frames[fold]["pred_valence_base"], frames[fold]["target_valence"]) for fold in FOLDS]
    weights: dict[str, float] = {}
    per_fold: list[dict[str, float | str]] = []

    for held in FOLDS:
        train_frames = [frames[fold] for fold in FOLDS if fold != held]
        weight = fit_weight(train_frames, args.weight_lower, args.weight_upper)
        weights[held] = weight
        frame = frames[held]
        prediction = frame["pred_valence_base"].to_numpy(dtype=np.float64) + weight * (
            frame["pred_valence_reference"].to_numpy(dtype=np.float64)
            - frame["pred_valence_base"].to_numpy(dtype=np.float64)
        )
        prediction = np.clip(prediction, -1.0, 1.0)
        result = {"fold": held, **metrics(prediction, frame["target_valence"])}
        per_fold.append(result)

        out = frame.copy()
        out["crossfold_reference_weight"] = weight
        out["pred_valence"] = prediction
        out.to_csv(args.out_dir / f"{held}_val_predictions.csv", index=False)

    base_macro = macro(base_rows)
    selected_macro = macro(per_fold)  # type: ignore[arg-type]
    summary = {
        "created_at": "2026-07-14",
        "selection_boundary": (
            "Each held-fold weight is fitted analytically on the other two locked development folds; "
            "the final locked test set is excluded"
        ),
        "base_dir": str(args.base_dir),
        "reference_dir": str(args.reference_dir),
        "formula": "pred = base + weight * (reference - base)",
        "weight_bounds": [args.weight_lower, args.weight_upper],
        "weights": weights,
        "base_macro": base_macro,
        "macro": selected_macro,
        "per_fold": per_fold,
        "delta_vs_base": {key: selected_macro[key] - base_macro[key] for key in base_macro},
        "gap_rows": threshold_rows(selected_macro),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
