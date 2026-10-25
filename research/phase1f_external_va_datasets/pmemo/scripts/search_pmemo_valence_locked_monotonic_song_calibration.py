#!/usr/bin/env python3
"""Cross-fold monotonic song-mean calibration for locked-development Valence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer, StandardScaler

from train_pmemo_mert_valence_baseline import ROOT, metrics, threshold_rows


FOLDS = ["fold_01", "fold_02", "fold_03"]
DEFAULT_BASE = ROOT / "outputs/valence_locked_musicnn_pair_blend_calibrated_20260714"
DEFAULT_OUT = ROOT / "outputs/valence_locked_monotonic_song_calibration_20260714"


def macro(rows: list[dict[str, float]]) -> dict[str, float]:
    return {key: float(np.mean([row[key] for row in rows])) for key in ["rmse", "mae", "pcc", "ccc"]}


def songs(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.groupby("musicId", sort=False).agg(
        pred=("pred_valence", "mean"), target=("target_valence", "mean")
    ).reset_index()


def fit_predict(method: str, train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    x = train["pred"].to_numpy(dtype=np.float64)
    y = train["target"].to_numpy(dtype=np.float64)
    xt = test["pred"].to_numpy(dtype=np.float64)
    if method == "isotonic":
        model = IsotonicRegression(y_min=-1.0, y_max=1.0, out_of_bounds="clip")
        return model.fit(x, y).predict(xt)
    if method.startswith("quantile_"):
        count = int(method.split("_")[1])
        q = np.linspace(0.0, 1.0, count)
        xq = np.quantile(x, q)
        yq = np.quantile(y, q)
        unique, index = np.unique(xq, return_index=True)
        return np.interp(xt, unique, yq[index], left=yq[index][0], right=yq[index][-1])
    if method.startswith("poly_"):
        degree = int(method.split("_")[1])
        model = make_pipeline(PolynomialFeatures(degree=degree, include_bias=False), StandardScaler(), Ridge(alpha=10.0))
        return model.fit(x[:, None], y).predict(xt[:, None])
    if method.startswith("spline_"):
        knots = int(method.split("_")[1])
        model = make_pipeline(
            SplineTransformer(n_knots=knots, degree=2, include_bias=False),
            StandardScaler(), Ridge(alpha=10.0),
        )
        return model.fit(x[:, None], y).predict(xt[:, None])
    raise ValueError(method)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    frames = {fold: pd.read_csv(args.base_dir / f"{fold}_val_predictions.csv") for fold in FOLDS}
    song_tables = {fold: songs(frames[fold]) for fold in FOLDS}
    base_rows = [metrics(frames[f]["pred_valence"], frames[f]["target_valence"]) for f in FOLDS]
    base_macro = macro(base_rows)
    methods = ["isotonic", "quantile_11", "quantile_21", "quantile_41", "poly_2", "poly_3", "spline_4", "spline_5", "spline_7"]
    raw: dict[str, dict[str, np.ndarray]] = {}
    for method in methods:
        raw[method] = {}
        for held in FOLDS:
            train = pd.concat([song_tables[f] for f in FOLDS if f != held], ignore_index=True)
            raw[method][held] = fit_predict(method, train, song_tables[held]) - song_tables[held]["pred"].to_numpy()

    rows = [{"method": "identity", "strength": 0.0, "cap": 0.0, **base_macro, "guarded": True}]
    cache: dict[tuple[str, float, float], dict[str, np.ndarray]] = {}
    for method in methods:
        for strength in [0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.70, 1.0]:
            for cap in [0.003, 0.005, 0.0075, 0.010, 0.015, 0.020, 0.030, 0.040, 0.060, 0.080, 0.120, 0.200]:
                per_fold = []
                preds: dict[str, np.ndarray] = {}
                for fold in FOLDS:
                    shift_map = dict(zip(song_tables[fold]["musicId"].astype(int), raw[method][fold]))
                    shift = frames[fold]["musicId"].astype(int).map(shift_map).to_numpy(dtype=np.float64)
                    base = frames[fold]["pred_valence"].to_numpy(dtype=np.float64)
                    pred = np.clip(base + np.clip(strength * shift, -cap, cap), -1.0, 1.0)
                    preds[fold] = pred
                    per_fold.append(metrics(pred, frames[fold]["target_valence"].to_numpy(dtype=np.float64)))
                result = macro(per_fold)
                guarded = (
                    result["mae"] <= base_macro["mae"] + 0.0005
                    and result["pcc"] >= base_macro["pcc"] - 0.0005
                    and result["ccc"] >= 0.80
                )
                rows.append({"method": method, "strength": strength, "cap": cap, **result, "guarded": guarded})
                cache[(method, strength, cap)] = preds

    table = pd.DataFrame(rows).sort_values(["rmse", "mae", "ccc"], ascending=[True, True, False])
    table.to_csv(args.out_dir / "candidates.csv", index=False)
    eligible = table[table["guarded"]]
    selected = (eligible if not eligible.empty else table).iloc[0].to_dict()
    if selected["method"] == "identity":
        selected_preds = {fold: frames[fold]["pred_valence"].to_numpy(dtype=np.float64) for fold in FOLDS}
    else:
        selected_preds = cache[(str(selected["method"]), float(selected["strength"]), float(selected["cap"]))]
    per_fold_rows = []
    for fold in FOLDS:
        out = frames[fold].copy().rename(columns={"pred_valence": "pre_monotonic_pred_valence"})
        out["pred_valence"] = selected_preds[fold]
        out.to_csv(args.out_dir / f"{fold}_val_predictions.csv", index=False)
        per_fold_rows.append({"fold": fold, **metrics(out["pred_valence"], out["target_valence"])})
    selected_macro = macro(per_fold_rows)
    summary = {
        "created_at": "2026-07-14",
        "selection_boundary": "Each held development fold uses a song-mean mapping fit only on the other two folds; final locked test excluded",
        "base_dir": str(args.base_dir), "base_macro": base_macro,
        "selected": selected, "macro": selected_macro, "per_fold": per_fold_rows,
        "delta_vs_base": {key: selected_macro[key] - base_macro[key] for key in base_macro},
        "gap_rows": threshold_rows(selected_macro),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
