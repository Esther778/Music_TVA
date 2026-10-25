#!/usr/bin/env python3
"""Low-variance OpenSMILE Ridge baselines on train+val development folds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from search_pmemo_valence_dev_val_postprocess import fit_calibration, moving_average_by_song
from train_pmemo_mert_valence_baseline import ROOT, metrics, threshold_rows


DEFAULT_FOLDS = ROOT / "data" / "splits" / "pmemo_trainval_devfolds_20260709"
DEFAULT_OUT = ROOT / "outputs" / "opensmile_ridge_devfolds_20260709"


def load_frames(index: pd.DataFrame, split: str, target: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    target_idx = 0 if target == "valence" else 1
    xs, ys, songs, times = [], [], [], []
    for row in index[index["split"] == split].sort_values("musicId").to_dict("records"):
        data = np.load(row["path"])
        x = data["X"].astype(np.float32)
        y = data["Y"].astype(np.float32)[:, target_idx]
        n = min(len(x), len(y), len(data["time_sec"]))
        xs.append(x[:n])
        ys.append(y[:n])
        songs.append(np.full(n, int(row["musicId"]), dtype=np.int32))
        times.append(data["time_sec"].astype(np.float32)[:n])
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(songs), np.concatenate(times)


def score(m: dict[str, float]) -> float:
    return (
        m["rmse"]
        + 0.015 * max(0.0, m["mae"] - 0.15)
        + 0.03 * max(0.0, 0.80 - m["ccc"])
        + 0.02 * max(0.0, 0.80 - m["pcc"])
    )


def postprocess_search(pred: np.ndarray, target: np.ndarray, song: np.ndarray) -> tuple[dict[str, Any], pd.DataFrame]:
    rows = []
    best: dict[str, Any] = {"postprocess": "raw", "metrics": metrics(pred, target), "score": score(metrics(pred, target))}
    for window in [1, 3, 5, 7, 9, 11, 15, 21, 31, 41]:
        smoothed = moving_average_by_song(pred, song, window)
        for smooth_blend in [0.25, 0.50, 0.75, 1.0]:
            base = (1.0 - smooth_blend) * pred + smooth_blend * smoothed
            for method in ["identity", "bias", "linear", "mean_std"]:
                calibrated, params = fit_calibration(base, target, method)
                for cal_blend in [0.25, 0.50, 0.75, 1.0]:
                    pp = (1.0 - cal_blend) * base + cal_blend * calibrated
                    m = metrics(pp, target)
                    row = {
                        "postprocess": f"w{window}_sb{smooth_blend:g}_{method}_cb{cal_blend:g}",
                        "window": window,
                        "smooth_blend": smooth_blend,
                        "method": method,
                        "cal_blend": cal_blend,
                        "params": json.dumps(params, ensure_ascii=False),
                        **m,
                    }
                    row["score"] = score(m)
                    rows.append(row)
                    if row["score"] < best["score"]:
                        best = {"postprocess": row["postprocess"], "metrics": m, "score": row["score"], **row}
    return best, pd.DataFrame(rows)


def run_fold(fold_dir: Path, out_dir: Path, target: str, alphas: list[float]) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    index = pd.read_csv(fold_dir / "sequence_index.csv")
    x_train, y_train, _, _ = load_frames(index, "train", target)
    x_val, y_val, song_val, time_val = load_frames(index, "val", target)
    rows = []
    best: dict[str, Any] | None = None
    for alpha in alphas:
        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        model.fit(x_train, y_train)
        pred = model.predict(x_val).astype(np.float32)
        raw_m = metrics(pred, y_val)
        post, pp_rows = postprocess_search(pred, y_val, song_val)
        row = {
            "alpha": alpha,
            **{f"raw_{k}": v for k, v in raw_m.items()},
            "best_postprocess": post["postprocess"],
            **{f"post_{k}": v for k, v in post["metrics"].items()},
            "post_score": post["score"],
        }
        rows.append(row)
        if best is None or row["post_score"] < best["row"]["post_score"]:
            best = {"row": row, "model": model, "post": post, "pp_rows": pp_rows, "pred": pred}
    assert best is not None
    pd.DataFrame(rows).to_csv(out_dir / "alpha_comparison.csv", index=False)
    best["pp_rows"].to_csv(out_dir / "postprocess_candidates.csv", index=False)
    joblib.dump(best["model"], out_dir / "best_ridge.joblib")
    pred_df = pd.DataFrame(
        {
            "musicId": song_val,
            "time_sec": time_val,
            f"target_{target}": y_val,
            f"pred_{target}_raw": best["pred"],
        }
    )
    pred_df.to_csv(out_dir / "val_predictions.csv", index=False)
    summary = {
        "fold": fold_dir.name,
        "target": target,
        "feature": "PMEmo official OpenSMILE dynamic features",
        "model": "StandardScaler + Ridge",
        "selection": "alpha and postprocess selected on this development fold validation only",
        "best": best["row"],
        "gap_rows": threshold_rows(
            {
                "rmse": float(best["row"]["post_rmse"]),
                "mae": float(best["row"]["post_mae"]),
                "pcc": float(best["row"]["post_pcc"]),
                "ccc": float(best["row"]["post_ccc"]),
            }
        ),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds-dir", type=Path, default=DEFAULT_FOLDS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--target", choices=["valence", "arousal", "both"], default="both")
    parser.add_argument("--alphas", nargs="*", type=float, default=[1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0])
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    targets = ["valence", "arousal"] if args.target == "both" else [args.target]
    summaries = []
    for target in targets:
        for fold_dir in sorted(args.folds_dir.glob("fold_*")):
            summaries.append(run_fold(fold_dir, args.out_dir / target / fold_dir.name, target, args.alphas))
    rows = []
    for s in summaries:
        row = {"target": s["target"], "fold": s["fold"], **s["best"]}
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(args.out_dir / "comparison.csv", index=False)
    aggregate = {
        "created_at": "2026-07-09",
        "purpose": "Low-variance Ridge development baseline on train+val folds; holdout/test excluded",
        "selection_boundary": "Development only. Do not claim blind/test performance.",
        "per_fold": rows,
        "aggregate": {},
    }
    for target in targets:
        sub = df[df["target"] == target]
        aggregate["aggregate"][target] = {
            key: float(sub[key].mean())
            for key in ["post_rmse", "post_mae", "post_pcc", "post_ccc", "raw_rmse", "raw_mae", "raw_pcc", "raw_ccc"]
        }
    (args.out_dir / "summary.json").write_text(json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(aggregate, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
