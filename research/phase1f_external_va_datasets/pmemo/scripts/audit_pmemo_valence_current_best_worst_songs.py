#!/usr/bin/env python3
"""Audit worst-song failure modes for the current best PMEmo Valence model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PRED_DIR = ROOT / "outputs" / "valence_prediction_only_rescale_ccc_preserving_20260628"
META_PATH = ROOT / "data/raw/PMEmo_2019_updated/extracted/PMEmo2019/metadata.csv"
STATIC_PATH = ROOT / "data/raw/PMEmo_2019_updated/extracted/PMEmo2019/annotations/static_annotations.csv"
OUT_DIR = ROOT / "outputs" / "valence_current_best_worst_song_audit_20260628"


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def ccc(x: np.ndarray, y: np.ndarray) -> float:
    mean_x = float(np.mean(x))
    mean_y = float(np.mean(y))
    var_x = float(np.var(x))
    var_y = float(np.var(y))
    cov = float(np.mean((x - mean_x) * (y - mean_y)))
    denom = var_x + var_y + (mean_x - mean_y) ** 2
    return float(2.0 * cov / denom) if denom > 1e-12 else 0.0


def metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    err = pred - target
    return {
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "pcc": pearson(pred, target),
        "ccc": ccc(pred, target),
    }


def gap_rows(current: dict[str, float]) -> list[dict[str, Any]]:
    thresholds = {
        "rmse": {"direction": "lower", "pass": 0.18, "good": 0.15, "excellent": 0.10},
        "mae": {"direction": "lower", "pass": 0.15, "good": 0.12, "excellent": 0.08},
        "pcc": {"direction": "higher", "pass": 0.80, "good": 0.85, "excellent": 0.90},
        "ccc": {"direction": "higher", "pass": 0.80, "good": 0.85, "excellent": 0.90},
    }
    rows: list[dict[str, Any]] = []
    for metric, cfg in thresholds.items():
        value = float(current[metric])
        if cfg["direction"] == "lower":
            status = (
                "excellent"
                if value <= cfg["excellent"]
                else "good"
                if value <= cfg["good"]
                else "pass"
                if value <= cfg["pass"]
                else "not_pass"
            )
            gap_to_pass = max(0.0, value - cfg["pass"])
            gap_to_good = max(0.0, value - cfg["good"])
            gap_to_excellent = max(0.0, value - cfg["excellent"])
        else:
            status = (
                "excellent"
                if value >= cfg["excellent"]
                else "good"
                if value >= cfg["good"]
                else "pass"
                if value >= cfg["pass"]
                else "not_pass"
            )
            gap_to_pass = max(0.0, cfg["pass"] - value)
            gap_to_good = max(0.0, cfg["good"] - value)
            gap_to_excellent = max(0.0, cfg["excellent"] - value)
        rows.append(
            {
                "metric": metric,
                "current": value,
                "minimum_pass": cfg["pass"],
                "good": cfg["good"],
                "excellent": cfg["excellent"],
                "gap_to_pass": gap_to_pass,
                "gap_to_good": gap_to_good,
                "gap_to_excellent": gap_to_excellent,
                "status": status,
            }
        )
    return rows


def read_metadata() -> pd.DataFrame:
    meta = pd.read_csv(META_PATH)
    static = pd.read_csv(STATIC_PATH)
    static = static.rename(
        columns={"Valence(mean)": "static_valence_0_1", "Arousal(mean)": "static_arousal_0_1"}
    )
    static["static_valence_minus1_1"] = static["static_valence_0_1"] * 2.0 - 1.0
    static["static_arousal_minus1_1"] = static["static_arousal_0_1"] * 2.0 - 1.0
    out = meta.merge(static, on="musicId", how="left")
    return out


def read_split_predictions(split_dir: Path) -> pd.DataFrame:
    frames = []
    for subset in ["train", "val", "test"]:
        path = split_dir / f"{subset}_predictions.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df["subset"] = subset
        df["split_name"] = split_dir.name
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No predictions found in {split_dir}")
    out = pd.concat(frames, ignore_index=True)
    out["musicId"] = out["musicId"].astype(int)
    out["time_sec"] = out["time_sec"].astype(float)
    return out


def classify_error(row: pd.Series) -> str:
    types = []
    if bool(row["polarity_mismatch"]):
        types.append("polarity_mismatch")
    if row["abs_mean_error"] >= 0.12:
        types.append("global_mean_bias")
    if row["std_ratio"] < 0.55 and row["target_std"] >= 0.04:
        types.append("flat_prediction")
    if row["target_range"] >= 0.55 and row["rmse"] >= 0.20:
        types.append("high_dynamic_range")
    if row["pcc"] < 0.30 and row["frames"] >= 20:
        types.append("trend_failure")
    return "|".join(types) if types else "ordinary_error"


def per_song_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (split_name, subset, sid), g in df.groupby(["split_name", "subset", "musicId"]):
        pred = g["pred_valence"].to_numpy(dtype=np.float64)
        target = g["target_valence"].to_numpy(dtype=np.float64)
        m = metrics(pred, target)
        target_std = float(np.std(target))
        pred_std = float(np.std(pred))
        target_mean = float(np.mean(target))
        pred_mean = float(np.mean(pred))
        target_range = float(np.max(target) - np.min(target))
        pred_range = float(np.max(pred) - np.min(pred))
        polarity_mismatch = (
            abs(target_mean) >= 0.05
            and abs(pred_mean) >= 0.05
            and np.sign(target_mean) != np.sign(pred_mean)
        )
        row = {
            "split_name": split_name,
            "subset": subset,
            "musicId": int(sid),
            "frames": int(len(g)),
            "duration_sec": float(len(g) * 0.5),
            "target_mean": target_mean,
            "pred_mean": pred_mean,
            "mean_error": float(pred_mean - target_mean),
            "abs_mean_error": float(abs(pred_mean - target_mean)),
            "target_std": target_std,
            "pred_std": pred_std,
            "std_ratio": float(pred_std / (target_std + 1e-12)),
            "target_min": float(np.min(target)),
            "target_max": float(np.max(target)),
            "target_range": target_range,
            "pred_min": float(np.min(pred)),
            "pred_max": float(np.max(pred)),
            "pred_range": pred_range,
            "polarity_mismatch": bool(polarity_mismatch),
            **m,
        }
        rows.append(row)
    out = pd.DataFrame(rows)
    out["error_type"] = out.apply(classify_error, axis=1)
    return out


def subset_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (split_name, subset), g in df.groupby(["split_name", "subset"]):
        pred = g["pred_valence"].to_numpy(dtype=np.float64)
        target = g["target_valence"].to_numpy(dtype=np.float64)
        m = metrics(pred, target)
        rows.append(
            {
                "split_name": split_name,
                "subset": subset,
                "songs": int(g["musicId"].nunique()),
                "frames": int(len(g)),
                **m,
            }
        )
    return pd.DataFrame(rows).sort_values(["split_name", "subset"])


def macro_test_metrics(subset_df: pd.DataFrame) -> dict[str, float]:
    test = subset_df[subset_df["subset"] == "test"]
    return {
        "rmse": float(test["rmse"].mean()),
        "mae": float(test["mae"].mean()),
        "pcc": float(test["pcc"].mean()),
        "ccc": float(test["ccc"].mean()),
    }


def safe_corr(df: pd.DataFrame, a: str, b: str) -> float:
    work = df[[a, b]].dropna()
    if len(work) < 3:
        return 0.0
    return pearson(work[a].to_numpy(dtype=np.float64), work[b].to_numpy(dtype=np.float64))


def correlation_table(per_song: pd.DataFrame) -> pd.DataFrame:
    test = per_song[per_song["subset"] == "test"].copy()
    fields = [
        "abs_mean_error",
        "target_mean",
        "target_std",
        "target_range",
        "pred_std",
        "std_ratio",
        "frames",
        "static_valence_minus1_1",
        "abs_static_dynamic_gap",
    ]
    rows = []
    for split_name, g in test.groupby("split_name"):
        for field in fields:
            rows.append(
                {
                    "scope": split_name,
                    "feature": field,
                    "corr_with_song_rmse": safe_corr(g, field, "rmse"),
                }
            )
    for field in fields:
        rows.append(
            {
                "scope": "all_test_splits",
                "feature": field,
                "corr_with_song_rmse": safe_corr(test, field, "rmse"),
            }
        )
    return pd.DataFrame(rows)


def error_type_table(per_song: pd.DataFrame) -> pd.DataFrame:
    rows = []
    test = per_song[per_song["subset"] == "test"].copy()
    for scope, g in [("all_test_splits", test)] + list(test.groupby("split_name")):
        if isinstance(scope, tuple):
            scope = scope[0]
        expanded: list[str] = []
        for item in g["error_type"]:
            expanded.extend(str(item).split("|"))
        vc = pd.Series(expanded).value_counts()
        for error_type, count in vc.items():
            rows.append(
                {
                    "scope": scope,
                    "error_type": error_type,
                    "count": int(count),
                    "song_rows": int(len(g)),
                    "rate": float(count / max(len(g), 1)),
                }
            )
    return pd.DataFrame(rows).sort_values(["scope", "count"], ascending=[True, False])


def top_records(df: pd.DataFrame, n: int = 10) -> list[dict[str, Any]]:
    cols = [
        "split_name",
        "musicId",
        "title",
        "artist",
        "rmse",
        "mae",
        "pcc",
        "ccc",
        "target_mean",
        "pred_mean",
        "abs_mean_error",
        "target_std",
        "pred_std",
        "std_ratio",
        "target_range",
        "pred_range",
        "error_type",
    ]
    return df.sort_values("rmse", ascending=False).head(n)[cols].to_dict("records")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    meta = read_metadata()
    split_frames = []
    for split_dir in sorted(PRED_DIR.glob("split_*_seed_*")):
        split_frames.append(read_split_predictions(split_dir))
    frame_df = pd.concat(split_frames, ignore_index=True)

    subset_df = subset_summary(frame_df)
    per_song = per_song_metrics(frame_df)
    per_song = per_song.merge(meta, on="musicId", how="left")
    per_song["abs_static_dynamic_gap"] = (
        per_song["static_valence_minus1_1"] - per_song["target_mean"]
    ).abs()

    test_per_song = per_song[per_song["subset"] == "test"].copy()
    split02 = test_per_song[test_per_song["split_name"] == "split_02_seed_202"].copy()

    macro = macro_test_metrics(subset_df)
    gaps = gap_rows(macro)
    corr = correlation_table(per_song)
    error_types = error_type_table(per_song)

    subset_df.to_csv(OUT_DIR / "subset_frame_metrics.csv", index=False)
    per_song.sort_values(["subset", "split_name", "rmse"], ascending=[True, True, False]).to_csv(
        OUT_DIR / "per_song_metrics.csv", index=False
    )
    test_per_song.sort_values("rmse", ascending=False).to_csv(
        OUT_DIR / "worst_songs_overall.csv", index=False
    )
    split02.sort_values("rmse", ascending=False).to_csv(
        OUT_DIR / "split_02_worst_songs.csv", index=False
    )
    corr.to_csv(OUT_DIR / "correlations.csv", index=False)
    error_types.to_csv(OUT_DIR / "error_type_summary.csv", index=False)
    pd.DataFrame(gaps).to_csv(OUT_DIR / "gap_to_thresholds.csv", index=False)

    summary = {
        "created_at": "2026-06-28",
        "model_id": "pmemo_valence_prediction_only_rescale_ccc_preserving_20260628",
        "input_predictions": str(PRED_DIR.relative_to(ROOT)),
        "macro_test_metrics": macro,
        "gap_to_thresholds": gaps,
        "per_split_test_metrics": subset_df[subset_df["subset"] == "test"].to_dict("records"),
        "worst_10_overall": top_records(test_per_song, 10),
        "worst_10_split_02": top_records(split02, 10),
        "top_correlations_all_test": corr[corr["scope"] == "all_test_splits"]
        .sort_values("corr_with_song_rmse", key=lambda s: s.abs(), ascending=False)
        .to_dict("records"),
        "error_type_summary": error_types.to_dict("records"),
    }
    (OUT_DIR / "audit_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary["macro_test_metrics"], indent=2, ensure_ascii=False))
    print("Wrote", OUT_DIR)


if __name__ == "__main__":
    main()
