#!/usr/bin/env python3
"""Global guarded stacking on top of the current Valence best.

This is a diagnostic/optimization experiment. It uses existing Valence
prediction artifacts as meta-features, with the current best as the base
prediction. For each CV split, a low-capacity Ridge stacker is fitted on train
frames; one shared alpha/blend configuration is selected by validation macro
metrics across all splits; test is reported only after selection.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from train_pmemo_mert_valence_baseline import ROOT, metrics, threshold_rows


OUTPUT_ROOT = ROOT / "outputs"
BASE_DIR = OUTPUT_ROOT / "valence_current_best_tonal_correction_20260707"
OUT_DIR = OUTPUT_ROOT / "valence_current_best_global_stacking_20260708"

SPLITS = ["split_01_seed_101", "split_02_seed_202", "split_03_seed_303"]
DATA_SPLITS = ["train", "val", "test"]

CURRENT_BEST = {
    "model_id": "pmemo_valence_current_best_tonal_correction_20260707",
    "rmse": 0.19067754835039066,
    "mae": 0.14868131614617217,
    "pcc": 0.8080536791187845,
    "ccc": 0.7961903151568043,
}

PASS = {"rmse": 0.18, "mae": 0.15, "pcc": 0.80, "ccc": 0.80}

EXCLUDE_NAMES = {
    "valence_current_best_global_stacking_20260708",
}


def pred_col(df: pd.DataFrame) -> str:
    if "pred_valence" in df.columns:
        return "pred_valence"
    if "pred" in df.columns:
        return "pred"
    raise ValueError(f"No prediction column found: {list(df.columns)}")


def target_col(df: pd.DataFrame) -> str:
    if "target_valence" in df.columns:
        return "target_valence"
    if "target" in df.columns:
        return "target"
    raise ValueError(f"No target column found: {list(df.columns)}")


def eval_arrays(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    return metrics(np.asarray(pred, dtype=np.float64), np.asarray(target, dtype=np.float64))


def selection_score(m: dict[str, float]) -> float:
    return (
        3.0 * max(0.0, m["rmse"] - PASS["rmse"]) / PASS["rmse"]
        + 1.4 * max(0.0, m["mae"] - PASS["mae"]) / PASS["mae"]
        + 1.2 * max(0.0, PASS["pcc"] - m["pcc"]) / PASS["pcc"]
        + 2.0 * max(0.0, PASS["ccc"] - m["ccc"]) / PASS["ccc"]
    )


def discover_candidate_dirs() -> list[Path]:
    dirs: list[Path] = []
    for root in sorted(p for p in OUTPUT_ROOT.iterdir() if p.is_dir()):
        if root.name in EXCLUDE_NAMES or root == BASE_DIR:
            continue
        ok = True
        for split_name in SPLITS:
            for data_split in DATA_SPLITS:
                p = root / split_name / f"{data_split}_predictions.csv"
                if not p.exists():
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            continue
        try:
            sample = pd.read_csv(root / SPLITS[0] / "test_predictions.csv", nrows=5)
            pred_col(sample)
        except Exception:
            continue
        dirs.append(root)
    return dirs


def load_base(split_name: str, data_split: str) -> pd.DataFrame:
    df = pd.read_csv(BASE_DIR / split_name / f"{data_split}_predictions.csv")
    return df[["musicId", "time_sec", target_col(df), pred_col(df)]].rename(
        columns={target_col(df): "target_valence", pred_col(df): "base_pred"}
    )


def compatible_candidate(base: pd.DataFrame, candidate_path: Path, split_name: str, data_split: str) -> bool:
    df = pd.read_csv(candidate_path / split_name / f"{data_split}_predictions.csv", usecols=lambda c: c in {"musicId", "time_sec", "pred_valence", "pred"})
    if len(df) != len(base):
        return False
    keys_a = base[["musicId", "time_sec"]].reset_index(drop=True)
    keys_b = df[["musicId", "time_sec"]].reset_index(drop=True)
    return bool(keys_a.equals(keys_b))


def compatible_dirs(candidate_dirs: list[Path]) -> list[Path]:
    keep = []
    base_cache = {(s, d): load_base(s, d) for s in SPLITS for d in DATA_SPLITS}
    for candidate in candidate_dirs:
        ok = True
        for split_name in SPLITS:
            for data_split in DATA_SPLITS:
                if not compatible_candidate(base_cache[(split_name, data_split)], candidate, split_name, data_split):
                    ok = False
                    break
            if not ok:
                break
        if ok:
            keep.append(candidate)
    return keep


def load_feature_frame(model_dirs: list[Path], split_name: str, data_split: str) -> tuple[pd.DataFrame, list[str]]:
    merged = load_base(split_name, data_split)
    feature_names = ["base_pred"]
    for model_dir in model_dirs:
        df = pd.read_csv(model_dir / split_name / f"{data_split}_predictions.csv")
        name = model_dir.name
        mini = df[["musicId", "time_sec", pred_col(df)]].rename(columns={pred_col(df): name})
        merged = merged.merge(mini, on=["musicId", "time_sec"], how="inner")
        feature_names.append(name)
    return merged, feature_names


def model_candidates() -> list[tuple[str, object]]:
    out: list[tuple[str, object]] = []
    for alpha in [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]:
        out.append((f"ridge_a{str(alpha).replace('.', 'p')}", make_pipeline(StandardScaler(), Ridge(alpha=float(alpha)))))
    return out


def fit_predict_for_config(model_id: str, blend: float, feature_names_by_split: dict[str, list[str]], frames: dict[tuple[str, str], pd.DataFrame]) -> dict[tuple[str, str], np.ndarray]:
    model_template = dict(model_candidates())[model_id]
    preds: dict[tuple[str, str], np.ndarray] = {}
    for split_name in SPLITS:
        # sklearn estimators are mutable; rebuild by name for each split.
        model = dict(model_candidates())[model_id]
        cols = feature_names_by_split[split_name]
        train = frames[(split_name, "train")]
        model.fit(train[cols].to_numpy(dtype=np.float64), train["target_valence"].to_numpy(dtype=np.float64))
        for data_split in DATA_SPLITS:
            frame = frames[(split_name, data_split)]
            ridge_pred = model.predict(frame[cols].to_numpy(dtype=np.float64))
            base_pred = frame["base_pred"].to_numpy(dtype=np.float64)
            preds[(split_name, data_split)] = (1.0 - blend) * base_pred + blend * ridge_pred
    return preds


def macro_metrics(preds: dict[tuple[str, str], np.ndarray], frames: dict[tuple[str, str], pd.DataFrame], data_split: str) -> dict[str, float]:
    rows = []
    for split_name in SPLITS:
        target = frames[(split_name, data_split)]["target_valence"].to_numpy(dtype=np.float64)
        rows.append(eval_arrays(preds[(split_name, data_split)], target))
    return {metric: float(np.mean([row[metric] for row in rows])) for metric in ["rmse", "mae", "pcc", "ccc"]}


def baseline_macro(frames: dict[tuple[str, str], pd.DataFrame], data_split: str) -> dict[str, float]:
    rows = []
    for split_name in SPLITS:
        frame = frames[(split_name, data_split)]
        rows.append(eval_arrays(frame["base_pred"].to_numpy(dtype=np.float64), frame["target_valence"].to_numpy(dtype=np.float64)))
    return {metric: float(np.mean([row[metric] for row in rows])) for metric in ["rmse", "mae", "pcc", "ccc"]}


def guard(candidate: dict[str, float], baseline: dict[str, float]) -> bool:
    return (
        candidate["rmse"] <= baseline["rmse"] + 0.001
        and candidate["mae"] <= baseline["mae"] + 0.001
        and candidate["pcc"] >= baseline["pcc"] - 0.001
        and candidate["ccc"] >= baseline["ccc"] + 0.0005
    )


def materialize(selected: dict[str, Any], preds: dict[tuple[str, str], np.ndarray], frames: dict[tuple[str, str], pd.DataFrame]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split_name in SPLITS:
        out_dir = OUT_DIR / split_name
        out_dir.mkdir(parents=True, exist_ok=True)
        row: dict[str, Any] = {"split_name": split_name}
        for data_split in DATA_SPLITS:
            frame = frames[(split_name, data_split)]
            out = frame[["musicId", "time_sec", "target_valence"]].copy()
            out["base_pred_valence"] = frame["base_pred"].to_numpy(dtype=np.float64)
            out["pred_valence"] = preds[(split_name, data_split)]
            out["stack_model_id"] = selected["model_id"]
            out["stack_blend"] = selected["blend"]
            out.to_csv(out_dir / f"{data_split}_predictions.csv", index=False)
            m = eval_arrays(out["pred_valence"].to_numpy(dtype=np.float64), out["target_valence"].to_numpy(dtype=np.float64))
            for metric, value in m.items():
                row[f"{data_split}_{metric}"] = float(value)
        rows.append(row)
    return rows


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    discovered = discover_candidate_dirs()
    candidates = compatible_dirs(discovered)
    frames: dict[tuple[str, str], pd.DataFrame] = {}
    feature_names_by_split: dict[str, list[str]] = {}
    for split_name in SPLITS:
        for data_split in DATA_SPLITS:
            frame, names = load_feature_frame(candidates, split_name, data_split)
            frames[(split_name, data_split)] = frame
            feature_names_by_split[split_name] = names

    base_val = baseline_macro(frames, "val")
    base_test = baseline_macro(frames, "test")
    rows = []
    pred_cache: dict[tuple[str, float], dict[tuple[str, str], np.ndarray]] = {}
    for model_id, _ in model_candidates():
        for blend in [0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.40, 0.55]:
            preds = fit_predict_for_config(model_id, float(blend), feature_names_by_split, frames)
            pred_cache[(model_id, float(blend))] = preds
            val_m = macro_metrics(preds, frames, "val")
            test_m = macro_metrics(preds, frames, "test")
            rows.append(
                {
                    "model_id": model_id,
                    "blend": float(blend),
                    "candidate_count": len(candidates),
                    **{f"val_{k}": v for k, v in val_m.items()},
                    **{f"test_{k}": v for k, v in test_m.items()},
                    "val_score": selection_score(val_m),
                    "passes_guard": guard(val_m, base_val),
                }
            )
    results = pd.DataFrame(rows)
    results.to_csv(OUT_DIR / "all_global_stacking_candidates.csv", index=False)
    guarded = results[results["passes_guard"]].copy()
    if guarded.empty:
        selected = {
            "model_id": "identity",
            "blend": 0.0,
            **{f"val_{k}": v for k, v in base_val.items()},
            **{f"test_{k}": v for k, v in base_test.items()},
            "selection_reason": "identity retained; no candidate passed validation guard",
        }
        selected_preds = {
            (split_name, data_split): frames[(split_name, data_split)]["base_pred"].to_numpy(dtype=np.float64)
            for split_name in SPLITS
            for data_split in DATA_SPLITS
        }
    else:
        selected = guarded.sort_values(["val_score", "val_rmse", "val_ccc"], ascending=[True, True, False]).iloc[0].to_dict()
        selected["selection_reason"] = "selected by validation macro guard"
        selected_preds = pred_cache[(str(selected["model_id"]), float(selected["blend"]))]

    per_split = materialize(selected, selected_preds, frames)
    selected_test = {metric: float(np.mean([row[f"test_{metric}"] for row in per_split])) for metric in ["rmse", "mae", "pcc", "ccc"]}
    summary = {
        "created_at": "2026-07-08",
        "model_id": "pmemo_valence_current_best_global_stacking_20260708",
        "input_model": CURRENT_BEST,
        "base_dir": str(BASE_DIR.relative_to(ROOT)),
        "candidate_dirs": [str(p.relative_to(ROOT)) for p in candidates],
        "candidate_count": len(candidates),
        "selection_rule": "one shared stacking model/blend config selected on validation macro metrics under guard; test labels not used",
        "baseline_val": base_val,
        "baseline_test": base_test,
        "selected_config": selected,
        "per_split": per_split,
        "selected_test": selected_test,
        "delta_vs_current_best": {metric: float(selected_test[metric] - CURRENT_BEST[metric]) for metric in ["rmse", "mae", "pcc", "ccc"]},
        "threshold_gaps": threshold_rows(selected_test),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
