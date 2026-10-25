#!/usr/bin/env python3
"""Guarded prediction stacking for PMEmo Valence candidates.

This script uses already generated prediction CSVs from multiple Valence
candidate models. It fits a low-capacity Ridge stacker on train predictions,
selects alpha/blend on validation, and reports test after selection.

It does not retrain audio feature extractors or sequence models.
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
OUT_DIR = OUTPUT_ROOT / "valence_prediction_stacking_guarded_20260707"

BASE_MODEL_DIR = OUTPUT_ROOT / "valence_parsimony_output_guarded_20260707"
FALLBACK_BASE_MODEL_DIR = OUTPUT_ROOT / "valence_per_split_direct_multilayer_parsimony_20260629"

SPLITS = ["split_01_seed_101", "split_02_seed_202", "split_03_seed_303"]
DATA_SPLITS = ["train", "val", "test"]

CURRENT_BEST = {
    "model_id": "pmemo_valence_parsimony_output_guarded_20260707",
    "rmse": 0.19127265472577384,
    "mae": 0.14892050958155043,
    "pcc": 0.8060728304873086,
    "ccc": 0.7915010047864408,
}

PASS = {"rmse": 0.18, "mae": 0.15, "pcc": 0.80, "ccc": 0.80}

PREFERRED_MODELS = [
    "valence_parsimony_output_guarded_20260707",
    "valence_per_split_direct_multilayer_parsimony_20260629",
    "valence_per_split_direct_multilayer_ensemble_20260629",
    "valence_targeted_multilayer_postprocess_20260629",
    "valence_targeted_multilayer_global_ensemble_20260629",
    "valence_prediction_only_rescale_ccc_preserving_20260628",
    "valence_global_prediction_only_rescale_20260628",
    "valence_frame_calibrator_song_bias_20260628",
    "valence_regularized_stack_ensemble_20260628",
    "valence_multisplit_ensemble_ccc_preserving_20260627",
    "valence_multisplit_bilstm_tcn_ensemble_20260627",
    "valence_multisplit_best_20260627",
    "valence_multisplit_tcn_20260627",
    "valence_multisplit_full_librosa_20260628",
    "valence_multisplit_harm_energy_tempo_20260628",
    "valence_opensmile_aux_ensemble_20260629",
    "valence_oof_arousal_residual_20260628",
]


def pred_col(df: pd.DataFrame) -> str:
    if "pred_valence" in df.columns:
        return "pred_valence"
    if "pred" in df.columns:
        return "pred"
    raise ValueError(f"No prediction column found in {list(df.columns)}")


def target_col(df: pd.DataFrame) -> str:
    if "target_valence" in df.columns:
        return "target_valence"
    if "target" in df.columns:
        return "target"
    raise ValueError(f"No target column found in {list(df.columns)}")


def eval_arrays(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    return metrics(np.asarray(pred, dtype=np.float64), np.asarray(target, dtype=np.float64))


def selection_score(m: dict[str, float]) -> float:
    return (
        3.00 * max(0.0, m["rmse"] - PASS["rmse"]) / PASS["rmse"]
        + 1.25 * max(0.0, m["mae"] - PASS["mae"]) / PASS["mae"]
        + 1.25 * max(0.0, PASS["pcc"] - m["pcc"]) / PASS["pcc"]
        + 2.50 * max(0.0, PASS["ccc"] - m["ccc"]) / PASS["ccc"]
        + 0.05 * m["rmse"]
    )


def discover_model_dirs() -> list[Path]:
    dirs = []
    for name in PREFERRED_MODELS:
        path = OUTPUT_ROOT / name
        if not path.exists():
            continue
        ok = all((path / split / "train_predictions.csv").exists() for split in SPLITS)
        if ok:
            dirs.append(path)
    return dirs


def load_base_frame(base_dir: Path, split_name: str, data_split: str) -> pd.DataFrame:
    df = pd.read_csv(base_dir / split_name / f"{data_split}_predictions.csv")
    return df[["musicId", "time_sec", target_col(df), pred_col(df)]].rename(
        columns={target_col(df): "target_valence", pred_col(df): "base_pred"}
    )


def load_feature_matrix(model_dirs: list[Path], split_name: str, data_split: str) -> tuple[pd.DataFrame, list[str]]:
    base_dir = BASE_MODEL_DIR if BASE_MODEL_DIR.exists() else FALLBACK_BASE_MODEL_DIR
    merged = load_base_frame(base_dir, split_name, data_split)
    feature_names = ["base_pred"]
    for model_dir in model_dirs:
        df = pd.read_csv(model_dir / split_name / f"{data_split}_predictions.csv")
        col_name = model_dir.name
        mini = df[["musicId", "time_sec", pred_col(df)]].rename(columns={pred_col(df): col_name})
        before = len(merged)
        merged = merged.merge(mini, on=["musicId", "time_sec"], how="inner")
        if len(merged) != before:
            raise ValueError(f"Frame mismatch after merging {model_dir.name} {split_name} {data_split}: {before} -> {len(merged)}")
        if col_name not in feature_names:
            feature_names.append(col_name)
    return merged, feature_names


def split_metrics(preds: dict[str, np.ndarray], frames: dict[str, pd.DataFrame]) -> dict[str, dict[str, float]]:
    return {
        split: eval_arrays(preds[split], frames[split]["target_valence"].to_numpy(dtype=np.float64))
        for split in DATA_SPLITS
    }


def passes_guard(candidate: dict[str, float], baseline: dict[str, float], base_score: float, candidate_score: float) -> bool:
    if candidate_score > base_score - 0.0005:
        return False
    if candidate["rmse"] > baseline["rmse"] + 0.001:
        return False
    if candidate["ccc"] < baseline["ccc"] - 0.001:
        return False
    return True


def evaluate_split(split_name: str, model_dirs: list[Path]) -> tuple[dict[str, Any], pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    feature_names: list[str] | None = None
    for data_split in DATA_SPLITS:
        frame, names = load_feature_matrix(model_dirs, split_name, data_split)
        frames[data_split] = frame
        if feature_names is None:
            feature_names = names
        elif feature_names != names:
            raise ValueError("Feature name mismatch")
    assert feature_names is not None

    y_train = frames["train"]["target_valence"].to_numpy(dtype=np.float64)
    baseline_preds = {split: frames[split]["base_pred"].to_numpy(dtype=np.float64) for split in DATA_SPLITS}
    baseline_metrics = split_metrics(baseline_preds, frames)
    base_score = selection_score(baseline_metrics["val"])

    x_train = frames["train"][feature_names].to_numpy(dtype=np.float64)
    rows = []
    pred_cache: dict[tuple[float, float, bool], dict[str, np.ndarray]] = {}
    for alpha in [0.001, 0.01, 0.1, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0]:
        for fit_intercept in [True, False]:
            model = make_pipeline(StandardScaler(), Ridge(alpha=float(alpha), fit_intercept=fit_intercept))
            model.fit(x_train, y_train)
            ridge_preds = {
                split: model.predict(frames[split][feature_names].to_numpy(dtype=np.float64))
                for split in DATA_SPLITS
            }
            for blend in [0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.00]:
                preds = {
                    split: (1.0 - float(blend)) * baseline_preds[split] + float(blend) * ridge_preds[split]
                    for split in DATA_SPLITS
                }
                metrics_by_split = split_metrics(preds, frames)
                val_score = selection_score(metrics_by_split["val"])
                key = (float(alpha), float(blend), bool(fit_intercept))
                pred_cache[key] = preds
                rows.append(
                    {
                        "split_name": split_name,
                        "alpha": float(alpha),
                        "blend": float(blend),
                        "fit_intercept": bool(fit_intercept),
                        "feature_count": len(feature_names),
                        **{f"{split}_{metric}": float(value) for split, vals in metrics_by_split.items() for metric, value in vals.items()},
                        "val_selection_score": float(val_score),
                        "passes_guard": passes_guard(metrics_by_split["val"], baseline_metrics["val"], base_score, val_score),
                    }
                )

    df = pd.DataFrame(rows)
    guarded = df[df["passes_guard"]].sort_values(["val_selection_score", "val_rmse", "val_mae", "val_ccc"], ascending=[True, True, True, False])
    if guarded.empty:
        selected = {
            "split_name": split_name,
            "alpha": "identity",
            "blend": 0.0,
            "fit_intercept": "identity",
            "feature_count": len(feature_names),
            **{f"{split}_{metric}": float(value) for split, vals in baseline_metrics.items() for metric, value in vals.items()},
            "val_selection_score": float(base_score),
            "passes_guard": True,
            "selected_reason": "identity retained because no stacking candidate passed validation guard",
        }
        selected_preds = baseline_preds
    else:
        selected = guarded.iloc[0].to_dict()
        selected["selected_reason"] = "validation guard passed"
        key = (float(selected["alpha"]), float(selected["blend"]), bool(selected["fit_intercept"]))
        selected_preds = pred_cache[key]

    out_split = OUT_DIR / split_name
    out_split.mkdir(parents=True, exist_ok=True)
    for split in DATA_SPLITS:
        out = frames[split][["musicId", "time_sec", "target_valence"]].copy()
        out["pred_valence"] = selected_preds[split]
        out["stack_alpha"] = selected["alpha"]
        out["stack_blend"] = selected["blend"]
        out["stack_fit_intercept"] = selected["fit_intercept"]
        out.to_csv(out_split / f"{split}_predictions.csv", index=False)

    summary = {
        "split_name": split_name,
        "feature_names": feature_names,
        "baseline": baseline_metrics,
        "selected": selected,
        "delta_vs_input_test": {
            metric: float(selected[f"test_{metric}"] - baseline_metrics["test"][metric]) for metric in ["rmse", "mae", "pcc", "ccc"]
        },
    }
    (out_split / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary, df


def aggregate(rows: list[dict[str, Any]], model_dirs: list[Path]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "created_at": "2026-07-07",
        "model_id": "pmemo_valence_prediction_stacking_guarded_20260707",
        "purpose": "guarded Ridge stacking over existing Valence prediction candidates",
        "input_model": CURRENT_BEST,
        "candidate_models": [p.name for p in model_dirs],
        "selection_rule": "fit Ridge stacker on train predictions; choose alpha/blend on validation; keep identity unless validation guard passes",
        "num_splits": len(rows),
        "per_split": rows,
    }
    for variant in ["baseline", "selected"]:
        for metric in ["rmse", "mae", "pcc", "ccc"]:
            if variant == "selected":
                vals = np.asarray([float(r["selected"][f"test_{metric}"]) for r in rows], dtype=np.float64)
            else:
                vals = np.asarray([float(r["baseline"]["test"][metric]) for r in rows], dtype=np.float64)
            out[f"{variant}_test_{metric}_mean"] = float(vals.mean())
            out[f"{variant}_test_{metric}_std"] = float(vals.std(ddof=0))
    selected_mean = {metric: float(out[f"selected_test_{metric}_mean"]) for metric in ["rmse", "mae", "pcc", "ccc"]}
    baseline_mean = {metric: float(out[f"baseline_test_{metric}_mean"]) for metric in ["rmse", "mae", "pcc", "ccc"]}
    out["selected_mean_test_gap_rows"] = threshold_rows(selected_mean)
    out["selected_delta_vs_input_model"] = {metric: float(selected_mean[metric] - CURRENT_BEST[metric]) for metric in ["rmse", "mae", "pcc", "ccc"]}
    out["selected_delta_vs_baseline_recomputed"] = {metric: float(selected_mean[metric] - baseline_mean[metric]) for metric in ["rmse", "mae", "pcc", "ccc"]}
    return out


def run() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model_dirs = discover_model_dirs()
    if not model_dirs:
        raise RuntimeError("No candidate prediction directories found.")
    rows = []
    candidates = []
    for split_name in SPLITS:
        summary, df = evaluate_split(split_name, model_dirs)
        rows.append(summary)
        candidates.append(df)
    pd.concat(candidates, ignore_index=True).to_csv(OUT_DIR / "all_stacking_candidates.csv", index=False)
    summary = aggregate(rows, model_dirs)
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    run()
