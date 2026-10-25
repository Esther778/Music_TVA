#!/usr/bin/env python3
"""Nested dynamic MAEST correction on locked PMEmo development folds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from search_pmemo_valence_locked_effnet_moodtheme_tags import FOLDS, macro
from train_pmemo_mert_valence_baseline import ROOT, metrics, threshold_rows


DEFAULT_BASE = ROOT / "outputs/valence_locked_effnet_moodtheme_tags_balanced_final_20260714"
DEFAULT_MAEST = ROOT.parents[1] / "phase1g_valence_data_pretraining/outputs/pmemo_maest_30s_layer7_20260714"
DEFAULT_OUT = ROOT / "outputs/valence_locked_maest_dynamic_nested_20260714"


def interpolate_embeddings(frame: pd.DataFrame, maest_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    output = np.zeros((len(frame), 2304), dtype=np.float32)
    weights = np.zeros(len(frame), dtype=np.float32)
    for music_id, indices in frame.groupby("musicId", sort=False).groups.items():
        loc = np.asarray(indices, dtype=np.int64)
        embedding = np.load(maest_dir / "per_song" / f"{int(music_id)}.npz")["embedding"].astype(np.float32)
        source = np.linspace(0.0, 1.0, len(embedding), dtype=np.float32)
        target = np.linspace(0.0, 1.0, len(loc), dtype=np.float32)
        output[loc] = np.column_stack([
            np.interp(target, source, embedding[:, column])
            for column in range(embedding.shape[1])
        ])
        weights[loc] = 1.0 / len(loc)
    weights *= len(weights) / weights.sum()
    return output, weights


def selected_columns(x: np.ndarray, target: np.ndarray, top_k: int) -> np.ndarray:
    centered_x = x - x.mean(axis=0, keepdims=True)
    centered_y = target - target.mean()
    denominator = np.sqrt(np.square(centered_x).sum(axis=0) * np.square(centered_y).sum())
    correlation = np.divide(
        centered_x.T @ centered_y,
        denominator,
        out=np.zeros(x.shape[1], dtype=np.float64),
        where=denominator > 1e-12,
    )
    return np.argsort(np.abs(correlation))[-top_k:]


def fit_shift(
    train_folds: list[str],
    test_fold: str,
    frames: dict[str, pd.DataFrame],
    features: dict[str, np.ndarray],
    weights: dict[str, np.ndarray],
    target_mode: str,
    top_k: int,
    alpha: float,
) -> np.ndarray:
    x_train = np.concatenate([features[fold] for fold in train_folds])
    target = np.concatenate([
        frames[fold]["target_valence"].to_numpy(dtype=np.float64) for fold in train_folds
    ])
    base = np.concatenate([
        frames[fold]["pred_valence"].to_numpy(dtype=np.float64) for fold in train_folds
    ])
    fit_target = target - base if target_mode == "residual" else target
    columns = selected_columns(x_train, fit_target, top_k)
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    model.fit(
        x_train[:, columns],
        fit_target,
        ridge__sample_weight=np.concatenate([weights[fold] for fold in train_folds]),
    )
    estimate = model.predict(features[test_fold][:, columns])
    if target_mode == "direct":
        estimate -= frames[test_fold]["pred_valence"].to_numpy(dtype=np.float64)
    return estimate


def apply_shift(frame: pd.DataFrame, shift: np.ndarray, strength: float, cap: float) -> np.ndarray:
    base = frame["pred_valence"].to_numpy(dtype=np.float64)
    return np.clip(base + np.clip(strength * shift, -cap, cap), -1.0, 1.0)


def inner_select(
    train_folds: list[str],
    frames: dict[str, pd.DataFrame],
    features: dict[str, np.ndarray],
    weights: dict[str, np.ndarray],
) -> tuple[dict[str, Any], pd.DataFrame]:
    base_macro = macro([
        metrics(frames[fold]["pred_valence"], frames[fold]["target_valence"])
        for fold in train_folds
    ])
    rows: list[dict[str, Any]] = [{
        "target_mode": "identity", "top_k": 0, "alpha": 0.0,
        "strength": 0.0, "cap": 0.0, **base_macro, "guarded": True,
    }]
    for target_mode in ["direct", "residual"]:
        for top_k in [16, 32, 64]:
            for alpha in [10.0, 100.0, 1000.0]:
                shifts = {}
                for inner_held in train_folds:
                    inner_train = [fold for fold in train_folds if fold != inner_held]
                    shifts[inner_held] = fit_shift(
                        inner_train, inner_held, frames, features, weights,
                        target_mode, top_k, alpha,
                    )
                for strength in [0.05, 0.10, 0.20, 0.40]:
                    for cap in [0.003, 0.005, 0.010, 0.020]:
                        result = macro([
                            metrics(
                                apply_shift(frames[fold], shifts[fold], strength, cap),
                                frames[fold]["target_valence"],
                            )
                            for fold in train_folds
                        ])
                        guarded = (
                            result["mae"] <= base_macro["mae"]
                            and result["pcc"] >= base_macro["pcc"]
                            and result["ccc"] >= base_macro["ccc"]
                        )
                        rows.append({
                            "target_mode": target_mode, "top_k": top_k,
                            "alpha": alpha, "strength": strength, "cap": cap,
                            **result, "guarded": guarded,
                        })
    table = pd.DataFrame(rows).sort_values(["rmse", "mae", "ccc"], ascending=[True, True, False])
    eligible = table[table["guarded"]]
    selected = (eligible if not eligible.empty else table).iloc[0].to_dict()
    return selected, table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--maest-dir", type=Path, default=DEFAULT_MAEST)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    frames = {fold: pd.read_csv(args.base_dir / f"{fold}_val_predictions.csv") for fold in FOLDS}
    features, weights = {}, {}
    for fold in FOLDS:
        features[fold], weights[fold] = interpolate_embeddings(frames[fold], args.maest_dir)

    base_rows = [metrics(frames[fold]["pred_valence"], frames[fold]["target_valence"]) for fold in FOLDS]
    base_macro = macro(base_rows)
    per_fold = []
    selected_rows = []
    for held in FOLDS:
        train_folds = [fold for fold in FOLDS if fold != held]
        selected, table = inner_select(train_folds, frames, features, weights)
        table.to_csv(args.out_dir / f"{held}_inner_candidates.csv", index=False)
        if selected["target_mode"] == "identity":
            prediction = frames[held]["pred_valence"].to_numpy(dtype=np.float64)
        else:
            shift = fit_shift(
                train_folds, held, frames, features, weights,
                str(selected["target_mode"]), int(selected["top_k"]), float(selected["alpha"]),
            )
            prediction = apply_shift(
                frames[held], shift, float(selected["strength"]), float(selected["cap"])
            )
        out = frames[held].copy()
        out["pre_maest_dynamic_pred_valence"] = out["pred_valence"]
        out["pred_valence"] = prediction
        out.to_csv(args.out_dir / f"{held}_val_predictions.csv", index=False)
        result = metrics(prediction, out["target_valence"])
        per_fold.append({"fold": held, **result})
        selected_rows.append({"fold": held, **selected})

    result = macro(per_fold)
    summary = {
        "created_at": "2026-07-14",
        "model_id": "pmemo_valence_locked_maest_dynamic_nested_20260714",
        "selection_boundary": "For each held fold, hyperparameters selected by bidirectional cross-evaluation on the other two song folds; final locked test excluded",
        "feature": "MAEST layer-7 30-second window embeddings interpolated over each complete song",
        "base_macro": base_macro,
        "selected_per_fold": selected_rows,
        "macro": result,
        "per_fold": per_fold,
        "delta_vs_base": {key: result[key] - base_macro[key] for key in base_macro},
        "guarded_promotion": (
            result["rmse"] < base_macro["rmse"]
            and result["mae"] <= base_macro["mae"]
            and result["pcc"] >= base_macro["pcc"]
            and result["ccc"] >= base_macro["ccc"]
        ),
        "gap_rows": threshold_rows(result),
        "final_locked_test": "excluded",
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
