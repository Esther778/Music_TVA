#!/usr/bin/env python3
"""Cross-devfold MuQ + predicted-Arousal stacker for Valence.

The meta-model for each held fold is fitted on the other two development folds.
No held-fold target is used for calibration and the final locked test is never read.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from search_pmemo_valence_locked_arousal_assisted_crossfold import (
    AROUSAL_DIRS,
    add_temporal_features,
    standard,
)
from train_pmemo_mert_valence_baseline import metrics, threshold_rows


ROOT = Path(__file__).resolve().parents[1]
FOLDS = ["fold_01", "fold_02", "fold_03"]
BASE_DIR = ROOT / "outputs/valence_locked_muq_threeway_fusion_20260713"
OUT_DIR = ROOT / "outputs/valence_locked_muq_arousal_stacker_fine_20260714"


def macro(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {key: float(np.mean([row[key] for row in rows])) for key in ["rmse", "mae", "pcc", "ccc"]}


def load_fold(fold: str) -> pd.DataFrame:
    frame = pd.read_csv(BASE_DIR / f"{fold}_val_predictions.csv")
    frame = frame.rename(
        columns={
            "base_pred_valence": "pred_reference",
            "muq_raw_pred_valence": "pred_muq_bilstm_raw",
            "muq_post_pred_valence": "pred_muq_bilstm_post",
            "mlp_post_pred_valence": "pred_muq_mlp_post",
            "pred_valence": "pred_threeway",
        }
    )
    keep = [
        "musicId", "time_sec", "target_valence", "pred_reference",
        "pred_muq_bilstm_raw", "pred_muq_bilstm_post",
        "pred_muq_mlp_post", "pred_threeway",
    ]
    frame = frame[keep]
    for name, root in AROUSAL_DIRS.items():
        arousal = standard(root / f"{fold}_val_predictions.csv", f"a_{name}", "target_arousal")
        frame = frame.merge(
            arousal.drop(columns=["target_arousal"]),
            on=["musicId", "time_sec"], how="inner", validate="one_to_one",
        )
    # Compatibility alias used by the shared Arousal interaction feature.
    frame["pred_v_crossfold_cal"] = frame["pred_reference"]
    pred_cols = [col for col in frame.columns if col.startswith("pred_")]
    return add_temporal_features(frame, pred_cols)


def feature_sets(columns: list[str]) -> dict[str, list[str]]:
    compact_names = [
        "pred_reference", "pred_muq_bilstm_raw", "pred_muq_bilstm_post",
        "pred_muq_mlp_post", "pred_threeway", "pred_a_mean",
        "pred_reference_song_mean", "pred_muq_bilstm_raw_song_mean",
        "pred_muq_mlp_post_song_mean", "pred_threeway_song_mean",
        "pred_a_mean_song_mean", "pred_threeway_centered",
        "pred_a_mean_centered", "song_pos",
    ]
    compact = [name for name in compact_names if name in columns]
    raw = [
        name for name in columns
        if name.startswith("pred_") and not any(
            suffix in name for suffix in ["_song_mean", "_song_std", "_centered", "_delta"]
        )
    ] + ["song_pos"]
    all_features = [name for name in columns if name.startswith("pred_") or name == "song_pos"]
    return {"raw": raw, "compact": compact, "all": all_features}


def models() -> list[tuple[str, object]]:
    return [
        ("ridge_10", make_pipeline(StandardScaler(), Ridge(alpha=10.0))),
        ("ridge_100", make_pipeline(StandardScaler(), Ridge(alpha=100.0))),
        ("hgb_leaf6_l2_1", HistGradientBoostingRegressor(
            learning_rate=0.025, max_leaf_nodes=6, min_samples_leaf=40,
            l2_regularization=1.0, max_iter=180, random_state=20260714,
        )),
        ("hgb_leaf4_l2_3", HistGradientBoostingRegressor(
            learning_rate=0.02, max_leaf_nodes=4, min_samples_leaf=60,
            l2_regularization=3.0, max_iter=200, random_state=20260714,
        )),
    ]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    folds = {fold: load_fold(fold) for fold in FOLDS}
    fsets = feature_sets(list(folds[FOLDS[0]].columns))
    base_rows = [
        metrics(folds[fold]["pred_threeway"].to_numpy(), folds[fold]["target_valence"].to_numpy())
        for fold in FOLDS
    ]
    base_macro = macro(base_rows)
    rows: list[dict[str, Any]] = []
    cache: dict[tuple[str, str, float, float], dict[str, np.ndarray]] = {}

    for feature_name, cols in fsets.items():
        for model_name, model in models():
            raw: dict[str, np.ndarray] = {}
            for held in FOLDS:
                train = pd.concat([folds[name] for name in FOLDS if name != held], ignore_index=True)
                model.fit(train[cols].to_numpy(dtype=np.float64), train["target_valence"].to_numpy(dtype=np.float64))
                raw[held] = model.predict(folds[held][cols].to_numpy(dtype=np.float64))
            for blend in [0.025, 0.05, 0.075, 0.10, 0.125, 0.15, 0.20, 0.35, 0.50, 0.70, 1.00]:
                for cap in [0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.06, 0.10, 1.0]:
                    predictions: dict[str, np.ndarray] = {}
                    per_fold: list[dict[str, float]] = []
                    for fold in FOLDS:
                        base = folds[fold]["pred_threeway"].to_numpy(dtype=np.float64)
                        candidate = (1.0 - blend) * base + blend * raw[fold]
                        pred = base + np.clip(candidate - base, -cap, cap)
                        pred = np.clip(pred, -1.0, 1.0)
                        predictions[fold] = pred
                        per_fold.append(metrics(pred, folds[fold]["target_valence"].to_numpy(dtype=np.float64)))
                    result = macro(per_fold)
                    guarded = (
                        result["mae"] <= base_macro["mae"] + 0.001
                        and result["pcc"] >= base_macro["pcc"] - 0.001
                        and result["ccc"] >= base_macro["ccc"] - 0.001
                    )
                    row = {
                        "feature_set": feature_name, "model": model_name,
                        "blend": blend, "cap": cap, **result, "guarded": guarded,
                    }
                    rows.append(row)
                    cache[(feature_name, model_name, blend, cap)] = predictions

    table = pd.DataFrame(rows).sort_values(["rmse", "mae", "ccc"], ascending=[True, True, False])
    table.to_csv(OUT_DIR / "candidates.csv", index=False)
    guarded = table[table["guarded"]]
    selected = (guarded if not guarded.empty else table).iloc[0]
    key = (str(selected.feature_set), str(selected.model), float(selected.blend), float(selected.cap))
    selected_predictions = cache[key]
    per_fold_rows: list[dict[str, Any]] = []
    for fold in FOLDS:
        out = folds[fold][["musicId", "time_sec", "target_valence", "pred_threeway"]].copy()
        out = out.rename(columns={"pred_threeway": "base_pred_valence"})
        out["pred_valence"] = selected_predictions[fold]
        out.to_csv(OUT_DIR / f"{fold}_val_predictions.csv", index=False)
        per_fold_rows.append({"fold": fold, **metrics(out["pred_valence"], out["target_valence"])})
    selected_macro = macro(per_fold_rows)
    summary = {
        "created_at": "2026-07-14",
        "selection_boundary": "Cross-devfold meta-fit only; no held-fold calibration; final locked test excluded.",
        "base": str(BASE_DIR),
        "base_macro": base_macro,
        "selected": selected.to_dict(),
        "macro": selected_macro,
        "per_fold": per_fold_rows,
        "delta_vs_base": {key: selected_macro[key] - base_macro[key] for key in base_macro},
        "gap_rows": threshold_rows(selected_macro),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
