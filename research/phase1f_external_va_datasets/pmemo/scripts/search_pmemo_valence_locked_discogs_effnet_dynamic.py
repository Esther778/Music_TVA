#!/usr/bin/env python3
"""Fine search for dynamic Discogs-EffNet locked-development correction."""

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

from search_pmemo_valence_locked_musicnn_dynamic_ridge import build_fold, macro, matrix
from train_pmemo_mert_valence_baseline import ROOT, metrics, threshold_rows


FOLDS = ["fold_01", "fold_02", "fold_03"]
DEFAULT_BASE = ROOT / "outputs/valence_locked_discogs_effnet_fine_20260714"
DEFAULT_FEATURES = ROOT.parents[1] / "phase1g_valence_data_pretraining/outputs/pmemo_discogs_effnet_features_20260714/per_song"
DEFAULT_OUT = ROOT / "outputs/valence_locked_discogs_effnet_dynamic_fine_20260714"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--alphas", nargs="+", type=float, default=[30.0, 60.0, 100.0, 150.0, 300.0])
    parser.add_argument("--blends", nargs="+", type=float, default=[0.10, 0.15, 0.20, 0.25, 0.30])
    parser.add_argument("--caps", nargs="+", type=float, default=[0.005, 0.006, 0.0075, 0.009, 0.010])
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    frames = {fold: pd.read_csv(args.base_dir / f"{fold}_val_predictions.csv") for fold in FOLDS}
    data = {fold: build_fold(frames[fold], args.feature_dir) for fold in FOLDS}
    base_rows = [metrics(frames[fold]["pred_valence"], frames[fold]["target_valence"]) for fold in FOLDS]
    base_macro = macro(base_rows)
    candidates: list[dict[str, Any]] = []
    raw_cache: dict[tuple[str, str, float], dict[str, np.ndarray]] = {}

    for feature_set in ["embedding", "embedding_base"]:
        for target_mode in ["direct", "residual"]:
            for alpha in args.alphas:
                raw: dict[str, np.ndarray] = {}
                for held in FOLDS:
                    train = {
                        key: np.concatenate([data[fold][key] for fold in FOLDS if fold != held])
                        for key in data[held]
                    }
                    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
                    target = train["target"].copy()
                    if target_mode == "residual":
                        target -= train["base"][:, 0]
                    model.fit(matrix(train, feature_set), target, ridge__sample_weight=train["weight"])
                    estimate = model.predict(matrix(data[held], feature_set))
                    if target_mode == "residual":
                        estimate += data[held]["base"][:, 0]
                    raw[held] = estimate
                raw_cache[(feature_set, target_mode, alpha)] = raw
                for blend in args.blends:
                    for cap in args.caps:
                        per_fold = []
                        for fold in FOLDS:
                            base = frames[fold]["pred_valence"].to_numpy(dtype=np.float64)
                            pred = np.clip(
                                base + np.clip(blend * (raw[fold] - base), -cap, cap), -1.0, 1.0
                            )
                            per_fold.append(metrics(pred, frames[fold]["target_valence"]))
                        result = macro(per_fold)
                        guarded = (
                            result["mae"] <= base_macro["mae"] + 0.0005
                            and result["pcc"] >= base_macro["pcc"] - 0.0005
                            and result["ccc"] >= 0.80
                        )
                        candidates.append({
                            "feature_set": feature_set,
                            "target_mode": target_mode,
                            "alpha": alpha,
                            "blend": blend,
                            "cap": cap,
                            **result,
                            "guarded": guarded,
                        })

    table = pd.DataFrame(candidates).sort_values(["rmse", "mae", "ccc"], ascending=[True, True, False])
    table.to_csv(args.out_dir / "candidates.csv", index=False)
    eligible = table[table["guarded"]]
    selected = (eligible if not eligible.empty else table).iloc[0].to_dict()
    raw = raw_cache[(str(selected["feature_set"]), str(selected["target_mode"]), float(selected["alpha"]))]
    per_fold_rows = []
    for fold in FOLDS:
        base = frames[fold]["pred_valence"].to_numpy(dtype=np.float64)
        prediction = np.clip(
            base + np.clip(
                float(selected["blend"]) * (raw[fold] - base),
                -float(selected["cap"]), float(selected["cap"]),
            ),
            -1.0, 1.0,
        )
        out = frames[fold].copy()
        out["pre_effnet_dynamic_pred_valence"] = base
        out["pred_valence"] = prediction
        out.to_csv(args.out_dir / f"{fold}_val_predictions.csv", index=False)
        per_fold_rows.append({"fold": fold, **metrics(prediction, out["target_valence"])})

    selected_macro = macro(per_fold_rows)
    summary = {
        "created_at": "2026-07-14",
        "model_id": "pmemo_valence_locked_discogs_effnet_dynamic_fine_20260714",
        "selection_boundary": (
            "Dynamic Ridge trained on the other two song-level development folds with song-balanced weights; "
            "final locked test excluded"
        ),
        "backbone": "Essentia Discogs-EffNet bs64",
        "alignment": "Per-song normalized-time interpolation from EffNet patches to 0.5-second label frames",
        "base_dir": str(args.base_dir),
        "feature_dir": str(args.feature_dir),
        "base_macro": base_macro,
        "selected": selected,
        "macro": selected_macro,
        "per_fold": per_fold_rows,
        "delta_vs_base": {key: selected_macro[key] - base_macro[key] for key in base_macro},
        "gap_rows": threshold_rows(selected_macro),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
