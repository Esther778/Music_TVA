#!/usr/bin/env python3
"""Cross-fold ordinal song-polarity head for locked-development Valence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from search_pmemo_valence_locked_song_disagreement_stacker import FOLDS, MOOD, load_fold, macro
from train_pmemo_mert_valence_baseline import ROOT, metrics, threshold_rows


DEFAULT_BASE = ROOT / "outputs" / "valence_locked_song_disagreement_final_blend_20260714"
DEFAULT_OUT = ROOT / "outputs" / "valence_locked_ordinal_song_head_20260714"


def ordinal_estimate(train: pd.DataFrame, test: pd.DataFrame, columns: list[str], bins: int, c: float, balanced: bool, mode: str) -> np.ndarray:
    target = train["target_song"].to_numpy(dtype=np.float64)
    boundaries = np.quantile(target, np.linspace(0.0, 1.0, bins + 1)[1:-1])
    labels = np.digitize(target, boundaries)
    centers = np.asarray([target[labels == label].mean() for label in range(bins)], dtype=np.float64)
    x_train = train[columns].to_numpy(dtype=np.float64)
    x_test = test[columns].to_numpy(dtype=np.float64)
    if mode == "multinomial":
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=c, class_weight="balanced" if balanced else None, max_iter=5000, solver="lbfgs"),
        )
        model.fit(x_train, labels)
        probabilities = model.predict_proba(x_test)
        classes = model.named_steps["logisticregression"].classes_.astype(int)
        return probabilities @ centers[classes]
    if mode == "cumulative":
        above = []
        for threshold in range(bins - 1):
            binary = (labels > threshold).astype(int)
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(C=c, class_weight="balanced" if balanced else None, max_iter=5000, solver="lbfgs"),
            )
            model.fit(x_train, binary)
            above.append(model.predict_proba(x_test)[:, 1])
        q = np.minimum.accumulate(np.stack(above, axis=1), axis=1)
        probabilities = np.column_stack([1.0 - q[:, 0], q[:, :-1] - q[:, 1:], q[:, -1]])
        return probabilities @ centers
    raise ValueError(mode)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--mood-csv", type=Path, default=MOOD)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--feature-sets", nargs="+", default=["valence_arousal", "all"])
    parser.add_argument("--bins", nargs="+", type=int, default=[3, 5, 7])
    parser.add_argument("--cs", nargs="+", type=float, default=[0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0])
    parser.add_argument("--strengths", nargs="+", type=float, default=[0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.70, 1.0])
    parser.add_argument("--caps", nargs="+", type=float, default=[0.003, 0.005, 0.0075, 0.010, 0.015, 0.020, 0.030, 0.040, 0.060])
    parser.add_argument("--modes", nargs="+", default=["multinomial"])
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    mood = pd.read_csv(args.mood_csv)
    frames, songs, sets = {}, {}, {}
    for fold in FOLDS:
        frames[fold], songs[fold], sets = load_fold(args.base_dir, fold, mood)
    base_rows = [metrics(frames[f]["pred_valence"], frames[f]["target_valence"]) for f in FOLDS]
    base_macro = macro(base_rows)
    candidates = [{"feature_set": "identity", "mode": "identity", "bins": 0, "c": 0.0, "balanced": False, "strength": 0.0, "cap": 0.0, **base_macro, "guarded": True}]
    raw_cache: dict[tuple[str, str, int, float, bool], dict[str, np.ndarray]] = {}
    for feature_name in args.feature_sets:
        columns = sets[feature_name]
        for mode in args.modes:
          for bins in args.bins:
            for c in args.cs:
              for balanced in [False, True]:
                    shifts = {}
                    for held in FOLDS:
                        train = pd.concat([songs[f] for f in FOLDS if f != held], ignore_index=True)
                        test = songs[held]
                        estimate = ordinal_estimate(train, test, columns, bins, c, balanced, mode)
                        shifts[held] = estimate - test["current_pred_valence_mean"].to_numpy(dtype=np.float64)
                    raw_cache[(feature_name, mode, bins, c, balanced)] = shifts
                    for strength in args.strengths:
                        for cap in args.caps:
                            per_fold = []
                            for fold in FOLDS:
                                shift_map = dict(zip(songs[fold]["musicId"].astype(int), shifts[fold]))
                                shift = frames[fold]["musicId"].astype(int).map(shift_map).to_numpy(dtype=np.float64)
                                base = frames[fold]["pred_valence"].to_numpy(dtype=np.float64)
                                pred = np.clip(base + np.clip(strength * shift, -cap, cap), -1.0, 1.0)
                                per_fold.append(metrics(pred, frames[fold]["target_valence"].to_numpy(dtype=np.float64)))
                            result = macro(per_fold)
                            guarded = result["mae"] <= base_macro["mae"] + 0.0005 and result["pcc"] >= base_macro["pcc"] - 0.0005 and result["ccc"] >= 0.80
                            candidates.append({"feature_set": feature_name, "mode": mode, "bins": bins, "c": c, "balanced": balanced, "strength": strength, "cap": cap, **result, "guarded": guarded})

    table = pd.DataFrame(candidates).sort_values(["rmse", "mae", "ccc"], ascending=[True, True, False])
    table.to_csv(args.out_dir / "candidates.csv", index=False)
    eligible = table[table["guarded"]]
    selected = (eligible if not eligible.empty else table).iloc[0].to_dict()
    if selected["feature_set"] == "identity":
        selected_preds = {fold: frames[fold]["pred_valence"].to_numpy(dtype=np.float64) for fold in FOLDS}
    else:
        shifts = raw_cache[(str(selected["feature_set"]), str(selected["mode"]), int(selected["bins"]), float(selected["c"]), bool(selected["balanced"]))]
        selected_preds = {}
        for fold in FOLDS:
            shift_map = dict(zip(songs[fold]["musicId"].astype(int), shifts[fold]))
            shift = frames[fold]["musicId"].astype(int).map(shift_map).to_numpy(dtype=np.float64)
            base = frames[fold]["pred_valence"].to_numpy(dtype=np.float64)
            selected_preds[fold] = np.clip(base + np.clip(float(selected["strength"]) * shift, -float(selected["cap"]), float(selected["cap"])), -1.0, 1.0)
    per_fold_rows = []
    for fold in FOLDS:
        out = frames[fold].copy().rename(columns={"pred_valence": "pre_ordinal_pred_valence"})
        out["pred_valence"] = selected_preds[fold]
        out.to_csv(args.out_dir / f"{fold}_val_predictions.csv", index=False)
        per_fold_rows.append({"fold": fold, **metrics(out["pred_valence"], out["target_valence"])})
    selected_macro = macro(per_fold_rows)
    summary = {
        "created_at": "2026-07-14",
        "selection_boundary": "Ordinal head fit and bin boundaries derived from other two development folds only; final locked test excluded",
        "base_dir": str(args.base_dir), "base_macro": base_macro,
        "selected": selected, "macro": selected_macro, "per_fold": per_fold_rows,
        "delta_vs_base": {key: selected_macro[key] - base_macro[key] for key in base_macro},
        "gap_rows": threshold_rows(selected_macro),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
