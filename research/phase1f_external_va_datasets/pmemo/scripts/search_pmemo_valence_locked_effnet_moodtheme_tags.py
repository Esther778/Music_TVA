#!/usr/bin/env python3
"""Cross-fold MTG-Jamendo mood/theme tag correction for locked Valence."""

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

from train_pmemo_mert_valence_baseline import ROOT, metrics, threshold_rows


FOLDS = ["fold_01", "fold_02", "fold_03"]
DEFAULT_BASE = ROOT / "outputs/valence_locked_discogs_effnet_dynamic_fine_20260714"
DEFAULT_TAGS = ROOT.parents[1] / "phase1g_valence_data_pretraining/outputs/pmemo_effnet_moodtheme_tags_20260714"
DEFAULT_OUT = ROOT / "outputs/valence_locked_effnet_moodtheme_tags_20260714"
VALENCE_INDICES = np.array([
    5, 11, 16, 17, 18, 22, 23, 26, 29, 30, 31, 32,
    33, 35, 38, 39, 41, 43, 44, 45, 46, 47, 54, 55,
], dtype=np.int64)


def macro(rows: list[dict[str, float]]) -> dict[str, float]:
    return {key: float(np.mean([row[key] for row in rows])) for key in ["rmse", "mae", "pcc", "ccc"]}


def interpolate_tags(frame: pd.DataFrame, tag_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    tags_out = np.zeros((len(frame), 56), dtype=np.float32)
    song_weight = np.zeros(len(frame), dtype=np.float32)
    for _, indices in frame.groupby("musicId", sort=False).groups.items():
        loc = np.asarray(indices, dtype=np.int64)
        music_id = int(frame.loc[loc[0], "musicId"])
        tags = np.load(tag_dir / "per_song" / f"{music_id}.npz")["tags"].astype(np.float32)
        source_x = np.linspace(0.0, 1.0, len(tags), dtype=np.float32)
        target_x = np.linspace(0.0, 1.0, len(loc), dtype=np.float32)
        tags_out[loc] = np.column_stack([
            np.interp(target_x, source_x, tags[:, column]) for column in range(tags.shape[1])
        ])
        song_weight[loc] = 1.0 / len(loc)
    song_weight *= len(song_weight) / song_weight.sum()
    return tags_out, song_weight


def shifted(frame: pd.DataFrame, tags: np.ndarray, offsets: list[int]) -> np.ndarray:
    parts = []
    for offset in offsets:
        output = np.empty_like(tags)
        for _, indices in frame.groupby("musicId", sort=False).groups.items():
            loc = np.asarray(indices, dtype=np.int64)
            source = np.clip(np.arange(len(loc)) + offset, 0, len(loc) - 1)
            output[loc] = tags[loc[source]]
        parts.append(output)
    return np.concatenate(parts, axis=1)


def feature_matrix(frame: pd.DataFrame, tags: np.ndarray, mode: str) -> np.ndarray:
    base_tags = tags[:, VALENCE_INDICES] if mode.startswith("valence_") else tags
    mode = mode.removeprefix("valence_").removeprefix("all_")
    if mode == "current":
        return base_tags
    if mode == "past_sparse":
        return shifted(frame, base_tags, [-16, -8, 0])
    if mode == "past_dense":
        return shifted(frame, base_tags, [-16, -12, -8, -4, 0])
    raise ValueError(mode)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--tag-dir", type=Path, default=DEFAULT_TAGS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--modes", nargs="+", default=None)
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.1, 1.0, 10.0, 100.0, 1000.0])
    parser.add_argument("--strengths", nargs="+", type=float, default=[0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.50])
    parser.add_argument("--caps", nargs="+", type=float, default=[0.003, 0.005, 0.0075, 0.010, 0.015, 0.020])
    parser.add_argument("--max-mae-increase", type=float, default=0.0005)
    parser.add_argument("--max-pcc-drop", type=float, default=0.0005)
    parser.add_argument("--max-ccc-drop", type=float, default=None)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    modes = args.modes or [
        "valence_current", "valence_past_sparse", "valence_past_dense",
        "all_current", "all_past_sparse", "all_past_dense",
    ]
    frames = {fold: pd.read_csv(args.base_dir / f"{fold}_val_predictions.csv") for fold in FOLDS}
    tags, weights, features = {}, {}, {mode: {} for mode in modes}
    for fold in FOLDS:
        tags[fold], weights[fold] = interpolate_tags(frames[fold], args.tag_dir)
        for mode in modes:
            features[mode][fold] = feature_matrix(frames[fold], tags[fold], mode)

    base_rows = [metrics(frames[fold]["pred_valence"], frames[fold]["target_valence"]) for fold in FOLDS]
    base_macro = macro(base_rows)
    candidates: list[dict[str, Any]] = [{
        "mode": "identity", "target_mode": "identity", "weight_mode": "identity",
        "alpha": 0.0, "strength": 0.0, "cap": 0.0, **base_macro, "guarded": True,
    }]
    raw_cache: dict[tuple[str, str, str, float], dict[str, np.ndarray]] = {}
    for mode in modes:
        for target_mode in ["direct", "residual"]:
            for weight_mode in ["frame", "song"]:
                for alpha in args.alphas:
                    shifts: dict[str, np.ndarray] = {}
                    for held in FOLDS:
                        train_folds = [fold for fold in FOLDS if fold != held]
                        x_train = np.concatenate([features[mode][fold] for fold in train_folds])
                        target = np.concatenate([
                            frames[fold]["target_valence"].to_numpy(dtype=np.float64)
                            for fold in train_folds
                        ])
                        base_train = np.concatenate([
                            frames[fold]["pred_valence"].to_numpy(dtype=np.float64)
                            for fold in train_folds
                        ])
                        if target_mode == "residual":
                            target -= base_train
                        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
                        fit_kwargs = {}
                        if weight_mode == "song":
                            fit_kwargs["ridge__sample_weight"] = np.concatenate([weights[fold] for fold in train_folds])
                        model.fit(x_train, target, **fit_kwargs)
                        estimate = model.predict(features[mode][held])
                        if target_mode == "direct":
                            estimate -= frames[held]["pred_valence"].to_numpy(dtype=np.float64)
                        shifts[held] = estimate
                    raw_cache[(mode, target_mode, weight_mode, alpha)] = shifts
                    for strength in args.strengths:
                        for cap in args.caps:
                            per_fold = []
                            for fold in FOLDS:
                                base = frames[fold]["pred_valence"].to_numpy(dtype=np.float64)
                                prediction = np.clip(
                                    base + np.clip(strength * shifts[fold], -cap, cap), -1.0, 1.0
                                )
                                per_fold.append(metrics(prediction, frames[fold]["target_valence"]))
                            result = macro(per_fold)
                            ccc_floor = (
                                0.80 if args.max_ccc_drop is None
                                else base_macro["ccc"] - args.max_ccc_drop
                            )
                            guarded = (
                                result["mae"] <= base_macro["mae"] + args.max_mae_increase
                                and result["pcc"] >= base_macro["pcc"] - args.max_pcc_drop
                                and result["ccc"] >= ccc_floor
                            )
                            candidates.append({
                                "mode": mode, "target_mode": target_mode, "weight_mode": weight_mode,
                                "alpha": alpha, "strength": strength, "cap": cap,
                                **result, "guarded": guarded,
                            })

    table = pd.DataFrame(candidates).sort_values(["rmse", "mae", "ccc"], ascending=[True, True, False])
    table.to_csv(args.out_dir / "candidates.csv", index=False)
    eligible = table[table["guarded"]]
    selected = (eligible if not eligible.empty else table).iloc[0].to_dict()
    per_fold_rows = []
    for fold in FOLDS:
        base = frames[fold]["pred_valence"].to_numpy(dtype=np.float64)
        if selected["mode"] == "identity":
            prediction = base
        else:
            key = (
                str(selected["mode"]), str(selected["target_mode"]),
                str(selected["weight_mode"]), float(selected["alpha"]),
            )
            prediction = np.clip(
                base + np.clip(
                    float(selected["strength"]) * raw_cache[key][fold],
                    -float(selected["cap"]), float(selected["cap"]),
                ), -1.0, 1.0,
            )
        out = frames[fold].copy()
        out["pre_moodtheme_pred_valence"] = base
        out["pred_valence"] = prediction
        out.to_csv(args.out_dir / f"{fold}_val_predictions.csv", index=False)
        per_fold_rows.append({"fold": fold, **metrics(prediction, out["target_valence"])})

    selected_macro = macro(per_fold_rows)
    summary = {
        "created_at": "2026-07-14",
        "model_id": f"pmemo_{args.out_dir.name}",
        "selection_boundary": "Other-two-fold Ridge fit; final locked test excluded",
        "tag_head": "MTG-Jamendo mood/theme Discogs-EffNet, 56 tags",
        "base_dir": str(args.base_dir),
        "tag_dir": str(args.tag_dir),
        "guard": {
            "max_mae_increase": args.max_mae_increase,
            "max_pcc_drop": args.max_pcc_drop,
            "max_ccc_drop": args.max_ccc_drop,
        },
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
