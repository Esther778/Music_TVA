#!/usr/bin/env python3
"""Nested pure-audio song-mean correction for locked-development VA curves."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from train_pmemo_mert_valence_baseline import ROOT, metrics, threshold_rows


RESEARCH_ROOT = ROOT.parents[1]
FEATURE_ROOT = RESEARCH_ROOT / "phase1g_valence_data_pretraining/outputs"
FOLDS = ["fold_01", "fold_02", "fold_03"]
TAG_DIR = FEATURE_ROOT / "pmemo_effnet_moodtheme_tags_20260714/per_song"
CLAP_DIR = FEATURE_ROOT / "pmemo_clap_music_10s_20260714/per_song"
MAEST_DIR = FEATURE_ROOT / "pmemo_maest_30s_layer7_20260714/per_song"
STATIC_FEATURES = ROOT / "data/raw/PMEmo_2019_updated/extracted/PMEmo2019/features/static_features.csv"
SONG_MANIFEST = ROOT / "data/processed/pmemo2019_song_manifest_normalized.csv"


def summarize(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    t = np.linspace(-1.0, 1.0, len(x))
    slope = float(np.polyfit(t, x, 1)[0]) if len(x) > 1 else 0.0
    return np.array([
        x.mean(), x.std(), x.min(), x.max(),
        *np.quantile(x, [0.10, 0.25, 0.50, 0.75, 0.90]),
        slope,
    ], dtype=np.float64)


def pool(sequence: np.ndarray) -> np.ndarray:
    x = np.asarray(sequence, dtype=np.float64)
    if x.ndim == 1:
        return x
    return np.concatenate([x.mean(axis=0), x.std(axis=0)])


def load_audio_blocks(music_id: int) -> dict[str, np.ndarray]:
    tags = np.load(TAG_DIR / f"{music_id}.npz")["tags"]
    clap = np.load(CLAP_DIR / f"{music_id}.npz")["embedding"]
    maest = np.load(MAEST_DIR / f"{music_id}.npz")["embedding"]
    return {"tags": pool(tags), "clap": pool(clap), "maest": pool(maest)}


def macro(rows: list[dict[str, float]]) -> dict[str, float]:
    return {key: float(np.mean([row[key] for row in rows])) for key in ["rmse", "mae", "pcc", "ccc"]}


def make_model(family: str, alpha: float):
    steps = [StandardScaler()]
    if "opensmile_static" in family:
        steps.append(PCA(n_components=32, svd_solver="randomized", random_state=20260715))
    steps.append(Ridge(alpha=alpha, solver="lsqr", max_iter=10000))
    return make_pipeline(*steps)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimension", choices=["valence", "arousal"], required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--aux-arousal-dir", type=Path, default=None)
    parser.add_argument("--include-opensmile-static", action="store_true")
    parser.add_argument("--use-static-aux-target", action="store_true")
    parser.add_argument("--families", nargs="+", default=None)
    parser.add_argument("--alphas", nargs="+", type=float, default=[10, 100, 1000, 10000])
    parser.add_argument("--strengths", nargs="+", type=float, default=[0.10, 0.20, 0.30, 0.40, 0.50, 0.75, 1.0])
    parser.add_argument("--caps", nargs="+", type=float, default=[0.02, 0.04, 0.06, 0.10, 0.20])
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    target_col, pred_col = f"target_{args.dimension}", f"pred_{args.dimension}"

    frame_tables, song_rows = {}, []
    opensmile_lookup = {}
    static_target_lookup = {}
    if args.include_opensmile_static:
        static_features = pd.read_csv(STATIC_FEATURES).set_index("musicId")
        opensmile_lookup = {
            int(music_id): row.to_numpy(dtype=np.float64)
            for music_id, row in static_features.iterrows()
        }
    if args.use_static_aux_target:
        manifest = pd.read_csv(SONG_MANIFEST).set_index("musicId")
        static_target_lookup = manifest[
            "static_valence_deam_scale" if args.dimension == "valence"
            else "static_arousal_deam_scale"
        ].to_dict()
    aux_tables = {}
    if args.aux_arousal_dir is not None:
        if args.dimension != "valence":
            raise ValueError("--aux-arousal-dir is only valid for Valence")
        aux_tables = {
            fold: pd.read_csv(args.aux_arousal_dir / f"{fold}_val_predictions.csv")
            for fold in FOLDS
        }
    for fold in FOLDS:
        frame = pd.read_csv(args.input_dir / f"{fold}_val_predictions.csv")
        frame_tables[fold] = frame
        for music_id, song in frame.groupby("musicId", sort=False):
            blocks = load_audio_blocks(int(music_id))
            row: dict[str, Any] = {
                "fold": fold,
                "musicId": int(music_id),
                "target_mean": float(song[target_col].mean()),
                "base_mean": float(song[pred_col].mean()),
                "pred": summarize(song[pred_col].to_numpy(dtype=np.float64)),
                **blocks,
            }
            if aux_tables:
                aux_song = aux_tables[fold].loc[
                    aux_tables[fold]["musicId"].eq(int(music_id)),
                    "pred_arousal",
                ]
                if len(aux_song) != len(song):
                    raise ValueError(f"Arousal alignment mismatch for {fold}/{music_id}")
                row["arousal"] = summarize(aux_song.to_numpy(dtype=np.float64))
            if opensmile_lookup:
                row["opensmile_static"] = opensmile_lookup[int(music_id)]
            if static_target_lookup:
                row["static_target"] = float(static_target_lookup[int(music_id)])
            song_rows.append(row)

    songs = pd.DataFrame(song_rows).sort_values(["fold", "musicId"]).reset_index(drop=True)
    families = {
        "pred_tags": ["pred", "tags"],
        "pred_tags_clap": ["pred", "tags", "clap"],
        "pred_tags_maest": ["pred", "tags", "maest"],
        "pred_tags_clap_maest": ["pred", "tags", "clap", "maest"],
    }
    if aux_tables:
        families.update({
            "pred_tags_arousal": ["pred", "tags", "arousal"],
            "pred_tags_clap_arousal": ["pred", "tags", "clap", "arousal"],
            "pred_tags_maest_arousal": ["pred", "tags", "maest", "arousal"],
            "pred_tags_clap_maest_arousal": ["pred", "tags", "clap", "maest", "arousal"],
        })
    if opensmile_lookup:
        families.update({
            "pred_opensmile_static": ["pred", "opensmile_static"],
            "pred_tags_opensmile_static": ["pred", "tags", "opensmile_static"],
        })
    if args.families is not None:
        unknown = sorted(set(args.families) - set(families))
        if unknown:
            raise ValueError(f"Unknown feature families: {unknown}")
        families = {name: families[name] for name in args.families}
    feature_cache = {
        name: np.stack([
            np.concatenate([row[block] for block in blocks])
            for _, row in songs.iterrows()
        ])
        for name, blocks in families.items()
    }
    target = songs["target_mean"].to_numpy(dtype=np.float64)
    base_mean = songs["base_mean"].to_numpy(dtype=np.float64)
    static_target = (
        songs["static_target"].to_numpy(dtype=np.float64)
        if static_target_lookup else None
    )
    nested_rows, estimates = [], {}

    for family, x in feature_cache.items():
        estimate = np.zeros(len(songs), dtype=np.float64)
        for held in FOLDS:
            train_folds = [fold for fold in FOLDS if fold != held]
            best = None
            target_modes = ["direct", "residual"]
            if static_target is not None:
                target_modes.extend(["static_aux", "static_mix25", "static_mix50", "static_mix75"])
            for mode in target_modes:
                for alpha in args.alphas:
                    inner_scores = []
                    for inner_held in train_folds:
                        inner_train = songs["fold"].eq(train_folds[1] if inner_held == train_folds[0] else train_folds[0]).to_numpy()
                        inner_val = songs["fold"].eq(inner_held).to_numpy()
                        if mode == "direct":
                            y = target
                        elif mode == "residual":
                            y = target - base_mean
                        elif mode == "static_aux":
                            y = static_target
                        else:
                            static_weight = float(mode.removeprefix("static_mix")) / 100.0
                            slope, intercept = np.polyfit(
                                static_target[inner_train],
                                target[inner_train],
                                1,
                            )
                            mapped_static = slope * static_target + intercept
                            y = (1.0 - static_weight) * target + static_weight * mapped_static
                        model = make_model(family, alpha)
                        model.fit(x[inner_train], y[inner_train])
                        pred = model.predict(x[inner_val])
                        if mode == "residual":
                            pred = base_mean[inner_val] + pred
                        elif mode == "static_aux":
                            slope, intercept = np.polyfit(
                                static_target[inner_train],
                                target[inner_train],
                                1,
                            )
                            pred = slope * pred + intercept
                        inner_scores.append(float(np.sqrt(np.mean(np.square(pred - target[inner_val])))))
                    score = float(np.mean(inner_scores))
                    candidate = (score, mode, float(alpha))
                    if best is None or candidate < best:
                        best = candidate
            assert best is not None
            _, mode, alpha = best
            train_mask = songs["fold"].isin(train_folds).to_numpy()
            held_mask = songs["fold"].eq(held).to_numpy()
            if mode == "direct":
                y = target
            elif mode == "residual":
                y = target - base_mean
            elif mode == "static_aux":
                y = static_target
            else:
                static_weight = float(mode.removeprefix("static_mix")) / 100.0
                slope, intercept = np.polyfit(
                    static_target[train_mask],
                    target[train_mask],
                    1,
                )
                mapped_static = slope * static_target + intercept
                y = (1.0 - static_weight) * target + static_weight * mapped_static
            model = make_model(family, alpha)
            model.fit(x[train_mask], y[train_mask])
            held_estimate = model.predict(x[held_mask])
            if mode == "residual":
                held_estimate = base_mean[held_mask] + held_estimate
            elif mode == "static_aux":
                slope, intercept = np.polyfit(
                    static_target[train_mask],
                    target[train_mask],
                    1,
                )
                held_estimate = slope * held_estimate + intercept
            estimate[held_mask] = held_estimate
            nested_rows.append({
                "family": family,
                "held_fold": held,
                "selected_mode": mode,
                "selected_alpha": alpha,
                "inner_song_rmse": best[0],
                "held_song_rmse": float(np.sqrt(np.mean(np.square(held_estimate - target[held_mask])))),
            })
        estimates[family] = estimate

    nested_best = np.zeros(len(songs), dtype=np.float64)
    for held in FOLDS:
        eligible = [row for row in nested_rows if row["held_fold"] == held]
        selected_family = min(eligible, key=lambda row: row["inner_song_rmse"])["family"]
        held_mask = songs["fold"].eq(held).to_numpy()
        nested_best[held_mask] = estimates[selected_family][held_mask]
    estimates["nested_best_family"] = nested_best

    base_rows = [metrics(frame[pred_col], frame[target_col]) for frame in frame_tables.values()]
    base_macro = macro(base_rows)
    candidates = [{
        "family": "identity", "strength": 0.0, "cap": 0.0,
        **base_macro, "guarded": True,
    }]
    prediction_cache: dict[tuple[str, float, float], dict[str, np.ndarray]] = {}
    for family, estimate in estimates.items():
        song_shift = estimate - base_mean
        for strength in args.strengths:
            for cap in args.caps:
                fold_predictions, per_fold = {}, []
                for fold in FOLDS:
                    frame = frame_tables[fold]
                    fold_songs = songs[songs["fold"].eq(fold)]
                    shifts = {
                        int(row.musicId): float(np.clip(strength * song_shift[row.Index], -cap, cap))
                        for row in fold_songs.itertuples()
                    }
                    prediction = np.clip(
                        frame[pred_col].to_numpy(dtype=np.float64)
                        + frame["musicId"].map(shifts).to_numpy(dtype=np.float64),
                        -1.0,
                        1.0,
                    )
                    fold_predictions[fold] = prediction
                    per_fold.append(metrics(prediction, frame[target_col]))
                result = macro(per_fold)
                guarded = (
                    result["mae"] <= base_macro["mae"]
                    and result["pcc"] >= base_macro["pcc"]
                    and result["ccc"] >= base_macro["ccc"]
                )
                candidates.append({
                    "family": family, "strength": strength, "cap": cap,
                    **result, "guarded": guarded,
                })
                prediction_cache[(family, strength, cap)] = fold_predictions

    table = pd.DataFrame(candidates).sort_values(["rmse", "mae", "ccc"], ascending=[True, True, False])
    table.to_csv(args.out_dir / "candidates.csv", index=False)
    pd.DataFrame(nested_rows).to_csv(args.out_dir / "nested_song_model_selection.csv", index=False)
    eligible = table[table["guarded"]]
    selected = (eligible if not eligible.empty else table).iloc[0].to_dict()
    per_fold = []
    for fold, frame in frame_tables.items():
        output = frame.copy()
        output[f"pre_song_mean_{pred_col}"] = output[pred_col]
        if selected["family"] != "identity":
            key = (str(selected["family"]), float(selected["strength"]), float(selected["cap"]))
            output[pred_col] = prediction_cache[key][fold]
        output.to_csv(args.out_dir / f"{fold}_val_predictions.csv", index=False)
        per_fold.append({"fold": fold, **metrics(output[pred_col], output[target_col])})

    selected_macro = macro(per_fold)
    summary = {
        "created_at": "2026-07-15",
        "model_id": f"pmemo_{args.out_dir.name}",
        "dimension": args.dimension,
        "selection_boundary": "Nested other-fold song-mean Ridge; final locked test excluded",
        "input_dir": str(args.input_dir),
        "feature_families": families,
        "base_macro": base_macro,
        "selected": selected,
        "macro": selected_macro,
        "per_fold": per_fold,
        "delta_vs_base": {key: selected_macro[key] - base_macro[key] for key in base_macro},
        "gap_rows": threshold_rows(selected_macro),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
