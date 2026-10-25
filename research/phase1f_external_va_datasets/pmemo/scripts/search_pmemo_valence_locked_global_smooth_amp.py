#!/usr/bin/env python3
"""Low-degree global smoothing and within-song amplitude check for Valence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from search_pmemo_valence_locked_audio_ensemble_postprocess import smooth_by_song
from train_pmemo_mert_valence_baseline import ROOT, metrics, threshold_rows


FOLDS = ["fold_01", "fold_02", "fold_03"]
DEFAULT_INPUT = ROOT / "outputs/valence_locked_effnet_moodtheme_tags_balanced_final_20260714"
DEFAULT_OUT = ROOT / "outputs/valence_locked_global_smooth_amp_20260715"


def macro(rows: list[dict[str, float]]) -> dict[str, float]:
    return {key: float(np.mean([row[key] for row in rows])) for key in ["rmse", "mae", "pcc", "ccc"]}


def scale_within_song(frame: pd.DataFrame, values: np.ndarray, scale: float) -> np.ndarray:
    output = np.empty_like(values, dtype=np.float64)
    for _, indices in frame.groupby("musicId", sort=False).groups.items():
        loc = np.asarray(indices, dtype=np.int64)
        mean = float(values[loc].mean())
        output[loc] = mean + scale * (values[loc] - mean)
    return output


def shift_by_song(frame: pd.DataFrame, values: np.ndarray, shift: int) -> np.ndarray:
    output = np.empty_like(values, dtype=np.float64)
    for _, indices in frame.groupby("musicId", sort=False).groups.items():
        loc = np.asarray(indices, dtype=np.int64)
        source = np.clip(np.arange(len(loc)) + shift, 0, len(loc) - 1)
        output[loc] = values[loc[source]]
    return output


def transform(
    frame: pd.DataFrame,
    values: np.ndarray,
    shift: int,
    window: int,
    smooth_blend: float,
    amplitude: float,
) -> np.ndarray:
    shifted = shift_by_song(frame, values, shift)
    if window == 1 or smooth_blend == 0.0:
        smoothed = shifted
    else:
        local = smooth_by_song(frame, shifted, window)
        smoothed = (1.0 - smooth_blend) * shifted + smooth_blend * local
    return np.clip(scale_within_song(frame, smoothed, amplitude), -1.0, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--windows", nargs="+", type=int, default=[1, 3, 5, 7, 9])
    parser.add_argument("--shifts", nargs="+", type=int, default=[0])
    parser.add_argument("--smooth-blends", nargs="+", type=float, default=[0.25, 0.50, 0.75, 1.0])
    parser.add_argument("--amplitudes", nargs="+", type=float, default=[0.95, 0.975, 1.0, 1.025, 1.05])
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    frames = {fold: pd.read_csv(args.input_dir / f"{fold}_val_predictions.csv") for fold in FOLDS}
    base_rows = [metrics(frame["pred_valence"], frame["target_valence"]) for frame in frames.values()]
    base_macro = macro(base_rows)
    rows = []
    cache: dict[tuple[int, int, float, float], dict[str, np.ndarray]] = {}
    for shift in args.shifts:
        for window in args.windows:
            blends = [0.0] if window == 1 else args.smooth_blends
            for smooth_blend in blends:
                for amplitude in args.amplitudes:
                    predictions, per_fold = {}, []
                    for fold, frame in frames.items():
                        base = frame["pred_valence"].to_numpy(dtype=np.float64)
                        prediction = transform(frame, base, shift, window, smooth_blend, amplitude)
                        predictions[fold] = prediction
                        per_fold.append(metrics(prediction, frame["target_valence"]))
                    result = macro(per_fold)
                    guarded = (
                        result["mae"] <= base_macro["mae"]
                        and result["pcc"] >= base_macro["pcc"]
                        and result["ccc"] >= base_macro["ccc"]
                    )
                    rows.append({
                        "shift_frames": shift,
                        "shift_sec": shift * 0.5,
                        "window": window,
                        "smooth_blend": smooth_blend,
                        "amplitude": amplitude,
                        **result,
                        "guarded": guarded,
                    })
                    cache[(shift, window, smooth_blend, amplitude)] = predictions

    table = pd.DataFrame(rows).sort_values(["rmse", "mae", "ccc"], ascending=[True, True, False])
    table.to_csv(args.out_dir / "candidates.csv", index=False)
    eligible = table[table["guarded"]]
    selected = (eligible if not eligible.empty else table).iloc[0].to_dict()
    key = (
        int(selected["shift_frames"]),
        int(selected["window"]),
        float(selected["smooth_blend"]),
        float(selected["amplitude"]),
    )
    per_fold_rows = []
    for fold, frame in frames.items():
        output = frame.copy()
        output["pre_smooth_amp_pred_valence"] = output["pred_valence"]
        output["pred_valence"] = cache[key][fold]
        output.to_csv(args.out_dir / f"{fold}_val_predictions.csv", index=False)
        per_fold_rows.append({
            "fold": fold,
            **metrics(output["pred_valence"], output["target_valence"]),
        })
    selected_macro = macro(per_fold_rows)
    summary = {
        "created_at": "2026-07-15",
        "model_id": f"pmemo_{args.out_dir.name}",
        "selection_boundary": "One global low-degree policy on development folds; final locked test excluded",
        "input_dir": str(args.input_dir),
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
