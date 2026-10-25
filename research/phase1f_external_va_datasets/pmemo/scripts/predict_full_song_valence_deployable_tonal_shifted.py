#!/usr/bin/env python3
"""Predict full-song Valence with the deployable tonal shift+smoothing chain.

This wraps the existing raw-MP3 `latest_refined` Valence predictor and applies
only fixed, validation-selected post-processing that can be used on arbitrary
complete songs:

1. temporal shift = -22 frames at 0.5s hop;
2. moving-average smoothing window = 21 frames;
3. light global affine calibration, blend = 0.03.

The CV reference for this deployable-only chain is:
RMSE/MAE/PCC/CCC = 0.188912 / 0.146588 / 0.811721 / 0.796749.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE_SCRIPT = ROOT / "scripts" / "predict_full_song_valence_latest_refined.py"
DEFAULT_OUT_DIR = ROOT / "outputs" / "valence_fullsong_deployable_tonal_shifted"

MODEL_ID = "pmemo_valence_deployable_tonal_shift_smoothing_affine_fullsong_20260708"
CV_REFERENCE = {
    "model_id": "pmemo_valence_deployable_tonal_shift_smoothing_affine_20260708",
    "rmse": 0.1889116153150956,
    "mae": 0.1465878035891668,
    "pcc": 0.8117208849284806,
    "ccc": 0.7967490459172722,
}
TEMPORAL_SHIFT_FRAMES = -22
SMOOTH_WINDOW = 21
AFFINE_A = 1.0506696898779115
AFFINE_B = -0.013458198797455023
AFFINE_BLEND = 0.03


def shift_one(pred: np.ndarray, shift_frames: int) -> np.ndarray:
    if shift_frames == 0 or len(pred) <= 1:
        return pred.copy()
    out = np.empty_like(pred)
    if shift_frames > 0:
        out[:-shift_frames] = pred[shift_frames:]
        out[-shift_frames:] = pred[-1]
    else:
        k = min(-shift_frames, len(pred))
        out[:k] = pred[0]
        out[k:] = pred[:-k] if k < len(pred) else pred[0]
    return out


def moving_average(pred: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(pred) <= 2:
        return pred.copy()
    local = min(int(window), len(pred))
    pad = local // 2
    padded = np.pad(pred, (pad, pad), mode="edge")
    kernel = np.ones(local, dtype=np.float64) / float(local)
    return np.convolve(padded, kernel, mode="valid")[: len(pred)]


def apply_postprocess(pred: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    shifted = shift_one(pred.astype(np.float64), TEMPORAL_SHIFT_FRAMES)
    smoothed = moving_average(shifted, SMOOTH_WINDOW)
    affine = AFFINE_A * smoothed + AFFINE_B
    corrected = (1.0 - AFFINE_BLEND) * smoothed + AFFINE_BLEND * affine
    corrected = np.clip(corrected, -1.0, 1.0).astype(np.float32)
    return corrected, {
        "temporal_shift_frames": float(TEMPORAL_SHIFT_FRAMES),
        "smooth_window": float(SMOOTH_WINDOW),
        "affine_a": float(AFFINE_A),
        "affine_b": float(AFFINE_B),
        "affine_blend": float(AFFINE_BLEND),
        "mean_before": float(np.mean(pred)) if len(pred) else 0.0,
        "mean_after": float(np.mean(corrected)) if len(corrected) else 0.0,
    }


def run_base_predictor(args: argparse.Namespace, temp_output: Path) -> dict:
    cmd = [
        sys.executable,
        str(BASE_SCRIPT),
        "--audio",
        str(args.audio),
        "--output",
        str(temp_output),
        "--device",
        str(args.device),
        "--start-sec",
        str(args.start_sec),
        "--hop-sec",
        str(args.hop_sec),
        "--chunk-sec",
        str(args.chunk_sec),
        "--overlap-sec",
        str(args.overlap_sec),
        "--pool-window-sec",
        str(args.pool_window_sec),
    ]
    if args.end_sec is not None:
        cmd.extend(["--end-sec", str(args.end_sec)])
    if args.allow_download:
        cmd.append("--allow-download")
    if args.save_member_preds:
        cmd.append("--save-member-preds")
    result = subprocess.run(cmd, cwd=ROOT.parents[2], text=True, capture_output=True, check=False)
    if result.returncode != 0:
        raise SystemExit(
            "Base latest-refined Valence predictor failed\n"
            f"cmd={' '.join(cmd)}\n"
            f"stdout_tail={result.stdout[-2000:]}\n"
            f"stderr_tail={result.stderr[-4000:]}"
        )
    meta_path = temp_output.with_suffix(temp_output.suffix + ".meta.json")
    if meta_path.exists():
        return json.loads(meta_path.read_text(encoding="utf-8"))
    return {"base_stdout_tail": result.stdout[-2000:]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--hop-sec", type=float, default=0.5)
    parser.add_argument("--start-sec", type=float, default=0.5)
    parser.add_argument("--end-sec", type=float, default=None)
    parser.add_argument("--chunk-sec", type=float, default=30.0)
    parser.add_argument("--overlap-sec", type=float, default=1.0)
    parser.add_argument("--pool-window-sec", type=float, default=0.5)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--save-member-preds", action="store_true")
    args = parser.parse_args()

    with TemporaryDirectory(prefix="musicva_valence_deployable_shifted_") as tmp:
        temp_output = Path(tmp) / "latest_refined_valence.csv"
        base_meta = run_base_predictor(args, temp_output)
        base_df = pd.read_csv(temp_output)

    base_pred = base_df["valence_minus1_1"].to_numpy(dtype=np.float64)
    corrected, post_meta = apply_postprocess(base_pred)

    out = base_df.copy()
    out["latest_refined_valence_minus1_1"] = base_pred
    out["valence_minus1_1"] = corrected
    out["valence_0_1"] = np.clip((corrected + 1.0) / 2.0, 0.0, 1.0)
    out["model_id"] = MODEL_ID
    out["metric_best_reference_model_id"] = CV_REFERENCE["model_id"]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)

    meta = {
        "audio": str(args.audio),
        "output": str(args.output),
        "frames": int(len(out)),
        "model_id": MODEL_ID,
        "base_predictor": str(BASE_SCRIPT.relative_to(ROOT)),
        "base_meta": base_meta,
        "cv_reference": CV_REFERENCE,
        "postprocess": post_meta,
        "label_scale": "[-1, 1]; valence_0_1 is exported for convenience",
        "deployability": (
            "Uses raw-MP3 MERT-based latest_refined predictor plus fixed "
            "validation-selected temporal shift, smoothing, and affine parameters."
        ),
        "caveat": (
            "CV reference is tied to the tonal-correction prediction artifacts; "
            "this full-song script is the matching deployable approximation."
        ),
    }
    meta_path = args.output.with_suffix(args.output.suffix + ".meta.json")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
