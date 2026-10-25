#!/usr/bin/env python3
"""Predict full-song Valence with the current metric-best global policy.

This script wraps the existing raw-MP3 Valence approximation and applies the
new global song-level mean policy:

    pred_only + Ridge(alpha=1.0), strength=0.5, cap=0.070

The correction uses only statistics of the predicted Valence curve, so it is
compatible with arbitrary complete-song inference.
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
BASE_SCRIPT = ROOT / "scripts" / "predict_full_song_valence_current_best_approx.py"
DEFAULT_OUT_DIR = ROOT / "outputs" / "valence_fullsong_global_policy"
DEFAULT_POLICY_PARAMS = ROOT / "outputs" / "valence_global_policy_params_20260707" / "pred_only_ridge_a1_strength050_cap070.json"

MODEL_ID = "pmemo_valence_current_best_song_mean_global_policy_fullsong_20260707"
METRIC_BEST_REFERENCE = {
    "model_id": "pmemo_valence_current_best_song_mean_global_policy_20260707",
    "rmse": 0.19078885880411342,
    "mae": 0.1486967345561368,
    "pcc": 0.8074956914596779,
    "ccc": 0.7951366852830063,
}
PREVIOUS_REFERENCE = {
    "model_id": "pmemo_valence_parsimony_output_guarded_20260707",
    "rmse": 0.19127265472577384,
    "mae": 0.14892050958155043,
    "pcc": 0.8060728304873086,
    "ccc": 0.7915010047864408,
}

def song_stats(pred: np.ndarray, time: np.ndarray) -> dict[str, float]:
    n = len(pred)
    head_n = min(6, n)
    tail_n = min(6, n)
    return {
        "frames": float(n),
        "duration_proxy": float(time.max() - time.min() + 0.5) if n else 0.0,
        "pred_mean": float(np.mean(pred)),
        "pred_std": float(np.std(pred)),
        "pred_min": float(np.min(pred)),
        "pred_max": float(np.max(pred)),
        "pred_range": float(np.max(pred) - np.min(pred)),
        "pred_q10": float(np.quantile(pred, 0.10)),
        "pred_q90": float(np.quantile(pred, 0.90)),
        "pred_head_mean": float(np.mean(pred[:head_n])),
        "pred_tail_mean": float(np.mean(pred[-tail_n:])),
    }


def load_policy_params(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(
            f"Policy params not found: {path}\n"
            "Run scripts/materialize_pmemo_valence_global_policy_params.py first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def ridge_target_mean(stats: dict[str, float], params: dict) -> float:
    feature_cols = params["feature_cols"]
    x = np.asarray([float(stats[col]) for col in feature_cols], dtype=np.float64)
    mean = np.asarray(params["scaler_mean"], dtype=np.float64)
    scale = np.asarray(params["scaler_scale"], dtype=np.float64)
    coef = np.asarray(params["ridge_coef"], dtype=np.float64)
    intercept = float(params["ridge_intercept"])
    z = (x - mean) / np.maximum(scale, 1e-12)
    return float(np.dot(z, coef) + intercept)


def apply_global_policy(pred: np.ndarray, time: np.ndarray, params: dict) -> tuple[np.ndarray, dict[str, float]]:
    stats = song_stats(pred, time)
    target_mean_pred = ridge_target_mean(stats, params)
    raw_shift = target_mean_pred - stats["pred_mean"]
    shift_cap = float(params["shift_cap"])
    policy_strength = float(params["policy_strength"])
    capped_shift = float(np.clip(raw_shift, -shift_cap, shift_cap))
    applied_shift = policy_strength * capped_shift
    corrected = np.clip(pred + applied_shift, -1.0, 1.0).astype(np.float32)
    policy_meta = {
        **stats,
        "ridge_alpha": float(params["ridge_alpha"]),
        "policy_strength": policy_strength,
        "shift_cap": shift_cap,
        "target_mean_pred": float(target_mean_pred),
        "raw_song_mean_shift": float(raw_shift),
        "capped_song_mean_shift": float(capped_shift),
        "applied_song_mean_shift": float(applied_shift),
    }
    return corrected, policy_meta


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
            "Base Valence predictor failed\n"
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
    parser.add_argument("--policy-params", type=Path, default=DEFAULT_POLICY_PARAMS)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--save-member-preds", action="store_true")
    args = parser.parse_args()

    with TemporaryDirectory(prefix="musicva_valence_global_") as tmp:
        temp_output = Path(tmp) / "base_valence.csv"
        base_meta = run_base_predictor(args, temp_output)
        base_df = pd.read_csv(temp_output)

    time = base_df["time_sec"].to_numpy(dtype=np.float64)
    base_pred = base_df["valence_minus1_1"].to_numpy(dtype=np.float64)
    policy_params = load_policy_params(args.policy_params)
    corrected, policy_meta = apply_global_policy(base_pred, time, policy_params)

    out = base_df.copy()
    out["base_global_policy_input_valence_minus1_1"] = base_pred
    out["global_policy_shift"] = policy_meta["applied_song_mean_shift"]
    out["valence_minus1_1"] = corrected
    out["valence_0_1"] = np.clip((corrected + 1.0) / 2.0, 0.0, 1.0)
    out["model_id"] = MODEL_ID
    out["metric_best_reference_model_id"] = METRIC_BEST_REFERENCE["model_id"]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)

    meta = {
        "audio": str(args.audio),
        "output": str(args.output),
        "frames": int(len(out)),
        "model_id": MODEL_ID,
        "metric_best_reference": METRIC_BEST_REFERENCE,
        "previous_reference": PREVIOUS_REFERENCE,
        "base_predictor": str(BASE_SCRIPT.relative_to(ROOT)),
        "base_meta": base_meta,
        "global_policy": {
            "feature_set": "pred_only",
            "params_path": str(args.policy_params),
            "params_model_id": policy_params.get("model_id"),
            "model": "StandardScaler + Ridge(alpha=1.0)",
            "feature_cols": policy_params.get("feature_cols"),
            "strength": policy_params.get("policy_strength"),
            "cap": policy_params.get("shift_cap"),
            "note": (
                "The formal CV metric-best selected this policy configuration "
                "under validation guard. This script uses materialized train-fold "
                "Ridge parameters for raw-MP3 inference."
            ),
            **policy_meta,
        },
        "label_scale": "[-1, 1]; valence_0_1 is exported for convenience",
        "training_caveat": "PMEmo dynamic annotations exclude the unstable first 15 seconds; predictions before 15s should be interpreted cautiously.",
    }
    meta_path = args.output.with_suffix(args.output.suffix + ".meta.json")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
