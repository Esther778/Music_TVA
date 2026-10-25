#!/usr/bin/env python3
"""Evaluate already-frozen label-free predictions on the final test once."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


PMEMO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN = PMEMO_ROOT / "final_clean_teacher/outputs/final_clean_teacher_v2"
DEFAULT_INDEX = PMEMO_ROOT / "data/splits/pmemo_final_locked_20260713/final_locked_test_index.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    error = prediction - target
    pred_std = prediction.std()
    target_std = target.std()
    pcc = float(np.corrcoef(prediction, target)[0, 1]) if pred_std > 1e-12 and target_std > 1e-12 else 0.0
    covariance = float(np.mean((prediction - prediction.mean()) * (target - target.mean())))
    ccc = float(
        2.0 * covariance
        / (prediction.var() + target.var() + (prediction.mean() - target.mean()) ** 2 + 1e-12)
    )
    return {
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "mae": float(np.mean(np.abs(error))),
        "pcc": pcc,
        "ccc": ccc,
    }


def thresholds(result: dict[str, float]) -> list[dict[str, object]]:
    limits = {
        "rmse": (0.18, 0.15, 0.10, True),
        "mae": (0.15, 0.12, 0.08, True),
        "pcc": (0.80, 0.85, 0.90, False),
        "ccc": (0.80, 0.85, 0.90, False),
    }
    rows = []
    for key, value in result.items():
        minimum, good, excellent, lower = limits[key]
        gap = value - minimum if lower else minimum - value
        rows.append({
            "metric": key,
            "value": value,
            "minimum": minimum,
            "gap_to_minimum": gap,
            "good": good,
            "excellent": excellent,
            "status": "pass" if gap <= 0 else "not_pass",
        })
    return rows


def aggregate(frame: pd.DataFrame, window_sec: Optional[float]) -> pd.DataFrame:
    data = frame.copy()
    group_columns = ["musicId"]
    if window_sec is not None:
        start = data.groupby("musicId")["time_sec"].transform("min")
        data["window_index"] = np.floor((data["time_sec"] - start) / window_sec).astype(int)
        group_columns.append("window_index")
    return data.groupby(group_columns, sort=False)[
        ["pred_valence", "target_valence", "pred_arousal", "target_arousal"]
    ].mean().reset_index()


def evaluate_table(table: pd.DataFrame) -> dict[str, dict[str, float]]:
    return {
        dimension: metrics(table[f"pred_{dimension}"], table[f"target_{dimension}"])
        for dimension in ("valence", "arousal")
    }


def bootstrap_clip_ci(
    clip: pd.DataFrame, dimension: str, samples: int, seed: int
) -> dict[str, list[float]]:
    rng = np.random.default_rng(seed)
    prediction = clip[f"pred_{dimension}"].to_numpy(dtype=np.float64)
    target = clip[f"target_{dimension}"].to_numpy(dtype=np.float64)
    draws = {key: np.empty(samples, dtype=np.float64) for key in ("rmse", "mae", "pcc", "ccc")}
    for sample in range(samples):
        index = rng.integers(0, len(clip), size=len(clip))
        result = metrics(prediction[index], target[index])
        for key in draws:
            draws[key][sample] = result[key]
    return {
        key: [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]
        for key, values in draws.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    args = parser.parse_args()

    output_path = args.run_dir / "FINAL_LOCKED_TEST_RESULTS.json"
    if output_path.exists():
        raise RuntimeError(f"Final results already exist; refusing a second evaluation: {output_path}")

    manifest_path = args.run_dir / "FINAL_PREDICTION_FREEZE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    prediction_path = Path(manifest["prediction_path"])
    if not prediction_path.is_absolute():
        prediction_path = PMEMO_ROOT / prediction_path
    actual_prediction_hash = sha256(prediction_path)
    actual_index_hash = sha256(args.index)
    if actual_prediction_hash != manifest["prediction_sha256"]:
        raise RuntimeError("Frozen prediction hash mismatch")
    if actual_index_hash != manifest["final_index_sha256"]:
        raise RuntimeError("Final index hash mismatch")

    prediction = pd.read_csv(prediction_path)
    final_index = pd.read_csv(args.index)
    if list(prediction.columns) != ["musicId", "time_sec", "pred_valence", "pred_arousal"]:
        raise RuntimeError("Unexpected frozen prediction schema")
    if prediction["musicId"].nunique() != 115 or len(prediction) != 5390:
        raise RuntimeError("Frozen prediction coverage changed")

    target_rows = []
    # This is the single intentional label-unsealing point in the final protocol.
    for item in final_index.itertuples(index=False):
        sequence = np.load(Path(item.path))
        target = sequence["Y"].astype(np.float64)
        time_sec = sequence["time_sec"].astype(np.float64)
        if len(target) != int(item.frames):
            raise RuntimeError(f"Target frame count mismatch for {item.musicId}")
        target_rows.append(pd.DataFrame({
            "musicId": int(item.musicId),
            "time_sec": time_sec,
            "target_valence": target[:, 0],
            "target_arousal": target[:, 1],
        }))
    target = pd.concat(target_rows, ignore_index=True)
    frame = prediction.merge(target, on=["musicId", "time_sec"], validate="one_to_one")
    if len(frame) != 5390 or not np.isfinite(frame.select_dtypes(include=[np.number])).all().all():
        raise RuntimeError("Final frame alignment or finiteness check failed")

    clip = aggregate(frame, None)
    levels = {
        "frame_0.5s": evaluate_table(frame),
        "window_2s": evaluate_table(aggregate(frame, 2.0)),
        "window_5s": evaluate_table(aggregate(frame, 5.0)),
        "window_10s": evaluate_table(aggregate(frame, 10.0)),
        "window_15s": evaluate_table(aggregate(frame, 15.0)),
        "complete_clip": evaluate_table(clip),
    }
    ci = {
        dimension: bootstrap_clip_ci(
            clip, dimension, args.bootstrap_samples, 20260715 + index
        )
        for index, dimension in enumerate(("valence", "arousal"))
    }
    result = {
        "unsealed_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "protocol": "prospective locked final test; predictions frozen before label access",
        "model_ids": {
            "valence": "pmemo_final_clean_valence_teacher_v1",
            "arousal": "pmemo_final_clean_arousal_teacher_v1",
        },
        "prediction_sha256": actual_prediction_hash,
        "final_index_sha256": actual_index_hash,
        "songs": int(clip["musicId"].nunique()),
        "frames": len(frame),
        "primary_level": "complete_clip",
        "levels": levels,
        "complete_clip_bootstrap_95ci": ci,
        "complete_clip_thresholds": {
            dimension: thresholds(levels["complete_clip"][dimension])
            for dimension in ("valence", "arousal")
        },
        "post_test_tuning_allowed": False,
        "note": "Historical multi-stage OOF metric best was not the executable model evaluated here.",
    }
    frame.to_csv(args.run_dir / "final_locked_test_frame_predictions_WITH_LABELS.csv", index=False)
    clip.to_csv(args.run_dir / "final_locked_test_clip_predictions_WITH_LABELS.csv", index=False)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
