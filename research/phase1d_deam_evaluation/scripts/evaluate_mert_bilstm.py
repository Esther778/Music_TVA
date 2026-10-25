from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parents[1]
MERT_ROOT = PROJECT_ROOT / "research" / "phase1b_bert_deam_va_training"
FRAME_HOP_SEC = 0.5
WINDOW_SECONDS = (2, 5, 10, 15)
SHIFT_SECONDS = tuple(range(-4, 5))


@dataclass
class SongCurve:
    song_id: int
    prediction: np.ndarray
    target: np.ndarray


def metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    error = prediction - target
    pred_centered = prediction - prediction.mean(axis=0, keepdims=True)
    target_centered = target - target.mean(axis=0, keepdims=True)
    pcc = np.sum(pred_centered * target_centered, axis=0) / np.maximum(
        np.sqrt(
            np.sum(pred_centered**2, axis=0)
            * np.sum(target_centered**2, axis=0)
        ),
        1e-8,
    )
    pred_mean = prediction.mean(axis=0)
    target_mean = target.mean(axis=0)
    covariance = ((prediction - pred_mean) * (target - target_mean)).mean(axis=0)
    ccc = 2 * covariance / np.maximum(
        prediction.var(axis=0)
        + target.var(axis=0)
        + (pred_mean - target_mean) ** 2,
        1e-8,
    )
    rmse_dim = np.sqrt(np.mean(error**2, axis=0))
    mae_dim = np.mean(np.abs(error), axis=0)
    return {
        "valence_rmse": float(rmse_dim[0]),
        "arousal_rmse": float(rmse_dim[1]),
        "mean_rmse": float(np.mean(rmse_dim)),
        "overall_rmse": float(np.sqrt(np.mean(error**2))),
        "valence_mae": float(mae_dim[0]),
        "arousal_mae": float(mae_dim[1]),
        "mean_mae": float(np.mean(mae_dim)),
        "valence_pearson": float(pcc[0]),
        "arousal_pearson": float(pcc[1]),
        "mean_pearson": float(np.mean(pcc)),
        "valence_ccc": float(ccc[0]),
        "arousal_ccc": float(ccc[1]),
        "mean_ccc": float(np.mean(ccc)),
    }


def load_song_curves(split: str) -> list[SongCurve]:
    dataset_root = MERT_ROOT / "data" / "datasets" / "mert_va"
    index = pd.read_csv(dataset_root / "sequence_index.csv")
    rows = (
        index[index["split"] == split]
        .sort_values("song_id")
        .to_dict("records")
    )
    flattened_prediction = np.load(
        MERT_ROOT
        / "outputs"
        / "predictions"
        / "bilstm"
        / f"bilstm_baseline_{split}_pred.npy"
    ).astype(np.float32)
    curves = []
    offset = 0
    for row in rows:
        with np.load(row["sequence_path"]) as data:
            target = data["Y"].astype(np.float32)
        frame_count = len(target)
        prediction = flattened_prediction[offset : offset + frame_count]
        if len(prediction) != frame_count:
            raise ValueError(f"Prediction length mismatch for song {row['song_id']}.")
        curves.append(
            SongCurve(
                song_id=int(row["song_id"]),
                prediction=prediction.copy(),
                target=target,
            )
        )
        offset += frame_count
    if offset != len(flattened_prediction):
        raise ValueError(
            f"Unused flattened predictions for {split}: {len(flattened_prediction) - offset}"
        )
    return curves


def concatenate(curves: list[SongCurve]) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.concatenate([curve.prediction for curve in curves], axis=0),
        np.concatenate([curve.target for curve in curves], axis=0),
    )


def aggregate_curves(
    curves: list[SongCurve], level: str, window_frames: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    predictions = []
    targets = []
    for curve in curves:
        if level == "frame":
            predictions.append(curve.prediction)
            targets.append(curve.target)
        elif level == "clip":
            predictions.append(curve.prediction.mean(axis=0, keepdims=True))
            targets.append(curve.target.mean(axis=0, keepdims=True))
        elif level == "fixed_window":
            assert window_frames is not None
            full_windows = len(curve.target) // window_frames
            for index in range(full_windows):
                start = index * window_frames
                end = start + window_frames
                predictions.append(
                    curve.prediction[start:end].mean(axis=0, keepdims=True)
                )
                targets.append(curve.target[start:end].mean(axis=0, keepdims=True))
        else:
            raise ValueError(level)
    return np.concatenate(predictions), np.concatenate(targets)


def evaluation_rows(
    model: str, split: str, variant: str, curves: list[SongCurve]
) -> list[dict[str, object]]:
    rows = []
    for level, window_sec in [("frame", None), *[("fixed_window", value) for value in WINDOW_SECONDS], ("clip", None)]:
        window_frames = (
            None if window_sec is None else int(round(window_sec / FRAME_HOP_SEC))
        )
        prediction, target = aggregate_curves(curves, level, window_frames)
        rows.append(
            {
                "model": model,
                "split": split,
                "variant": variant,
                "level": level,
                "window_size_sec": "" if window_sec is None else window_sec,
                "units": len(target),
                **metrics(prediction, target),
            }
        )
    return rows


def constant_baseline_curves(
    source: list[SongCurve], mean_value: np.ndarray
) -> list[SongCurve]:
    return [
        SongCurve(
            song_id=curve.song_id,
            prediction=np.broadcast_to(mean_value, curve.target.shape).copy(),
            target=curve.target,
        )
        for curve in source
    ]


def shifted_curves(curves: list[SongCurve], shift_seconds: int) -> list[SongCurve]:
    shift_frames = int(round(shift_seconds / FRAME_HOP_SEC))
    shifted = []
    for curve in curves:
        if shift_frames > 0:
            prediction = curve.prediction[:-shift_frames]
            target = curve.target[shift_frames:]
        elif shift_frames < 0:
            prediction = curve.prediction[-shift_frames:]
            target = curve.target[:shift_frames]
        else:
            prediction = curve.prediction
            target = curve.target
        if len(target):
            shifted.append(SongCurve(curve.song_id, prediction, target))
    return shifted


def moving_average(values: np.ndarray, width: int) -> np.ndarray:
    kernel = np.ones(width, dtype=np.float32) / width
    return np.stack(
        [np.convolve(values[:, dim], kernel, mode="same") for dim in range(2)],
        axis=1,
    )


def smooth_array(values: np.ndarray, method: str, parameter: float) -> np.ndarray:
    if method == "none":
        return values.copy()
    if method == "moving_average":
        return moving_average(values, int(parameter))
    if method == "gaussian":
        return gaussian_filter1d(values, sigma=parameter, axis=0, mode="nearest")
    if method == "savgol":
        window = int(parameter)
        if len(values) < window:
            return values.copy()
        return savgol_filter(values, window_length=window, polyorder=2, axis=0, mode="interp")
    raise ValueError(method)


def smooth_curves(
    curves: list[SongCurve], method: str, parameter: float
) -> list[SongCurve]:
    return [
        SongCurve(
            curve.song_id,
            smooth_array(curve.prediction, method, parameter).astype(np.float32),
            curve.target,
        )
        for curve in curves
    ]


def fit_affine(curves: list[SongCurve]) -> tuple[np.ndarray, np.ndarray]:
    prediction, target = concatenate(curves)
    scale = np.zeros(2, dtype=np.float32)
    bias = np.zeros(2, dtype=np.float32)
    for dim in range(2):
        design = np.column_stack(
            [prediction[:, dim], np.ones(len(prediction), dtype=np.float32)]
        )
        solution, *_ = np.linalg.lstsq(design, target[:, dim], rcond=None)
        scale[dim], bias[dim] = solution
    return scale, bias


def calibrate_curves(
    curves: list[SongCurve], scale: np.ndarray, bias: np.ndarray
) -> list[SongCurve]:
    return [
        SongCurve(
            curve.song_id,
            (curve.prediction * scale + bias).astype(np.float32),
            curve.target,
        )
        for curve in curves
    ]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    output_dir = ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    curves = {split: load_song_curves(split) for split in ("train", "val", "test")}

    train_target = np.concatenate(
        [curve.target for curve in curves["train"]], axis=0
    )
    train_mean = train_target.mean(axis=0)

    evaluation = []
    for split in ("val", "test"):
        evaluation.extend(
            evaluation_rows(
                "training_mean_baseline",
                split,
                "raw",
                constant_baseline_curves(curves[split], train_mean),
            )
        )
        evaluation.extend(
            evaluation_rows(
                "mert_bilstm_baseline", split, "raw", curves[split]
            )
        )

    shift_rows = []
    for shift_sec in SHIFT_SECONDS:
        for split in ("val", "test"):
            prediction, target = concatenate(
                shifted_curves(curves[split], shift_sec)
            )
            shift_rows.append(
                {
                    "split": split,
                    "shift_seconds": shift_sec,
                    "convention": "prediction_at_t_compared_with_target_at_t_plus_shift",
                    "frames": len(target),
                    **metrics(prediction, target),
                }
            )

    smoothing_candidates = [
        ("none", 0.0),
        ("moving_average", 3.0),
        ("moving_average", 5.0),
        ("moving_average", 9.0),
        ("gaussian", 0.5),
        ("gaussian", 1.0),
        ("gaussian", 2.0),
        ("savgol", 5.0),
        ("savgol", 9.0),
    ]
    smoothing_rows = []
    for method, parameter in smoothing_candidates:
        smoothed = smooth_curves(curves["val"], method, parameter)
        prediction, target = concatenate(smoothed)
        smoothing_rows.append(
            {
                "method": method,
                "parameter": parameter,
                **metrics(prediction, target),
            }
        )
    best_smoothing = min(smoothing_rows, key=lambda row: row["overall_rmse"])
    selected_val = smooth_curves(
        curves["val"],
        str(best_smoothing["method"]),
        float(best_smoothing["parameter"]),
    )
    selected_test = smooth_curves(
        curves["test"],
        str(best_smoothing["method"]),
        float(best_smoothing["parameter"]),
    )
    scale, bias = fit_affine(selected_val)
    calibrated_val = calibrate_curves(selected_val, scale, bias)
    calibrated_test = calibrate_curves(selected_test, scale, bias)
    evaluation.extend(
        evaluation_rows(
            "mert_bilstm_baseline",
            "val",
            "validation_selected_smoothing",
            selected_val,
        )
    )
    evaluation.extend(
        evaluation_rows(
            "mert_bilstm_baseline",
            "test",
            "validation_selected_smoothing",
            selected_test,
        )
    )
    evaluation.extend(
        evaluation_rows(
            "mert_bilstm_baseline",
            "val",
            "validation_selected_smoothing_plus_affine_calibration",
            calibrated_val,
        )
    )
    evaluation.extend(
        evaluation_rows(
            "mert_bilstm_baseline",
            "test",
            "validation_selected_smoothing_plus_affine_calibration",
            calibrated_test,
        )
    )

    write_csv(output_dir / "multi_level_evaluation.csv", evaluation)
    write_csv(output_dir / "label_shift_scan.csv", shift_rows)
    write_csv(output_dir / "smoothing_validation_scan.csv", smoothing_rows)
    summary = {
        "label_scale": "[-1,+1]",
        "label_start_sec": 15.0,
        "frame_hop_sec": FRAME_HOP_SEC,
        "split_songs": {split: len(curves[split]) for split in curves},
        "train_target_mean": train_mean.tolist(),
        "selected_smoothing": best_smoothing,
        "affine_calibration": {
            "scale": scale.tolist(),
            "bias": bias.tolist(),
            "fit_split": "validation",
        },
        "label_shift_convention": (
            "positive shift compares prediction at t with annotation at t+shift; "
            "use validation for diagnosis and retrain before final use"
        ),
    }
    with (output_dir / "diagnostic_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
