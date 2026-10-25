from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
from sklearn.isotonic import IsotonicRegression

from train_bilstm_full_song_transfer import metrics


def moving_average(series: np.ndarray, frames: int) -> np.ndarray:
    if frames <= 1:
        return series.copy()
    if frames % 2 == 0:
        frames += 1
    half = frames // 2
    padded = np.pad(series, (half, half), mode="edge")
    kernel = np.ones(frames, dtype=np.float64) / frames
    return np.convolve(padded, kernel, mode="valid")


def smooth_dimension(
    values: np.ndarray,
    song_ids: np.ndarray,
    method: str,
    parameter: float,
) -> np.ndarray:
    result = values.copy()
    for song_id in np.unique(song_ids):
        selected = song_ids == song_id
        series = values[selected]
        if method == "none":
            result[selected] = series
        elif method == "moving_average":
            result[selected] = moving_average(series, int(parameter))
        elif method == "gaussian":
            result[selected] = gaussian_filter1d(
                series, sigma=float(parameter), mode="nearest"
            )
        elif method == "savgol":
            window = int(parameter)
            if window % 2 == 0:
                window += 1
            if window >= len(series):
                window = len(series) - 1 if len(series) % 2 == 0 else len(series)
            if window < 5:
                result[selected] = series
            else:
                result[selected] = savgol_filter(
                    series,
                    window_length=window,
                    polyorder=2,
                    mode="nearest",
                )
        else:
            raise ValueError(f"Unknown smoothing method: {method}")
    return result


def calibrate_dimension(
    prediction: np.ndarray,
    target: np.ndarray,
    method: str,
) -> tuple[np.ndarray, dict]:
    x = prediction.astype(np.float64)
    y = target.astype(np.float64)
    if method == "none":
        return prediction.copy(), {"method": method}
    if method == "linear":
        variance = np.var(x)
        if variance <= 1e-12:
            scale = 1.0
            bias = float(y.mean() - x.mean())
        else:
            covariance = np.mean((x - x.mean()) * (y - y.mean()))
            scale = float(covariance / variance)
            bias = float(y.mean() - scale * x.mean())
        return (scale * prediction + bias).astype(np.float32), {
            "method": method,
            "scale": scale,
            "bias": bias,
        }
    if method == "quadratic":
        design = np.column_stack([x**2, x, np.ones_like(x)])
        coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
        calibrated = design @ coefficients
        return calibrated.astype(np.float32), {
            "method": method,
            "a2": float(coefficients[0]),
            "a1": float(coefficients[1]),
            "bias": float(coefficients[2]),
        }
    if method == "isotonic":
        model = IsotonicRegression(out_of_bounds="clip")
        calibrated = model.fit_transform(x, y)
        return calibrated.astype(np.float32), {
            "method": method,
            "threshold_count": int(len(model.X_thresholds_)),
        }
    raise ValueError(f"Unknown calibration method: {method}")


def search(args: argparse.Namespace) -> dict:
    output_dir = args.output_dir
    prediction = np.load(output_dir / "hybrid_validation_pred.npy")
    target = np.load(output_dir / "hybrid_validation_target.npy")
    song_ids = np.load(output_dir / "validation_song_ids.npy")

    valence_smoothing = []
    valence_smoothing.append(("none", 0))
    valence_smoothing.extend(
        ("moving_average", frames)
        for frames in [131, 141, 151, 161, 171, 181, 191, 201, 211, 221]
    )
    valence_smoothing.extend(
        ("gaussian", sigma)
        for sigma in [16, 20, 24, 28, 32, 36, 40, 44]
    )
    valence_smoothing.extend(
        ("savgol", frames)
        for frames in [131, 151, 171, 191, 211]
    )

    arousal_smoothing = []
    arousal_smoothing.append(("none", 0))
    arousal_smoothing.extend(
        ("moving_average", frames)
        for frames in [21, 31, 41, 51, 61]
    )
    arousal_smoothing.extend(
        ("gaussian", sigma)
        for sigma in [6, 8, 10, 12, 14]
    )
    arousal_smoothing.extend(
        ("savgol", frames)
        for frames in [21, 31, 41, 51, 61]
    )

    calibration_methods = ["linear", "quadratic", "isotonic"]
    rows = []
    best_mean = None
    best_valence_rmse = None
    best_valence_ccc = None
    best_minimum_candidate = None
    best_prediction = None

    valence_cache = {}
    arousal_cache = {}
    for method, parameter in valence_smoothing:
        valence_cache[(method, parameter)] = smooth_dimension(
            prediction[:, 0], song_ids, method, parameter
        )
    for method, parameter in arousal_smoothing:
        arousal_cache[(method, parameter)] = smooth_dimension(
            prediction[:, 1], song_ids, method, parameter
        )

    for valence_key, valence_smoothed in valence_cache.items():
        for arousal_key, arousal_smoothed in arousal_cache.items():
            for valence_calibration in calibration_methods:
                calibrated_valence, valence_parameters = calibrate_dimension(
                    valence_smoothed,
                    target[:, 0],
                    valence_calibration,
                )
                for arousal_calibration in calibration_methods:
                    calibrated_arousal, arousal_parameters = calibrate_dimension(
                        arousal_smoothed,
                        target[:, 1],
                        arousal_calibration,
                    )
                    calibrated = np.column_stack(
                        [calibrated_valence, calibrated_arousal]
                    )
                    result = metrics(calibrated, target)
                    row = {
                        "valence_smoothing": {
                            "method": valence_key[0],
                            "parameter": valence_key[1],
                        },
                        "arousal_smoothing": {
                            "method": arousal_key[0],
                            "parameter": arousal_key[1],
                        },
                        "valence_calibration": valence_parameters,
                        "arousal_calibration": arousal_parameters,
                        **result,
                    }
                    rows.append(row)
                    if best_mean is None or row["rmse"] < best_mean["rmse"]:
                        best_mean = row
                        best_prediction = calibrated
                    if (
                        best_valence_rmse is None
                        or row["rmse_valence"]
                        < best_valence_rmse["rmse_valence"]
                    ):
                        best_valence_rmse = row
                    if (
                        best_valence_ccc is None
                        or row["ccc_valence"] > best_valence_ccc["ccc_valence"]
                    ):
                        best_valence_ccc = row
                    if (
                        row["rmse"] <= 0.18
                        and row["mae"] <= 0.15
                        and row["mean_pcc"] >= 0.80
                        and row["mean_ccc"] >= 0.80
                    ):
                        if (
                            best_minimum_candidate is None
                            or row["rmse_valence"]
                            < best_minimum_candidate["rmse_valence"]
                        ):
                            best_minimum_candidate = row

    summary = {
        "search_id": "validation_postprocess_v2_smoothing_nonlinear_calibration",
        "validation_selected": True,
        "test_used": False,
        "input_dir": str(output_dir),
        "selection_note": (
            "Validation-only search. Isotonic and quadratic calibration are "
            "diagnostic because they can overfit this small validation set."
        ),
        "best_mean_rmse": best_mean,
        "best_valence_rmse": best_valence_rmse,
        "best_valence_ccc": best_valence_ccc,
        "best_minimum_threshold_candidate": best_minimum_candidate,
        "all_trials": rows,
    }
    (output_dir / "validation_postprocess_v2_search.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    np.save(
        output_dir / "hybrid_validation_pred_postprocess_v2_best_mean.npy",
        best_prediction.astype(np.float32),
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(search(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
