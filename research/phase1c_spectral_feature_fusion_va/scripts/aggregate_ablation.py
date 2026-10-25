from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parents[1]
OPENL3_ROOT = (
    PROJECT_ROOT / "research" / "phase1_va_prediction" / "openl3_va_baseline"
)
MERT_ROOT = PROJECT_ROOT / "research" / "phase1b_bert_deam_va_training"


def select_metrics(source: dict, prefix: str) -> dict[str, float]:
    key_map = {
        "rmse": "rmse",
        "mae": "mae",
        "rmse_valence": "rmse_valence",
        "rmse_arousal": "rmse_arousal",
        "mae_valence": "mae_valence",
        "mae_arousal": "mae_arousal",
        "pcc_valence": "pcc_valence",
        "pcc_arousal": "pcc_arousal",
        "ccc_valence": "ccc_valence",
        "ccc_arousal": "ccc_arousal",
        "mean_ccc": "mean_ccc",
    }
    row = {}
    for output_key, source_key in key_map.items():
        candidates = [
            f"{prefix}_{source_key}",
            f"{prefix}_{source_key.replace('pcc_', 'pearson_')}",
        ]
        for candidate in candidates:
            if candidate in source and not pd.isna(source[candidate]):
                row[output_key] = float(source[candidate])
                break
        else:
            raise KeyError(f"Missing {prefix} metric {source_key}")
    return row


def mean_baseline() -> dict[str, float]:
    dataset_dir = MERT_ROOT / "data" / "datasets" / "mert_va" / "frame"
    train = np.load(dataset_dir / "Y_train.npy")
    test = np.load(dataset_dir / "Y_test.npy")
    prediction = np.broadcast_to(train.mean(axis=0), test.shape)
    error = prediction - test
    return {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "rmse_valence": float(np.sqrt(np.mean(error[:, 0] ** 2))),
        "rmse_arousal": float(np.sqrt(np.mean(error[:, 1] ** 2))),
        "mae_valence": float(np.mean(np.abs(error[:, 0]))),
        "mae_arousal": float(np.mean(np.abs(error[:, 1]))),
        "pcc_valence": 0.0,
        "pcc_arousal": 0.0,
        "ccc_valence": 0.0,
        "ccc_arousal": 0.0,
        "mean_ccc": 0.0,
    }


def array_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
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
    return {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "rmse_valence": float(np.sqrt(np.mean(error[:, 0] ** 2))),
        "rmse_arousal": float(np.sqrt(np.mean(error[:, 1] ** 2))),
        "mae_valence": float(np.mean(np.abs(error[:, 0]))),
        "mae_arousal": float(np.mean(np.abs(error[:, 1]))),
        "pcc_valence": float(pcc[0]),
        "pcc_arousal": float(pcc[1]),
        "ccc_valence": float(ccc[0]),
        "ccc_arousal": float(ccc[1]),
        "mean_ccc": float(np.mean(ccc)),
    }


def main() -> None:
    rows = [
        {
            "model": "Training mean baseline",
            "feature": "none",
            "temporal_head": "constant",
            "val_rmse": "",
            **{f"test_{key}": value for key, value in mean_baseline().items()},
        }
    ]

    openl3_path = (
        OPENL3_ROOT
        / "outputs"
        / "results"
        / "mlp_song_level_aligned15"
        / "openl3_mlp_song_level_aligned15_seed2026_summary.json"
    )
    with openl3_path.open("r", encoding="utf-8") as handle:
        openl3 = json.load(handle)
    rows.append(
        {
            "model": "OpenL3 + MLP",
            "feature": "OpenL3 512-d",
            "temporal_head": "frame MLP",
            "val_rmse": float(openl3["val_rmse"]),
            **{
                f"test_{key}": value
                for key, value in select_metrics(openl3, "test").items()
            },
        }
    )

    comparison = pd.read_csv(MERT_ROOT / "outputs" / "results" / "model_comparison.csv")
    mert = comparison[
        (comparison["model_family"] == "bilstm")
        & (comparison["trial_id"] == "bilstm_baseline")
    ].iloc[0].to_dict()
    mert_prediction = np.load(
        MERT_ROOT / "outputs" / "predictions" / "bilstm" / "bilstm_baseline_test_pred.npy"
    )
    mert_target = np.load(
        MERT_ROOT / "data" / "datasets" / "mert_va" / "frame" / "Y_test.npy"
    )
    rows.append(
        {
            "model": "MERT + BiLSTM",
            "feature": "MERT-v1-95M last layer",
            "temporal_head": "BiLSTM",
            "val_rmse": float(mert["val_rmse"]),
            **{
                f"test_{key}": value
                for key, value in array_metrics(
                    mert_prediction, mert_target
                ).items()
            },
        }
    )

    for mode, label, feature in (
        ("handcrafted", "Handcrafted + BiLSTM", "Librosa 118-d"),
        ("logmel", "Log-Mel DAMFF-inspired", "log-Mel 64 bins"),
        ("fusion", "Log-Mel + handcrafted fusion", "log-Mel + Librosa"),
    ):
        with (ROOT / "outputs" / mode / "summary.json").open(
            "r", encoding="utf-8"
        ) as handle:
            result = json.load(handle)
        rows.append(
            {
                "model": label,
                "feature": feature,
                "temporal_head": "multi-scale CNN/CBAM/BiLSTM"
                if mode != "handcrafted"
                else "BiLSTM",
                "val_rmse": float(result["val_rmse"]),
                **{
                    f"test_{key}": value
                    for key, value in select_metrics(result, "test").items()
                },
            }
        )

    for row in rows:
        row["test_mean_pcc"] = (
            float(row["test_pcc_valence"]) + float(row["test_pcc_arousal"])
        ) / 2.0

    output_dir = ROOT / "outputs" / "ablation"
    output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with (output_dir / "comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "comparison.json").open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, ensure_ascii=False, indent=2)
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
