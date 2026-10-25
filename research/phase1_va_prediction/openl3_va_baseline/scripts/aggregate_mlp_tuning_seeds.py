from __future__ import annotations

import csv
import json
import statistics as stats
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "outputs" / "results" / "mlp_tuning"
BASE_TRIAL = "mlp_tune_w512_d010_lr2e4_e120"
SEEDS = [2026, 3407, 614]
METRICS = [
    "val_rmse",
    "val_mean_ccc",
    "test_rmse",
    "test_mae",
    "test_rmse_valence",
    "test_rmse_arousal",
    "test_pearson_valence",
    "test_pearson_arousal",
    "test_ccc_valence",
    "test_ccc_arousal",
    "test_mean_ccc",
]


def main() -> None:
    rows = []
    for seed in SEEDS:
        path = RESULT_DIR / f"{BASE_TRIAL}_seed{seed}_summary.json"
        with path.open("r", encoding="utf-8") as handle:
            rows.append(json.load(handle))

    summary: dict[str, object] = {
        "base_trial": BASE_TRIAL,
        "seeds": SEEDS,
        "n": len(rows),
        "rows": [
            {
                "seed": row["seed"],
                "best_epoch": row["best_epoch"],
                "checkpoint": row["checkpoint"],
                **{metric: row[metric] for metric in METRICS},
            }
            for row in rows
        ],
    }
    for metric in METRICS:
        values = [float(row[metric]) for row in rows]
        summary[f"{metric}_mean"] = stats.mean(values)
        summary[f"{metric}_std"] = stats.stdev(values) if len(values) > 1 else 0.0

    json_path = RESULT_DIR / "mlp_tuning_seed_summary.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    csv_path = RESULT_DIR / "mlp_tuning_seed_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "mean", "std"])
        for metric in METRICS:
            writer.writerow([metric, summary[f"{metric}_mean"], summary[f"{metric}_std"]])

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
