#!/usr/bin/env python3
"""Materialize targeted PMEmo Valence song-mean correction.

Rule:
- start from current best predictions;
- use the validation-safe song-mean selected output only when the split's
  validation RMSE improves by at least 0.003 and validation CCC improves by at
  least 0.003;
- otherwise keep no-change.

This rule targets the audited bottleneck split without forcing small/unstable
corrections onto splits that are already good.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from search_pmemo_valence_song_mean_fusion import eval_df
from train_pmemo_mert_valence_baseline import ROOT, threshold_rows


CURRENT_DIR = ROOT / "outputs" / "valence_prediction_only_rescale_ccc_preserving_20260628"
VALSAFE_DIR = ROOT / "outputs" / "valence_song_mean_valsafe_selection_20260628"
OUT_DIR = ROOT / "outputs" / "valence_targeted_split02_song_mean_20260628"

CURRENT_BEST = {
    "model_id": "pmemo_valence_prediction_only_rescale_ccc_preserving_20260628",
    "rmse": 0.19531630337282654,
    "mae": 0.15179290177231075,
    "pcc": 0.7977655929469952,
    "ccc": 0.7848342582143512,
}


def aggregate(rows: list[dict[str, Any]], variant: str) -> dict[str, float]:
    out = {}
    for metric in ["rmse", "mae", "pcc", "ccc"]:
        vals = np.asarray([row[variant]["test"][metric] for row in rows], dtype=np.float64)
        out[f"{variant}_test_{metric}_mean"] = float(vals.mean())
        out[f"{variant}_test_{metric}_std"] = float(vals.std(ddof=0))
    return out


def copy_split(src_dir: Path, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for split in ["train", "val", "test"]:
        shutil.copy2(src_dir / f"{split}_predictions.csv", dst_dir / f"{split}_predictions.csv")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    valsafe_summary = json.loads((VALSAFE_DIR / "summary.json").read_text(encoding="utf-8"))
    by_split = {row["split_name"]: row for row in valsafe_summary["per_split"]}
    rows = []
    for split_dir in sorted(CURRENT_DIR.glob("split_*_seed_*")):
        split_name = split_dir.name
        baseline = {
            split: eval_df(pd.read_csv(split_dir / f"{split}_predictions.csv"))
            for split in ["train", "val", "test"]
        }
        valsafe_row = by_split[split_name]
        val_delta = valsafe_row["delta_vs_current_best_val"]
        use_selected = val_delta["rmse"] <= -0.003 and val_delta["ccc"] >= 0.003
        source_dir = VALSAFE_DIR / split_name if use_selected else split_dir
        copy_split(source_dir, OUT_DIR / split_name)
        selected = {
            split: eval_df(pd.read_csv((OUT_DIR / split_name) / f"{split}_predictions.csv"))
            for split in ["train", "val", "test"]
        }
        rows.append(
            {
                "split_name": split_name,
                "decision": "valsafe_song_mean" if use_selected else "no_change",
                "decision_rule": "use selected if val_rmse_delta <= -0.003 and val_ccc_delta >= 0.003",
                "valsafe_selected_config": valsafe_row.get("selected_config"),
                "baseline": baseline,
                "selected": selected,
                "delta_vs_current_best_test": {
                    metric: float(selected["test"][metric] - baseline["test"][metric])
                    for metric in ["rmse", "mae", "pcc", "ccc"]
                },
                "delta_vs_current_best_val": {
                    metric: float(selected["val"][metric] - baseline["val"][metric])
                    for metric in ["rmse", "mae", "pcc", "ccc"]
                },
            }
        )

    baseline_mean = aggregate(rows, "baseline")
    selected_mean = aggregate(rows, "selected")
    selected_metrics = {
        "rmse": selected_mean["selected_test_rmse_mean"],
        "mae": selected_mean["selected_test_mae_mean"],
        "pcc": selected_mean["selected_test_pcc_mean"],
        "ccc": selected_mean["selected_test_ccc_mean"],
    }
    summary = {
        "created_at": "2026-06-28",
        "model_id": "pmemo_valence_targeted_split02_song_mean_20260628",
        "input_model": CURRENT_BEST,
        "source_valsafe_output": str(VALSAFE_DIR.relative_to(ROOT)),
        "selection_rule": "Apply validation-safe song-mean correction only if val_rmse_delta <= -0.003 and val_ccc_delta >= 0.003.",
        "num_splits": len(rows),
        "per_split": rows,
        **baseline_mean,
        **selected_mean,
        "selected_mean_test_gap_rows": threshold_rows(selected_metrics),
        "delta_vs_current_best_test_mean": {
            metric: float(selected_metrics[metric] - CURRENT_BEST[metric])
            for metric in ["rmse", "mae", "pcc", "ccc"]
        },
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
