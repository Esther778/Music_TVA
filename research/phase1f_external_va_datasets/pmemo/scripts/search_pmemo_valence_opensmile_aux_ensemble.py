#!/usr/bin/env python3
"""Split-matched OpenSMILE Valence auxiliary ensemble.

OpenSMILE Valence is much weaker than the current MERT-based Valence teacher,
but it is a different audio feature family. This script trains the best
lightweight OpenSMILE Valence trial inside the same song-level splits as the
current Valence evaluation, then searches a validation-selected global ensemble
weight.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from train_pmemo_opensmile_valence_trials import DEFAULT_TRIALS, ROOT, resolve_device, run_trial, threshold_rows
from train_pmemo_mert_valence_baseline import metrics


BASE_SPLIT_DIR = ROOT / "outputs" / "valence_multisplit_best_20260627"
OPENSMILE_DATASET_DIR = ROOT / "outputs" / "opensmile_baseline" / "dataset"
CURRENT_DIR = ROOT / "outputs" / "valence_targeted_multilayer_postprocess_20260629"
OUT_DIR = ROOT / "outputs" / "valence_opensmile_aux_ensemble_20260629"

OPENSMILE_TRIAL_ID = "v_only_h96_l1_do015_mse"
CURRENT_BEST = {
    "model_id": "pmemo_valence_targeted_multilayer_postprocess_20260629",
    "rmse": 0.19270108142080475,
    "mae": 0.15030960055717216,
    "pcc": 0.8029485501074519,
    "ccc": 0.7870809597955878,
}
PASS = {"rmse": 0.18, "mae": 0.15, "pcc": 0.80, "ccc": 0.80}


def trial_config():
    for cfg in DEFAULT_TRIALS:
        if cfg.trial_id == OPENSMILE_TRIAL_ID:
            return cfg
    raise ValueError(OPENSMILE_TRIAL_ID)


def materialize_dataset(split_dir: Path, split_name: str) -> Path:
    out = OUT_DIR / "opensmile_by_split" / split_name / "dataset"
    out.mkdir(parents=True, exist_ok=True)
    split_index = pd.read_csv(split_dir / "sequence_index.csv")[["musicId", "split", "frames"]]
    base = pd.read_csv(OPENSMILE_DATASET_DIR / "sequence_index.csv")[["musicId", "path"]]
    merged = split_index.merge(base, on="musicId", how="left", validate="one_to_one")
    if merged["path"].isna().any():
        missing = merged.loc[merged["path"].isna(), "musicId"].tolist()[:10]
        raise ValueError(f"Missing OpenSMILE paths for {split_name}: {missing}")
    merged.to_csv(out / "sequence_index.csv", index=False)
    return out


def train_or_load_opensmile(split_dir: Path, split_name: str, device, seed: int) -> Path:
    out = OUT_DIR / "opensmile_by_split" / split_name
    trial_dir = out / OPENSMILE_TRIAL_ID
    required = [trial_dir / f"{split}_predictions.csv" for split in ["train", "val", "test"]]
    if all(p.exists() for p in required) and (trial_dir / "summary.json").exists():
        return trial_dir
    data_dir = materialize_dataset(split_dir, split_name)
    run_trial(trial_config(), data_dir, out, device, seed)
    return trial_dir


def eval_df(df: pd.DataFrame) -> dict[str, float]:
    return metrics(df["pred_valence"].to_numpy(dtype=np.float64), df["target_valence"].to_numpy(dtype=np.float64))


def macro(split_rows: list[dict[str, float]], prefix: str) -> dict[str, float]:
    return {k: float(np.mean([r[f"{prefix}_{k}"] for r in split_rows])) for k in ["rmse", "mae", "pcc", "ccc"]}


def selection_score(m: dict[str, float]) -> float:
    return (
        2.2 * max(0.0, m["rmse"] - PASS["rmse"]) / PASS["rmse"]
        + 1.6 * max(0.0, m["mae"] - PASS["mae"]) / PASS["mae"]
        + 1.0 * max(0.0, PASS["pcc"] - m["pcc"]) / PASS["pcc"]
        + 2.4 * max(0.0, PASS["ccc"] - m["ccc"]) / PASS["ccc"]
        + 0.04 * m["rmse"]
    )


def combine(current: pd.DataFrame, aux: pd.DataFrame, weight_aux: float) -> pd.DataFrame:
    right = aux[["musicId", "time_sec", "pred_valence"]].rename(columns={"pred_valence": "aux_pred"})
    merged = current.merge(right, on=["musicId", "time_sec"], how="inner")
    if len(merged) != len(current):
        raise ValueError(f"Alignment mismatch current={len(current)} merged={len(merged)}")
    merged["pred_valence"] = (
        (1.0 - weight_aux) * merged["pred_valence"].to_numpy(dtype=np.float64)
        + weight_aux * merged["aux_pred"].to_numpy(dtype=np.float64)
    )
    merged["weight_opensmile_aux"] = weight_aux
    return merged.drop(columns=["aux_pred"])


def evaluate_weight(aux_dirs: dict[str, Path], weight: float) -> tuple[dict[str, Any], dict[tuple[str, str], pd.DataFrame]]:
    split_rows = []
    cache = {}
    for split_name, aux_dir in aux_dirs.items():
        row = {"split_name": split_name}
        for split in ["train", "val", "test"]:
            current = pd.read_csv(CURRENT_DIR / split_name / f"{split}_predictions.csv")
            aux = pd.read_csv(aux_dir / f"{split}_predictions.csv")
            out = combine(current, aux, weight)
            cache[(split_name, split)] = out
            m = eval_df(out)
            row.update({f"{split}_{k}": v for k, v in m.items()})
        split_rows.append(row)
    val_macro = macro(split_rows, "val")
    test_macro = macro(split_rows, "test")
    result = {
        "weight_opensmile_aux": float(weight),
        **{f"val_macro_{k}": v for k, v in val_macro.items()},
        **{f"test_macro_{k}": v for k, v in test_macro.items()},
        "val_selection_score": selection_score(val_macro),
        "per_split": split_rows,
    }
    return result, cache


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = resolve_device("auto")
    aux_dirs = {}
    for split_dir in sorted(BASE_SPLIT_DIR.glob("split_*")):
        split_name = split_dir.name
        seed = int(split_name.split("_")[3])
        print(f"=== train/load split-matched OpenSMILE Valence: {split_name} ===", flush=True)
        aux_dirs[split_name] = train_or_load_opensmile(split_dir, split_name, device, seed)

    rows = []
    caches = {}
    for weight in np.round(np.arange(0.0, 0.501, 0.01), 2):
        result, cache = evaluate_weight(aux_dirs, float(weight))
        rows.append(result)
        caches[float(weight)] = cache
    table = pd.DataFrame([{k: v for k, v in r.items() if k != "per_split"} for r in rows])
    base = table[table["weight_opensmile_aux"] == 0.0].iloc[0]
    viable = table[
        (table["val_macro_rmse"] <= base["val_macro_rmse"] + 0.0005)
        & (table["val_macro_mae"] <= base["val_macro_mae"] + 0.0005)
        & (table["val_macro_pcc"] >= base["val_macro_pcc"] - 0.001)
        & (table["val_macro_ccc"] >= base["val_macro_ccc"] - 0.001)
    ].copy()
    if viable.empty:
        viable = table.copy()
    selected = viable.sort_values(
        ["val_selection_score", "val_macro_rmse", "val_macro_mae", "val_macro_ccc"],
        ascending=[True, True, True, False],
    ).iloc[0].to_dict()
    selected_weight = float(selected["weight_opensmile_aux"])
    selected_full = next(r for r in rows if float(r["weight_opensmile_aux"]) == selected_weight)
    cache = caches[selected_weight]
    for split_name in sorted(aux_dirs):
        out_split = OUT_DIR / split_name
        out_split.mkdir(parents=True, exist_ok=True)
        for split in ["train", "val", "test"]:
            cache[(split_name, split)].to_csv(out_split / f"{split}_predictions.csv", index=False)
    table.to_csv(OUT_DIR / "all_weight_candidates.csv", index=False)
    test_metrics = {k: float(selected[f"test_macro_{k}"]) for k in ["rmse", "mae", "pcc", "ccc"]}
    summary = {
        "created_at": "2026-06-29",
        "model_id": "pmemo_valence_opensmile_aux_ensemble_20260629",
        "purpose": "validation-selected ensemble of current Valence teacher with split-matched OpenSMILE Valence auxiliary model",
        "selection_rule": "one global OpenSMILE auxiliary weight selected on validation macro metrics with no-degradation guard; test labels not used",
        "current_best_reference": CURRENT_BEST,
        "opensmile_trial_id": OPENSMILE_TRIAL_ID,
        "selected": selected,
        "test_macro": {"per_split": selected_full["per_split"], **test_metrics},
        "test_gap_rows": threshold_rows(test_metrics),
        "delta_vs_current_best": {k: float(test_metrics[k] - CURRENT_BEST[k]) for k in ["rmse", "mae", "pcc", "ccc"]},
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
