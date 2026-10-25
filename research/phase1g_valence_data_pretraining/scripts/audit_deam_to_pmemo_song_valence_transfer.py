#!/usr/bin/env python3
"""Audit whether DEAM MERT song-level Valence polarity transfers to PMEmo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


PROJECT = Path(__file__).resolve().parents[3]
DEAM_SEQ = PROJECT / "research/phase1b_bert_deam_va_training/data/datasets/mert_va/sequence"
PMEMO_ROOT = PROJECT / "research/phase1f_external_va_datasets/pmemo"
PMEMO_INDEX = PMEMO_ROOT / "data/splits/pmemo_final_locked_20260713/development_pool_index.csv"
PMEMO_MERT = PMEMO_ROOT / "outputs/mert_pmemo_hidden6/features"
PMEMO_LABEL = PMEMO_ROOT / "outputs/opensmile_baseline/dataset/sequence"
DEFAULT_OUT = PROJECT / "research/phase1g_valence_data_pretraining/outputs/deam_to_pmemo_song_transfer_20260714"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--deam-max-song-id",
        type=int,
        default=None,
        help="Optionally keep only DEAM song IDs up to this value (1000 selects the 744-song EmoMusic subset).",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    pred = pred.astype(np.float64)
    target = target.astype(np.float64)
    err = pred - target
    pcc = float(np.corrcoef(pred, target)[0, 1])
    pm, tm = pred.mean(), target.mean()
    cov = float(np.mean((pred - pm) * (target - tm)))
    ccc = float(2 * cov / (pred.var() + target.var() + (pm - tm) ** 2 + 1e-12))
    return {"rmse": float(np.sqrt(np.mean(err**2))), "mae": float(np.mean(np.abs(err))), "pcc": pcc, "ccc": ccc}


def pair_accuracy(pred: np.ndarray, target: np.ndarray, margin: float = 0.05) -> dict[str, float]:
    i, j = np.triu_indices(len(target), k=1)
    delta = target[i] - target[j]
    keep = np.abs(delta) >= margin
    correct = np.sign(pred[i][keep] - pred[j][keep]) == np.sign(delta[keep])
    return {"pair_margin": margin, "pair_count": int(keep.sum()), "pair_accuracy": float(correct.mean())}


def pooled(x: np.ndarray, feature_set: str) -> np.ndarray:
    mean = x.mean(axis=0)
    return mean if feature_set == "mean" else np.concatenate([mean, x.std(axis=0)])


def load_deam(feature_set: str, max_song_id: int | None = None) -> pd.DataFrame:
    rows = []
    for path in sorted(DEAM_SEQ.glob("*.npz")):
        z = np.load(path)
        song_id = int(z["song_id"])
        if max_song_id is not None and song_id > max_song_id:
            continue
        rows.append({
            "song_id": song_id, "split": str(z["split"]),
            "target": float(z["Y"][:, 0].mean()), "feature": pooled(z["X"], feature_set),
        })
    return pd.DataFrame(rows)


def load_pmemo(feature_set: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    index = pd.read_csv(PMEMO_INDEX)
    features, targets, song_ids = [], [], []
    for music_id in index["musicId"].astype(int):
        mert = np.load(PMEMO_MERT / f"{music_id}.npz")["embedding"].astype(np.float32)
        label = np.load(PMEMO_LABEL / f"{music_id}.npz")["Y"][:, 0].astype(np.float32)
        n = min(len(mert), len(label))
        features.append(pooled(mert[:n], feature_set))
        targets.append(float(label[:n].mean()))
        song_ids.append(music_id)
    return np.stack(features), np.asarray(targets), np.asarray(song_ids)


def main() -> None:
    args = parse_args()
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    rows, cache = [], {}
    for feature_set in ["mean", "mean_std"]:
        deam = load_deam(feature_set, args.deam_max_song_id)
        x_pmemo, y_pmemo, ids = load_pmemo(feature_set)
        for alpha in [1.0, 10.0, 100.0, 1000.0, 10000.0]:
            train = deam[deam.split == "train"]
            val = deam[deam.split == "val"]
            test = deam[deam.split == "test"]
            model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
            model.fit(np.stack(train.feature), train.target.to_numpy())
            val_pred = model.predict(np.stack(val.feature))
            test_pred = model.predict(np.stack(test.feature))
            pmemo_pred = model.predict(x_pmemo)
            row = {"feature_set": feature_set, "alpha": alpha}
            row.update({f"deam_val_{k}": v for k, v in metrics(val_pred, val.target.to_numpy()).items()})
            row.update({f"deam_test_{k}": v for k, v in metrics(test_pred, test.target.to_numpy()).items()})
            row.update({f"pmemo_dev_{k}": v for k, v in metrics(pmemo_pred, y_pmemo).items()})
            row.update({f"pmemo_dev_{k}": v for k, v in pair_accuracy(pmemo_pred, y_pmemo).items()})
            rows.append(row)
            cache[(feature_set, alpha)] = (pmemo_pred, y_pmemo, ids)
    table = pd.DataFrame(rows).sort_values(["deam_val_rmse", "deam_val_pcc"], ascending=[True, False])
    table.to_csv(out / "candidates.csv", index=False)
    selected = table.iloc[0].to_dict()
    key = (str(selected["feature_set"]), float(selected["alpha"]))
    pred, target, ids = cache[key]
    pd.DataFrame({"musicId": ids, "target_song_valence": target, "deam_transfer_pred": pred}).to_csv(out / "pmemo_development_predictions.csv", index=False)
    decision = "promising_for_muq_gate" if selected["pmemo_dev_pcc"] >= 0.30 and selected["pmemo_dev_pair_accuracy"] >= 0.60 else "reject_cross_dataset_polarity_transfer"
    summary = {
        "created_at": "2026-07-14",
        "selection_boundary": "Ridge configuration selected on DEAM validation; PMEmo final locked test excluded.",
        "deam_max_song_id": args.deam_max_song_id,
        "deam_song_counts": deam["split"].value_counts().sort_index().to_dict(),
        "selected": selected,
        "decision_rule": "PMEmo development PCC >= 0.30 and pair accuracy >= 0.60",
        "decision": decision,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
