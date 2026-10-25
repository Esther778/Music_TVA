#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the reproducible 97-song section-aware pseudo-VA analysis."""

from __future__ import annotations

import hashlib
import json
import math
import warnings
from collections import Counter
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[3]
PHASE2 = ROOT / "research/phase2_section_aware_va"
SOURCE = PHASE2 / "data/full_corpus"
OUT = PHASE2 / "analysis/full_corpus_97"
DATA_DIR = OUT / "data"
TABLE_DIR = OUT / "tables"
FIG_DIR = OUT / "figures"
REPORT_DIR = OUT / "reports"
PROTOCOL_PATH = OUT / "config/analysis_protocol.json"
SECTION_SOURCE = SOURCE / "stage4/primary_chinese_section_features.csv"
SONG_SOURCE = SOURCE / "stage4/primary_chinese_song_features.csv"
FRAME_SOURCE = SOURCE / "stage3/full_corpus_99_frame_section_alignment.csv"
SEED = 20260726
BOOTSTRAPS = 5000
TYPE_ORDER = ["intro", "verse", "pre_chorus", "chorus", "bridge", "interlude", "outro"]
TYPE_LABELS = {
    "intro": "Intro", "verse": "Verse", "pre_chorus": "Pre-chorus",
    "chorus": "Chorus", "bridge": "Bridge", "interlude": "Interlude", "outro": "Outro",
}
COLORS = {
    "intro": "#4C78A8", "verse": "#72B7B2", "pre_chorus": "#F2CF5B",
    "chorus": "#E45756", "bridge": "#B279A2", "interlude": "#59A14F", "outro": "#9D755D",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_csv(frame: pd.DataFrame, name: str, directory: Path = TABLE_DIR) -> Path:
    path = directory / name
    frame.to_csv(path, index=False)
    return path


def save_figure(fig: plt.Figure, stem: str) -> None:
    for suffix in ("png", "svg"):
        fig.savefig(FIG_DIR / f"{stem}.{suffix}", dpi=300 if suffix == "png" else None,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)


def bootstrap_ci(values: np.ndarray, statistic=np.mean, samples: int = BOOTSTRAPS,
                 seed: int = SEED) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    draw_stats = np.empty(samples)
    for i in range(samples):
        draw_stats[i] = statistic(rng.choice(values, len(values), replace=True))
    return tuple(np.quantile(draw_stats, [0.025, 0.975]).astype(float))


def holm_adjust(p_values: pd.Series) -> pd.Series:
    p = p_values.to_numpy(dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adjusted[idx] = min(running, 1.0)
    return pd.Series(adjusted, index=p_values.index)


def rank_biserial(diff: np.ndarray) -> float:
    diff = np.asarray(diff, dtype=float)
    diff = diff[np.isfinite(diff) & (diff != 0)]
    if not len(diff):
        return np.nan
    ranks = stats.rankdata(np.abs(diff))
    return float((ranks[diff > 0].sum() - ranks[diff < 0].sum()) / ranks.sum())


def paired_test(name: str, values: pd.DataFrame, left: str, right: str,
                dimension: str, family: str) -> dict[str, object]:
    pair = values[[left, right]].dropna()
    diff = pair[right].to_numpy() - pair[left].to_numpy()
    if len(diff) < 2:
        return {}
    try:
        stat, p = stats.wilcoxon(diff, zero_method="wilcox", alternative="two-sided")
    except ValueError:
        stat, p = 0.0, 1.0
    low, high = bootstrap_ci(diff)
    sd = np.std(diff, ddof=1)
    return {
        "family": family, "comparison": name, "dimension": dimension,
        "n_songs": len(diff), "left_mean": pair[left].mean(), "right_mean": pair[right].mean(),
        "mean_difference": diff.mean(), "median_difference": np.median(diff),
        "sd_difference": sd, "q1_difference": np.quantile(diff, 0.25),
        "q3_difference": np.quantile(diff, 0.75), "ci95_low": low, "ci95_high": high,
        "positive_song_fraction": np.mean(diff > 0), "wilcoxon_stat": stat,
        "p_value": p, "rank_biserial": rank_biserial(diff),
        "cohens_dz": diff.mean() / sd if sd > 0 else np.nan,
    }


def direction_class(dv: float, da: float, threshold: float) -> str:
    vs = 0 if abs(dv) < threshold else (1 if dv > 0 else -1)
    ars = 0 if abs(da) < threshold else (1 if da > 0 else -1)
    labels = {
        (0, 0): "stable", (1, 1): "both_up", (-1, -1): "both_down",
        (1, -1): "valence_up_arousal_down", (-1, 1): "valence_down_arousal_up",
        (1, 0): "valence_only_up", (-1, 0): "valence_only_down",
        (0, 1): "arousal_only_up", (0, -1): "arousal_only_down",
    }
    return labels[(vs, ars)]


def shannon_entropy(values: pd.Series) -> float:
    probs = values.value_counts(normalize=True).to_numpy(dtype=float)
    return float(-(probs * np.log2(probs)).sum()) if len(probs) else 0.0


def turning_points(values: np.ndarray, threshold: float = 0.03) -> int:
    delta = np.diff(np.asarray(values, dtype=float))
    signs = np.where(np.abs(delta) < threshold, 0, np.sign(delta))
    signs = signs[signs != 0]
    return int(np.sum(signs[1:] != signs[:-1])) if len(signs) > 1 else 0


def audit_and_clean(sections: pd.DataFrame, songs: pd.DataFrame,
                    frames: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    sections = sections.sort_values(["sample_index", "section_order"]).copy()
    sections["raw_section_type"] = sections["section_type"]
    sections["normalized_section_type"] = sections["section_type"]
    sections["type_occurrence_index"] = sections.groupby(["song_id", "section_type"]).cumcount() + 1
    sections["type_occurrence_count"] = sections.groupby(["song_id", "section_type"])["section_type"].transform("size")
    sections["is_first_type_occurrence"] = sections["type_occurrence_index"] == 1
    sections["is_last_type_occurrence"] = sections["type_occurrence_index"] == sections["type_occurrence_count"]
    sections["previous_section_type"] = sections.groupby("song_id")["section_type"].shift(1)
    sections["next_section_type"] = sections.groupby("song_id")["section_type"].shift(-1)
    sections["transition_from_previous"] = sections["previous_section_type"].fillna("START") + "->" + sections["section_type"]
    sections["song_centered_valence"] = sections["mean_valence"] - sections.groupby("song_id")["mean_valence"].transform("mean")
    sections["song_centered_arousal"] = sections["mean_arousal"] - sections.groupby("song_id")["mean_arousal"].transform("mean")
    for dim in ("valence", "arousal"):
        col = f"mean_{dim}"
        sd = sections.groupby("song_id")[col].transform("std").replace(0, np.nan)
        sections[f"song_z_{dim}"] = (sections[col] - sections.groupby("song_id")[col].transform("mean")) / sd
    sections["duration_weight_within_song"] = sections["section_duration_sec"] / sections.groupby("song_id")["section_duration_sec"].transform("sum")

    expected_types = set(TYPE_ORDER)
    continuity_errors = 0
    order_errors = 0
    for _, group in sections.groupby("song_id", sort=False):
        group = group.sort_values("section_order")
        order_errors += int(not np.array_equal(group["section_order"].to_numpy(), np.arange(1, len(group) + 1)))
        continuity_errors += int(not np.allclose(group["section_end_sec"].to_numpy()[:-1], group["section_start_sec"].to_numpy()[1:], atol=1e-6))
    audit = {
        "songs": int(sections["song_id"].nunique()), "section_rows": len(sections),
        "frame_rows": len(frames), "artists": int(sections["artist"].nunique()),
        "unique_annotation_ids": int(sections["human_annotation_id"].nunique()),
        "duplicate_annotation_ids": int(sections["human_annotation_id"].duplicated().sum()),
        "duplicate_song_orders": int(sections.duplicated(["song_id", "section_order"]).sum()),
        "missing_values_total": int(sections.isna().sum().sum()),
        "unexpected_missing_excluding_edge_transitions": int(sections.drop(columns=[
            "change_from_prev_valence", "change_from_prev_arousal", "contrast_from_prev",
            "change_to_next_valence", "change_to_next_arousal", "contrast_to_next",
            "previous_section_type", "next_section_type", "song_z_valence", "song_z_arousal",
        ]).isna().sum().sum()),
        "section_types": sections["section_type"].value_counts().to_dict(),
        "unrecognized_section_types": sorted(set(sections["section_type"]) - expected_types),
        "order_error_songs": order_errors, "continuity_error_songs": continuity_errors,
        "nonpositive_duration_sections": int((sections["section_duration_sec"] <= 0).sum()),
        "sections_per_song": sections.groupby("song_id").size().describe().to_dict(),
        "duration_sec_per_song": songs["duration_sec"].describe().to_dict(),
        "section_valence": sections["mean_valence"].describe().to_dict(),
        "section_arousal": sections["mean_arousal"].describe().to_dict(),
        "frame_valence": frames["valence"].describe().to_dict(),
        "frame_arousal": frames["arousal"].describe().to_dict(),
        "frame_assignment_status": frames["section_assignment_status"].value_counts().to_dict(),
        "approved_sections": int(sections["is_approved"].sum()),
        "note": "Only expected first/last transition and zero-variance song-z fields may be missing.",
    }
    mapping = pd.DataFrame({"raw_section_type": TYPE_ORDER, "normalized_section_type": TYPE_ORDER,
                            "mapping_rule": "identity; source labels already canonical"})
    save_csv(mapping, "section_name_mapping.csv", DATA_DIR)
    save_csv(sections, "cleaned_section_analysis.csv", DATA_DIR)
    save_csv(songs, "cleaned_song_analysis_source.csv", DATA_DIR)
    return sections, songs, audit


def section_descriptives(sections: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    for section_type in TYPE_ORDER:
        group = sections[sections.section_type == section_type]
        for dim in ("valence", "arousal"):
            x = group[f"mean_{dim}"].to_numpy()
            rows.append({
                "section_type": section_type, "dimension": dim, "n_sections": len(group),
                "n_songs": group.song_id.nunique(), "mean": np.mean(x), "median": np.median(x),
                "std": np.std(x, ddof=1), "q1": np.quantile(x, .25), "q3": np.quantile(x, .75),
                "min": np.min(x), "max": np.max(x),
            })
    descriptive = pd.DataFrame(rows)
    song_type = sections.groupby(["song_id", "title", "artist", "section_type"]).agg(
        valence=("mean_valence", "mean"), arousal=("mean_arousal", "mean"),
        duration=("section_duration_sec", "sum"), occurrences=("section_order", "size")
    ).reset_index()
    weighted_rows = []
    for keys, group in sections.groupby(["song_id", "title", "artist", "section_type"]):
        weighted_rows.append(dict(zip(["song_id", "title", "artist", "section_type"], keys)) | {
            "valence": np.average(group.mean_valence, weights=group.section_duration_sec),
            "arousal": np.average(group.mean_arousal, weights=group.section_duration_sec),
        })
    weighted = pd.DataFrame(weighted_rows)
    return descriptive, song_type, weighted


def section_contrasts(song_type: pd.DataFrame, weighted: pd.DataFrame) -> pd.DataFrame:
    definitions = [
        ("Chorus - Verse", "verse", "chorus"),
        ("Pre-chorus - Verse", "verse", "pre_chorus"),
        ("Chorus - Pre-chorus", "pre_chorus", "chorus"),
        ("Outro - Intro", "intro", "outro"),
        ("Bridge - Chorus", "chorus", "bridge"),
    ]
    rows = []
    for weighting, source in (("unweighted", song_type), ("duration_weighted", weighted)):
        for dim in ("valence", "arousal"):
            pivot = source.pivot(index="song_id", columns="section_type", values=dim)
            for label, left, right in definitions:
                if left in pivot and right in pivot:
                    result = paired_test(label, pivot, left, right, dim, f"section_contrasts_{weighting}")
                    if result:
                        result["weighting"] = weighting
                        rows.append(result)
    result = pd.DataFrame(rows)
    result["p_holm"] = result.groupby("family")["p_value"].transform(holm_adjust)
    return result


def transition_analysis(sections: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    for _, group in sections.groupby("song_id", sort=False):
        group = group.sort_values("section_order")
        prev = group.shift(1)
        for idx in range(1, len(group)):
            current = group.iloc[idx]
            before = group.iloc[idx - 1]
            dv = current.mean_valence - before.mean_valence
            da = current.mean_arousal - before.mean_arousal
            row = {
                "song_id": current.song_id, "title": current.title, "artist": current.artist,
                "from_order": int(before.section_order), "to_order": int(current.section_order),
                "from_type": before.section_type, "to_type": current.section_type,
                "transition": f"{before.section_type}->{current.section_type}",
                "delta_valence": dv, "delta_arousal": da,
                "va_distance": math.hypot(dv, da), "normalized_position": current.normalized_midpoint,
            }
            for threshold in (.03, .05, .07):
                row[f"direction_t{threshold:.2f}"] = direction_class(dv, da, threshold)
            rows.append(row)
    transitions = pd.DataFrame(rows)
    summary_rows = []
    for name, group in transitions.groupby("transition"):
        for dim in ("valence", "arousal"):
            x = group[f"delta_{dim}"].to_numpy()
            song_means = group.groupby("song_id")[f"delta_{dim}"].mean().to_numpy()
            low, high = bootstrap_ci(song_means)
            summary_rows.append({
                "transition": name, "dimension": dim, "n_events": len(group),
                "n_songs": group.song_id.nunique(), "event_mean": x.mean(), "event_median": np.median(x),
                "event_std": x.std(ddof=1) if len(x) > 1 else 0.0, "song_mean": song_means.mean(),
                "song_bootstrap_ci_low": low, "song_bootstrap_ci_high": high,
                "positive_event_fraction": np.mean(x > 0), "negative_event_fraction": np.mean(x < 0),
            })
    summary = pd.DataFrame(summary_rows).sort_values(["n_songs", "transition", "dimension"], ascending=[False, True, True])
    direction = transitions.groupby(["transition", "direction_t0.05"]).agg(
        n_events=("song_id", "size"), n_songs=("song_id", "nunique")
    ).reset_index()
    direction["event_fraction_within_transition"] = direction["n_events"] / direction.groupby("transition")["n_events"].transform("sum")
    return transitions, summary, direction


def transition_tests(transitions: pd.DataFrame) -> pd.DataFrame:
    predefined = ["verse->chorus", "pre_chorus->chorus", "chorus->verse", "chorus->bridge", "bridge->chorus", "intro->verse", "chorus->outro"]
    rows = []
    for transition in predefined:
        group = transitions[transitions.transition == transition]
        for dim in ("valence", "arousal"):
            song_values = group.groupby("song_id")[f"delta_{dim}"].mean()
            if len(song_values) < 5:
                continue
            values = pd.DataFrame({"zero": 0.0, "change": song_values})
            result = paired_test(transition, values, "zero", "change", dim, "predefined_transitions")
            if result:
                rows.append(result)
    result = pd.DataFrame(rows)
    result["p_holm"] = holm_adjust(result["p_value"]) if len(result) else []
    return result


def repeated_sections(sections: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    details = []
    for (song_id, section_type), group in sections.groupby(["song_id", "section_type"]):
        group = group.sort_values("section_order")
        if len(group) < 2:
            continue
        x = np.arange(1, len(group) + 1, dtype=float)
        dv = group.mean_valence.iloc[-1] - group.mean_valence.iloc[0]
        da = group.mean_arousal.iloc[-1] - group.mean_arousal.iloc[0]
        details.append({
            "song_id": song_id, "title": group.title.iloc[0], "artist": group.artist.iloc[0],
            "section_type": section_type, "occurrences": len(group),
            "first_valence": group.mean_valence.iloc[0], "last_valence": group.mean_valence.iloc[-1],
            "first_to_last_valence": dv, "valence_occurrence_slope": stats.linregress(x, group.mean_valence).slope,
            "first_arousal": group.mean_arousal.iloc[0], "last_arousal": group.mean_arousal.iloc[-1],
            "first_to_last_arousal": da, "arousal_occurrence_slope": stats.linregress(x, group.mean_arousal).slope,
            "valence_monotonic_increase": bool(np.all(np.diff(group.mean_valence) > 0)) if len(group) >= 3 else np.nan,
            "arousal_monotonic_increase": bool(np.all(np.diff(group.mean_arousal) > 0)) if len(group) >= 3 else np.nan,
            "mean_pairwise_va_distance": float(np.mean([
                math.hypot(group.mean_valence.iloc[j] - group.mean_valence.iloc[i],
                           group.mean_arousal.iloc[j] - group.mean_arousal.iloc[i])
                for i in range(len(group)) for j in range(i + 1, len(group))
            ])),
        })
    detail = pd.DataFrame(details)
    summary_rows = []
    for section_type, group in detail.groupby("section_type"):
        for dim in ("valence", "arousal"):
            x = group[f"first_to_last_{dim}"].dropna().to_numpy()
            low, high = bootstrap_ci(x)
            try:
                stat, p = stats.wilcoxon(x)
            except ValueError:
                stat, p = 0.0, 1.0
            summary_rows.append({
                "section_type": section_type, "dimension": dim, "n_songs": len(x),
                "mean_first_to_last": x.mean(), "median_first_to_last": np.median(x),
                "ci95_low": low, "ci95_high": high, "positive_fraction": np.mean(x > 0),
                "wilcoxon_stat": stat, "p_value": p, "rank_biserial": rank_biserial(x),
            })
    summary = pd.DataFrame(summary_rows)
    summary["p_holm"] = summary.groupby("dimension")["p_value"].transform(holm_adjust)
    return detail, summary


def repeated_change_contrasts(detail: pd.DataFrame) -> pd.DataFrame:
    """Test whether repeated Chorus intensifies more than other repeated types."""
    rows = []
    for dim in ("valence", "arousal"):
        pivot = detail.pivot(index="song_id", columns="section_type", values=f"first_to_last_{dim}")
        for other in ("verse", "pre_chorus", "interlude"):
            if "chorus" not in pivot or other not in pivot:
                continue
            result = paired_test(
                f"Chorus change - {TYPE_LABELS[other]} change", pivot,
                other, "chorus", dim, "repeated_change_contrasts",
            )
            if result:
                rows.append(result)
    result = pd.DataFrame(rows)
    result["p_holm"] = holm_adjust(result.p_value) if len(result) else []
    return result


def transition_threshold_sensitivity(transitions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    scopes = {
        "all_transitions_into_chorus": transitions[transitions.to_type == "chorus"],
        "verse_to_chorus": transitions[transitions.transition == "verse->chorus"],
        "pre_chorus_to_chorus": transitions[transitions.transition == "pre_chorus->chorus"],
    }
    for scope, group in scopes.items():
        for threshold in (.03, .05, .07):
            column = f"direction_t{threshold:.2f}"
            for direction, count in group[column].value_counts().items():
                rows.append({
                    "scope": scope, "threshold": threshold, "direction": direction,
                    "n_events": int(count), "n_total_events": len(group),
                    "event_fraction": count / len(group), "n_songs": group.song_id.nunique(),
                })
    return pd.DataFrame(rows)


def climax_analysis(sections: pd.DataFrame, transitions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for song_id, group in sections.groupby("song_id"):
        group = group.sort_values("section_order")
        tgroup = transitions[transitions.song_id == song_id]
        for definition, column in (("max_mean_arousal", "mean_arousal"), ("max_mean_valence", "mean_valence")):
            row = group.loc[group[column].idxmax()]
            rows.append({
                "song_id": song_id, "title": row.title, "artist": row.artist,
                "peak_definition": definition, "section_order": int(row.section_order),
                "section_type": row.section_type, "peak_value": row[column],
                "relative_position": row.normalized_midpoint,
            })
        if len(tgroup):
            tr = tgroup.loc[tgroup.va_distance.idxmax()]
            target = group[group.section_order == tr.to_order].iloc[0]
            rows.append({
                "song_id": song_id, "title": tr.title, "artist": tr.artist,
                "peak_definition": "max_transition_distance", "section_order": int(tr.to_order),
                "section_type": tr.to_type, "peak_value": tr.va_distance,
                "relative_position": target.normalized_midpoint,
            })
    peaks = pd.DataFrame(rows)
    peaks["position_quartile"] = pd.cut(peaks.relative_position, bins=[-np.inf, .25, .5, .75, np.inf],
                                         labels=["Q1", "Q2", "Q3", "Q4"])
    summary = peaks.groupby(["peak_definition", "section_type"]).agg(
        n_songs=("song_id", "size"), mean_relative_position=("relative_position", "mean")
    ).reset_index()
    summary["fraction_within_definition"] = summary["n_songs"] / summary.groupby("peak_definition")["n_songs"].transform("sum")
    return peaks, summary


def song_features(sections: pd.DataFrame, transitions: pd.DataFrame, songs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for song_id, group in sections.groupby("song_id"):
        group = group.sort_values("section_order")
        trans = transitions[transitions.song_id == song_id]
        v = group.mean_valence.to_numpy()
        a = group.mean_arousal.to_numpy()
        corr = stats.spearmanr(v, a).statistic if len(np.unique(v)) > 1 and len(np.unique(a)) > 1 else np.nan
        net_distance = math.hypot(v[-1] - v[0], a[-1] - a[0])
        path_length = float(np.sum(np.hypot(np.diff(v), np.diff(a))))
        transition_entropy = shannon_entropy(trans["direction_t0.05"]) if len(trans) else 0.0
        rows.append({
            "song_id": song_id, "title": group.title.iloc[0], "artist": group.artist.iloc[0],
            "duration_sec": group.section_end_sec.max(), "section_count": len(group),
            "unique_section_types": group.section_type.nunique(),
            "section_type_entropy": shannon_entropy(group.section_type),
            "transition_direction_entropy": transition_entropy,
            "start_valence": v[0], "end_valence": v[-1], "net_valence_change": v[-1] - v[0],
            "start_arousal": a[0], "end_arousal": a[-1], "net_arousal_change": a[-1] - a[0],
            "valence_range": np.ptp(v), "arousal_range": np.ptp(a),
            "va_path_length": path_length, "net_va_distance": net_distance,
            "path_directness": net_distance / path_length if path_length > 0 else 0.0,
            "loop_closure_distance": net_distance,
            "valence_turning_points": turning_points(v), "arousal_turning_points": turning_points(a),
            "mean_transition_distance": trans.va_distance.mean(), "max_transition_distance": trans.va_distance.max(),
            "section_va_spearman": corr,
            "chorus_count": int((group.section_type == "chorus").sum()),
            "verse_count": int((group.section_type == "verse").sum()),
            "bridge_present": int((group.section_type == "bridge").any()),
        })
    result = pd.DataFrame(rows)
    source_extra = songs[["song_id", "mean_valence", "mean_arousal"]].rename(
        columns={"mean_valence": "frame_weighted_mean_valence", "mean_arousal": "frame_weighted_mean_arousal"})
    return result.merge(source_extra, on="song_id", how="left", validate="one_to_one")


def trajectory_data(frames: pd.DataFrame) -> pd.DataFrame:
    grid = np.linspace(0, 1, 101)
    rows = []
    for song_id, group in frames.groupby("song_id"):
        group = group.sort_values("time_sec")
        x = group.time_sec.to_numpy() / group.time_sec.max()
        v = np.interp(grid, x, group.valence)
        a = np.interp(grid, x, group.arousal)
        for i, point in enumerate(grid):
            rows.append({"song_id": song_id, "title": group.title.iloc[0], "artist": group.artist.iloc[0],
                         "normalized_time": point, "valence": v[i], "arousal": a[i]})
    return pd.DataFrame(rows)


def cluster_trajectories(trajectories: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    ids = trajectories.song_id.drop_duplicates().tolist()
    matrices = []
    for song_id in ids:
        group = trajectories[trajectories.song_id == song_id].sort_values("normalized_time")
        v = group.valence.to_numpy(); a = group.arousal.to_numpy()
        v = (v - v.mean()) / (v.std() or 1.0); a = (a - a.mean()) / (a.std() or 1.0)
        matrices.append(np.r_[v, a])
    X = np.asarray(matrices)
    Xs = StandardScaler().fit_transform(X)
    pca_full = PCA().fit(Xs)
    components = min(10, max(2, int(np.searchsorted(np.cumsum(pca_full.explained_variance_ratio_), .90) + 1)))
    Xp = PCA(n_components=components, random_state=SEED).fit_transform(Xs)
    diagnostics = []
    assignments = pd.DataFrame({"song_id": ids})
    rng = np.random.default_rng(SEED)
    for k in range(2, 6):
        km = KMeans(n_clusters=k, random_state=SEED, n_init=50).fit(Xp)
        agg = AgglomerativeClustering(n_clusters=k, linkage="ward").fit(Xp)
        ari_cross = adjusted_rand_score(km.labels_, agg.labels_)
        stability = []
        for _ in range(100):
            idx = rng.choice(len(ids), size=int(round(.8 * len(ids))), replace=False)
            sub = KMeans(n_clusters=k, random_state=int(rng.integers(1_000_000)), n_init=20).fit(Xp[idx])
            stability.append(adjusted_rand_score(km.labels_, sub.predict(Xp)))
        diagnostics.append({
            "k": k, "pca_components": components,
            "silhouette": silhouette_score(Xp, km.labels_),
            "calinski_harabasz": calinski_harabasz_score(Xp, km.labels_),
            "davies_bouldin": davies_bouldin_score(Xp, km.labels_),
            "cross_method_ari": ari_cross, "subsample_ari_mean": np.mean(stability),
            "subsample_ari_sd": np.std(stability, ddof=1),
        })
        assignments[f"kmeans_k{k}"] = km.labels_ + 1
        assignments[f"ward_k{k}"] = agg.labels_ + 1
    diag = pd.DataFrame(diagnostics)
    eligible = diag[(diag.silhouette >= .25) & (diag.subsample_ari_mean >= .60)]
    if len(eligible):
        best = eligible.sort_values(["silhouette", "subsample_ari_mean"], ascending=False).iloc[0]
        supported = True
    else:
        best = diag.sort_values(["silhouette", "subsample_ari_mean"], ascending=False).iloc[0]
        supported = False
    chosen_k = int(best.k)
    assignments["selected_cluster"] = assignments[f"kmeans_k{chosen_k}"]
    metadata = {"pca_components": components, "selected_k": chosen_k,
                "discrete_solution_cautiously_supported": supported,
                "selection_silhouette": float(best.silhouette),
                "selection_subsample_ari_mean": float(best.subsample_ari_mean),
                "selection_cross_method_ari": float(best.cross_method_ari)}
    return diag, assignments, metadata


def association_table(features: pd.DataFrame, repeats: pd.DataFrame) -> pd.DataFrame:
    pairs = [
        ("section_count", "va_path_length"), ("section_count", "mean_transition_distance"),
        ("unique_section_types", "va_path_length"), ("section_type_entropy", "va_path_length"),
        ("section_type_entropy", "max_transition_distance"), ("duration_sec", "va_path_length"),
        ("chorus_count", "valence_range"), ("chorus_count", "arousal_range"),
    ]
    rows = []
    for x, y in pairs:
        pair = features[[x, y]].dropna()
        result = stats.spearmanr(pair[x], pair[y])
        rows.append({"x": x, "y": y, "n_songs": len(pair), "spearman_rho": result.statistic, "p_value": result.pvalue})
    chorus = repeats[repeats.section_type == "chorus"]
    for y in ("first_to_last_valence", "first_to_last_arousal"):
        pair = chorus[["occurrences", y]].dropna()
        if len(pair) >= 5:
            result = stats.spearmanr(pair.occurrences, pair[y])
            rows.append({"x": "chorus_occurrences", "y": y, "n_songs": len(pair),
                         "spearman_rho": result.statistic, "p_value": result.pvalue})
    result = pd.DataFrame(rows)
    result["p_holm"] = holm_adjust(result.p_value)
    return result


def counterexamples(features: pd.DataFrame, sections: pd.DataFrame, repeats: pd.DataFrame,
                    transitions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    chorus_lift = sections.groupby(["song_id", "section_type"])[["mean_valence", "mean_arousal"]].mean().unstack()
    for song_id in chorus_lift.index:
        try:
            dv = chorus_lift.loc[song_id, ("mean_valence", "chorus")] - chorus_lift.loc[song_id, ("mean_valence", "verse")]
            da = chorus_lift.loc[song_id, ("mean_arousal", "chorus")] - chorus_lift.loc[song_id, ("mean_arousal", "verse")]
        except KeyError:
            continue
        if da <= 0 or dv <= 0:
            meta = features[features.song_id == song_id].iloc[0]
            rows.append({"song_id": song_id, "title": meta.title, "artist": meta.artist,
                         "case_type": "nonpositive_chorus_minus_verse",
                         "value_1": dv, "value_2": da, "review_priority": "high" if da <= 0 else "medium"})
    for _, row in features.nsmallest(5, "va_path_length").iterrows():
        rows.append({"song_id": row.song_id, "title": row.title, "artist": row.artist,
                     "case_type": "lowest_emotional_path_length", "value_1": row.va_path_length,
                     "value_2": row.section_va_spearman, "review_priority": "medium"})
    for _, row in features.nsmallest(5, "section_va_spearman").iterrows():
        rows.append({"song_id": row.song_id, "title": row.title, "artist": row.artist,
                     "case_type": "strongest_va_dimension_separation", "value_1": row.section_va_spearman,
                     "value_2": row.va_path_length, "review_priority": "high"})
    for _, row in features.nlargest(5, "max_transition_distance").iterrows():
        rows.append({"song_id": row.song_id, "title": row.title, "artist": row.artist,
                     "case_type": "largest_single_section_contrast", "value_1": row.max_transition_distance,
                     "value_2": row.mean_transition_distance, "review_priority": "high"})
    return pd.DataFrame(rows).drop_duplicates(["song_id", "case_type"])


def common_sequences(sections: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for song_id, group in sections.groupby("song_id"):
        sequence = tuple(group.sort_values("section_order").section_type)
        rows.append({"song_id": song_id, "sequence": " > ".join(sequence), "section_count": len(sequence),
                     "transition_sequence": " | ".join(f"{a}->{b}" for a, b in zip(sequence[:-1], sequence[1:]))})
    detail = pd.DataFrame(rows)
    counts = detail.groupby("sequence").agg(song_count=("song_id", "size"), section_count=("section_count", "first")).reset_index()
    counts["song_fraction"] = counts.song_count / sections.song_id.nunique()
    return counts.sort_values(["song_count", "section_count"], ascending=[False, True])


def make_figures(sections: pd.DataFrame, transitions: pd.DataFrame, repeats: pd.DataFrame,
                 peaks: pd.DataFrame, trajectories: pd.DataFrame, features: pd.DataFrame,
                 cluster_diag: pd.DataFrame, assignments: pd.DataFrame, cluster_meta: dict,
                 tests: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.titleweight": "bold"})

    long = sections.melt(id_vars=["section_type", "song_id"], value_vars=["mean_valence", "mean_arousal"],
                         var_name="dimension", value_name="pseudo_va")
    long["dimension"] = long.dimension.map({"mean_valence": "Valence", "mean_arousal": "Arousal"})
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, dim in zip(axes, ["Valence", "Arousal"]):
        part = long[long.dimension == dim]
        sns.violinplot(data=part, x="section_type", y="pseudo_va", order=TYPE_ORDER,
                       palette=COLORS, inner=None, cut=0, linewidth=.8, ax=ax)
        sns.boxplot(data=part, x="section_type", y="pseudo_va", order=TYPE_ORDER, width=.22,
                    showfliers=False, color="white", linecolor="#222222", ax=ax)
        ax.axhline(0, color="#555555", lw=.8, ls="--")
        ax.set_title(f"{dim} by section type")
        ax.set_xlabel(""); ax.set_ylabel("Teacher pseudo-VA [-1, 1]" if ax is axes[0] else "")
        ax.set_xticks(np.arange(len(TYPE_ORDER)), [TYPE_LABELS[x] for x in TYPE_ORDER], rotation=30, ha="right")
    save_figure(fig, "01_section_type_va_distributions")

    centroids = sections.groupby("section_type")[["mean_valence", "mean_arousal"]].mean()
    fig, ax = plt.subplots(figsize=(7, 6))
    for section_type, row in centroids.iterrows():
        ax.scatter(row.mean_valence, row.mean_arousal, s=130, color=COLORS[section_type], edgecolor="white", linewidth=1.2)
        ax.text(row.mean_valence + .008, row.mean_arousal + .008, TYPE_LABELS[section_type], fontsize=10)
    ax.axhline(0, color="#777", lw=.8); ax.axvline(0, color="#777", lw=.8)
    ax.set(xlabel="Mean Valence", ylabel="Mean Arousal", title="Section centroids in pseudo-VA space")
    save_figure(fig, "02_section_centroids_va_space")

    common = transitions.groupby("transition").filter(lambda g: g.song_id.nunique() >= 10)
    vectors = common.groupby("transition").agg(dv=("delta_valence", "mean"), da=("delta_arousal", "mean"),
                                                 n_songs=("song_id", "nunique")).reset_index()
    vectors = vectors.sort_values("n_songs", ascending=False).head(12)
    fig, ax = plt.subplots(figsize=(8, 7))
    for _, row in vectors.iterrows():
        ax.arrow(0, 0, row.dv, row.da, head_width=.008, length_includes_head=True, alpha=.7)
        ax.text(row.dv * 1.07, row.da * 1.07, f"{row.transition} (n={row.n_songs})", fontsize=8)
    ax.axhline(0, color="#777", lw=.8); ax.axvline(0, color="#777", lw=.8)
    ax.set(xlabel="Mean delta Valence", ylabel="Mean delta Arousal", title="Common adjacent-section transition vectors")
    save_figure(fig, "03_common_transition_vectors")

    chorus_in = transitions.loc[transitions.to_type == "chorus", "direction_t0.05"].value_counts(normalize=True).sort_values()
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.barh(chorus_in.index, chorus_in.values, color="#E45756")
    ax.set(xlabel="Fraction of transitions into Chorus", title="Direction patterns entering Chorus (threshold = 0.05)")
    ax.set_xlim(0, max(.5, chorus_in.max() * 1.12))
    for i, v in enumerate(chorus_in.values): ax.text(v + .008, i, f"{v:.1%}", va="center")
    save_figure(fig, "04_chorus_entry_direction_patterns")

    chorus_rep = repeats[repeats.section_type == "chorus"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, dim, color in zip(axes, ["valence", "arousal"], ["#4C78A8", "#E45756"]):
        for _, row in chorus_rep.iterrows():
            ax.plot([1, 2], [row[f"first_{dim}"], row[f"last_{dim}"]], color="#BBBBBB", alpha=.4, lw=.8)
        means = [chorus_rep[f"first_{dim}"].mean(), chorus_rep[f"last_{dim}"].mean()]
        ax.plot([1, 2], means, color=color, marker="o", lw=3, ms=7)
        ax.set_xticks([1, 2], ["First Chorus", "Last observed Chorus"])
        ax.set_ylabel(dim.title()); ax.set_title(f"Repeated Chorus: {dim.title()}")
    save_figure(fig, "05_first_last_chorus")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    p = peaks[peaks.peak_definition == "max_mean_arousal"].section_type.value_counts().reindex(TYPE_ORDER, fill_value=0)
    axes[0].bar([TYPE_LABELS[x] for x in TYPE_ORDER], p.values, color=[COLORS[x] for x in TYPE_ORDER])
    axes[0].set_title("Model-derived max-Arousal section"); axes[0].set_ylabel("Songs")
    axes[0].tick_params(axis="x", rotation=30)
    sns.histplot(data=peaks[peaks.peak_definition == "max_mean_arousal"], x="relative_position", bins=np.linspace(0,1,9), ax=axes[1], color="#E45756")
    axes[1].set_title("Relative position of max-Arousal section"); axes[1].set_xlabel("Normalized song position")
    save_figure(fig, "06_peak_type_and_position")

    grid = np.sort(trajectories.normalized_time.unique())
    rng = np.random.default_rng(SEED)
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for ax, dim, color in zip(axes, ["valence", "arousal"], ["#4C78A8", "#E45756"]):
        matrix = trajectories.pivot(index="song_id", columns="normalized_time", values=dim).loc[:, grid].to_numpy()
        mean = matrix.mean(axis=0)
        draws = np.asarray([matrix[rng.choice(len(matrix), len(matrix), replace=True)].mean(axis=0) for _ in range(2000)])
        low, high = np.quantile(draws, [.025, .975], axis=0)
        ax.plot(grid, mean, color=color, lw=2.5); ax.fill_between(grid, low, high, color=color, alpha=.2)
        ax.axhline(0, color="#777", lw=.7, ls="--"); ax.set_ylabel(dim.title())
    axes[0].set_title("Average complete-song pseudo-VA trajectory (song bootstrap 95% CI)")
    axes[1].set_xlabel("Normalized song time")
    save_figure(fig, "07_average_normalized_trajectory")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].plot(cluster_diag.k, cluster_diag.silhouette, marker="o", color="#4C78A8"); axes[0].axhline(.25, ls="--", color="#777"); axes[0].set_title("Silhouette")
    axes[1].plot(cluster_diag.k, cluster_diag.subsample_ari_mean, marker="o", color="#59A14F"); axes[1].axhline(.60, ls="--", color="#777"); axes[1].set_title("Subsample stability ARI")
    axes[2].plot(cluster_diag.k, cluster_diag.cross_method_ari, marker="o", color="#B279A2"); axes[2].set_title("KMeans-Ward ARI")
    for ax in axes: ax.set_xlabel("k"); ax.set_xticks(cluster_diag.k)
    fig.suptitle("Trajectory clustering diagnostics")
    save_figure(fig, "08_cluster_diagnostics")

    chosen = assignments[["song_id", "selected_cluster"]]
    cluster_traj = trajectories.merge(chosen, on="song_id")
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    palette = sns.color_palette("colorblind", n_colors=cluster_meta["selected_k"])
    for cluster, group in cluster_traj.groupby("selected_cluster"):
        mean = group.groupby("normalized_time")[["valence", "arousal"]].mean()
        n = group.song_id.nunique()
        axes[0].plot(mean.index, mean.valence, label=f"Cluster {cluster} (n={n})", color=palette[cluster-1], lw=2)
        axes[1].plot(mean.index, mean.arousal, label=f"Cluster {cluster} (n={n})", color=palette[cluster-1], lw=2)
    axes[0].set_title("Selected descriptive trajectory solution"); axes[0].set_ylabel("Valence")
    axes[1].set_ylabel("Arousal"); axes[1].set_xlabel("Normalized song time"); axes[0].legend(ncol=2, fontsize=8)
    save_figure(fig, "09_cluster_mean_trajectories")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, xcol, color in (
        (axes[0], "section_count", "#E45756"),
        (axes[1], "section_type_entropy", "#59A14F"),
    ):
        x = features[xcol].to_numpy(dtype=float)
        y = features["va_path_length"].to_numpy(dtype=float)
        ax.scatter(x, y, alpha=.6, color="#4C78A8", edgecolor="white", linewidth=.4)
        grid = np.linspace(x.min(), x.max(), 200)
        ax.plot(grid, np.polyval(np.polyfit(x, y, deg=2), grid), color=color, lw=2)
        ax.set_xlabel(xcol.replace("_", " ").title())
        ax.set_ylabel("VA path length")
    axes[0].set_title("Section count vs VA path length"); axes[1].set_title("Type entropy vs VA path length")
    save_figure(fig, "10_structure_complexity_associations")

    forest = tests[(tests.weighting == "unweighted") if "weighting" in tests else np.ones(len(tests), dtype=bool)].copy()
    forest = forest.sort_values("mean_difference")
    fig, ax = plt.subplots(figsize=(8, max(4.5, .42 * len(forest))))
    y = np.arange(len(forest))
    labels = forest.comparison + " / " + forest.dimension.str.title()
    ax.errorbar(forest.mean_difference, y,
                xerr=[forest.mean_difference - forest.ci95_low, forest.ci95_high - forest.mean_difference],
                fmt="o", color="#2F4B7C", ecolor="#7A8DA8", capsize=3)
    ax.axvline(0, color="#777", ls="--", lw=.8); ax.set_yticks(y, labels); ax.set_xlabel("Mean paired difference (95% song-bootstrap CI)")
    ax.set_title("Predefined section contrasts")
    save_figure(fig, "11_section_contrast_effects")


def write_reports(audit: dict, descriptive: pd.DataFrame, section_tests: pd.DataFrame,
                  transition_summary: pd.DataFrame, transition_test: pd.DataFrame,
                  repeats_summary: pd.DataFrame, peaks: pd.DataFrame, features: pd.DataFrame,
                  cluster_meta: dict, associations: pd.DataFrame, counter: pd.DataFrame,
                  sequence: pd.DataFrame) -> None:
    cv = section_tests[(section_tests.comparison == "Chorus - Verse") & (section_tests.weighting == "unweighted")].set_index("dimension")
    vt = transition_test[transition_test.comparison == "verse->chorus"].set_index("dimension")
    pct_a = cv.loc["arousal", "positive_song_fraction"]
    pct_v = cv.loc["valence", "positive_song_fraction"]
    peak_a = peaks[peaks.peak_definition == "max_mean_arousal"]
    peak_counts = peak_a.section_type.value_counts()
    peak_chorus = int(peak_counts.get("chorus", 0))
    peak_late = float((peak_a.relative_position >= .75).mean())
    rep_ch = repeats_summary[repeats_summary.section_type == "chorus"].set_index("dimension")
    corr_median = features.section_va_spearman.median()
    corr_negative = int((features.section_va_spearman < 0).sum())
    top_sequences = sequence.head(5)
    supported = cluster_meta["discrete_solution_cautiously_supported"]

    audit_md = f"""# Data Audit Report\n\nReport time: {datetime.now().astimezone().isoformat(timespec='seconds')}\n\n## Scope\n\n- Primary songs: **{audit['songs']}**\n- Artists: **{audit['artists']}**\n- Sections: **{audit['section_rows']}**\n- In-scope 0.5-second frames: **{audit['frame_rows']}**\n- Approved sections: **{audit['approved_sections']}/{audit['section_rows']}**\n- Duplicate annotation IDs: **{audit['duplicate_annotation_ids']}**\n- Duplicate song/order rows: **{audit['duplicate_song_orders']}**\n- Order-error songs: **{audit['order_error_songs']}**\n- Boundary-discontinuity songs: **{audit['continuity_error_songs']}**\n- Nonpositive section durations: **{audit['nonpositive_duration_sections']}**\n- Unrecognized section types: **{audit['unrecognized_section_types'] or 'none'}**\n\n## Actual Data Semantics\n\nThe VA scale is `[-1, 1]`. Each section value is the arithmetic mean of all\nTeacher predictions at 0.5-second frame centers assigned to the reviewed\nsection interval. These are pseudo-VA estimates, not independent human ground\ntruth. Source section labels are already canonical: Intro, Verse, Pre-chorus,\nChorus, Bridge, Interlude, and Outro. The original label and normalized label\nare both retained in the cleaned file.\n\n## Distribution\n\n- Section Valence range: `{audit['section_valence']['min']:.3f}` to `{audit['section_valence']['max']:.3f}`.\n- Section Arousal range: `{audit['section_arousal']['min']:.3f}` to `{audit['section_arousal']['max']:.3f}`.\n- Sections per song: median `{audit['sections_per_song']['50%']:.0f}`, range `{audit['sections_per_song']['min']:.0f}`-`{audit['sections_per_song']['max']:.0f}`.\n- Song duration: median `{audit['duration_sec_per_song']['50%']:.1f}` s, range `{audit['duration_sec_per_song']['min']:.1f}`-`{audit['duration_sec_per_song']['max']:.1f}` s.\n\n## Decisions\n\n1. The song is the independent statistical unit.\n2. Raw pseudo-VA is primary; within-song centered/z-scored values are sensitivity data.\n3. Occurrence means are primary; duration weighting is a sensitivity analysis.\n4. `Last observed Chorus` is used because no `Final Chorus` label exists.\n5. Maximum predicted Arousal is called a model-derived peak candidate, not a verified human climax.\n"""
    (REPORT_DIR / "DATA_AUDIT.md").write_text(audit_md, encoding="utf-8")

    findings = f"""# Main Findings Summary\n\nReport time: {datetime.now().astimezone().isoformat(timespec='seconds')}\n\n1. **Chorus is the clearest section-level contrast.** In {int(cv.loc['arousal','n_songs'])} paired songs, Chorus minus Verse was `{cv.loc['valence','mean_difference']:+.3f}` Valence and `{cv.loc['arousal','mean_difference']:+.3f}` Arousal. Positive lift occurred in `{pct_v:.1%}` and `{pct_a:.1%}` of songs, respectively.\n2. **The lift is stronger in Arousal than Valence.** Direct Verse-to-Chorus transitions changed by `{vt.loc['valence','mean_difference']:+.3f}` Valence and `{vt.loc['arousal','mean_difference']:+.3f}` Arousal across {int(vt.loc['arousal','n_songs'])} songs. This supports an energy-release interpretation more strongly than a universal positivity increase.\n3. **Later Chorus is not automatically stronger.** First-to-last observed Chorus change was `{rep_ch.loc['valence','mean_first_to_last']:+.3f}` Valence and `{rep_ch.loc['arousal','mean_first_to_last']:+.3f}` Arousal. These results must be read with their confidence intervals and corrected p-values in `repeated_section_summary.csv`.\n4. **Model-derived Arousal peaks favor Chorus but are not exclusive to it.** Chorus was the maximum-mean-Arousal section in `{peak_chorus}/97` songs; `{peak_late:.1%}` of peaks were in the final quarter. Other peak locations are substantive counterexamples, not automatic errors.\n5. **Valence and Arousal are not interchangeable.** Median within-song section-level Spearman correlation was `{corr_median:.3f}`, and {corr_negative}/97 songs had a negative association. Transition-direction tables show both coupled and dimension-separated changes.\n6. **Trajectory clusters are {'cautiously supported' if supported else 'not sufficiently stable for a strong taxonomy'}.** The selected descriptive solution uses k={cluster_meta['selected_k']} with silhouette `{cluster_meta['selection_silhouette']:.3f}` and subsample ARI `{cluster_meta['selection_subsample_ari_mean']:.3f}`. It is exploratory, not a discovered universal set of pop-song arc types.\n\nAll findings describe this convenience sample and frozen Teacher outputs. They do not prove universal compositional rules or true human emotion trajectories.\n"""
    (REPORT_DIR / "MAIN_FINDINGS.md").write_text(findings, encoding="utf-8")

    limitations = """# Limitations And Next Steps\n\nReport date: 2026-07-26\n\n## Main Limitations\n\n1. VA values are frozen Teacher pseudo-labels. The 15-song human check supports trend-level use only and is based on one rater.\n2. The 97-song set is a convenience sample, not a probability sample of all Chinese popular music. Artist, era, genre, production style, and platform exposure are not controlled.\n3. Section labels have reviewed boundaries and canonical types, but form labels can be perceptually ambiguous. The source does not distinguish Chorus 1, Chorus 2, or Final Chorus.\n4. No lyrics, harmony, mode, tempo, instrumentation, loudness, or production descriptors are used. VA differences cannot be attributed to a specific compositional mechanism.\n5. Multiple sections from one song are dependent. Inference therefore uses song-level aggregation; section-level plots are descriptive.\n6. Clustering 97 trajectories has limited power. Cluster labels are descriptive and require replication on an independently sampled corpus.\n\n## Highest-Value Next Steps\n\n1. Add a second independent rater to the frozen 15-song validation set and report inter-rater reliability.\n2. Manually review the listed counterexample trajectories and boundary neighborhoods before interpreting them musically.\n3. Freeze an independent, ranking-stratified Chinese-pop extension only if stronger external validity is required. Do not tune the Teacher on it.\n4. In a later study, add interpretable audio descriptors to test mechanisms behind Arousal and Valence changes.\n"""
    (REPORT_DIR / "LIMITATIONS_AND_NEXT_STEPS.md").write_text(limitations, encoding="utf-8")

    discovery_rows = []
    for _, row in associations.sort_values("p_holm").head(6).iterrows():
        discovery_rows.append(f"- `{row.x}` vs `{row.y}`: Spearman rho `{row.spearman_rho:.3f}`, Holm p `{row.p_holm:.4g}`, n={int(row.n_songs)}. This is exploratory association evidence, not causation.")
    discovery = "# Exploratory Discovery Log\n\nReport date: 2026-07-26\n\n## Data-Driven Checks\n\n" + "\n".join(discovery_rows) + "\n\n## Common Exact Sequences\n\n"
    for _, row in top_sequences.iterrows():
        discovery += f"- n={int(row.song_count)} ({row.song_fraction:.1%}): `{row.sequence}`\n"
    discovery += "\n## Follow-Up Queue\n\nThe machine-selected counterexamples are listed in `counterexample_review_queue.csv`. They are retained for listening and boundary review rather than deleted as noise.\n"
    (REPORT_DIR / "DISCOVERY_LOG.md").write_text(discovery, encoding="utf-8")

    report = f"""# 97首中文流行歌曲 Section-Aware 情绪结构探索报告\n\n报告时间：{datetime.now().astimezone().strftime('%Y-%m-%d %H:%M %Z')}\n\n当前科研进度：第一阶段 VA Teacher 已冻结；第二阶段完成 97 首中文歌曲的完整曲 pseudo-VA 推理、section 对齐、人工趋势验证和本轮探索性结构分析。\n\n## 1. 数据与方法\n\n本研究分析 {audit['songs']} 首中文流行歌曲、{audit['section_rows']} 个经过质检的 section。每个 section 的 Valence/Arousal 是落入该段落的 0.5 秒 Teacher 帧的算术均值，范围为 `[-1,1]`。这些值属于 **pseudo-VA**，不能写成独立人工真实值。\n\n统计推断以歌曲为独立单位：先在每首歌内计算 Verse、Chorus 等段落均值或转换均值，再对歌曲进行配对 Wilcoxon、歌曲级 bootstrap 置信区间和 Holm 多重比较校正。段落行级小提琴图仅用于描述。\n\n原始标签已经统一为 Intro、Verse、Pre-chorus、Chorus、Bridge、Interlude、Outro。由于没有 `Final Chorus` 字段，本文只分析“第一次 Chorus”与“最后一次出现的 Chorus”，不把后者自动等同于终极副歌。\n\n流行音乐结构标签本身允许一定感知歧义，因此结果解释重点放在本数据的功能段落差异，而不是把标签当作绝对形式真理。Harmonix Set 说明功能段落标注是流行音乐结构研究的标准数据表示之一；de Clercq 的形式研究同时提醒 Verse、Chorus、Bridge 的分类具有模糊边界。\n\n## 2. 不同 Section 的情感特征\n\n最稳定的结果是 Chorus 相对 Verse 的抬升。在 {int(cv.loc['arousal','n_songs'])} 首同时包含二者的歌曲中：\n\n- Valence 平均差 `{cv.loc['valence','mean_difference']:+.3f}`，95% CI `[{cv.loc['valence','ci95_low']:.3f}, {cv.loc['valence','ci95_high']:.3f}]`，正向歌曲占 `{pct_v:.1%}`；\n- Arousal 平均差 `{cv.loc['arousal','mean_difference']:+.3f}`，95% CI `[{cv.loc['arousal','ci95_low']:.3f}, {cv.loc['arousal','ci95_high']:.3f}]`，正向歌曲占 `{pct_a:.1%}`。\n\nArousal 差明显大于 Valence 差，数据更直接支持“副歌通常提高能量/唤醒度”，而不是“副歌必然更加积极”。这可能对应主歌铺垫到副歌释放的形式功能，但由于本研究没有响度、速度、配器等机制变量，不能断言由哪一种作曲或制作手段造成。\n\n## 3. Section 转换\n\n直接 Verse → Chorus 的歌曲级平均变化为 Valence `{vt.loc['valence','mean_difference']:+.3f}`、Arousal `{vt.loc['arousal','mean_difference']:+.3f}`。完整转换表同时保留 VA 二维距离、方向和稳定阈值敏感性。`0.05` 被设为主阈值，并以 `0.03/0.07` 复核，避免把微小噪声强行解释为上升或下降。\n\n转换进入 Chorus 并非只有“双维共同上升”：也存在 Arousal 上升但 Valence 稳定或下降的歌曲。这说明两个维度承担不同表达功能，不能压缩成单一“情绪强度”。\n\n## 4. 重复段落与副歌抬升\n\n第一次到最后一次出现的 Chorus：\n\n- Valence 平均变化 `{rep_ch.loc['valence','mean_first_to_last']:+.3f}`；\n- Arousal 平均变化 `{rep_ch.loc['arousal','mean_first_to_last']:+.3f}`。\n\n是否构成系统强化必须结合 `repeated_section_summary.csv` 的置信区间和 Holm 校正结果判断。即使平均上升，也不能称所有歌曲都采用“逐次增强”；配对图保留了下降和近似稳定的反例。\n\n## 5. 高点位置与全曲组织\n\n以 section 平均 Arousal 最大值定义“模型推测的高唤醒候选段”时，Chorus 出现在 `{peak_chorus}/97` 首；`{peak_late:.1%}` 位于全曲最后四分之一。Bridge、Interlude、Outro 等也可能成为高点，因此“高潮等于唯一副歌”不成立。本文分别保留最大 Arousal、最大 Valence和最大相邻转换距离三种定义，不用任意加权把它们混成一个分数。\n\n## 6. Valence 与 Arousal 的关系\n\n歌曲内 section VA 的 Spearman 相关中位数为 `{corr_median:.3f}`；{corr_negative}/97 首为负相关。结果提示 V/A 经常同向但不是必然同步。专业解释应区分“积极程度”与“激烈程度”，尤其关注 Valence 稳定而 Arousal 抬升、或二者反向的 section 转换。\n\n## 7. 轨迹模式\n\n完整歌曲帧曲线被插值到 101 个归一化时间点；聚类使用歌曲内标准化后的 V/A 形状，比较 KMeans 与 Ward、k=2–5，并通过 100 次 80% 子样本检验稳定性。选中的描述性解为 k={cluster_meta['selected_k']}，silhouette=`{cluster_meta['selection_silhouette']:.3f}`，子样本 ARI=`{cluster_meta['selection_subsample_ari_mean']:.3f}`。\n\n离散模式结论：**{'达到谨慎支持阈值，可作为探索性类型使用' if supported else '未达到预设稳定性阈值，不宜宣称存在固定的情感弧线类型'}**。即使画出聚类均值轨迹，也只用于观察连续变化，不作为普遍分类法。\n\n## 8. 自主发现与反例\n\n歌曲 section 数量、类型熵、VA 路径长度等关联见 `song_level_associations_holm.csv`；常见完整 section 序列见 `common_exact_section_sequences.csv`。这些关系是探索性发现，不代表复杂结构导致更强情绪变化。\n\n`counterexample_review_queue.csv` 主动保留无副歌抬升、V/A 强烈分离、路径极短和单次转换极大的歌曲。它们可能是特殊创作策略，也可能暴露 Teacher 或边界问题，必须逐首听审后才能解释。\n\n## 9. 稳健性\n\n- 原始 VA 与歌曲内中心化/标准化副本均被保留；主结论使用原始 VA。\n- section 类型均值同时提供不加权与时长加权结果。\n- 转换方向使用 0.03、0.05、0.07 三个阈值。\n- 关键比较以歌曲为单位 bootstrap，不把 1,076 个 section 当作独立样本。\n- 预定义检验族执行 Holm 校正；效应量、覆盖率、反例同时报告。\n\n## 10. 结论与边界\n\n本数据最清晰的结构规律是：Chorus 相对 Verse 的情绪差异主要体现为 Arousal 抬升，Valence 也倾向上升但幅度更小；高唤醒候选段偏向 Chorus 和歌曲后部，却存在大量其他组织方式；V/A 在部分歌曲中明显分离。\n\n可用于论文的克制表述是：“在本 97 首中文流行歌曲便利样本及冻结 Teacher pseudo-VA 上，观察到上述结构倾向。”不能写成“证明了中文流行歌曲的普遍创作规律”。独立第二标注者与外部抽样复现仍是提高证据等级的关键。\n\n## 参考文献边界\n\n- Nieto et al. (2019), The Harmonix Set: https://archives.ismir.net/ismir2019/paper/000068.pdf\n- de Clercq (2017), Embracing Ambiguity in Pop/Rock Form: https://www.mtosmt.org/issues/mto.17.23.3/mto.17.23.3.de_clercq.php\n- Eerola & Vuoskoski (2011), dimensional and discrete emotion models: https://doi.org/10.1177/0305735610362821\n"""
    (REPORT_DIR / "EXPLORATORY_ANALYSIS_REPORT_ZH.md").write_text(report, encoding="utf-8")


def write_extended_reports(audit: dict, section_tests: pd.DataFrame,
                           transition_test: pd.DataFrame, repeats_summary: pd.DataFrame,
                           repeat_contrasts: pd.DataFrame, peaks: pd.DataFrame,
                           features: pd.DataFrame, cluster_meta: dict,
                           associations: pd.DataFrame, sequence: pd.DataFrame) -> None:
    """Write the final interpretation after cross-result and counterexample checks."""
    core = section_tests[section_tests.weighting == "unweighted"]
    cv = core[core.comparison == "Chorus - Verse"].set_index("dimension")
    pv = core[core.comparison == "Pre-chorus - Verse"].set_index("dimension")
    cp = core[core.comparison == "Chorus - Pre-chorus"].set_index("dimension")
    bc = core[core.comparison == "Bridge - Chorus"].set_index("dimension")
    vt = transition_test[transition_test.comparison == "verse->chorus"].set_index("dimension")
    rep = repeats_summary[repeats_summary.section_type == "chorus"].set_index("dimension")
    did = repeat_contrasts[
        repeat_contrasts.comparison == "Chorus change - Verse change"
    ].set_index("dimension")
    peak_a = peaks[peaks.peak_definition == "max_mean_arousal"]
    peak_transition = peaks[peaks.peak_definition == "max_transition_distance"]
    chorus_peak_n = int((peak_a.section_type == "chorus").sum())
    late_peak_fraction = float((peak_a.relative_position >= .75).mean())
    outro_transition_n = int((peak_transition.section_type == "outro").sum())
    corr_median = features.section_va_spearman.median()
    corr_negative = int((features.section_va_spearman < 0).sum())
    supported = cluster_meta["discrete_solution_cautiously_supported"]

    summary_lines = [
        "# Main Findings Summary", "", f"Report time: {datetime.now().astimezone().isoformat(timespec='seconds')}", "",
        f"1. **Chorus is the clearest section contrast.** Across {int(cv.loc['arousal','n_songs'])} paired songs, Chorus minus Verse was `{cv.loc['valence','mean_difference']:+.3f}` Valence and `{cv.loc['arousal','mean_difference']:+.3f}` Arousal; positive lift covered `{cv.loc['valence','positive_song_fraction']:.1%}` and `{cv.loc['arousal','positive_song_fraction']:.1%}` of songs.",
        f"2. **The transition is mainly an Arousal release.** Direct Verse-to-Chorus change was `{vt.loc['valence','mean_difference']:+.3f}` Valence and `{vt.loc['arousal','mean_difference']:+.3f}` Arousal. V and A therefore should not be collapsed into one score.",
        f"3. **Later Chorus grows, but the growth is not Chorus-specific.** First-to-last Chorus changed `{rep.loc['valence','mean_first_to_last']:+.3f}/{rep.loc['arousal','mean_first_to_last']:+.3f}` V/A. Chorus-minus-Verse repeat change was only `{did.loc['valence','mean_difference']:+.3f}/{did.loc['arousal','mean_difference']:+.3f}` with Holm p `{did.loc['valence','p_holm']:.3f}/{did.loc['arousal','p_holm']:.3f}`. This supports general late-song accumulation.",
        f"4. **Peak organization is late but not synonymous with Chorus.** Chorus is the model-derived max-Arousal section in `{chorus_peak_n}/97` songs and `{late_peak_fraction:.1%}` of peaks lie in the final quarter. The largest transition lands on Outro in `{outro_transition_n}/97` songs, usually reflecting release/closure.",
        f"5. **Form is heterogeneous.** There are `{len(sequence)}` exact complete section sequences; the most common covers only `{int(sequence.iloc[0].song_count)}/97` songs.",
        f"6. **VA is usually coupled but can separate.** Median within-song section Spearman is `{corr_median:.3f}`; `{corr_negative}/97` songs are negative.",
        f"7. **Discrete arc types are {'cautiously supported' if supported else 'not supported strongly enough'}.** Best descriptive solution: k={cluster_meta['selected_k']}, silhouette `{cluster_meta['selection_silhouette']:.3f}`, subsample ARI `{cluster_meta['selection_subsample_ari_mean']:.3f}`.",
        "", "These are exploratory patterns in frozen Teacher pseudo-VA, not universal rules or independent human ground truth.", "",
    ]
    (REPORT_DIR / "MAIN_FINDINGS.md").write_text("\n".join(summary_lines), encoding="utf-8")

    report_lines = [
        "# 97首中文流行歌曲 Section-Aware 情绪结构探索报告", "",
        f"报告时间：{datetime.now().astimezone().strftime('%Y-%m-%d %H:%M %Z')}", "",
        "当前科研进度：第一阶段 VA Teacher 已冻结；第二阶段完成完整曲推理、section 对齐、15 首人工趋势验证，以及本轮 97 首探索性结构分析。", "",
        "## 1. 数据与方法", "",
        f"本研究分析 **{audit['songs']} 首歌曲、{audit['section_rows']} 个质检 section、{audit['frame_rows']} 个 0.5 秒帧**。section VA 是其内部所有 Teacher 帧的算术均值，范围为 `[-1,1]`。它是 pseudo-VA，不是独立人工真实值。", "",
        "统计推断以歌曲为独立单位，采用歌曲内聚合、配对 Wilcoxon、5,000 次歌曲级 bootstrap、效应量和 Holm 校正。原始 VA 为主分析，时长加权与歌曲内标准化为敏感性分析。原始 section 标签已经规范为七类；数据没有 `Final Chorus` 标签，因此仅使用“最后一次出现的 Chorus”。", "",
        "流行音乐的功能段落标注是已有 MIR 语料采用的结构表示，但段落分类仍可能具有感知歧义。因此本文分析功能标签与 pseudo-VA 的关系，不把标签视为绝对形式真理。", "",
        "## 2. Section 的典型情感功能", "",
        f"Chorus 相对 Verse 的 Valence 差为 `{cv.loc['valence','mean_difference']:+.3f}`（95% CI `[{cv.loc['valence','ci95_low']:.3f}, {cv.loc['valence','ci95_high']:.3f}]`），Arousal 差为 `{cv.loc['arousal','mean_difference']:+.3f}`（95% CI `[{cv.loc['arousal','ci95_low']:.3f}, {cv.loc['arousal','ci95_high']:.3f}]`）。正向覆盖分别为 `{cv.loc['valence','positive_song_fraction']:.1%}` 和 `{cv.loc['arousal','positive_song_fraction']:.1%}`。", "",
        "这更直接支持“副歌提高唤醒度/能量”，而不是“副歌必然更加积极”。全 96 首 Teacher Arousal 均为正向抬升过于整齐，虽然 15 首人工验证中也有 13/15 同方向，仍需保留 Teacher 结构偏差的可能。", "",
        f"Pre-chorus 相对 Verse 的 Arousal 为 `{pv.loc['arousal','mean_difference']:+.3f}`，Chorus 相对 Pre-chorus 再增加 `{cp.loc['arousal','mean_difference']:+.3f}`，提示分级蓄力—释放。Bridge 相对 Chorus 的 Arousal 为 `{bc.loc['arousal','mean_difference']:+.3f}`（n={int(bc.loc['arousal','n_songs'])}），但 Bridge 覆盖低且位置偏后，不能推广到所有歌曲。", "",
        "## 3. 相邻 Section 转换", "",
        f"直接 Verse → Chorus 的歌曲级平均变化为 Valence `{vt.loc['valence','mean_difference']:+.3f}`、Arousal `{vt.loc['arousal','mean_difference']:+.3f}`。在 0.05 稳定阈值下，所有 368 次进入 Chorus 的转换中，双维共同上升占 `30.2%`、仅 Arousal 上升占 `24.2%`、稳定占 `19.3%`；其余为下降或维度分离。完整结果同时使用 0.03/0.05/0.07 阈值复核。", "",
        "## 4. 重复段落与全曲累积", "",
        f"第一次到最后一次 Chorus 平均变化为 Valence `{rep.loc['valence','mean_first_to_last']:+.3f}`、Arousal `{rep.loc['arousal','mean_first_to_last']:+.3f}`，正向歌曲比例为 `{rep.loc['valence','positive_fraction']:.1%}/{rep.loc['arousal','positive_fraction']:.1%}`。", "",
        f"关键反思是 Verse、Pre-chorus、Interlude 的后续重复也上升。对 89 首同时重复 Verse 与 Chorus 的歌曲做差分，Chorus 相对 Verse 的额外变化只有 Valence `{did.loc['valence','mean_difference']:+.3f}`、Arousal `{did.loc['arousal','mean_difference']:+.3f}`，Holm p=`{did.loc['valence','p_holm']:.3f}/{did.loc['arousal','p_holm']:.3f}`。没有证据表明副歌得到独有的重复强化，更合理的是全曲后程普遍累积。", "",
        "## 5. 高点与收束", "",
        f"最大 section 平均 Arousal 位于 Chorus 的歌曲为 `{chorus_peak_n}/97`，位于最后四分之一的为 `{late_peak_fraction:.1%}`。归一化平均轨迹在约 75%–80% 位置达到高位，随后回落。最大 VA 转换最常落到 Outro（`{outro_transition_n}/97`），通常表示进入收束段的明显下降，而不是新的正向高潮。", "",
        "最大 Arousal、最大 Valence、最大转换距离被分开定义；本文不使用任意加权制造单一高潮分数。", "",
        "## 6. Valence 与 Arousal", "",
        f"歌曲内 section VA 的 Spearman 相关中位数为 `{corr_median:.3f}`，`{corr_negative}/97` 首为负。V/A 多数同向，但反向和单维变化确实存在，因此后续结构特征必须保留两个维度。", "",
        "## 7. 形式异质性与预设方法审计", "",
        f"97 首歌产生 `{len(sequence)}` 种完整 section 序列，最常见序列只有 `{int(sequence.iloc[0].song_count)}` 首（`{sequence.iloc[0].song_fraction:.1%}`）。这是样本中的形式分布描述，不用于建立预设的歌曲类型。", "",
        f"作为方法审计，曾比较 k=2–5 的 KMeans 与 Ward，并做 100 次 80% 子样本检验。描述性最佳解为 k={cluster_meta['selected_k']}，silhouette=`{cluster_meta['selection_silhouette']:.3f}`、子样本 ARI=`{cluster_meta['selection_subsample_ari_mean']:.3f}`。该预设分类尝试未通过稳定性门槛，结果只保留在实验记录中，不作为关于音乐情感结构的实质结论。", "",
        "## 8. 自主发现与反例", "",
        "时长与累计 VA 路径长度相关（rho `0.429`），但这是累积指标，不能解释为长歌更复杂。section 数量与平均转换幅度弱负相关（rho `-0.276`）；类型熵与最大单次转换距离弱正相关（rho `0.318`）。这些均是探索性关联，不是因果关系。", "",
        "反例清单保留 9 首 Valence 副歌未抬升歌曲、强 V/A 分离、低路径长度和极大单次转换歌曲。反例必须听审，不能为了结论整齐而删除。", "",
        "## 9. 稳健性", "",
        "- 不加权和时长加权的核心 Chorus–Verse 差异基本一致。",
        "- 方向分类使用 0.03、0.05、0.07 三种阈值。",
        "- 重复副歌结果增加 Chorus–Verse 差分，排除一般时间趋势。",
        "- 1,076 个 section 不被当作独立样本。",
        "- 预设聚类尝试只作为方法审计，不进入音乐学结论。", "",
        "## 10. 结论与证据边界", "",
        "本数据最清晰的规律是：Chorus 相对 Verse 的差异主要体现为 Arousal 抬升；后续副歌通常更强，但属于更广泛的后程累积；高唤醒候选段偏向 Chorus 和歌曲后部；V/A 在少数歌曲中分离。", "",
        "论文应写“在本 97 首中文流行歌曲便利样本及冻结 Teacher pseudo-VA 上观察到结构倾向”，不能写“证明了中文流行歌曲的普遍创作规律”。第二独立标注者和独立外部样本仍是提升证据等级的关键。", "",
        "## 参考文献", "",
        "- Nieto et al. (2019), The Harmonix Set: https://archives.ismir.net/ismir2019/paper/000068.pdf",
        "- de Clercq (2017), Embracing Ambiguity in Pop/Rock Form: https://www.mtosmt.org/issues/mto.17.23.3/mto.17.23.3.de_clercq.php",
        "- Eerola & Vuoskoski (2011), dimensional and discrete emotion models: https://doi.org/10.1177/0305735610362821", "",
    ]
    (REPORT_DIR / "EXPLORATORY_ANALYSIS_REPORT_ZH.md").write_text("\n".join(report_lines), encoding="utf-8")

    discovery = [
        "# Exploratory Discovery Log", "", "Report date: 2026-07-26", "",
        "## Findings That Changed The Initial Interpretation", "",
        f"- Later Chorus rose in V/A, but Chorus-minus-Verse repeat change was `{did.loc['valence','mean_difference']:+.3f}/{did.loc['arousal','mean_difference']:+.3f}` and nonsignificant after Holm correction. Interpretation changed from Chorus-specific reinforcement to general late-song accumulation.",
        f"- Exact complete form is highly heterogeneous: {len(sequence)} sequences among 97 songs; the top sequence appears in only {int(sequence.iloc[0].song_count)} songs.",
        f"- A preset trajectory-clustering attempt failed its stability gate (silhouette `{cluster_meta['selection_silhouette']:.3f}`, ARI `{cluster_meta['selection_subsample_ari_mean']:.3f}`) and is retained only as a method-audit record, not a substantive musicological conclusion.",
        f"- Maximum transition most often lands on Outro ({outro_transition_n}/97), separating peak intensity from largest structural contrast.", "",
        "## Association Screen", "",
    ]
    for _, row in associations.sort_values("p_holm").iterrows():
        discovery.append(f"- `{row.x}` vs `{row.y}`: rho `{row.spearman_rho:.3f}`, Holm p `{row.p_holm:.4g}`, n={int(row.n_songs)}.")
    discovery.extend(["", "All associations are exploratory and require independent replication.", ""])
    (REPORT_DIR / "DISCOVERY_LOG.md").write_text("\n".join(discovery), encoding="utf-8")


def main() -> None:
    for directory in (DATA_DIR, TABLE_DIR, FIG_DIR, REPORT_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    sections = pd.read_csv(SECTION_SOURCE, dtype={"song_id": str})
    songs = pd.read_csv(SONG_SOURCE, dtype={"song_id": str})
    frames_all = pd.read_csv(FRAME_SOURCE, dtype={"song_id": str})
    in_scope = set(sections.song_id)
    frames = frames_all[frames_all.song_id.isin(in_scope)].copy()
    sections, songs, audit = audit_and_clean(sections, songs, frames)

    descriptive, song_type, weighted = section_descriptives(sections)
    tests = section_contrasts(song_type, weighted)
    transitions, transition_summary, directions = transition_analysis(sections)
    transition_test = transition_tests(transitions)
    repeat_detail, repeat_summary = repeated_sections(sections)
    repeat_contrasts = repeated_change_contrasts(repeat_detail)
    threshold_sensitivity = transition_threshold_sensitivity(transitions)
    peaks, peak_summary = climax_analysis(sections, transitions)
    features = song_features(sections, transitions, songs)
    trajectories = trajectory_data(frames)
    cluster_diag, assignments, cluster_meta = cluster_trajectories(trajectories)
    associations = association_table(features, repeat_detail)
    counter = counterexamples(features, sections, repeat_detail, transitions)
    sequences = common_sequences(sections)

    outputs = {
        "section_type_descriptive_statistics.csv": descriptive,
        "song_by_section_type_means.csv": song_type,
        "song_by_section_type_duration_weighted_means.csv": weighted,
        "section_type_paired_tests_holm.csv": tests,
        "adjacent_section_transitions.csv": transitions,
        "transition_descriptive_statistics.csv": transition_summary,
        "transition_direction_proportions.csv": directions,
        "predefined_transition_tests_holm.csv": transition_test,
        "repeated_section_song_details.csv": repeat_detail,
        "repeated_section_summary.csv": repeat_summary,
        "repeated_section_change_contrasts_holm.csv": repeat_contrasts,
        "transition_threshold_sensitivity.csv": threshold_sensitivity,
        "model_derived_peak_candidates.csv": peaks,
        "model_derived_peak_summary.csv": peak_summary,
        "song_level_structure_features.csv": features,
        "trajectory_clustering_diagnostics.csv": cluster_diag,
        "trajectory_cluster_assignments.csv": assignments,
        "song_level_associations_holm.csv": associations,
        "counterexample_review_queue.csv": counter,
        "common_exact_section_sequences.csv": sequences,
    }
    paths = [save_csv(frame, name) for name, frame in outputs.items()]
    save_csv(trajectories, "normalized_complete_song_trajectories_101pts.csv", DATA_DIR)
    (TABLE_DIR / "trajectory_clustering_decision.json").write_text(json.dumps(cluster_meta, indent=2) + "\n", encoding="utf-8")
    (REPORT_DIR / "data_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    make_figures(sections, transitions, repeat_detail, peaks, trajectories, features,
                 cluster_diag, assignments, cluster_meta, tests)
    write_reports(audit, descriptive, tests, transition_summary, transition_test,
                  repeat_summary, peaks, features, cluster_meta, associations, counter, sequences)
    write_extended_reports(audit, tests, transition_test, repeat_summary,
                           repeat_contrasts, peaks, features, cluster_meta,
                           associations, sequences)

    manifest = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": "pass", "protocol": protocol,
        "inputs": {str(path.relative_to(ROOT)): {"sha256": sha256_file(path)} for path in (SECTION_SOURCE, SONG_SOURCE, FRAME_SOURCE)},
        "counts": {"songs": int(sections.song_id.nunique()), "sections": len(sections), "frames": len(frames)},
        "cluster_decision": cluster_meta,
        "outputs": {str(path.relative_to(OUT)): {"sha256": sha256_file(path)} for path in OUT.rglob("*") if path.is_file() and path.name != "manifest.json"},
        "scientific_guardrails": [
            "Teacher outputs are pseudo-VA, not human ground truth",
            "song is the inferential unit",
            "no Teacher tuning was performed",
            "cluster labels remain exploratory",
        ],
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "pass", "counts": manifest["counts"], "cluster_decision": cluster_meta,
                      "output_root": str(OUT)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    main()
