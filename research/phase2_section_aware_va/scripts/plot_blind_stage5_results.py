#!/usr/bin/env python3
"""Create paper-ready diagnostic plots for the Stage 5 blind structural analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.font_manager import FontProperties


PHASE2_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE5 = PHASE2_ROOT / "data/blind_validation/stage5"
DEFAULT_OUTPUT = PHASE2_ROOT / "outputs/plots/blind_validation_stage5"
TYPE_COLORS = {
    "intro": "#4C78A8",
    "verse": "#72B7B2",
    "pre_chorus": "#F2CF5B",
    "chorus": "#E45756",
    "bridge": "#B279A2",
    "interlude": "#54A24B",
    "outro": "#9D755D",
}
CJK_FONT = FontProperties(fname="/System/Library/Fonts/Supplemental/Arial Unicode.ttf")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def paired_plot(stage5: Path, output: Path) -> None:
    sections = pd.read_csv(stage5 / "blind_sample_15_section_features_with_types.csv")
    song_type = sections.groupby(["song_id", "section_type"])[["mean_valence", "mean_arousal"]].mean()
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2), sharex=True)
    for axis, dimension, label in zip(axes, ("mean_valence", "mean_arousal"), ("Valence", "Arousal")):
        values = []
        for song_id in sections["song_id"].drop_duplicates():
            verse = song_type.loc[(song_id, "verse"), dimension]
            chorus = song_type.loc[(song_id, "chorus"), dimension]
            values.append((verse, chorus))
            axis.plot([0, 1], [verse, chorus], color="#A9ADB3", alpha=0.65, linewidth=1.0, zorder=1)
            axis.scatter(0, verse, color=TYPE_COLORS["verse"], s=24, zorder=2)
            axis.scatter(1, chorus, color=TYPE_COLORS["chorus"], s=24, zorder=2)
        values = np.asarray(values)
        axis.plot([0, 1], values.mean(axis=0), color="#111827", linewidth=2.6, marker="o", zorder=3)
        axis.axhline(0, color="#D1D5DB", linewidth=0.8)
        axis.set_xticks([0, 1], ["Verse", "Chorus"])
        axis.set_ylabel(f"Mean predicted {label}")
        axis.set_title(f"Within-song {label}")
        axis.grid(axis="y", color="#E5E7EB", linewidth=0.7)
    fig.suptitle("Held-out section types: paired Verse and Chorus pseudo-VA", fontsize=12)
    fig.tight_layout()
    fig.savefig(output / "chorus_verse_paired_va.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def type_scatter(stage5: Path, output: Path) -> None:
    sections = pd.read_csv(stage5 / "blind_sample_15_section_features_with_types.csv")
    fig, axis = plt.subplots(figsize=(7.4, 5.5))
    for section_type, group in sections.groupby("section_type"):
        color = TYPE_COLORS.get(section_type, "#6B7280")
        axis.scatter(
            group["mean_valence"],
            group["mean_arousal"],
            s=22,
            alpha=0.45,
            color=color,
            label=f"{section_type} (n={len(group)})",
        )
        axis.scatter(
            group["mean_valence"].mean(),
            group["mean_arousal"].mean(),
            s=105,
            marker="X",
            color=color,
            edgecolor="white",
            linewidth=0.8,
        )
    axis.axhline(0, color="#9CA3AF", linewidth=0.8)
    axis.axvline(0, color="#9CA3AF", linewidth=0.8)
    axis.set_xlabel("Section mean predicted Valence")
    axis.set_ylabel("Section mean predicted Arousal")
    axis.set_title("Pseudo-VA distribution by held-out section type")
    axis.grid(color="#E5E7EB", linewidth=0.6)
    axis.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output / "section_type_va_distribution.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def climax_plot(stage5: Path, output: Path) -> None:
    candidates = pd.read_csv(stage5 / "blind_sample_15_climax_candidates.csv").sort_values(
        "climax_section_time_ratio"
    )
    fig, axis = plt.subplots(figsize=(8.0, 5.4))
    y = np.arange(len(candidates))
    for index, row in enumerate(candidates.itertuples(index=False)):
        section_type = row.climax_candidate_section_type
        axis.scatter(
            row.climax_section_time_ratio,
            index,
            s=58,
            color=TYPE_COLORS.get(section_type, "#6B7280"),
            edgecolor="white",
            linewidth=0.7,
        )
    axis.axvline(0.5, color="#D1D5DB", linewidth=0.8, linestyle="--")
    axis.axvline(0.75, color="#9CA3AF", linewidth=0.8, linestyle="--")
    axis.set_yticks(y, candidates["title"], fontproperties=CJK_FONT)
    axis.set_xlim(0, 1)
    axis.set_xlabel("Normalized midpoint of maximum-mean-Arousal section")
    axis.set_title("Model-derived climax candidates (not human-validated)")
    axis.grid(axis="x", color="#E5E7EB", linewidth=0.6)
    handles = []
    for section_type in candidates["climax_candidate_section_type"].drop_duplicates():
        handles.append(
            plt.Line2D(
                [0], [0], marker="o", color="none", markerfacecolor=TYPE_COLORS.get(section_type, "#6B7280"),
                markeredgecolor="white", markersize=7, label=section_type
            )
        )
    axis.legend(handles=handles, frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(output / "climax_candidate_positions.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage5-dir", type=Path, default=DEFAULT_STAGE5)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paired_plot(args.stage5_dir, args.output_dir)
    type_scatter(args.stage5_dir, args.output_dir)
    climax_plot(args.stage5_dir, args.output_dir)
    paths = sorted(args.output_dir.glob("*.png"))
    manifest = {
        path.name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
        for path in paths
    }
    manifest_path = args.stage5_dir / "stage5_plot_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
