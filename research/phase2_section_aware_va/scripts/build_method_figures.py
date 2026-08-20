#!/usr/bin/env python3
"""Build publication-ready Method figures from the frozen 100-song dataset."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import librosa
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[3]
PACKAGE = ROOT / "exports/gpt_top60_quota_100_section_va_data_package_20260815"
TEACHER_EVAL = ROOT / "exports/gpt_section_va_data_package_20260802/01_teacher_evaluation"
OUTPUT = ROOT / "论文写作/Method图表_20260820"
REPRESENTATIVE_SONG_ID = "life_should_face_light"

MANIFEST_PATH = PACKAGE / "02_corpus_inventory/top60_quota_100_manifest.csv"
SECTION_PATH = PACKAGE / "05_section_level/top60_quota_100_section_features.csv"
FRAME_PATH = PACKAGE / "03_frame_level_va/top60_quota_100_frame_va.csv"
ALIGNMENT_PATH = PACKAGE / "04_frame_section_alignment/top60_quota_100_frame_section_alignment.csv"
TEACHER_RESULTS_PATH = TEACHER_EVAL / "FINAL_LOCKED_TEST_RESULTS.json"

SECTION_COLORS = {
    "intro": "#CAD6E2",
    "verse": "#AFCBE3",
    "pre_chorus": "#F2D27A",
    "chorus": "#E99A9A",
    "bridge": "#C4B2DE",
    "interlude": "#9CCFC7",
    "outro": "#B9C0C8",
}

SECTION_LABELS = {
    "intro": "Intro",
    "verse": "Verse",
    "pre_chorus": "Pre-chorus",
    "chorus": "Chorus",
    "bridge": "Bridge",
    "interlude": "Interlude",
    "outro": "Outro",
}

INK = "#1D2630"
MUTED = "#66717D"
GRID = "#D8DEE5"
BLUE = "#2F6B9A"
ORANGE = "#D2763C"
RED = "#B4473F"
TEAL = "#39867D"
PAPER = "#FFFFFF"


def parse_time(value: object) -> float:
    text = str(value).strip()
    if ":" not in text:
        return float(text)
    fields = [float(part) for part in text.split(":")]
    if len(fields) == 2:
        return fields[0] * 60.0 + fields[1]
    if len(fields) == 3:
        return fields[0] * 3600.0 + fields[1] * 60.0 + fields[2]
    raise ValueError(f"Unsupported time value: {value}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def configure_matplotlib() -> None:
    candidates = [
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    ]
    font_path = next((Path(path) for path in candidates if Path(path).exists()), None)
    if font_path is None:
        raise FileNotFoundError("No Chinese-capable publication font found")
    font_manager.fontManager.addfont(str(font_path))
    font_name = font_manager.FontProperties(fname=str(font_path)).get_name()
    plt.rcParams.update(
        {
            "font.family": font_name,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "path",
            "figure.facecolor": PAPER,
            "axes.facecolor": PAPER,
            "savefig.facecolor": PAPER,
        }
    )


def save_figure(fig: plt.Figure, stem: str) -> list[Path]:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for suffix, kwargs in [
        (".pdf", {"bbox_inches": "tight"}),
        (".svg", {"bbox_inches": "tight"}),
        (".png", {"dpi": 600, "bbox_inches": "tight"}),
        (".tiff", {"dpi": 600, "bbox_inches": "tight", "pil_kwargs": {"compression": "tiff_lzw"}}),
    ]:
        path = OUTPUT / f"{stem}{suffix}"
        fig.savefig(path, **kwargs)
        outputs.append(path)
    plt.close(fig)
    return outputs


def arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], color: str = MUTED) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=1.2,
            color=color,
            shrinkA=3,
            shrinkB=3,
            zorder=1,
        )
    )


def node(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    title: str,
    detail: str,
    face: str,
    edge: str,
) -> None:
    x, y = xy
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.02,rounding_size=0.05",
            linewidth=1.0,
            facecolor=face,
            edgecolor=edge,
            zorder=2,
        )
    )
    multiline = "\n" in title
    ax.text(
        x + width / 2,
        y + height * (0.66 if multiline else 0.62),
        title,
        ha="center",
        va="center",
        color=INK,
        fontsize=8.0 if multiline else 8.4,
        weight="bold",
        linespacing=0.95,
    )
    ax.text(
        x + width / 2,
        y + height * (0.19 if multiline else 0.27),
        detail,
        ha="center",
        va="center",
        color=MUTED,
        fontsize=6.1,
        linespacing=1.10,
    )


def build_figure_1() -> list[Path]:
    fig, ax = plt.subplots(figsize=(18 / 2.54, 8.2 / 2.54))
    ax.set_xlim(0, 15.7)
    ax.set_ylim(0, 5.8)
    ax.axis("off")

    ax.text(0.25, 5.42, "段落结构通路", color=BLUE, fontsize=8.2, weight="bold")
    ax.text(0.25, 1.46, "VA测量通路", color=ORANGE, fontsize=8.2, weight="bold")

    node(ax, (0.25, 2.55), 1.55, 1.05, "100首\n歌曲语料", "60位配额\n歌手", "#EDF2F6", "#99A8B6")
    node(ax, (2.40, 4.08), 1.85, 1.05, "候选段落\n生成", "歌词 + 人声\n+ 声学", "#E7F0F7", "#78A5C8")
    node(ax, (4.85, 4.08), 1.85, 1.05, "人工段落\n质检", "边界与类型\n修正", "#E7F3F0", "#70A99F")
    node(ax, (7.30, 4.08), 1.85, 1.05, "冻结段落\n标注", "1046段\n7类标签", "#EEF4F2", "#70A99F")

    node(ax, (3.55, 0.52), 2.05, 1.05, "冻结VA\n教师模型", "V / A参数固定", "#FCF0E8", "#D59468")
    node(ax, (6.25, 0.52), 1.95, 1.05, "0.5 s\n帧级VA", "51448帧", "#FCF0E8", "#D59468")

    node(ax, (9.75, 2.55), 1.80, 1.05, "帧—段落\n对齐", "51444帧\n映射", "#F2EFF8", "#9B88BA")
    node(ax, (12.00, 2.55), 1.95, 1.05, "多层指标", "差异 / 转段 / 重复\n整曲轨迹", "#F5F1E4", "#B59A4B")
    node(ax, (14.40, 2.55), 1.10, 1.05, "歌曲级\n统计", "N=100", "#F4ECEB", "#B97972")

    arrow(ax, (1.80, 3.20), (2.40, 4.60), BLUE)
    arrow(ax, (1.80, 2.93), (3.55, 1.05), ORANGE)
    arrow(ax, (4.25, 4.60), (4.85, 4.60), BLUE)
    arrow(ax, (6.70, 4.60), (7.30, 4.60), BLUE)
    arrow(ax, (5.60, 1.05), (6.25, 1.05), ORANGE)
    arrow(ax, (9.15, 4.60), (9.75, 3.32), BLUE)
    arrow(ax, (8.20, 1.05), (9.75, 2.82), ORANGE)
    arrow(ax, (11.55, 3.08), (12.00, 3.08), MUTED)
    arrow(ax, (13.95, 3.08), (14.40, 3.08), MUTED)

    ax.annotate(
        "VA推断不读取段落边界、类型或人工情绪标签",
        xy=(6.20, 2.02),
        xytext=(6.20, 2.23),
        ha="center",
        va="center",
        fontsize=7.0,
        color=RED,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#F9ECEA", edgecolor="#D6A5A0", linewidth=0.8),
    )
    ax.plot([6.20, 6.20], [1.61, 2.00], color=RED, linewidth=1.0, linestyle=(0, (3, 2)))
    ax.text(14.95, 2.18, "歌手簇=60", ha="center", va="top", fontsize=6.8, color=MUTED)

    return save_figure(fig, "Figure1_research_workflow")


def read_representative_data() -> dict[str, object]:
    manifest = pd.read_csv(MANIFEST_PATH)
    sections = pd.read_csv(SECTION_PATH)
    frames = pd.read_csv(FRAME_PATH)

    row = manifest.loc[manifest["song_id"] == REPRESENTATIVE_SONG_ID].iloc[0]
    final_sections = sections.loc[sections["song_id"] == REPRESENTATIVE_SONG_ID].sort_values("section_order").copy()
    frame_va = frames.loc[frames["song_id"] == REPRESENTATIVE_SONG_ID].sort_values("time_sec").copy()
    auto_path = ROOT / "outputs/functional_sections" / f"{REPRESENTATIVE_SONG_ID}_agentic_sections.csv"
    auto_sections = pd.read_csv(auto_path)
    transcript_path = ROOT / "outputs/functional_sections/周深-生活总该迎着光亮_whisper_segments.csv"
    transcript = pd.read_csv(transcript_path)
    audio_path = ROOT / str(row["audio_path"]).lstrip("/")
    vocals_path = ROOT / "outputs/vocal_separation/htdemucs/周深-生活总该迎着光亮/vocals.wav"

    for column in ["start_time", "end_time"]:
        auto_sections[f"{column}_sec"] = auto_sections[column].map(parse_time)
        transcript[f"{column}_sec"] = transcript[column].map(parse_time)

    return {
        "manifest_row": row,
        "final_sections": final_sections,
        "frame_va": frame_va,
        "auto_sections": auto_sections,
        "transcript": transcript,
        "audio_path": audio_path,
        "vocals_path": vocals_path,
        "auto_path": auto_path,
        "transcript_path": transcript_path,
    }


def scale01(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    low = float(np.nanmin(values))
    high = float(np.nanmax(values))
    if not math.isfinite(low) or not math.isfinite(high) or high <= low:
        return np.zeros_like(values)
    return (values - low) / (high - low)


def load_acoustic_curves(audio_path: Path, vocals_path: Path) -> dict[str, np.ndarray]:
    y, sr = librosa.load(audio_path, sr=22050, mono=True)
    hop = 512
    rms = librosa.feature.rms(y=y, hop_length=hop).reshape(-1)
    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop).reshape(-1)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop).reshape(-1)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop, n_mfcc=13)
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop)

    def column_delta(matrix: np.ndarray) -> np.ndarray:
        if matrix.shape[1] < 2:
            return np.zeros(len(rms))
        delta = np.linalg.norm(np.diff(matrix, axis=1), axis=0)
        return np.concatenate([[0.0], delta])[: len(rms)]

    novelty = np.mean(
        [
            scale01(np.abs(np.diff(rms, prepend=rms[0]))),
            scale01(onset),
            scale01(np.abs(np.diff(centroid, prepend=centroid[0]))),
            scale01(column_delta(chroma)),
            scale01(column_delta(mfcc)),
        ],
        axis=0,
    )
    novelty_smooth = pd.Series(novelty).rolling(43, center=True, min_periods=1).mean().to_numpy()

    vocals, vocals_sr = librosa.load(vocals_path, sr=22050, mono=True)
    vocal_rms = librosa.feature.rms(y=vocals, hop_length=hop).reshape(-1)
    vocal_times = librosa.frames_to_time(np.arange(len(vocal_rms)), sr=vocals_sr, hop_length=hop)
    lo, hi = np.quantile(vocal_rms, [0.02, 0.98])
    vocal_activity = np.clip((vocal_rms - lo) / max(hi - lo, 1e-12), 0.0, 1.0)
    vocal_activity = pd.Series(vocal_activity).rolling(43, center=True, min_periods=1).mean().to_numpy()

    return {
        "times": times,
        "novelty": novelty,
        "novelty_smooth": novelty_smooth,
        "vocal_times": vocal_times,
        "vocal_activity": vocal_activity,
    }


def style_time_axis(ax: plt.Axes, duration: float, xlabel: bool = False) -> None:
    ax.set_xlim(0, duration)
    ax.set_xticks(np.arange(0, duration + 1, 30))
    ax.grid(axis="x", color=GRID, linewidth=0.55, alpha=0.75)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, length=2.5)
    if xlabel:
        ax.set_xlabel("时间（s）", color=INK)
    else:
        ax.tick_params(labelbottom=False)


def add_section_band(
    ax: plt.Axes,
    sections: pd.DataFrame,
    start_col: str,
    end_col: str,
    type_col: str,
    y: float,
    height: float,
    label: str,
) -> None:
    ax.text(-3.8, y + height / 2, label, ha="right", va="center", color=INK, fontsize=8.1, clip_on=False)
    for _, section in sections.iterrows():
        start = float(section[start_col])
        end = float(section[end_col])
        section_type = str(section[type_col]).lower().replace("-", "_")
        ax.add_patch(
            Rectangle(
                (start, y),
                end - start,
                height,
                facecolor=SECTION_COLORS.get(section_type, "#D7DCE1"),
                edgecolor=PAPER,
                linewidth=0.8,
            )
        )
        if end - start >= 10.5:
            ax.text(
                (start + end) / 2,
                y + height / 2,
                SECTION_LABELS.get(section_type, section_type),
                ha="center",
                va="center",
                fontsize=6.8,
                color=INK,
                clip_on=True,
            )


def build_figure_2(data: dict[str, object], curves: dict[str, np.ndarray]) -> list[Path]:
    duration = float(data["manifest_row"]["duration_sec"])
    transcript = data["transcript"]
    auto_sections = data["auto_sections"]
    final_sections = data["final_sections"]

    fig = plt.figure(figsize=(18 / 2.54, 12.8 / 2.54))
    grid = fig.add_gridspec(4, 1, height_ratios=[0.75, 1.25, 1.25, 1.45], hspace=0.11)
    axes = [fig.add_subplot(grid[index, 0]) for index in range(4)]

    ax = axes[0]
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    for idx, row in transcript.iterrows():
        start, end = float(row.start_time_sec), float(row.end_time_sec)
        no_speech = float(row.get("no_speech_prob", 0.0))
        face = "#D6E6F2" if no_speech < 0.80 else "#E6E9EC"
        ax.add_patch(Rectangle((start, 0.26), end - start, 0.46, facecolor=face, edgecolor=PAPER, linewidth=0.6))
    ax.text(-3.8, 0.49, "歌词/转写句子", ha="right", va="center", color=INK, fontsize=8.1, clip_on=False)
    ax.annotate("首次主唱进入", xy=(15.0, 0.74), xytext=(24, 0.88), arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7), fontsize=7.2, color=MUTED)
    ax.annotate("主题句（hook）", xy=(57.2, 0.74), xytext=(68, 0.88), arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7), fontsize=7.2, color=MUTED)
    style_time_axis(ax, duration)

    ax = axes[1]
    ax.fill_between(curves["vocal_times"], curves["vocal_activity"], color="#73A7C7", alpha=0.45, linewidth=0)
    ax.plot(curves["vocal_times"], curves["vocal_activity"], color=BLUE, linewidth=0.75)
    ax.set_ylim(0, 1.04)
    ax.set_ylabel("人声强度\n（归一化）", color=INK, rotation=0, ha="right", va="center", labelpad=14)
    ax.set_yticks([0, 0.5, 1.0])
    style_time_axis(ax, duration)

    ax = axes[2]
    ax.plot(curves["times"], curves["novelty"], color="#A8B1B9", linewidth=0.35, alpha=0.55, label="原始")
    ax.plot(curves["times"], curves["novelty_smooth"], color=TEAL, linewidth=1.0, label="1 s平滑")
    ax.set_ylim(0, max(0.55, float(np.max(curves["novelty_smooth"])) * 1.16))
    ax.set_ylabel("综合\nnovelty", color=INK, rotation=0, ha="right", va="center", labelpad=14)
    ax.set_yticks([0, 0.25, 0.5])
    style_time_axis(ax, duration)
    ax.legend(loc="upper right", frameon=False, fontsize=7.2, ncol=2)

    ax = axes[3]
    ax.set_ylim(0, 2.0)
    ax.set_yticks([])
    add_section_band(ax, auto_sections, "start_time_sec", "end_time_sec", "section_type", 1.08, 0.62, "自动候选")
    add_section_band(ax, final_sections, "section_start_sec", "section_end_sec", "section_type", 0.18, 0.62, "人工最终")
    style_time_axis(ax, duration, xlabel=True)

    auto_boundaries = auto_sections["end_time_sec"].iloc[:-1].to_numpy(float)
    final_boundaries = final_sections["section_end_sec"].iloc[:-1].to_numpy(float)
    for boundary in auto_boundaries:
        for upper_ax in axes[:3]:
            upper_ax.axvline(boundary, color="#8C939A", linewidth=0.55, linestyle=(0, (3, 2)), alpha=0.40, zorder=0)
    for boundary in final_boundaries:
        for upper_ax in axes[:3]:
            upper_ax.axvline(boundary, color="#33383D", linewidth=0.45, alpha=0.35, zorder=0)

    ax.annotate(
        "类型修正\nBridge → Interlude",
        xy=(94.0, 0.82),
        xytext=(107, 1.88),
        ha="center",
        va="bottom",
        fontsize=7.2,
        color=RED,
        arrowprops=dict(arrowstyle="-|>", color=RED, lw=0.8, shrinkA=2, shrinkB=2),
    )
    ax.annotate(
        "边界提前11 s，并改为Outro\n229 s → 218 s",
        xy=(218.0, 0.50),
        xytext=(193, 1.88),
        ha="center",
        va="bottom",
        fontsize=7.2,
        color=RED,
        arrowprops=dict(arrowstyle="-|>", color=RED, lw=0.8, shrinkA=2, shrinkB=2),
    )
    ax.axvline(229.0, color=RED, linewidth=0.9, linestyle=(0, (3, 2)), alpha=0.75)
    ax.axvline(218.0, color=RED, linewidth=1.0, alpha=0.85)

    fig.suptitle("候选段落生成与人工质检示例：周深《生活总该迎着光亮》", x=0.51, y=0.995, fontsize=10.5, color=INK, weight="bold")
    fig.text(0.51, 0.964, "虚线为自动候选边界，实线为最终人工边界", ha="center", va="top", fontsize=7.6, color=MUTED)
    fig.subplots_adjust(left=0.12, right=0.985, bottom=0.085, top=0.92)
    return save_figure(fig, "Figure2_candidate_and_human_qc_example")


def build_figure_3(data: dict[str, object]) -> list[Path]:
    duration = float(data["manifest_row"]["duration_sec"])
    final_sections = data["final_sections"]
    frame_va = data["frame_va"]

    fig = plt.figure(figsize=(18 / 2.54, 10.5 / 2.54))
    grid = fig.add_gridspec(2, 1, height_ratios=[0.80, 3.5], hspace=0.09)
    ax_band = fig.add_subplot(grid[0, 0])
    ax_va = fig.add_subplot(grid[1, 0], sharex=ax_band)

    ax_band.set_ylim(0, 1)
    ax_band.set_yticks([])
    add_section_band(ax_band, final_sections, "section_start_sec", "section_end_sec", "section_type", 0.18, 0.64, "最终人工段落")
    ax_band.set_xlim(0, duration)
    ax_band.tick_params(labelbottom=False, bottom=False)
    ax_band.spines[:].set_visible(False)

    ax_va.plot(frame_va["time_sec"], frame_va["valence"], color=BLUE, linewidth=0.9, label="Valence（0.5 s）", zorder=2)
    ax_va.plot(frame_va["time_sec"], frame_va["arousal"], color=ORANGE, linewidth=0.9, label="Arousal（0.5 s）", zorder=2)

    for _, section in final_sections.iterrows():
        start = float(section.section_start_sec)
        end = float(section.section_end_sec)
        midpoint = (start + end) / 2
        mean_v = float(section.mean_valence)
        mean_a = float(section.mean_arousal)
        ax_va.hlines(mean_v, start, end, color=BLUE, linewidth=1.7, alpha=0.55, zorder=3)
        ax_va.hlines(mean_a, start, end, color=ORANGE, linewidth=1.7, alpha=0.55, zorder=3)
        ax_va.scatter(midpoint, mean_v, s=18, marker="o", facecolor=PAPER, edgecolor=BLUE, linewidth=0.9, zorder=4)
        ax_va.scatter(midpoint, mean_a, s=20, marker="s", facecolor=PAPER, edgecolor=ORANGE, linewidth=0.9, zorder=4)
        ax_va.axvline(end, color=GRID, linewidth=0.45, zorder=0)

    values = np.concatenate([frame_va["valence"].to_numpy(float), frame_va["arousal"].to_numpy(float)])
    lower = min(-0.45, float(np.min(values)) - 0.05)
    upper = max(0.65, float(np.max(values)) + 0.05)
    ax_va.set_ylim(lower, upper)
    ax_va.axhline(0, color="#8E969E", linewidth=0.65, linestyle=(0, (3, 2)), zorder=0)
    ax_va.set_ylabel("模型估计VA值", color=INK)
    style_time_axis(ax_va, duration, xlabel=True)
    ax_va.legend(loc="upper left", ncol=2, frameon=False, fontsize=7.8)
    ax_va.text(
        0.995,
        0.03,
        "细线：0.5 s帧级预测；粗横线与空心标记：段落均值",
        transform=ax_va.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.2,
        color=MUTED,
    )

    fig.suptitle("段落—VA后置对齐示例：周深《生活总该迎着光亮》", x=0.53, y=0.985, fontsize=10.5, color=INK, weight="bold")
    fig.subplots_adjust(left=0.12, right=0.985, bottom=0.11, top=0.91)
    return save_figure(fig, "Figure3_section_va_alignment_example")


def build_table_1() -> tuple[list[Path], Path]:
    with TEACHER_RESULTS_PATH.open(encoding="utf-8") as handle:
        results = json.load(handle)
    level = results["levels"][results["primary_level"]]

    table_rows = [
        ("Valence", "complete-clip", level["valence"]["rmse"], level["valence"]["mae"], level["valence"]["pcc"], level["valence"]["ccc"]),
        ("Arousal", "complete-clip", level["arousal"]["rmse"], level["arousal"]["mae"], level["arousal"]["pcc"], level["arousal"]["ccc"]),
    ]

    fig, ax = plt.subplots(figsize=(18 / 2.54, 4.6 / 2.54))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.0, 0.92, "表1  VA教师模型独立测试性能", ha="left", va="center", fontsize=10, color=INK, weight="bold")

    columns = ["维度", "测试层级", "RMSE", "MAE", "PCC", "CCC"]
    x = [0.08, 0.30, 0.52, 0.66, 0.80, 0.93]
    ax.hlines([0.80, 0.62, 0.25], 0.0, 1.0, colors=INK, linewidths=[1.05, 0.60, 1.05])
    for xpos, label in zip(x, columns):
        ax.text(xpos, 0.70, label, ha="center", va="center", fontsize=8.4, color=INK, weight="bold")
    for ypos, row in zip([0.51, 0.35], table_rows):
        values = [row[0], row[1]] + [f"{value:.4f}" for value in row[2:]]
        for xpos, value in zip(x, values):
            ax.text(xpos, ypos, value, ha="center", va="center", fontsize=8.4, color=INK)
    ax.text(
        0.0,
        0.08,
        "注：PMEmo独立锁定测试集，115首、5390个0.5 s标注帧；预测冻结后揭示测试标签。",
        ha="left",
        va="center",
        fontsize=7.2,
        color=MUTED,
    )
    outputs = save_figure(fig, "Table1_VA_teacher_independent_test_performance")

    tex_path = OUTPUT / "Table1_VA_teacher_independent_test_performance.tex"
    tex_path.write_text(
        "\\begin{table}[htbp]\n"
        "\\centering\n"
        "\\caption{VA教师模型独立测试性能}\n"
        "\\label{tab:teacher-performance}\n"
        "\\begin{tabular}{llrrrr}\n"
        "\\toprule\n"
        "维度 & 测试层级 & RMSE & MAE & PCC & CCC \\\\\n"
        "\\midrule\n"
        f"Valence & complete-clip & {table_rows[0][2]:.4f} & {table_rows[0][3]:.4f} & {table_rows[0][4]:.4f} & {table_rows[0][5]:.4f} \\\\\n"
        f"Arousal & complete-clip & {table_rows[1][2]:.4f} & {table_rows[1][3]:.4f} & {table_rows[1][4]:.4f} & {table_rows[1][5]:.4f} \\\\\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\begin{flushleft}\\footnotesize 注：PMEmo独立锁定测试集，115首、5390个0.5 s标注帧；预测冻结后揭示测试标签。\\end{flushleft}\n"
        "\\end{table}\n",
        encoding="utf-8",
    )
    return outputs, tex_path


def validate_sources(data: dict[str, object]) -> dict[str, object]:
    from importlib.metadata import version

    manifest = pd.read_csv(MANIFEST_PATH)
    sections = pd.read_csv(SECTION_PATH)
    frames = pd.read_csv(FRAME_PATH)
    alignment = pd.read_csv(ALIGNMENT_PATH)
    with TEACHER_RESULTS_PATH.open(encoding="utf-8") as handle:
        teacher = json.load(handle)

    assert len(manifest) == 100
    assert len(sections) == 1046
    assert len(frames) == 51448
    assert int((alignment["section_assignment_status"] == "assigned").sum()) == 51444
    assert int((alignment["section_assignment_status"] != "assigned").sum()) == 4
    assert sections.groupby("song_id").size().min() > 0
    assert len(data["final_sections"]) == 11
    assert len(data["auto_sections"]) == 11
    assert teacher["songs"] == 115
    assert teacher["frames"] == 5390
    assert teacher["primary_level"] == "complete_clip"

    final_boundaries = data["final_sections"]["section_end_sec"].iloc[:-1].to_numpy(float)
    auto_boundaries = data["auto_sections"]["end_time_sec"].iloc[:-1].to_numpy(float)
    assert np.any(np.isclose(final_boundaries, 218.0))
    assert np.any(np.isclose(auto_boundaries, 229.0))

    return {
        "generated_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
        "corpus": {"songs": 100, "sections": 1046, "frames": 51448, "assigned_frames": 51444, "excluded_trailing_frames": 4},
        "representative_song": {
            "song_id": REPRESENTATIVE_SONG_ID,
            "title": str(data["manifest_row"]["title"]),
            "artist": str(data["manifest_row"]["artist"]),
            "duration_sec": float(data["manifest_row"]["duration_sec"]),
            "automatic_sections": len(data["auto_sections"]),
            "final_sections": len(data["final_sections"]),
            "documented_adjustments": ["Bridge to Interlude", "final boundary 229.0 s to 218.0 s and Outro relabel"],
        },
        "teacher_test": {"songs": teacher["songs"], "frames": teacher["frames"], "primary_level": teacher["primary_level"], "metrics": teacher["levels"]["complete_clip"]},
        "method_provenance": {
            "qq_music_singer_chart_snapshot_date": "2026-08-08",
            "ranking_snapshot_evidence": {
                "quota_file_created_local": "2026-08-08T18:21:30+08:00",
                "first_git_commit": "5d5b1372a741aa0d8bb3e1e7f06d5ac004d96f44",
                "first_git_commit_time": "2026-08-08T19:54:07+08:00",
            },
            "whisper": {
                "backend": "faster-whisper",
                "package_version": version("faster-whisper"),
                "model": "small",
                "language": "zh",
                "device": "cpu",
                "compute_type": "int8",
                "beam_size": 5,
                "vad_filter": False,
            },
            "demucs": {
                "package_version": version("demucs"),
                "model": "htdemucs",
                "mode": "two-stems=vocals",
            },
        },
        "source_sha256": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in [MANIFEST_PATH, SECTION_PATH, FRAME_PATH, ALIGNMENT_PATH, TEACHER_RESULTS_PATH, data["auto_path"], data["transcript_path"]]
        },
    }


def write_readme() -> Path:
    path = OUTPUT / "README_论文插图说明.md"
    path.write_text(
        """# Method图表使用说明

本目录中的图表均直接由冻结的100首语料、1046个最终人工段落、51448个0.5 s VA预测帧及锁定教师模型测试结果生成。

## 文件格式

- PDF、SVG：优先用于论文排版和后续缩放。
- PNG、TIFF：600 dpi；TIFF使用LZW无损压缩。
- 表1另附LaTeX三线表源码。

## 建议图注

**图1 研究总体流程。** 段落结构通路与VA测量通路独立运行，并在VA预测完成后依据冻结的人工段落边界进行对齐。统计推断以歌曲为主要单位，并以歌手聚类bootstrap检验歌手内相关的稳健性。

**图2 候选段落生成与人工质检示例。** 以周深《生活总该迎着光亮》为例，依次显示Whisper句级时间戳、Demucs人声音轨强度、声学综合novelty曲线、自动候选段落和最终人工段落。虚线表示自动候选边界，实线表示最终人工边界。示例中包括Bridge至Interlude的类型修正，以及末段边界由229 s提前至218 s并改标为Outro。

**图3 帧级VA与人工段落的后置对齐示例。** 以周深《生活总该迎着光亮》为例，细线为0.5 s帧级Valence和Arousal预测，粗横线与空心标记为人工段落内的VA均值。段落信息不参与教师模型推断，仅在预测完成后用于统计汇总。

**表1 VA教师模型独立测试性能。** 主报告采用PMEmo独立锁定测试集的complete-clip层级。测试集含115首歌曲或片段、5390个0.5 s标注帧；预测冻结后方揭示测试标签。

## 排版建议

- 图1：双栏宽度，建议最终宽度170-180 mm。
- 图2、图3：双栏宽度，建议最终宽度170-180 mm。
- 表1：优先重排为正文三线表；若直接插图，使用PDF或SVG。

## Method最终可追溯信息

- QQ音乐歌手榜截取并冻结于2026年8月8日。依据为配额表本地创建时间（2026-08-08 18:21:30 +08:00）及其首次Git提交（`5d5b1372`，2026-08-08 19:54:07 +08:00）。
- 转写后端：`faster-whisper 1.2.1`，`small`模型，中文语言设置，CPU int8推断，beam size=5；主切分流程关闭VAD以避免歌唱句被抑制。
- 人声分离：`Demucs 4.0.1`，`htdemucs`模型，`--two-stems=vocals`模式。

建议正文写作："缺少可靠同步歌词时，采用faster-whisper 1.2.1的small模型生成句级转写及时间戳（中文、CPU int8、beam size=5，VAD关闭）。具有人声分离结果时，采用Demucs 4.0.1的htdemucs模型，以two-stem vocals模式提取人声音轨。"

`source_data_audit.json`记录了生成时使用的关键计数、代表歌曲调整和源文件SHA-256，便于复核。
""",
        encoding="utf-8",
    )
    return path


def main() -> None:
    configure_matplotlib()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    data = read_representative_data()
    audit = validate_sources(data)
    curves = load_acoustic_curves(data["audio_path"], data["vocals_path"])

    generated: list[Path] = []
    generated.extend(build_figure_1())
    generated.extend(build_figure_2(data, curves))
    generated.extend(build_figure_3(data))
    table_outputs, tex_path = build_table_1()
    generated.extend(table_outputs)
    generated.append(tex_path)
    generated.append(write_readme())

    audit_path = OUTPUT / "source_data_audit.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    generated.append(audit_path)

    print(f"Wrote {len(generated)} files to {OUTPUT}")
    for path in generated:
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
