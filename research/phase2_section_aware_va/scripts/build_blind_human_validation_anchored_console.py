#!/usr/bin/env python3
"""Build a fresh blind annotation console with frozen cross-song VA anchors."""

from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PHASE_ROOT = PROJECT_ROOT / "research/phase2_section_aware_va"
BASE_HTML = (
    PHASE_ROOT
    / "outputs/human_validation/blind_sample_15_human_validation_console.html"
)
ANCHORS_JSON = (
    PHASE_ROOT
    / "outputs/human_validation/calibration_candidates/frozen_va_anchors_v1.json"
)
OUTPUT_HTML = (
    PHASE_ROOT
    / "outputs/human_validation/blind_sample_15_human_validation_console_anchored_v1.html"
)


ANCHOR_CSS = """
    .anchor-guide {
      grid-column: 1 / -1;
      padding: 10px 12px;
      border: 1px solid #96b8d0;
      border-left: 4px solid var(--blue);
      border-radius: var(--radius);
      background: #f7fbfe;
    }
    .anchor-guide h2 { margin: 0; color: var(--navy); font-size: 14px; }
    .anchor-guide > p { margin: 3px 0 9px; color: var(--muted); font-size: 12px; }
    .anchor-grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 8px;
    }
    .anchor-group {
      min-width: 0;
      padding: 7px 8px;
      border: 1px solid #d4e0e8;
      border-radius: 4px;
      background: #fff;
    }
    .anchor-group strong { display: block; margin-bottom: 4px; color: var(--navy); }
    .anchor-item {
      display: grid;
      grid-template-columns: minmax(74px, 1fr) minmax(100px, 1.5fr);
      gap: 6px;
      align-items: center;
      margin-top: 5px;
      color: var(--muted);
      font-size: 11px;
    }
    .anchor-item span { overflow-wrap: anywhere; }
    .anchor-item audio { width: 100%; min-width: 0; height: 28px; }
    .anchor-rule {
      margin-top: 8px;
      padding-top: 7px;
      border-top: 1px solid #dce6ed;
      color: #435461;
      font-size: 12px;
    }
    .rating-buttons {
      display: grid;
      grid-template-columns: repeat(var(--choice-count), minmax(0, 1fr));
      width: 100%;
      min-width: 0;
      overflow: hidden;
      border: 1px solid #9eafbb;
      border-radius: 4px;
      background: #fff;
    }
    .rating-choice {
      min-width: 0;
      height: 34px;
      padding: 0 3px;
      border: 0;
      border-right: 1px solid #c7d3dc;
      color: #334a5c;
      background: #fff;
      font-size: 12px;
      font-weight: 700;
      white-space: nowrap;
    }
    .rating-choice:last-child { border-right: 0; }
    .rating-choice:hover:not(:disabled) { background: #e8f2f9; }
    .rating-choice.active {
      color: #fff;
      background: var(--blue);
    }
    .rating-choice:focus-visible {
      position: relative;
      z-index: 1;
      outline: 2px solid #17324d;
      outline-offset: -2px;
    }
    .rating-choice:disabled {
      color: #65727c;
      background: #e9edf0;
      cursor: not-allowed;
    }
    @media (max-width: 980px) {
      .anchor-grid { grid-template-columns: 1fr 1fr; }
    }
    @media (max-width: 650px) {
      .anchor-grid { grid-template-columns: 1fr; }
    }
"""


def anchor_panel(
    anchors: list[dict[str, object]], expected_per_group: int = 2
) -> str:
    dimension_labels = {"valence": "Valence", "arousal": "Arousal"}
    level_labels = {"low": "低", "mid": "中", "high": "高"}
    groups = []
    for dimension in ("valence", "arousal"):
        for level in ("low", "mid", "high"):
            selected = [
                row
                for row in anchors
                if row["dimension"] == dimension and row["final_level"] == level
            ]
            if len(selected) != expected_per_group:
                raise RuntimeError(
                    f"Expected {expected_per_group} anchors for {dimension}/{level}"
                )
            items = []
            for row in selected:
                clip_path = str(
                    row.get("page_clip_relative_path")
                    or ("calibration_candidates/" + str(row["clip_relative_path"]))
                )
                items.append(
                    '<div class="anchor-item">'
                    f'<span>{row["title"]} · {row["artist"]}</span>'
                    f'<audio controls preload="metadata" src="{clip_path}"></audio>'
                    "</div>"
                )
            groups.append(
                '<div class="anchor-group">'
                f'<strong>{dimension_labels[dimension]} {level}（{level_labels[level]}）</strong>'
                + "".join(items)
                + "</div>"
            )
    return """
      <section class="anchor-guide" aria-label="VA跨歌曲统一参照">
        <h2>VA 跨歌曲统一参照</h2>
        <p>正式标注前先横向试听同一维度的 low / mid / high。参照用于统一量尺，不要求待标片段与示例的曲风或音色相同。</p>
        <div class="anchor-grid">%s</div>
        <div class="anchor-rule"><strong>使用规则：</strong>Valence 只比较消极—积极；Arousal 只比较平静—激烈。low / mid / high 是跨所有歌曲的统一等级，不是只在单首歌内部排序；强度变化列仍只比较同一首歌的紧邻上一段。</div>
      </section>
    """ % "".join(groups)


def replace_once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise RuntimeError(f"Expected one marker, found {text.count(old)}: {old[:80]!r}")
    return text.replace(old, new)


def main() -> None:
    payload = json.loads(ANCHORS_JSON.read_text(encoding="utf-8"))
    anchors = payload["anchors"]
    html = BASE_HTML.read_text(encoding="utf-8")
    html = replace_once(html, "</style>", ANCHOR_CSS + "\n  </style>")
    html = replace_once(
        html,
        "<title>Section 情绪人工盲法标注台</title>",
        "<title>Section 情绪人工盲法重新标注台（统一参照版）</title>",
    )
    html = replace_once(
        html,
        "<h1>Section 情绪人工盲法标注台</h1>",
        "<h1>Section 情绪人工盲法重新标注台</h1>",
    )
    html = replace_once(
        html,
        "15 首人工创作中文流行歌曲 · 179 个冻结 section · 页面不包含模型预测",
        "15 首中文流行歌曲 · 179 个冻结 section · 统一VA参照版 · 页面不包含模型预测",
    )
    html = replace_once(
        html,
        '<div class="protocol-item wide"><strong>高潮</strong><span>该 section 是否具有明显情绪峰值；每首歌允许 0 个、1 个或多个</span></div>',
        '<div class="protocol-item wide"><strong>局部情绪峰值</strong><span>判断该 section 是否相对前后段形成明显情绪峰值；每首歌允许 0 个、1 个或多个。不是识别 Chorus，也不是要求所有 Chorus 都标为“是”。</span></div>',
    )
    html = replace_once(
        html,
        '<div class="protocol-alert"><strong>已标注过歌曲：</strong>请按照上面的明确定义重新复核“强度变化”列，Valence、Arousal 和高潮不会被清空。</div>',
        '<div class="protocol-alert"><strong>重新标注：</strong>统一参照版使用独立记录，不载入旧版答案。请按本页定义从头完成 179 个 section。</div>',
    )
    html = replace_once(
        html,
        '<th>音频</th><th>Valence</th><th>Arousal</th><th>强度变化</th><th>高潮</th><th>备注</th>',
        '<th>音频</th><th>Valence</th><th>Arousal</th><th>强度变化</th><th>局部峰值</th><th>备注</th>',
    )
    html = replace_once(
        html,
        '<td class="input-cell cell-climax" data-label="高潮">${ratingSelect("is_climax", id, [["", "请选择"], ["0", "0（否）"], ["1", "1（是）"]])}</td>',
        '<td class="input-cell cell-climax" data-label="局部峰值" title="是否相对前后段形成局部情绪峰值；不是识别 Chorus">${ratingSelect("is_climax", id, [["", "请选择"], ["0", "0（否）"], ["1", "1（是）"]])}</td>',
    )
    html = replace_once(
        html,
        '      <div class="song-nav">',
        anchor_panel(anchors) + '\n      <div class="song-nav">',
    )
    html = replace_once(
        html,
        'const STORAGE_KEY = "musicva_blind15_human_validation_v4";',
        'const STORAGE_KEY = "musicva_blind15_human_validation_anchored_v1";',
    )
    html = replace_once(
        html,
        '''    function ratingSelect(field, id, options, disabled = false) {
      const current = state.rows[id][field];
      return `<select class="rating-select" data-id="${id}" data-field="${field}"${disabled ? " disabled" : ""}>
        ${options.map(([value, label]) => option(value, label, current)).join("")}
      </select>`;
    }''',
        '''    function ratingSelect(field, id, options, disabled = false) {
      const current = state.rows[id][field];
      const choices = options.filter(([value]) => value);
      const shortLabels = {
        low: "低", mid: "中", high: "高",
        up: "增强", down: "减弱", stable: "稳定",
        not_applicable: "不适用", "0": "否", "1": "是",
      };
      return `<div class="rating-buttons" style="--choice-count:${choices.length}" role="group">
        ${choices.map(([value, label]) => `<button type="button" class="rating-choice${current === value ? " active" : ""}" data-id="${id}" data-field="${field}" data-value="${value}" title="${label}" aria-pressed="${current === value}"${disabled ? " disabled" : ""}>${shortLabels[value] || label}</button>`).join("")}
      </div>`;
    }''',
    )
    html = replace_once(
        html,
        '''      body.querySelectorAll("select[data-field]").forEach((select) => {
        select.addEventListener("change", () => {
          const { id, field } = select.dataset;
          const value = select.value;
          if (field !== "notes" && value && !VALID[field].has(value)) return;
          state.rows[id][field] = value;
          saveState();
          updateProgress();
        });
      });''',
        '''      body.querySelectorAll("button.rating-choice[data-field]").forEach((button) => {
        button.addEventListener("click", () => {
          const { id, field, value } = button.dataset;
          if (!VALID[field]?.has(value)) return;
          state.rows[id][field] = value;
          const group = button.closest(".rating-buttons");
          group.querySelectorAll(".rating-choice").forEach((choice) => {
            const active = choice.dataset.value === value;
            choice.classList.toggle("active", active);
            choice.setAttribute("aria-pressed", String(active));
          });
          saveState();
          updateProgress();
        });
      });''',
    )
    html = replace_once(
        html,
        'link.download = `blind_sample_15_human_annotations_${state.rater_id}.csv`;',
        'link.download = `blind_sample_15_human_annotations_anchored_v1_${state.rater_id}.csv`;',
    )
    html = replace_once(
        html,
        "CSV 已导出。提交前确认 15 首歌曲均已完成。",
        "统一参照版 CSV 已导出。提交前确认 15 首歌曲均已完成。",
    )
    OUTPUT_HTML.write_text(html, encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(OUTPUT_HTML),
                "anchor_version": payload["version"],
                "anchors": len(anchors),
                "storage_key": "musicva_blind15_human_validation_anchored_v1",
                "old_annotations_imported": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
