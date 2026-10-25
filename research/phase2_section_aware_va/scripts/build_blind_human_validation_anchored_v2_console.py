#!/usr/bin/env python3
"""Build a fresh annotation console using five frozen references per VA level."""

from __future__ import annotations

import json
from pathlib import Path

from build_blind_human_validation_anchored_console import (
    ANCHOR_CSS,
    PHASE_ROOT,
    anchor_panel,
    replace_once,
)


BASE_HTML = (
    PHASE_ROOT
    / "outputs/human_validation/blind_sample_15_human_validation_console.html"
)
ANCHORS_JSON = (
    PHASE_ROOT
    / "outputs/human_validation/calibration_candidates_v2_supplement/"
    "frozen_va_anchors_v2_30clips.json"
)
OUTPUT_HTML = (
    PHASE_ROOT
    / "outputs/human_validation/"
    "blind_sample_15_human_validation_console_anchored_v2_30refs.html"
)


def main() -> None:
    payload = json.loads(ANCHORS_JSON.read_text(encoding="utf-8"))
    anchors = payload["anchors"]
    if len(anchors) != 30:
        raise RuntimeError(f"Expected 30 anchors, found {len(anchors)}")
    html = BASE_HTML.read_text(encoding="utf-8")
    html = replace_once(html, "</style>", ANCHOR_CSS + "\n  </style>")
    html = replace_once(
        html,
        "<title>Section 情绪人工盲法标注台</title>",
        "<title>Section 情绪人工盲法标注台（30参照版）</title>",
    )
    html = replace_once(
        html,
        "<h1>Section 情绪人工盲法标注台</h1>",
        "<h1>Section 情绪人工盲法标注台（30参照版）</h1>",
    )
    html = replace_once(
        html,
        "15 首人工创作中文流行歌曲 · 179 个冻结 section · 页面不包含模型预测",
        "15 首中文流行歌曲 · 179 个冻结 section · 每个VA等级5个参照 · 页面不包含模型预测",
    )
    html = replace_once(
        html,
        '<div class="protocol-item wide"><strong>高潮</strong><span>该 section 是否具有明显情绪峰值；每首歌允许 0 个、1 个或多个</span></div>',
        '<div class="protocol-item wide"><strong>局部情绪峰值</strong><span>判断该 section 是否相对前后段形成明显情绪峰值；每首歌允许 0 个、1 个或多个。不是识别 Chorus，也不是要求所有 Chorus 都标为“是”。</span></div>',
    )
    html = replace_once(
        html,
        '<div class="protocol-alert"><strong>已标注过歌曲：</strong>请按照上面的明确定义重新复核“强度变化”列，Valence、Arousal 和高潮不会被清空。</div>',
        '<div class="protocol-alert"><strong>独立30参照版本：</strong>本页使用全新记录，不载入旧版答案。只有重新完成179个section后，结果才能称为“每等级5个参照”的实验。</div>',
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
        anchor_panel(anchors, expected_per_group=5) + '\n      <div class="song-nav">',
    )
    html = replace_once(
        html,
        'const STORAGE_KEY = "musicva_blind15_human_validation_v4";',
        'const STORAGE_KEY = "musicva_blind15_human_validation_anchored_v2_30refs";',
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
        'link.download = `blind_sample_15_human_annotations_anchored_v2_30refs_${state.rater_id}.csv`;',
    )
    html = replace_once(
        html,
        "CSV 已导出。提交前确认 15 首歌曲均已完成。",
        "30参照版 CSV 已导出。提交前确认 15 首歌曲均已完成。",
    )
    OUTPUT_HTML.write_text(html, encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(OUTPUT_HTML),
                "anchor_version": payload["version"],
                "anchors": len(anchors),
                "anchors_per_dimension_level": payload[
                    "anchors_per_dimension_level"
                ],
                "storage_key": (
                    "musicva_blind15_human_validation_anchored_v2_30refs"
                ),
                "old_annotations_imported": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
