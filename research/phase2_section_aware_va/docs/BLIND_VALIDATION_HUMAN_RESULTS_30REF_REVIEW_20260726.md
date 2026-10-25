# 30参考片段辅助下的Section情绪人工复核结果

报告时间：2026-07-26（Asia/Shanghai）
当前科研进度：第二阶段15首完整歌曲、179个冻结section的单标注者人工验证；VA Teacher保持冻结。

## 1. 本轮实验性质

标注者在未查看模型预测的条件下，休息后重新听取测试片段，并结合30个已冻结的跨歌曲VA参考片段（Valence/Arousal各等级5个）修改人工等级。修改发生在重新计算模型一致性指标之前。

本轮可称为“同一标注者经30参考片段校准后的再次复核”，不能称为第二位独立标注者。2026-07-24的12参考延迟复核结果继续保留为敏感性分析，不能因本轮指标更高而删除。

## 2. 数据审计

- 15首歌曲、179个section、179个唯一`human_annotation_id`。
- Valence、Arousal、段间变化和局部峰值均无缺失、无非法值。
- 歌曲身份、section顺序及起止边界与冻结测试数据完全一致；最大边界误差为0秒。
- 当前标注文件SHA-256：`26f7a20a2a11ff044e7e6188ec914c388a1da01d9baf117ce15ac1299b54eb09`。
- Teacher预测、section边界和section类型均未因本次复核而修改。

## 3. 同标注者复核稳定性

与2026-07-24延迟复核版相比，共38个section至少修改一个字段。加权Kappa使用和前次相同的scikit-learn计算方法。

| 字段 | 修改行数 | 完全一致率 | Kappa |
|---|---:|---:|---:|
| Valence等级 | 23/179 | 87.15% | 0.858（quadratic） |
| Arousal等级 | 20/179 | 88.83% | 0.877（quadratic） |
| 相邻段强度变化 | 0/179 | 100.00% | 1.000 |
| 局部情绪峰值 | 1/179 | 99.44% | 0.987 |

变化集中在跨歌曲绝对VA等级，而相邻段变化方向完全未变、局部峰值仅改1处。这符合新增参考片段的预期作用：主要统一`low/mid/high`的跨歌曲尺度，而不是重写歌曲内部结构判断。

## 4. Teacher与人工等级一致性

Spearman越接近1，表示Teacher预测的情绪排序与人工等级排序越一致。

| 指标 | 12参考延迟复核 | 30参考再次复核 | 变化 |
|---|---:|---:|---:|
| Section pooled Valence Spearman | 0.570 | **0.614** | +0.044 |
| Section pooled Arousal Spearman | 0.556 | **0.576** | +0.020 |
| Within-song macro Valence Spearman | 0.385 | **0.437** | +0.052 |
| Within-song macro Arousal Spearman | 0.519 | **0.525** | +0.005 |
| Within-song centered Valence Spearman | 0.416 | **0.509** | +0.093 |
| Within-song centered Arousal Spearman | 0.548 | **0.562** | +0.013 |
| Between-song mean Valence Spearman | 0.640 | **0.742** | +0.103 |
| Between-song mean Arousal Spearman | 0.532 | **0.614** | +0.082 |
| 非stable变化方向准确率 | 68.2%（88/129） | **68.2%（88/129）** | 0 |

本轮Valence改善最明显，尤其是歌曲间平均Valence和歌曲内部中心化后的段落排序。Arousal有小幅改善，但段内宏平均基本稳定。Within-song macro的95% bootstrap区间为：Valence `[0.274, 0.588]`，Arousal `[0.433, 0.614]`。

这不能解释为“模型经过本轮变准”：Teacher从头到尾被冻结，指标变化来自人工等级校准。因为没有将标注者随机分配到12参考组和30参考组，也不能将全部提升因果归因于参考片段数量。

## 5. 情绪推进与Section结构

- 相邻段非stable变化方向：`88/129 = 68.2%`，与上一版完全相同。
- Teacher最高平均Arousal段命中人工局部峰值：精确`8/15`，相邻命中`12/15`；上一版为`8/15`和`13/15`。
- 人工局部峰值共56个：Chorus 53个、Pre-chorus 2个、Interlude 1个。
- Chorus相对Verse的人工Valence提升均值为`+0.667`，15首中13首为正，Wilcoxon `p=0.00121`。
- Chorus相对Verse的人工Arousal提升均值为`+0.802`，15首中13首为正，Wilcoxon `p=0.00130`。
- 人工/模型Chorus提升的歌曲间Spearman：Valence `0.228`，Arousal `0.586`。

结论仍支持：副歌通常承担更高的情绪强度和更积极的情绪。Arousal的结构提升量与Teacher具有中等排序一致性；Valence的Chorus提升一致性较弱，不应过度解释。

## 6. 论文使用原则

1. 将本轮称为`30-reference calibrated same-rater review`，不要写成独立双人标注。
2. 主表可以采用本轮结果，因为复核原因和操作发生在重新计算指标之前；同时必须保留7月24日结果作为敏感性分析。
3. 明确说明参考片段未使用Teacher预测、人工标注页面隐藏模型结果、Teacher没有重新训练。
4. 不报告人工验证RMSE、MAE或CCC，因为人工标签是三级序数，不是连续VA ground truth。
5. 当前最重要的可信度补强仍是增加第二位独立标注者，并计算真正的inter-rater weighted Kappa。

## 7. 可复算文件

```text
scripts/analyze_blind_human_validation.py
scripts/compare_delayed_and_30ref_review.py
outputs/human_validation/rater_01_anchored_v2_30refs_review_evaluation/summary.json
outputs/human_validation/rater_01_anchored_v2_30refs_review_evaluation/comparison_vs_delayed_v1.json
outputs/human_validation/rater_01_anchored_v2_30refs_review_evaluation/per_song_metrics.csv
```
