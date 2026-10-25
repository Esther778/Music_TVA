# 统一参照版 Section 人工验证结果

> 状态更新（2026-07-24）：标注者在间隔后完成延迟复核。延迟复核规则在重新计算模型一致性之前固定，最新主结果见
> `docs/BLIND_VALIDATION_HUMAN_RESULTS_DELAYED_REVIEW_20260724.md`。本报告保留为初次统一参照结果和敏感性对照。

报告时间：2026-07-23（Asia/Shanghai）
当前进度：第一阶段 VA Teacher 已冻结；第二阶段15首歌曲、179个 section 的统一参照人工重标与首轮模型对照分析已完成。

## 1. 实验定位

本实验检验冻结 VA Teacher 生成的 section-level pseudo-VA 是否与人工听感在以下方面一致：

- section Valence / Arousal 的序数排序；
- 相邻 section 的 Arousal 变化方向；
- 模型最高平均 Arousal 候选与人工局部情绪峰值的位置关系；
- Chorus 相对 Verse 的情绪提升。

人工标签为 `low / mid / high` 序数等级，不是连续 VA 真值，因此本报告使用 Spearman、方向一致率和命中率，不计算 RMSE、MAE 或 CCC。

## 2. 数据与冻结条件

- 歌曲：15首人工创作中文流行歌曲。
- Section：179个冻结人工边界。
- 人工标注者：1人（`rater_01`）。
- VA参照：正式15首之外的12个片段，每个 Valence/Arousal × low/mid/high 组合2个。
- Teacher输出：标注前已冻结，人工重标后未训练、未调参。
- 模型 section 值：边界内全部0.5秒 pseudo-VA 帧的算术平均。
- 当前均值包含边界内静音帧；去除技术性首尾静音的结果尚未作为独立敏感性分析计算。

正式人工文件：

```text
data/blind_validation/stage5/human_annotations/
blind_sample_15_human_annotations_anchored_v1_rater_01.csv
```

SHA-256：

```text
4ef639e1ae3170c9263653fbc3f19c972ad9f736a3fa6605a29e5dbcb6715e02
```

## 3. 完整性审计

| 项目 | 结果 |
|---|---:|
| 行数 | 179 |
| 唯一人工标注ID | 179 |
| 歌曲数 | 15 |
| 缺失必填项 | 0 |
| 最大起点误差 | 0.0秒 |
| 最大终点误差 | 0.0秒 |
| 审计状态 | PASS |

人工等级分布：

| 维度 | low | mid | high |
|---|---:|---:|---:|
| Valence | 64 | 92 | 23 |
| Arousal | 77 | 89 | 13 |

## 4. 正式结果

### 4.1 人工标签与 pseudo-VA 的相关

| 评价层级 | Valence Spearman | Arousal Spearman |
|---|---:|---:|
| 单曲内相关宏平均 | 0.398 | 0.504 |
| 单曲内相关中位数 | 0.507 | 0.514 |
| 179个section合并 | 0.584 | 0.533 |
| 去除歌曲均值后的合并相关 | 0.438 | 0.519 |
| 15首歌曲均值之间 | 0.697 | 0.577 |

单曲内宏平均的bootstrap 95% CI：

- Valence：`[0.223, 0.558]`；
- Arousal：`[0.396, 0.601]`。

解释：模型与人工听感存在中等正相关。Valence 的跨歌曲均值相关最高（0.697），说明统一参照后不同歌曲之间的 Valence 标尺更一致；但单曲内部动态排序仍只有中等水平。

三个歌曲的人工 Valence 全部为同一等级，单曲内 Valence Spearman 无法定义。可计算歌曲中，`如愿`最高（0.822），`春娇与志明`最低（-0.274）。Arousal 同样以`如愿`最高（0.822），`关山酒`最低（-0.005），表明模型表现存在明显歌曲差异。

### 4.2 相邻段落变化

| 指标 | 结果 |
|---|---:|
| 人工 up/down/stable 与模型 Arousal delta 的 Spearman | 0.312 |
| 非 stable 方向命中 | 87 / 129 |
| 非 stable 方向准确率 | 67.4% |

该结果高于随机二分类的50%参照，但只属于有限到中等一致性，不能声称模型准确恢复了完整的人类情绪变化。

### 4.3 局部情绪峰值

- 人工局部峰值：57个。
- 其中 Chorus：54个（占人工峰值94.7%）。
- 60个 Chorus 中有54个被标为峰值（90.0%）。
- 其余峰值：2个 Pre-chorus、1个 Interlude。
- 冻结模型最高平均 Arousal section：精确命中8/15（53.3%）；允许相邻 section 后命中13/15（86.7%）。

该结果支持“流行歌曲局部情绪峰值高度集中于 Chorus”，但不能把人工峰值当作独立的 Chorus 真值，也不能把相邻命中等同于精确定位。

### 4.4 Chorus 相对 Verse 的提升

| 指标 | Valence | Arousal |
|---|---:|---:|
| 人工平均提升（0/1/2序数单位） | +0.578 | +0.759 |
| 提升为正的歌曲 | 10/15 | 13/15 |
| Wilcoxon p-value | 0.00358 | 0.00130 |
| 人工提升与模型提升 Spearman | 0.117 | 0.456 |

人工结果显示 Chorus 在 Valence 和 Arousal 上均显著高于 Verse。模型对 Arousal Chorus-lift 的歌曲间排序具有中等一致性，但对 Valence Chorus-lift 的排序很弱。

## 5. 与无参照 Pilot 的协议敏感性比较

同一标注者完成了两轮，因此该比较只用于判断标注协议敏感性，不是独立重复实验。

| 指标 | 无参照 Pilot | 统一参照版 | 变化 |
|---|---:|---:|---:|
| section pooled Valence | 0.419 | 0.584 | +0.165 |
| section pooled Arousal | 0.589 | 0.533 | -0.056 |
| within-song centered Valence | 0.438 | 0.438 | +0.000 |
| within-song centered Arousal | 0.552 | 0.519 | -0.033 |
| between-song Valence | 0.400 | 0.697 | +0.297 |
| between-song Arousal | 0.732 | 0.577 | -0.155 |
| 非stable变化方向准确率 | 71.8% | 67.4% | -4.4个百分点 |

统一参照主要改善了跨歌曲 Valence 校准，而不是单曲内部排序。在12首两轮均可计算 Valence 相关的歌曲上，单曲内相关的平均配对变化为`-0.023`，95% bootstrap CI为`[-0.190, 0.121]`，Wilcoxon `p=0.929`，没有改善证据。

Arousal 的 high 使用从40段减少到13段，说明参照建立了更严格的高唤醒阈值，也压缩了等级方差。Arousal相关下降不能被隐藏，也不能仅因 Pilot 数值更高就回退使用无参照结果；统一参照版具有更清楚的测量定义，应作为正式主结果。

## 6. 可用于论文的结论

建议主张：

> 在15首中文流行歌曲、179个冻结section的一名标注者统一参照验证中，冻结pseudo-VA与人工section等级呈中等正相关。模型对跨歌曲Valence均值和Arousal段内排序具有一定有效性；人工感知的Chorus相对Verse提升显著，模型能够中等程度反映Arousal Chorus-lift，但对Valence Chorus-lift和精确峰值定位仍有限。

不应主张：

- pseudo-VA等同于真实连续人类VA；
- 模型准确识别全部高潮；
- 统一参照使所有指标都得到提升；
- 单一标注者足以建立普遍人类感知结论；
- 相邻峰值命中等同于精确命中。

## 7. 局限与下一步

1. 当前只有一名标注者，无法计算人际一致性；论文必须明确写为 exploratory one-rater ordinal validation。
2. 同一标注者完成两轮，Pilot与统一参照版存在记忆和学习效应。
3. 原始无参照 Pilot CSV 被同名文件覆盖；旧版汇总和逐歌结果仍保留，但原始CSV应从旧页面浏览器存储重新导出归档。
4. 当前 section mean 包含边界内技术性静音。应预注册统一的首尾静音识别规则，并同时报告 full-boundary 与 active-audio sensitivity 结果，不能按效果择优。
5. 对表现异常歌曲（如`春娇与志明`、`关山酒`）应做定性误差分析，但不得据此修改冻结 Teacher 或15首测试标签。

## 8. 可复算文件

```text
scripts/analyze_blind_human_validation.py
scripts/compare_human_validation_rounds.py
outputs/human_validation/rater_01_anchored_v1_evaluation/summary.json
outputs/human_validation/rater_01_anchored_v1_evaluation/per_song_metrics.csv
outputs/human_validation/rater_01_anchored_v1_evaluation/pilot_vs_anchored_comparison.json
```
