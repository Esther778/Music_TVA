# Section 人工验证延迟复核结果

报告时间：2026-07-24（Asia/Shanghai）
当前进度：15首歌曲、179个section的统一参照人工标注已完成延迟复核；VA Teacher始终冻结。

## 1. 为什么进行延迟复核

标注者在查看模型指标之前提出：连续长时间试听会产生听觉疲劳，间隔一段时间后重新判断更准确。因此，本项目在重新计算模型一致性之前固定以下规则：

- 延迟复核版作为正式主结果；
- 初次统一参照版完整保留，作为同一标注者稳定性分析；
- 不根据哪一版与模型更一致来选择结果；
- 不使用复核结果训练或调整Teacher。

## 2. 数据审计

延迟复核文件：

```text
data/blind_validation/stage5/human_annotations/
blind_sample_15_human_annotations_anchored_v1_delayed_review_rater_01.csv
```

SHA-256：

```text
25b1cf7d53e78bf0bbd2fc5550b870e65d99aa643e74dc144e8fbf4507377f43
```

- 179行、179个唯一人工标注ID、15首歌曲；
- 必填项缺失为0；
- 与冻结section起止边界的最大误差均为0秒；
- 审计状态：PASS。

## 3. 同一标注者的延迟复核稳定性

两次统一参照标注共有24个section至少修改了一个字段。

| 字段 | 修改数 | 完全一致率 | Cohen Kappa |
|---|---:|---:|---:|
| Valence | 11/179 | 93.9% | 0.930（quadratic weighted） |
| Arousal | 18/179 | 89.9% | 0.882（quadratic weighted） |
| 相邻强度变化 | 1/179 | 99.4% | 0.992 |
| 局部情绪峰值 | 0/179 | 100.0% | 1.000 |

该结果说明同一标注者在间隔后仍保持很高的一致性。它属于 test-retest / intra-rater stability，不能替代两名独立标注者的 inter-rater reliability。

## 4. 延迟复核版主结果

| 评价层级 | Valence | Arousal |
|---|---:|---:|
| 单曲内Spearman宏平均 | 0.385 | 0.519 |
| 179个section pooled Spearman | 0.570 | 0.556 |
| 去除歌曲均值后的pooled Spearman | 0.416 | 0.548 |
| 15首歌曲均值间Spearman | 0.640 | 0.532 |

单曲内宏平均bootstrap 95% CI：

- Valence：`[0.265, 0.503]`；
- Arousal：`[0.414, 0.609]`。

相邻段落变化：

- 人工变化等级与模型Arousal delta Spearman：`0.321`；
- 非stable方向命中：`88/129=68.2%`。

局部峰值：

- 人工峰值仍为57个，复核前后完全一致；
- 冻结模型最高平均Arousal section精确命中`8/15=53.3%`；
- 允许相邻section后命中`13/15=86.7%`。

## 5. Chorus相对Verse

| 指标 | Valence | Arousal |
|---|---:|---:|
| 人工平均提升（0/1/2序数单位） | +0.678 | +0.914 |
| 提升为正的歌曲 | 12/15 | 14/15 |
| Wilcoxon p-value | 0.00132 | 0.000948 |
| 人工提升与模型提升Spearman | 0.182 | 0.592 |

人工结果稳定支持Chorus相对Verse的Valence和Arousal提升。模型对Arousal Chorus-lift的歌曲间排序达到中等偏明显的一致性；对Valence Chorus-lift的排序仍弱。

## 6. 与初次统一参照版的差异

| 指标 | 初次统一参照 | 延迟复核 | 变化 |
|---|---:|---:|---:|
| pooled Valence | 0.584 | 0.570 | -0.013 |
| pooled Arousal | 0.533 | 0.556 | +0.022 |
| within-song centered Valence | 0.438 | 0.416 | -0.022 |
| within-song centered Arousal | 0.519 | 0.548 | +0.030 |
| between-song Valence | 0.697 | 0.640 | -0.057 |
| between-song Arousal | 0.577 | 0.532 | -0.045 |
| 非stable方向准确率 | 67.4% | 68.2% | +0.8个百分点 |

这些变化有升有降且幅度有限，没有出现复核后所有指标都变好的异常情况。两版共同支持“中等VA一致性、Arousal结构规律强于Valence精细结构”的结论。

## 7. 通俗结论

1. 同一标注者隔一段时间重听后，约90%以上的VA等级没有改变，峰值判断完全没有改变，说明人工标注总体稳定。
2. 模型对section情绪排序的相关约为`0.52–0.57`，能够提供有意义但不完美的结构信号。
3. 模型判断相邻段落增强或减弱约三次正确两次。
4. 模型更擅长找到峰值附近，而不是精确定位唯一峰值。
5. 最稳定的论文发现是Chorus相对Verse的Arousal提升；Valence的精细提升预测仍较弱。

## 8. 论文使用规则

主文使用延迟复核版结果，并在方法中说明：

> 为降低连续试听疲劳，同一标注者在间隔后依据同一冻结参照完成延迟复核。该复核原则在重新计算模型一致性之前确定。两轮标注表现出较高的intra-rater stability，延迟复核版用于主分析，初次统一参照版作为敏感性分析。

仍必须注明：

- 只有一名标注者；
- Kappa是同一标注者重测稳定性，不是人际一致性；
- 人工标签是序数等级，不是连续VA ground truth；
- 当前section mean仍包含边界内静音帧；
- Teacher没有因人工结果而调参。

## 9. 可复算文件

```text
scripts/analyze_blind_human_validation.py
scripts/compare_anchored_delayed_review.py
outputs/human_validation/rater_01_anchored_v1_delayed_review_evaluation/summary.json
outputs/human_validation/rater_01_anchored_v1_delayed_review_evaluation/per_song_metrics.csv
outputs/human_validation/rater_01_anchored_v1_delayed_review_evaluation/initial_vs_delayed_review.json
```
