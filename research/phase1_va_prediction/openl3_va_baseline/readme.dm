# 基于 OpenL3 的音乐连续愉悦度–唤醒度回归实验工程

本工程实现论文《基于OpenL3嵌入的音乐情绪连续愉悦度—唤醒度回归建模研究》中的全部实验流程，
包括 DEAM 数据预处理、OpenL3 特征提取、MLP/BiLSTM/Transformer 三种回归模型训练及结果可视化。

## 1. 目录结构（关键文件夹）

- `MEMD_audio/`：DEAM 音频文件
- `annotations/`：DEAM 原始连续 VA 标注
- `labels_2hz/`：预处理后 2 Hz 采样的 VA 标签
- `openl3_2hz_features/`：OpenL3 提取的 512 维帧级嵌入
- `dataset_mlp/`：MLP 用帧级数据集
- `dataset_bilstm_Transformer/`：BiLSTM / Transformer 用序列数据集
- `models_mlp/`, `models_bilstm/`, `models_transformer/`：训练好的模型参数
- `results_tuned/`：训练过程日志（loss / RMSE / PCC）
- `plots/`：所有实验图表
- `VA_results/`：对单首歌曲预测得到的 VA 曲线

## 2. 脚本说明（.py 文件）

### 2.1 标注与特征

- `prepare_labels.py`：处理原始 DEAM 标注，输出 2 Hz 对齐标签到 `labels_2hz/`。
- `batch_extract_openl3.py`：对所有音频提取 OpenL3 嵌入，保存到 `openl3_2hz_features/`。
- `（DEAM数据分布直方图）.py`：基于 `labels_2hz/` 绘制 VA 分布直方图 / 散点图。

### 2.2 数据集构建

- `build_mlp_dataset.py`：构造帧级 MLP 训练集，保存到 `dataset_mlp/`。
- `build_bilstm_Transformer_dataset.py`：构造时序 BiLSTM/Transformer 训练集，保存到 `dataset_bilstm_Transformer/`。

### 2.3 模型训练

- `train_mlp.py`：训练 MLP 回归模型并保存到 `models_mlp/` 与 `results_tuned/mlp/`。
- `train_bilstm.py`：训练 BiLSTM 回归模型。
- `train_transformer.py`：训练 Transformer Encoder 回归模型。

### 2.4 可视化与结果分析

- `plot_mlp_history.py`：绘制 MLP 训练 / 验证 RMSE、PCC 曲线。
- `plot_bilstm_history.py`：绘制 BiLSTM 训练 / 验证曲线。
- `plot_transformer_curves.py`：绘制单首歌曲的真实与预测 VA 曲线。
- `plot_compare_models.py`：汇总并对比三种模型在验证 / 测试集上的性能。

### 2.5 情绪预测接口

- `predict_mlp_va_curve.py`：加载已训练的 MLP 模型，对任意输入音频预测连续 VA 曲线，结果保存到 `VA_results/`。

## 3. 运行流程（与论文章节对应）

1. **准备数据**
   - 放置 DEAM 音频到 `MEMD_audio/`，标注到 `annotations/`。

2. **标签与特征**（论文 Method: Problem Definition & OpenL3 Feature Extraction）
   - `python prepare_labels.py`
   - `python batch_extract_openl3.py`

3. **构建数据集**（论文 Method: Model Architecture 数据部分）
   - `python build_mlp_dataset.py`
   - `python build_bilstm_Transformer_dataset.py`

4. **训练模型**（论文 Experiments & Results）
   - `python train_mlp.py`
   - `python train_bilstm.py`
   - `python train_transformer.py`

5. **可视化与对比**（论文 Quantitative Results & Method Analysis）
   - `python （DEAM数据分布直方图）.py`
   - `python plot_mlp_history.py`
   - `python plot_bilstm_history.py`
   - `python plot_transformer_curves.py`
   - `python plot_compare_models.py`

6. **接口测试**（论文 Conclusion 中提到的“连续情绪预测接口”）
   - `python predict_mlp_va_curve.py --audio your_song.wav`
