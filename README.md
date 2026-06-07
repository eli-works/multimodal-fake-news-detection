# 基于共享语义桥与双向跨模态注意力的多模态假新闻检测模型

**英文题目**: A Multi-modal Fake News Detection Model with Shared Semantic Bridge and Bidirectional Cross-modal Attention

本仓库用于展示并持续记录该多模态假新闻检测模型的设计思路、方法结构、实验结果与论文写作材料。

## 研究问题

假新闻在社交媒体中的传播往往伴随着图文联合表达。仅依赖单一文本模态的方法，容易忽略误导性配图、图文语义偏移与跨模态冲突信息；仅依赖图像模态的方法，又难以充分理解文本中的立场表达、断言强度与上下文语义。因此，多模态假新闻检测的关键在于同时建模文本内容、图像证据以及两者之间的一致性与互补性关系。

## 摘要

社交媒体中的假新闻通常同时包含文本与图像信息，单纯依赖文本或图像的检测方法在图文不一致、语义误导等场景下容易出现误判。针对这一问题，本文提出一种基于共享语义桥与双向跨模态注意力的多模态假新闻检测模型。模型在文本侧结合 BiLSTM 与 Transformer 构建双路径表示，在图像侧使用 EfficientNet-B0 提取视觉语义特征，并通过共享残差桥完成跨模态语义对齐，进一步利用双向跨模态注意力实现图文细粒度交互。实验在 Weibo、Gossip 和 CFND 三个数据集上进行，结果表明该方法在准确率与假新闻类别 F1 指标上整体优于多种典型基线模型。

## 模型亮点

- 双路径文本编码：结合 BiLSTM 与 Transformer，同时保留局部上下文与全局语义
- 共享语义桥：缓解文本特征与图像特征的分布差异
- 双向跨模态注意力：增强文本到图像、图像到文本的双向信息交互
- 端到端训练：统一优化文本编码、图像编码与融合分类模块

## 模型流程

```text
输入文本 T 与图像 I
  -> 文本分支: BiLSTM + Transformer
  -> 图像分支: EfficientNet-B0
  -> 共享语义桥: Shared Bridge
  -> 双向跨模态注意力: BiCrossAttn
  -> 分类头: MLP
  -> 输出真假标签
```

## 数据集介绍

本文在三个公开数据集上验证模型的有效性，覆盖中文与英文、平衡与不平衡、多场景新闻内容。

| 数据集 | 来源 | 规模 | 类别特点 | 简要说明 |
| --- | --- | ---: | --- | --- |
| Weibo | 新浪微博 | 9527 | 相对平衡 | 中文社交媒体谣言检测数据，适合验证图文联合判断能力 |
| Gossip | FakeNewsNet / GossipCop | 12840 | 明显不平衡 | 英文娱乐新闻场景，假新闻占比较高，文本线索通常更强 |
| CFND | Zhang 等 | 26664 | 多领域分布 | 中文多领域假新闻数据，适合评估模型的整体泛化能力 |

## 实验设置

- 任务形式：二分类，其中 Positive 表示假新闻，Negative 表示真新闻
- 评估指标：Accuracy、Precision、Recall、F1
- 文本长度设置：
  - Weibo: `max_len = 96`
  - Gossip: `max_len = 192`
  - CFND: `max_len = 42`
- 训练配置：
  - Optimizer: `AdamW`
  - Batch Size: `64`
  - Epoch: `25`
  - Learning Rate: `1e-4`
  - Weight Decay: `1e-3`
  - Dropout: `0.3`
  - Early Stopping: enabled
  - Model Selection Metric: validation `F1`

## 主要实验结果

| 数据集 | 模型 | Accuracy | 假新闻 F1 |
| --- | --- | ---: | ---: |
| CFND | Ours | 0.8489 | 0.8520 |
| CFND | Concat | 0.8403 | 0.8488 |
| Weibo | Ours | 0.8432 | 0.8300 |
| Weibo | Concat | 0.8252 | 0.8051 |
| Gossip | Ours | 0.8696 | 0.9225 |
| Gossip | EANN-noadv | 0.8678 | 0.9210 |

## 核心发现

- 在 `CFND` 和 `Weibo` 上，所提模型相较直接拼接基线有更明显提升，说明共享语义桥与双向跨模态交互在复杂图文场景下更有效
- 在 `Gossip` 上，提升幅度相对较小，表明当文本线索已经较强时，复杂融合模块的边际收益有限
- 单模态对比结果显示文本仍是主要信息来源，但图像模态在部分数据集上能带来稳定增益
- 消融实验表明双路径文本编码、共享语义桥和双向跨模态注意力三者共同决定模型性能，缺失任一模块都会造成性能退化

## 消融与分析概览

- `LSTM` 变体：仅保留 BiLSTM 文本路径，用于验证 Transformer 全局建模的贡献
- `TRM` 变体：仅保留 Transformer 文本路径，用于验证 BiLSTM 局部上下文建模的作用
- `-SB` 变体：去掉 Shared Bridge，用于观察语义对齐模块的重要性
- `-BCA` 变体：去掉 BiCrossAttn，用于分析细粒度跨模态交互的实际贡献
- 参数分析进一步考察了 EfficientNet 冻结层数与 Gossip 数据集 `max_len` 设置对结果的影响

## 文档导航

- 完整论文大纲：[docs/paper_outline.md](docs/paper_outline.md)
- 后续可扩展文档：
  - `docs/experiment_log.md`
  - `docs/figures.md`
  - `docs/tables.md`

## 当前定位

- 用于 GitHub 首页展示论文核心内容
- 用于持续沉淀模型设计、实验记录与写作素材
- 当前重点是模型设计与实验分析，不包含系统实现部分
