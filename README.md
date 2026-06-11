# 基于共享语义桥与双向跨模态注意力的多模态假新闻检测模型

**English title**: A Multi-modal Fake News Detection Model with Shared Semantic Bridge and Bidirectional Cross-modal Attention

本仓库用于公开展示并持续记录一个面向多模态假新闻检测的研究项目，重点覆盖模型设计、代码实现、实验结果、消融分析和论文附属材料。项目当前聚焦于文本-图像双模态检测，不包含完整线上系统实现。

## 项目定位

社交媒体假新闻通常以文本和图像共同传播。仅依赖文本容易忽略误导性配图、图文语义偏移和跨模态冲突；仅依赖图像又难以理解文本中的立场表达、断言强度和上下文语义。因此，本项目关注如何联合建模文本内容、图像证据以及二者之间的一致性与互补关系。

本文提出一种基于共享语义桥与双向跨模态注意力的多模态假新闻检测模型。模型在文本侧结合 BiLSTM 与 Transformer 构建序列表示，在图像侧使用 EfficientNet-B0 提取视觉语义特征，并通过共享残差桥完成跨模态语义对齐，进一步利用双向跨模态注意力实现图文细粒度交互。

## 核心贡献

- 双路径文本编码：结合 BiLSTM 的局部上下文建模能力和 Transformer 的全局语义建模能力。
- 共享语义桥：通过共享残差映射缓解文本特征与图像特征的语义空间差异。
- 双向跨模态注意力：同时建模文本到图像、图像到文本的交互关系，以增强图文一致性和冲突线索捕捉。
- 系统实验验证：在 Weibo、Gossip 和 CFND 三个数据集上进行主实验、单模态对比、消融实验和参数分析。

## 模型流程

```text
输入文本 T 与图像 I
  -> 文本分支: Embedding + BiLSTM + Transformer
  -> 图像分支: EfficientNet-B0
  -> 共享语义桥: Shared Residual Bridge
  -> 双向跨模态注意力: BiCrossAttn
  -> 分类头: MLP
  -> 输出真假标签
```

模型与代码的详细对应关系见 [docs/model_implementation.md](docs/model_implementation.md)。

## 数据集

| 数据集 | 来源论文 | 规模 | 类别特点 | 简要说明 |
| --- | --- | ---: | --- | --- |
| Weibo | [Jin et al., ACM MM 2017](https://dl.acm.org/doi/10.1145/3123266.3123454) | 9527 | 相对平衡 | 中文社交媒体谣言检测数据，适合验证图文联合判断能力 |
| Gossip | [Shu et al., Big Data 2020](https://journals.sagepub.com/doi/10.1089/big.2020.0062) | 12840 | 明显不平衡 | 英文娱乐新闻场景，假新闻占比较高，文本线索通常更强 |
| CFND | [Zhang et al., IJCAI 2024](https://www.ijcai.org/proceedings/2024/281) | 26664 | 多领域分布 | 中文多领域假新闻数据，适合评估模型泛化能力 |

由于数据集受原始发布方协议约束，仓库不直接包含完整原始数据。复现前请先按数据集发布方要求获取数据，并参考 [docs/reproducibility.md](docs/reproducibility.md) 放置到脚本配置指定位置。

## 主要结果

| 数据集 | 模型 | Accuracy | 假新闻 F1 |
| --- | --- | ---: | ---: |
| CFND | Ours | 0.8489 | 0.8520 |
| CFND | Concat | 0.8403 | 0.8488 |
| CFND | MVAE | 0.8211 | 0.8272 |
| CFND | att-RNN | 0.8174 | 0.8251 |
| CFND | EANN-noadv | 0.8114 | 0.8236 |
| Weibo | Ours | 0.8432 | 0.8300 |
| Weibo | Concat | 0.8252 | 0.8051 |
| Weibo | EANN-noadv | 0.7936 | 0.7749 |
| Weibo | att-RNN | 0.7756 | 0.7536 |
| Weibo | MVAE | 0.7720 | 0.7393 |
| Gossip | Ours | 0.8696 | 0.9225 |
| Gossip | EANN-noadv | 0.8678 | 0.9210 |
| Gossip | Concat | 0.8647 | 0.9182 |
| Gossip | att-RNN | 0.8558 | 0.9151 |
| Gossip | MVAE | 0.8516 | 0.9108 |

完整实验记录、单模态对比、消融实验和参数分析见 [docs/experiment_log.md](docs/experiment_log.md)。

## 快速复现

安装依赖：

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

进入主实验目录：

```bash
cd "code/01_我的实验代码_主实验+对比+消融/对比实验代码/主实验并入代码"
```

运行主实验：

```bash
python final_train_CFND.py
python final_train_gossip.py
python final_train_weibo.py
```

运行前请先检查对应脚本顶部的 `CFG.dataset_root`、`CFG.processed_dir`、`CFG.save_root` 等配置。更完整的环境、数据和复现说明见 [docs/reproducibility.md](docs/reproducibility.md)。

## 仓库结构

```text
.
├── README.md
├── requirements.txt
├── requirements-faknow.txt
├── docs/
│   ├── paper_outline.md
│   ├── reproducibility.md
│   ├── model_implementation.md
│   ├── experiment_log.md
│   ├── figures.md
│   ├── tables.md
│   └── repository_checklist.md
└── code/
    ├── 01_我的实验代码_主实验+对比+消融/
    ├── 03_我的可视化与分析代码/
    └── 04_第三方基线与参考实现/
```

## 文档导航

- 论文大纲：[docs/paper_outline.md](docs/paper_outline.md)
- 复现指南：[docs/reproducibility.md](docs/reproducibility.md)
- 模型实现对照：[docs/model_implementation.md](docs/model_implementation.md)
- 实验记录：[docs/experiment_log.md](docs/experiment_log.md)
- 图像材料：[docs/figures.md](docs/figures.md)
- 表格材料：[docs/tables.md](docs/tables.md)
- 公开仓库完善清单：[docs/repository_checklist.md](docs/repository_checklist.md)

## 当前结论

实验表明，所提模型在 CFND 和 Weibo 上相较直接拼接和传统多模态基线有更明显提升，说明共享语义桥与双向跨模态交互在复杂图文场景下更有效。在 Gossip 上，文本线索本身较强，复杂融合模块的边际收益相对有限。消融实验进一步表明，双路径文本编码、共享语义桥和双向跨模态注意力共同支撑模型性能。

## 公开说明

- 本仓库主要作为研究过程记录、论文附属材料和复现实验入口。
- 数据集、第三方基线代码和预训练模型可能受各自许可约束，使用前请遵守原始来源协议。
- 训练产生的权重、日志和大体积中间文件默认不提交到 GitHub；建议通过 release 或外部链接单独归档。
- 本地压缩包、Word 初稿、临时摘录和私人整理材料不建议提交到公开仓库。
