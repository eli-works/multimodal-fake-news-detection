# 基于共享语义桥与双向跨模态注意力的多模态假新闻检测模型

这是一个面向 GitHub 用户整理的项目仓库，主要用于展示多模态假新闻检测模型的核心实现、实验脚本组织方式以及与论文内容对应的说明文档。

## 项目简介

本项目围绕多模态假新闻检测任务展开，核心模型结合了：

- `BiLSTM + Transformer` 双路径文本编码
- `EfficientNet-B0` 图像特征提取
- 共享语义桥 `SharedResidualBridge`
- 双向跨模态注意力 `BiDirectionalCrossAttentionBlock`

仓库重点保留模型主体实现与实验入口，便于阅读、整理和后续公开归档。

## 仓库适合谁

- 想快速查看该模型整体实现思路的读者
- 想定位主模型、基线模型、消融模型代码入口的使用者
- 想根据论文结构对照代码实现的同学
- 想基于现有实验脚本继续整理公开版项目的作者本人

## 当前公开内容

当前仓库主要包含以下内容：

- 主模型训练脚本
- 基线模型与对比模型训练脚本
- 消融实验训练脚本
- 模型设计与代码实现对照文档
- 论文大纲、图表说明、实验记录等文档
- 第三方参考实现与基线参考代码

当前仓库不以“完整复现实验环境”作为唯一目标，因此原始数据、完整指标归档、可视化产物和数据处理过程不会作为首页重点展示内容。

## 快速导航

- 论文大纲：[docs/paper_outline.md](docs/paper_outline.md)
- 模型实现对照：[docs/model_implementation.md](docs/model_implementation.md)
- 复现说明：[docs/reproducibility.md](docs/reproducibility.md)
- 实验记录：[docs/experiment_log.md](docs/experiment_log.md)
- 图表说明：[docs/figures.md](docs/figures.md)
- 表格说明：[docs/tables.md](docs/tables.md)
- 数据集说明：[docs/dataset_statement.md](docs/dataset_statement.md)
- 第三方代码说明：[docs/third_party_code.md](docs/third_party_code.md)
- 仓库整理检查清单：[docs/repository_checklist.md](docs/repository_checklist.md)

## 代码结构

```text
code/
|-- 01_我的实验代码_主实验+对比+消融/
|   |-- 对比实验代码/
|   |   |-- 主实验并入代码/
|   |   |-- train_eann_noadv_*.py
|   |   |-- train_mvae_*.py
|   |   `-- train_textonly_*.py
|   `-- 消融实验代码/
|       |-- final_train_*_lstm_only.py
|       |-- final_train_*_transformer_only.py
|       |-- final_train_*_wosb.py
|       `-- final_train_*_wobicross.py
|-- 03_我的可视化与分析代码/
`-- 04_第三方基线与参考实现/
```

## 代码分类说明

### 1. 主模型代码

主模型脚本位于：

```text
code/01_我的实验代码_主实验+对比+消融/对比实验代码/主实验并入代码/
```

核心入口包括：

- `final_train_CFND.py`
- `final_train_gossip.py`
- `final_train_weibo.py`

这些脚本中包含主模型的大部分核心模块，例如：

- `LSTMTransformerEncoder`
- `ImageEncoder`
- `SharedResidualBridge`
- `BiDirectionalCrossAttentionBlock`
- `LightweightBiCrossAttentionFusion`
- `MultiModalModel`

### 2. 基线模型与对比模型代码

当前仓库中的对比实验脚本主要包括：

- `train_eann_noadv_cfnd.py`
- `train_eann_noadv_gossip.py`
- `train_eann_noadv_weibo.py`
- `train_mvae_cfnd.py`
- `train_mvae_gossip.py`
- `train_mvae_weibo.py`
- `train_textonly_cfnd.py`
- `train_textonly_gossip.py`
- `train_textonly_weibo.py`
- `final_train_*_concat.py`
- `final_train_*_attrnn.py`
- `final_train_*_image_only.py`

如果你是第一次阅读本仓库，建议优先看主模型脚本，再看这些对比实现。

### 3. 消融模型代码

消融实验脚本位于：

```text
code/01_我的实验代码_主实验+对比+消融/消融实验代码/
```

主要变体包括：

- `*_lstm_only.py`
- `*_transformer_only.py`
- `*_wosb.py`
- `*_wobicross.py`

分别对应：

- 仅保留 LSTM 文本路径
- 仅保留 Transformer 文本路径
- 去掉共享语义桥
- 去掉双向跨模态注意力

## 数据集来源

本项目实验涉及以下数据集，具体获取方式请参考原始论文或对应发布页面：

| 数据集 | 来源论文 | 简要说明 |
| --- | --- | --- |
| Weibo | [Jin et al., ACM MM 2017](https://dl.acm.org/doi/10.1145/3123266.3123454) | 中文社交媒体谣言检测数据 |
| Gossip | [Shu et al., Big Data 2020](https://journals.sagepub.com/doi/10.1089/big.2020.0062) | FakeNewsNet 中的 GossipCop 子集 |
| CFND | [Zhang et al., IJCAI 2024](https://www.ijcai.org/proceedings/2024/281) | 中文多领域多模态假新闻检测数据 |

说明：

- 本仓库不直接重新分发原始数据集
- 数据集路径通常在各训练脚本中的 `CFG` 配置里指定
- 运行前需要根据你本地的数据存放位置修改相应路径

## 使用方式

如果你只想看主模型，建议按下面顺序阅读：

1. 查看 [docs/model_implementation.md](docs/model_implementation.md)
2. 打开 `final_train_CFND.py`
3. 再对照 `final_train_gossip.py` 和 `final_train_weibo.py` 看数据集差异
4. 最后再看基线模型与消融模型脚本

如果你想运行脚本，建议先确认：

- Python 与 PyTorch 环境已准备好
- 数据集路径已正确配置
- 训练脚本中的 `CFG` 参数符合你的机器环境

## 主模型代码入口

代表性入口文件：

```text
code/01_我的实验代码_主实验+对比+消融/对比实验代码/主实验并入代码/final_train_CFND.py
```

这个脚本中同时包含：

- 配置定义
- 数据集读取
- 文本与图像编码模块
- 融合模块
- 训练与评估流程

如果后续继续公开整理，建议把这些内容进一步拆成：

- `models/`
- `datasets/`
- `trainers/`
- `configs/`

## 当前仓库状态

目前这个仓库更偏向“研究整理版”而不是“发布版框架工程”，因此会保留一些实验阶段形成的目录命名与脚本形态。  
如果后续要进一步公开给更多 GitHub 用户使用，建议继续做以下整理：

- 统一脚本命名
- 抽离公共模型模块
- 去掉不必要的结果记录逻辑
- 分离数据处理与训练逻辑
- 增加依赖说明和最小可运行示例

## 说明

- 首页 README 主要服务 GitHub 访客，不再按论文摘要形式组织
- 更详细的论文内容、图表说明和实验记录统一放在 `docs/` 下
- 若后续继续整理公开版代码，建议优先围绕“主模型 / 基线模型 / 消融模型”三类目录做瘦身
