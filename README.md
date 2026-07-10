# multimodal-fake-news-detection

> A research repository for multimodal fake news detection with a shared semantic bridge and bidirectional cross-modal attention.

## Overview

This repository documents a complete research project on multimodal fake news detection, with emphasis on:

- model design
- implementation details
- experiment organization
- ablation settings
- reproducibility notes
- paper supplementary materials

The core model combines:

- a dual-path text encoder based on `BiLSTM + Transformer`
- an image encoder based on `EfficientNet-B0`
- a `Shared Residual Bridge` for coarse semantic alignment
- a `Bidirectional Cross-modal Attention` module for fine-grained text-image interaction

The repository is intended to serve two purposes at the same time:

1. a public-facing GitHub project that clearly explains the work
2. supplementary material for a paper or thesis, so that others can understand and reproduce the model

## Project Goal

Fake news on social media is often expressed through both text and images. Text-only models may overlook misleading images, while image-only models may miss stance, context, or semantic contradiction in the text. This project studies how to jointly model text content, visual evidence, and the semantic consistency between them.

The proposed method focuses on three design ideas:

1. stronger text modeling through local and global sequence features
2. a shared semantic bridge that reduces modality gap before fusion
3. bidirectional cross-modal attention that captures text-to-image and image-to-text interaction

## Main Contributions

1. **Dual-path text encoder**
   Combines `BiLSTM` for local contextual order modeling and `Transformer` for global semantic dependency modeling.

2. **Shared semantic bridge**
   Uses a shared residual mapping to align text and image representations in a lightweight way before cross-modal fusion.

3. **Bidirectional cross-modal attention**
   Models both text querying image features and image querying text features, improving consistency and conflict detection between modalities.

4. **Systematic experiments**
   Includes main experiments, unimodal comparisons, baseline comparisons, and ablation studies on `Weibo`, `Gossip`, and `CFND`.

## Model Pipeline

```text
Input text T and image I
  -> Text branch: Embedding + BiLSTM + Transformer
  -> Image branch: EfficientNet-B0
  -> Shared Semantic Bridge
  -> Bidirectional Cross-modal Attention
  -> MLP classifier
  -> Fake / Real prediction
```

Detailed mapping from paper modules to implementation is provided in [docs/model_implementation.md](docs/model_implementation.md).

## Datasets

This project is organized around three datasets:

| Dataset | Source | Size | Notes |
| --- | --- | ---: | --- |
| Weibo | Jin et al., ACM MM 2017 | 9,527 | Chinese social media rumor dataset with relatively balanced labels |
| Gossip | Shu et al., Big Data 2020 | 12,840 | English entertainment news scenario with clear class imbalance |
| CFND | Zhang et al., IJCAI 2024 | 26,664 | Chinese cross-domain fake news dataset for generalization analysis |

Important:

- the repository does **not** directly include the full raw datasets
- dataset access may be restricted by the original publishers
- please obtain the datasets from their official or paper-linked sources before reproduction

See [docs/dataset_statement.md](docs/dataset_statement.md) and [docs/reproducibility.md](docs/reproducibility.md).

## Main Results

| Dataset | Model | Accuracy | Fake F1 |
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

More detailed records are maintained in [docs/experiment_log.md](docs/experiment_log.md).

## Quick Start

### 1. Create environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

For Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If you also want to run the FaKnow-adapted baselines:

```bash
pip install -r requirements-faknow.txt
```

### 2. Prepare data

Before training, check the dataset-related paths in the corresponding training scripts, especially:

- `CFG.dataset_root`
- `CFG.processed_dir`
- `CFG.train_csv`
- `CFG.val_csv`
- `CFG.test_csv`
- `CFG.save_root`

### 3. Run the main experiments

Main experiment scripts are located in:

```text
code/01_我的实验代码_主实验+对比+消融/对比实验代码/主实验并入代码/
```

Run:

```bash
python final_train_CFND.py
python final_train_gossip.py
python final_train_weibo.py
```

For full reproduction notes, see [docs/reproducibility.md](docs/reproducibility.md).

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── requirements-faknow.txt
├── CITATION.cff
├── docs/
│   ├── dataset_statement.md
│   ├── experiment_log.md
│   ├── figures.md
│   ├── model_implementation.md
│   ├── paper_outline.md
│   ├── reproducibility.md
│   ├── repository_checklist.md
│   ├── tables.md
│   └── third_party_code.md
└── code/
    ├── 01_我的实验代码_主实验+对比+消融/
    ├── 03_我的可视化与分析代码/
    └── 04_第三方基线与参考实现/
```

## Documentation Guide

- [docs/model_implementation.md](docs/model_implementation.md): maps the paper modules to the actual code
- [docs/reproducibility.md](docs/reproducibility.md): environment, data preparation, and reproduction workflow
- [docs/experiment_log.md](docs/experiment_log.md): experiment organization and current result summary
- [docs/figures.md](docs/figures.md): figure and visualization inventory
- [docs/tables.md](docs/tables.md): paper and README table sources
- [docs/dataset_statement.md](docs/dataset_statement.md): dataset access, restrictions, and publication notes
- [docs/third_party_code.md](docs/third_party_code.md): third-party baseline and reference implementation notes
- [docs/repository_checklist.md](docs/repository_checklist.md): public release checklist

## Code Evidence for the Main Model

The main `CFND` training script contains the core implementation modules:

- `CFG = SimpleNamespace(...)`
- `LSTMTransformerEncoder`
- `ImageEncoder`
- `SharedResidualBridge`
- `BiDirectionalCrossAttentionBlock`
- `LightweightBiCrossAttentionFusion`
- `MultiModalModel`
- `train_one_epoch`
- `evaluate`
- `compute_metrics_from_probs`

Representative file:

- `code/01_我的实验代码_主实验+对比+消融/对比实验代码/主实验并入代码/final_train_CFND.py`

See [docs/model_implementation.md](docs/model_implementation.md) for the full explanation.

## Public Release Notes

This repository is suitable for public academic presentation, but a strong public release should still confirm the following items before wide sharing:

1. dataset access instructions are complete
2. third-party code sources and licenses are clearly stated
3. final architecture figures are exported in high quality
4. full experiment logs and metric sources are preserved
5. license choice is explicitly confirmed by the author

At the moment, this repository is best understood as:

- a research record
- a reproducibility entry point
- a paper supplementary repository

## Citation

If you use or refer to this repository, please cite the associated paper, thesis, or repository record. A citation template is provided in [CITATION.cff](CITATION.cff).

## License

No open-source license has been selected yet in this repository revision.

Before making the repository broadly public, it is recommended to explicitly choose one of the following:

- `MIT` if you want maximum reuse flexibility
- `Apache-2.0` if you want an explicit patent grant
- `All rights reserved` if you only want the repository to function as a research disclosure or supplementary archive

Until a license is added, reuse rights are not clearly granted.

## 中文说明

这是一个以“模型设计与实现记录”为核心的多模态假新闻检测研究仓库，目标不是只放代码，而是把以下内容完整保留下来：

- 模型结构与论文方法的对应关系
- 主实验、对比实验、消融实验的脚本入口
- 关键实验结果
- 复现所需环境与数据说明
- 图表、表格、论文附属材料

如果你是第一次看这个仓库，建议按下面顺序阅读：

1. `README.md`
2. `docs/model_implementation.md`
3. `docs/reproducibility.md`
4. `docs/experiment_log.md`

如果你准备公开发布这个仓库，建议优先补齐：

1. 最终许可证
2. 数据获取方式
3. 第三方代码来源与许可
4. 最终高清模型结构图
5. 每个主实验 run 的指标来源文件或摘录
