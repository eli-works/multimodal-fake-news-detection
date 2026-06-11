# 模型设计与代码实现对照

本文档把论文中的核心模块映射到代码实现，方便审稿、答辩、复现和后续维护。

## 1. 主实现入口

| 数据集 | 主实验脚本 | 说明 |
| --- | --- | --- |
| CFND | `code/01_我的实验代码_主实验+对比+消融/对比实验代码/主实验并入代码/final_train_CFND.py` | 中文多领域数据集主实验 |
| Gossip | `code/01_我的实验代码_主实验+对比+消融/对比实验代码/主实验并入代码/final_train_gossip.py` | 英文 GossipCop 场景主实验 |
| Weibo | `code/01_我的实验代码_主实验+对比+消融/对比实验代码/主实验并入代码/final_train_weibo.py` | 中文微博谣言数据主实验 |

三个主实验脚本共享同一类模型结构，差异主要体现在数据路径、文本长度、图像冻结层数和学习率配置上。

## 2. 配置与训练流程

| 论文/实验项 | 代码位置 | 作用 |
| --- | --- | --- |
| 超参数配置 | `CFG = SimpleNamespace(...)` | 设置数据路径、模型维度、训练轮数、学习率、冻结层数等 |
| 数据读取 | `NewsDataset` | 文本分词、padding/mask 构造、图片读取和图像增强 |
| 词表构建 | `build_vocab` | 基于训练集构建词表，低频词按 `<UNK>` 处理 |
| 训练循环 | `train_one_epoch` | 前向、损失计算、反向传播、梯度裁剪和优化器更新 |
| 验证/测试 | `evaluate`、`compute_metrics_from_probs` | 输出 Accuracy、Precision、Recall、F1、Macro-F1、AUC 和混淆矩阵 |
| 结果保存 | `main` | 保存模型、训练曲线、混淆矩阵、指标 JSON 和日志 |

## 3. 文本分支：BiLSTM + Transformer

论文描述：输入文本序列先经词嵌入映射，再通过 BiLSTM 捕捉局部上下文和顺序依赖，随后投影到统一维度并加入位置编码，最后由 Transformer Encoder 建模全局语义关系。

代码模块：

```text
LSTMTransformerEncoder
```

实现流程：

```text
texts
  -> Embedding
  -> Dropout
  -> BiLSTM
  -> Linear projection
  -> SinusoidalPositionalEncoding
  -> TransformerEncoder
  -> LayerNorm
```

关键维度：

| 项目 | 默认值 |
| --- | ---: |
| `embed_dim` | 192 |
| `lstm_hidden_dim` | 128 |
| BiLSTM 输出 | 256 |
| `model_dim` | 192 |
| Transformer heads | 4 |
| Transformer layers | 1 |

## 4. 图像分支：EfficientNet-B0

论文描述：图像侧使用 ImageNet 预训练的 EfficientNet-B0 提取视觉语义特征，并将 1280 维视觉特征投影到与文本一致的 192 维空间。

代码模块：

```text
ImageEncoder
```

实现流程：

```text
image
  -> EfficientNet-B0
  -> remove classifier
  -> Linear(1280 -> model_dim)
  -> LayerNorm
  -> GELU
  -> Dropout
```

冻结策略由 `CFG.freeze_efficientnet_stages` 控制，不同数据集的默认设置不同，用于平衡通用视觉特征保留和任务适配能力。

## 5. 共享语义桥：Shared Residual Bridge

论文描述：文本特征与图像特征天然分布在不同空间，直接拼接容易受到模态差异影响。共享残差桥通过共享的两层映射提取公共语义残差信息，再与原始特征相加，完成轻量级语义对齐。

代码模块：

```text
SharedResidualBridge
```

实现流程：

```text
text_vec, image_vec
  -> shared residual unit
  -> tanh(dropout(x) + alpha * residual(x))
  -> text_share, image_share
```

对应论文公式：

| 公式 | 代码含义 |
| --- | --- |
| `Res_x = Dropout(W2 * tanh(W1 * x + b1) + b2)` | `_shared_residual` |
| `t_share = tanh(Dropout(t) + alpha * Res_t)` | `forward` 中的文本共享表示 |
| `v_share = tanh(Dropout(v) + alpha * Res_v)` | `forward` 中的图像共享表示 |

## 6. 双向跨模态注意力：BiCrossAttn

论文描述：共享语义桥完成粗粒度对齐后，模型进一步进行文本到图像、图像到文本的双向跨模态交互，以捕捉图文一致性、互补性和潜在冲突线索。

代码模块：

```text
BiDirectionalCrossAttentionBlock
LightweightBiCrossAttentionFusion
```

实现流程：

```text
text_seq + text_share
image_share
  -> image token projection
  -> text_to_image attention
  -> image_to_text attention
  -> FFN + residual + LayerNorm
  -> masked text pooling + image token mean pooling
  -> text_fused, image_fused
```

关键设计：

| 设计 | 说明 |
| --- | --- |
| `cross_attn_image_tokens` | 将单个图像向量展开为多个轻量图像 token |
| `text_to_image` | 文本 token 查询图像 token |
| `image_to_text` | 图像 token 查询文本 token，并使用文本 mask 屏蔽 padding |
| `ffn_ratio` | 控制跨模态注意力块中 FFN 的隐藏层宽度 |

## 7. 分类头与输出

代码模块：

```text
MultiModalModel
```

最终流程：

```text
text_fused, image_fused
  -> concat
  -> Linear(384 -> 128)
  -> LayerNorm
  -> GELU
  -> Dropout
  -> Linear(128 -> 2)
```

训练使用 `CrossEntropyLoss`，预测时取 `softmax(logits)[:, 1]` 作为假新闻概率。

## 8. 论文贡献与代码证据

| 论文贡献 | 代码证据 |
| --- | --- |
| 双路径文本编码 | `LSTMTransformerEncoder` |
| EfficientNet 图像语义提取 | `ImageEncoder` |
| 共享语义桥 | `SharedResidualBridge` |
| 双向跨模态注意力 | `BiDirectionalCrossAttentionBlock`、`LightweightBiCrossAttentionFusion` |
| 端到端训练与评估 | `MultiModalModel`、`train_one_epoch`、`evaluate`、`compute_metrics_from_probs` |

## 9. 公开展示建议

- README 只展示核心思想和结果，避免堆放所有脚本细节。
- 本文档作为论文方法部分的代码补充材料。
- 复现命令、数据口径和环境差异统一写在 `docs/reproducibility.md`。
- 实验结果不要只写最终表格，应在 `docs/experiment_log.md` 中保留脚本、配置、输出目录和指标来源。
