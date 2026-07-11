# 实验记录

本文档用于沉淀主实验、对比实验、消融实验和参数分析的可追溯记录。公开仓库建议保留指标来源、脚本入口、关键结论和分析逻辑，避免在摘要性质的文档中反复堆叠具体数值。

## 1. 记录规范

每次实验建议记录：

| 字段 | 说明 |
| --- | --- |
| 实验编号 | 例如 `main-cfnd-001` |
| 数据集 | `CFND`、`Gossip`、`Weibo` |
| 脚本 | 训练脚本路径 |
| 变体 | `Ours`、`Concat`、`TextOnly`、`ImageOnly`、`-SB`、`-BCA` 等 |
| 关键配置 | `max_len`、`batch_size`、`lr`、`freeze_efficientnet_stages` 等 |
| 输出目录 | `save_root` 下的 run 目录 |
| 指标文件 | `test_metrics.json` 或手动汇总表 |
| 结论 | 一句话概括该实验支撑的论文观点 |

## 2. 主实验摘要

| 数据集 | 模型 | 记录方式 | 指标来源 |
| --- | --- | --- | --- |
| CFND | Ours | 关键指标保存在 run 目录中 | `test_metrics.json` |
| CFND | Concat / MVAE / att-RNN / EANN-noadv | 作为基线对照保留 | 对比实验汇总 |
| Weibo | Ours | 关键指标保存在 run 目录中 | `test_metrics.json` |
| Weibo | Concat / MVAE / att-RNN / EANN-noadv | 作为基线对照保留 | 对比实验汇总 |
| Gossip | Ours | 关键指标保存在 run 目录中 | `test_metrics.json` |
| Gossip | Concat / MVAE / att-RNN / EANN-noadv | 作为基线对照保留 | 对比实验汇总 |

结论摘要：

- 完整模型在 CFND 和 Weibo 上更能体现图文融合的价值
- Gossip 上文本线索本身较强，因此融合模块的边际收益更小
- 与传统基线相比，当前模型更适合需要图文一致性建模的场景

## 3. 单模态对比

| 数据集 | 模型 | 定性说明 | 结论 |
| --- | --- | --- | --- |
| CFND | Ours / TextOnly / ImageOnly | 单模态与双模态差异明显 | 图像单独建模偏弱，融合更稳 |
| Gossip | Ours / TextOnly / ImageOnly | 文本线索占主导 | 双模态保持稳定，但提升有限 |
| Weibo | Ours / TextOnly / ImageOnly | 图文信息互补更明显 | 融合能覆盖更多误导场景 |

## 4. 消融实验

| 变体 | 对应脚本关键字 | 主要观察 |
| --- | --- | --- |
| `LSTM` | `lstm_only` | 去掉 Transformer 后，全局语义建模下降 |
| `TRM` | `transformer_only` | 去掉 BiLSTM 后，局部上下文和顺序信息下降 |
| `-SB` | `wosb` | 去掉共享语义桥后，对齐能力下降 |
| `-BCA` | `wobicross` | 去掉双向跨模态注意力后，细粒度交互能力下降 |

关键发现：

- 共享语义桥有助于缓解模态差异
- 双向跨模态注意力对捕捉图文冲突和互补关系是必要的
- 双路径文本编码比单一路径更稳

## 5. 参数分析

### 5.1 EfficientNet 冻结层数

| 数据集 | 观察 |
| --- | --- |
| Gossip | 冻结策略对结果影响较小，说明文本线索较强 |
| CFND | 中等冻结策略通常更稳，图像和文本分布更匹配 |
| Weibo | 冻结策略更敏感，说明图像适配更重要 |

### 5.2 Gossip 文本长度

| `max_len` | 观察 |
| ---: | --- |
| 较短 | 更依赖标题和局部短语 |
| 中等 | 通常更平衡 |
| 较长 | 容易引入噪声并拉低稳定性 |

结论：Gossip 上中等长度文本截断通常更合适。

## 6. 待补充材料

- 每个主实验的 `test_metrics.json` 原始文件或摘录
- 每个数据集的混淆矩阵图
- 每个消融脚本对应的完整评估表
- 训练日志中的 best epoch 和最终指标来源
