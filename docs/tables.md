# 论文表格材料

本文档记录论文、README 和答辩材料中的表格来源。建议所有表格都能追溯到脚本输出或实验日志。

## 1. 数据集统计表

| 数据集 | 总数 | 真新闻 | 假新闻 | 训练/验证/测试 | 特点 |
| --- | ---: | ---: | ---: | --- | --- |
| Weibo | 9527 | 4779 | 4748 | 6777 / 754 / 1996 | 中文社交媒体，类别相对平衡 |
| Gossip | 12840 | 2581 | 10259 | 9009 / 1001 / 2830 | 英文娱乐新闻，类别不平衡 |
| CFND | 26664 | 16394 | 10270 | 15997 / 5333 / 5334 | 中文多领域数据 |

## 2. 实验配置表

| 参数 | 默认值 |
| --- | --- |
| Optimizer | AdamW |
| Batch size | 64 |
| Epoch | 25 或 30，依数据集脚本而定 |
| Learning rate | `1e-4` |
| Backbone learning rate | `3e-5` 或 `5e-5` |
| Weight decay | `1e-3` |
| Dropout | `0.3` |
| Random seed | `42` |
| Gradient clipping | `0.8` |
| Model selection | Validation F1 |

## 3. 主实验结果表

主实验结果表维护在 `docs/experiment_log.md`，论文正文中可保留 Accuracy 和 Fake F1，附录中补全 Precision、Recall、Macro-F1 和 AUC。

## 4. 消融实验表

消融实验建议拆成两类表：

| 表名 | 内容 |
| --- | --- |
| 模块消融 | `Ours`、`-SB`、`-BCA`、`LSTM`、`TRM` |
| 单模态对比 | `Ours`、`TextOnly`、`ImageOnly` |

## 5. 参数分析表

建议保留：

- EfficientNet 冻结层数对三数据集的影响。
- Gossip `max_len` 对 Accuracy 和 Fake F1 的影响。
- 如后续补充，可加入 batch size、学习率、图像 token 数等敏感性分析。

## 6. 待补充

- 从每个 run 的 `test_metrics.json` 中补齐 Precision、Recall、Macro-F1、AUC。
- 给每张表增加“指标来源”列，指向具体 run 或实验编号。
- 最终论文表格和 README 表格保持同一套数字，避免出现版本不一致。
