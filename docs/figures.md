# 图表与可视化材料

本文档用于记录论文和 GitHub 展示中使用的图像、统计图和模型示意图来源。

## 1. 建议图清单

| 图编号 | 名称 | 当前状态 | 来源或生成方式 |
| --- | --- | --- | --- |
| Figure 1 | 模型整体结构图 | 已有展示素材 | `code/03_我的可视化与分析代码/论文代码展示片段/模型流程.png` |
| Figure 2 | 数据集标签分布 | 已有统计脚本与图片 | `code/03_我的可视化与分析代码/可视化脚本/` |
| Figure 3 | 文本长度分布 | 已有统计脚本与图片 | `generate_raw_dataset_visualizations.py` |
| Figure 4 | 高频词/词云对比 | 已有图片 | `weibo/gossip/cfnd/figures/wordcloud_label_*.png` |
| Figure 5 | 训练曲线 | 训练后生成 | 每个 run 目录中的 `loss_curve.png`、`metrics_curve.png` |
| Figure 6 | 混淆矩阵 | 训练后生成 | 每个 run 目录中的 `test_confusion_matrix.png` |
| Figure 7 | 案例分析图 | 待整理 | 从 CFND 测试集案例和模型预测结果中选取 |

## 2. 可视化脚本

原始数据统计与可视化脚本：

```text
code/03_我的可视化与分析代码/可视化脚本/generate_raw_dataset_visualizations.py
```

该脚本会为不同数据集生成：

- 标签分布图
- 文本长度分布图
- 高频词对比图
- 词云图
- `summary.json`
- 中间统计 CSV

脚本逻辑说明可参考：

```text
code/03_我的可视化与分析代码/可视化脚本/RAW_DATASET_VISUALIZATION_DOC.md
```

## 3. 公开展示建议

- README 中只放 1-2 张最关键图片，例如模型结构图和主结果表。
- 论文附录或补充材料中放完整数据分布图、训练曲线和混淆矩阵。
- 所有图都建议保留生成脚本、输入口径和输出路径，避免只保留图片。
- 如果图片来自训练 run 目录，建议把对应 run 编号写入 `docs/experiment_log.md`。

## 4. 待补充

- 将最终版模型结构图导出为高清 PNG 或 PDF。
- 为三数据集各补一张混淆矩阵图。
- 为消融实验补充柱状图或折线图。
- 为案例分析补充预测样本、真实标签、预测标签和置信度说明。
