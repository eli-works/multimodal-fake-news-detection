# 公开仓库完善清单

本清单用于把当前仓库整理成可公开展示、可理解、可复现、可作为论文附属材料的研究项目。

## 已补充

- `requirements.txt`：记录核心运行依赖。
- `requirements-faknow.txt`：记录 FaKnow 基线适配的可选依赖。
- `.gitignore`：忽略缓存、模型权重、运行输出和本地数据集。
- `docs/reproducibility.md`：说明环境、数据准备、主实验运行和复现注意事项。
- `docs/model_implementation.md`：建立论文模块和代码模块的对应关系。
- `docs/experiment_log.md`：提供实验记录模板和当前核心结果汇总。
- `docs/figures.md`：整理图像材料和可视化脚本来源。
- `docs/tables.md`：整理论文表格来源和待补充指标。

## 还需要人工确认或后续补充

| 项目 | 重要性 | 说明 |
| --- | --- | --- |
| 数据集下载说明 | 高 | 需要写清楚 Weibo、Gossip/FakeNewsNet、CFND 的获取方式和许可限制 |
| 原始 run 指标 | 高 | 将 `test_metrics.json`、best epoch、best val F1 补入 `docs/experiment_log.md` |
| 高清模型结构图 | 高 | README 和论文都建议放一张最终版结构图 |
| 许可证 | 高 | 公开 GitHub 前建议选择 MIT、Apache-2.0 或仅保留 All rights reserved |
| 引用信息 | 中 | 可补 `CITATION.cff`，便于他人引用项目 |
| 权重下载链接 | 中 | 如果公开权重，建议放 GitHub Release、网盘或 HuggingFace，并注明数据许可 |
| 英文 README 摘要 | 中 | 面向更广泛读者时，可以增加英文摘要或双语 README |
| 代码目录精简 | 中 | 可考虑把核心主实验脚本复制或重构到更短路径，如 `src/` 或 `experiments/` |
| 第三方代码声明 | 高 | 对 `04_第三方基线与参考实现` 中的外部代码补来源、许可证和修改说明 |
| 大文件处理 | 高 | 压缩包、`.pyc`、数据中间文件、权重文件不建议直接提交到公开仓库 |

## 推荐的公开仓库结构

```text
.
├── README.md
├── requirements.txt
├── docs/
│   ├── paper_outline.md
│   ├── reproducibility.md
│   ├── model_implementation.md
│   ├── experiment_log.md
│   ├── figures.md
│   ├── tables.md
│   └── repository_checklist.md
├── code/
│   ├── 01_我的实验代码_主实验+对比+消融/
│   ├── 03_我的可视化与分析代码/
│   └── 04_第三方基线与参考实现/
└── .gitignore
```

本地的 `05_压缩包归档`、`06_资料与临时文件`、Word 初稿和论文摘录更适合留在个人归档中，不建议进入公开仓库。

如果后续准备长期维护，建议逐步演进为：

```text
.
├── src/
│   └── model/
├── experiments/
│   ├── train_cfnd.py
│   ├── train_gossip.py
│   └── train_weibo.py
├── configs/
├── docs/
└── scripts/
```

当前阶段不必强行重构。先保证 README、复现文档、实验记录和代码证据完整，会更稳。
