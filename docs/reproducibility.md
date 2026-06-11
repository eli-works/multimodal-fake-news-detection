# 复现指南

本文档说明如何从公开仓库理解并复现实验代码。由于 Weibo、Gossip/FakeNewsNet、CFND 等数据集通常受原始发布方协议约束，仓库不直接包含完整原始数据和训练权重；复现时需要先按数据集来源自行获取数据，再放到脚本配置指定的位置。

## 1. 环境准备

推荐环境：

| 项目 | 建议版本 |
| --- | --- |
| Python | 3.10 |
| PyTorch | 2.5.x |
| CUDA | 12.x |
| GPU | 24GB 显存更稳，较小显存可降低 batch size |

安装依赖：

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

如果需要运行 FaKnow 适配基线，再安装可选依赖：

```bash
pip install -r requirements-faknow.txt
```

Windows PowerShell 可使用：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

如果需要运行 FaKnow 适配基线：

```powershell
pip install -r requirements-faknow.txt
```

## 2. 数据准备

主实验使用三个数据集：

| 数据集 | 任务脚本 | 默认数据配置 |
| --- | --- | --- |
| CFND | `code/01_我的实验代码_主实验+对比+消融/对比实验代码/主实验并入代码/final_train_CFND.py` | `CFG.dataset_root = "../CFND_dataset"` |
| Gossip | `code/01_我的实验代码_主实验+对比+消融/对比实验代码/主实验并入代码/final_train_gossip.py` | `CFG.dataset_root = "../AAAI_dataset"`，`CFG.processed_dir = "gossip"` |
| Weibo | `code/01_我的实验代码_主实验+对比+消融/对比实验代码/主实验并入代码/final_train_weibo.py` | `CFG.dataset_root = "weibo"`，`CFG.processed_dir = "weibo"` |

复现前请打开对应脚本顶部的 `CFG`，确认以下字段与本机数据位置一致：

- `dataset_root`
- `processed_dir`，仅 Gossip 和 Weibo 使用
- `train_csv`
- `val_csv`
- `test_csv`
- `image_column`
- `label_column`
- `save_root`

标签约定为 `0=real`、`1=fake`，评估时以 `fake` 作为 positive class。

## 3. 运行主实验

进入主实验脚本目录：

```bash
cd "code/01_我的实验代码_主实验+对比+消融/对比实验代码/主实验并入代码"
```

运行 CFND：

```bash
python final_train_CFND.py
```

运行 Gossip：

```bash
python final_train_gossip.py
```

运行 Weibo：

```bash
python final_train_weibo.py
```

每次运行会在 `save_root` 下创建带时间戳的实验目录，通常包含：

- `train.log`
- `best_model.pth`
- `final_model.pth`
- `history.json`
- `test_metrics.json`
- `loss_curve.png`
- `metrics_curve.png`
- `test_confusion_matrix.png`

公开仓库建议保留指标表、日志摘要和曲线图；模型权重文件通常不提交到 GitHub，可上传到 release 或网盘并在 README 中补链接。

## 4. 运行对比与消融

对比实验脚本位于：

```text
code/01_我的实验代码_主实验+对比+消融/对比实验代码/
```

消融实验脚本位于：

```text
code/01_我的实验代码_主实验+对比+消融/消融实验代码/
```

建议按如下顺序复现：

1. 先运行三个 `final_train_*` 主实验脚本，确认数据读取和 GPU 环境正常。
2. 再运行 `textonly`、`image_only`、`concat` 等单模态/简单融合基线。
3. 最后运行 `wosb`、`wobicross`、`lstm_only`、`transformer_only` 等消融脚本。
4. 将每次运行的 `test_metrics.json` 汇总到 `docs/experiment_log.md`。

## 5. FaKnow 基线适配

FaKnow 适配代码位于：

```text
code/01_我的实验代码_主实验+对比+消融/对比实验代码/主实验并入代码/faknow_adapter/
```

该目录用于统一适配 SpotFake、HMCAN、MCAN、EANN、MFAN、SAFE、CAFE 等内容驱动多模态模型。详细用法见该目录下的 `README.md`。

## 6. 复现注意事项

- 由于随机初始化、CUDA 算子、数据增强和硬件差异，复现结果可能存在小幅波动。
- 脚本中开启了 `cudnn.benchmark=True` 和 TF32 优化，优先保证训练吞吐；如果需要严格确定性，需要关闭相关优化并固定更多随机源。
- 数据集划分、标签口径和缺失图片处理会显著影响结果，复现报告中应明确记录。
- 如果显存不足，优先减小 `CFG.batch_size`，必要时降低 `num_workers`。
