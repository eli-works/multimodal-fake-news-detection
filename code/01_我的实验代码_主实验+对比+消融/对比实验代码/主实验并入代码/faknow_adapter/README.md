# FaKnow 三数据集七模型适配说明

## 1. 交付内容

本目录提供一个与原有 `final_train_*` 脚本完全独立的流水线，实现 `CFND / Gossip / Weibo` 三个数据集对 FaKnow 内容驱动多模态模型的适配：

- `prepare_all.py`：数据工件生成器（只做数据适配，不训练）。
- `run_all_models.py`：统一入口（可选 prepare + train）。
- `README.md`：使用说明与工件契约。

支持模型：

- SpotFake
- HMCAN
- MCAN
- EANN
- MFAN
- SAFE
- CAFE

## 2. 环境建议

推荐在云服务器使用独立虚拟环境：

```bash
pip install faknow==0.0.4
pip install torch torchvision torchaudio
pip install transformers jieba pandas pillow scipy scikit-learn tensorboard
```

说明：

- `run_all_models.py` 训练阶段依赖 `faknow==0.0.4`。
- `prepare_all.py` 会调用 HuggingFace 文本模型和 ResNet50 提取特征，首次运行会下载预训练权重。

## 3. 目录结构

默认目录（可通过参数改）：

- 工件目录：`train/faknow_adapter/artifacts`
- 运行目录：`train/faknow_adapter/runs`

适配后结构：

```text
artifacts/
  cfnd|gossip|weibo/
    standard/
      train.json
      val.json
      test.json
    spotfake/
      train.json
      val.json
      test.json
    hmcan/
      train.json
      val.json
      test.json
    mcan/
      train.json
      val.json
      test.json
    eann/
      train.json
      val.json
      test.json
      vocab.pkl
      word_vectors.pkl
      stop_words.txt
    mfan/
      train.json
      val.json
      test.json
      post_id_map.json
      vocab.pkl
      word_vectors.pkl
      node_embedding.pkl
      adjacency.json
    safe/
      train/
        case_headline.npy
        case_body.npy
        case_image.npy
        case_y_fn_dim1.npy
      test/
        case_headline.npy
        case_body.npy
        case_image.npy
        case_y_fn_dim1.npy
    cafe/
      train_text_with_label.npz
      train_image_with_label.npz
      test_text_with_label.npz
      test_image_with_label.npz
    metadata.json
  prepare_report.json
```

训练日志结构：

```text
runs/
  {dataset}/
    {model}/
      {timestamp}/
        config_snapshot.json
        metrics.json
        logs/
        tb_logs/
  run_report.json
```

## 4. 字段与工件契约

### 4.1 标准内部字段

`standard/*.json` 每条样本字段：

- `sample_id`
- `text`
- `image_abs_path`
- `label`
- `raw_post_id`

### 4.2 模型 JSON 字段

- SpotFake：`post_text`, `image_id`, `label`
- HMCAN/MCAN：`text`, `image`, `label`
- EANN：`text`, `image`, `domain`, `label`（本适配固定 `domain=0`）
- MFAN：`post_id`, `text`, `image`, `label`（`post_id` 连续重映射）

### 4.3 SAFE/CAFE 契约

- SAFE split 目录下必须包含：`case_headline.npy`, `case_body.npy`, `case_image.npy`, `case_y_fn_dim1.npy`
- CAFE 数据必须包含四个文件：`train/test_text_with_label.npz`, `train/test_image_with_label.npz`
- CAFE 目标形状：
  - 文本：`(N, 30, 200)`
  - 图像：`(N, 512)`

## 5. 关键实现说明

### 5.1 三数据集映射

- 默认原始数据路径（可覆盖）：
  - `CFND`: `C:\Users\gan\Desktop\dataset\CFND_dataset`
  - `Gossip`: `C:\Users\gan\Desktop\dataset\社交媒体谣言检测数据集\gossip`
  - `Weibo`: `C:\Users\gan\Desktop\dataset\社交媒体谣言检测数据集\weibo`
- Gossip 图像根目录可单独指定：
  - 默认：`C:\Users\gan\Desktop\dataset\AAAI_dataset`
  - 会自动处理 `AAAI_dataset/Images/...` 映射。
- 与你原 `final_train_gossip.py / final_train_weibo.py` 一致，若 `gossip/weibo` 的 `train/val/test` CSV 缺失，可自动调用：
  - `train/prepare_gossip_data.py`
  - `train/prepare_weibo_data.py`
  来补齐标准 CSV（可通过参数关闭）。

### 5.2 缺图策略

图片路径统一写绝对路径；缺失图片统一回退到占位图，避免 DataLoader 读图崩溃。

### 5.3 EANN/MFAN 修正

- EANN：`run_all_models.py` 使用本地修正版 tokenizer，修复官方中文分词拼接问题并支持 OOV 回退。
- MFAN：使用本地修正版 tokenizer（OOV 兼容）和修正版 Trainer（训练时正确把 batch 移到 `device`）。

### 5.4 无权重输出策略

- 所有训练入口显式 `save_best=None`。
- 不启用 early stopping 保存。
- 每次训练后自动扫描并删除 `.pth/.pt/.ckpt`。
- 默认只保留日志与指标文件。

### 5.5 SAFE 跨平台说明

- FaKnow 原生 `SAFENumpyDataset` 使用 Windows 风格反斜杠拼接路径，在 Linux 服务器可能读取失败。
- 本项目在 `run_all_models.py` 内置了兼容版 SAFE 数据集加载器，统一使用 `pathlib` 拼接，云端 Linux/Windows 都可用。

## 6. 命令用法

## 6.1 仅准备数据工件

```bash
python train/faknow_adapter/prepare_all.py --dataset cfnd --out train/faknow_adapter/artifacts
python train/faknow_adapter/prepare_all.py --dataset gossip --out train/faknow_adapter/artifacts --gossip-image-root C:/Users/gan/Desktop/dataset/AAAI_dataset
python train/faknow_adapter/prepare_all.py --dataset weibo --out train/faknow_adapter/artifacts
```

## 6.2 统一入口（prepare + train）

```bash
python train/faknow_adapter/run_all_models.py --dataset cfnd --model all --all --device cuda --epochs 10 --batch-size 32
```

## 6.3 仅训练（使用已生成工件）

```bash
python train/faknow_adapter/run_all_models.py --dataset weibo --model mcan --train-only --device cuda --epochs 5 --batch-size 32
python train/faknow_adapter/run_all_models.py --dataset gossip --model eann --train-only --device cuda --epochs 8 --batch-size 64
```

## 6.4 仅准备（从统一入口）

```bash
python train/faknow_adapter/run_all_models.py --dataset all --model all --prepare-only --artifacts-dir train/faknow_adapter/artifacts
```

## 6.5 快速准备模式（推荐先冒烟）

快速准备模式只生成轻量工件（`standard` + 各模型 JSON），跳过重特征提取（EANN/MFAN/SAFE/CAFE 的向量/图结构/NPY/NPZ）。

适合用途：

- 先验证数据路径、字段映射、图片路径可读性。
- 先验证 `spotfake/hmcan/mcan` 的基础链路。

命令示例：

```bash
python train/faknow_adapter/run_all_models.py \
  --dataset weibo \
  --model all \
  --prepare-only \
  --quick-prepare
```

注意：

- `--quick-prepare` 不能直接用于训练全模型；若要训练 EANN/MFAN/SAFE/CAFE，需重新执行完整 prepare（不加 `--quick-prepare`）。

## 7. 常见参数

通用参数：

- `--dataset {cfnd,gossip,weibo,all}`
- `--model {spotfake,hmcan,mcan,eann,mfan,safe,cafe,all}`
- `--prepare-only`
- `--train-only`
- `--all`
- `--device`
- `--epochs`
- `--batch-size`
- `--artifacts-dir`
- `--runs-dir`
- `--auto-prepare-data / --no-auto-prepare-data`
- `--force-prepare-data`
- `--val-ratio`

准备阶段参数（会透传到 `prepare_all.py` 逻辑）：

- `--cfnd-root`
- `--gossip-root`
- `--weibo-root`
- `--gossip-image-root`
- `--prepare-text-model-zh`
- `--prepare-text-model-en`
- `--prepare-device`
- `--prepare-batch-size`
- `--min-freq`
- `--mfan-knn`
- `--safe-head-len --safe-body-len --safe-image-len`
- `--cafe-text-len --cafe-text-dim`

CAFE 训练补充：

- 默认 `drop_last=True`（与 FaKnow 官方 `run_cafe` 一致，避免小 batch 触发 BatchNorm 问题）。
- 若你确实要保留最后一个不足 batch，可加 `--cafe-keep-last`。

## 8. 常见报错排查

1. `faknow is required for training`
- 原因：未安装 FaKnow。
- 处理：`pip install faknow==0.0.4`。

2. `Missing artifacts for model=...`
- 原因：未先生成工件，或工件目录不完整。
- 处理：先运行 `--prepare-only`，确认 `artifacts/{dataset}/{model}` 文件齐全。

3. HuggingFace/torchvision 首次下载失败
- 原因：网络或代理问题。
- 处理：配置代理或提前缓存模型权重。

4. Gossip 图片读取报错或大量缺图
- 原因：`--gossip-image-root` 未指向 `AAAI_dataset` 根目录。
- 处理：显式指定 `--gossip-image-root` 并检查 `AAAI_dataset/Images/...` 是否存在。

5. GPU OOM
- 原因：batch 过大或模型显存占用高。
- 处理：减小 `--batch-size`，必要时切换 `--device cpu` 或分模型单独训练。

## 9. 推荐云端执行顺序

1. 先全量准备：

```bash
python train/faknow_adapter/run_all_models.py --dataset all --model all --prepare-only --device cuda
```

2. 再按数据集/模型分批训练：

```bash
python train/faknow_adapter/run_all_models.py --dataset cfnd --model all --train-only --device cuda
python train/faknow_adapter/run_all_models.py --dataset gossip --model all --train-only --device cuda
python train/faknow_adapter/run_all_models.py --dataset weibo --model all --train-only --device cuda
```

## 10. Linux 云端最短落地（推荐）

先按官方文档准备 FaKnow 运行环境：

- 文档入口：[FaKnow Introduction](https://faknow.readthedocs.io/en/latest/get_started/introduction.html)
- 安装页面：[FaKnow Installation](https://faknow.readthedocs.io/en/latest/get_started/installation.html)

然后执行：

```bash
cd /path/to/your/repo
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install faknow==0.0.4
pip install torch torchvision torchaudio
pip install transformers jieba pandas pillow scipy scikit-learn tensorboard
```

设置数据路径并一键跑三数据集七模型：

```bash
export CFND_ROOT=/data/CFND_dataset
export GOSSIP_ROOT=/data/gossip
export WEIBO_ROOT=/data/weibo
export GOSSIP_IMAGE_ROOT=/data/AAAI_dataset
export DEVICE=cuda
export EPOCHS=5
export BATCH_SIZE=32

bash train/faknow_adapter/run_cloud_all.sh
```

结果位置：

- 工件：`train/faknow_adapter/artifacts`
- 日志与指标：`train/faknow_adapter/runs`
- 总报告：`train/faknow_adapter/runs/run_report.json`

## 11. 与 final_train_* 脚本配合

`final_train_CFND.py / final_train_gossip.py / final_train_weibo.py` 已支持通过环境变量覆盖关键路径和超参，不用再改代码文件。

示例（CFND）：

```bash
export CFND_DATASET_ROOT=/data/CFND_dataset
export CFND_SAVE_ROOT=/data/exp_outputs/cfnd
export CFND_BATCH_SIZE=32
export CFND_EPOCHS=20
python train/final_train_CFND.py
```

示例（Gossip）：

```bash
export GOSSIP_DATASET_ROOT=/data/AAAI_dataset
export GOSSIP_PROCESSED_DIR=/data/gossip_processed
export GOSSIP_SAVE_ROOT=/data/exp_outputs/gossip
python train/final_train_gossip.py
```

示例（Weibo）：

```bash
export WEIBO_DATASET_ROOT=/data/weibo
export WEIBO_PROCESSED_DIR=/data/weibo
export WEIBO_SAVE_ROOT=/data/exp_outputs/weibo
python train/final_train_weibo.py
```
