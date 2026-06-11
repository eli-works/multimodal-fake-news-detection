# 原始数据集可视化说明文档

## 1. 文档目标

本文档用于说明 `generate_raw_dataset_visualizations.py` 的真实处理逻辑、统计口径、输出文件含义和复现方法，确保你在论文或报告中描述可视化流程时与实际代码一致。

脚本路径：

- `C:\Users\gan\Desktop\code\viz\generate_raw_dataset_visualizations.py`

当前支持的数据集：

- Weibo（中文）
- Gossip（英文）
- CFND（中文）

---

## 2. 运行环境与依赖

脚本依赖：

- `python`
- `pandas`
- `numpy`
- `matplotlib`
- `jieba`
- `wordcloud`

中文词云和中文坐标轴显示依赖系统字体（脚本自动尝试）：

- `C:\Windows\Fonts\msyh.ttc`
- `C:\Windows\Fonts\simhei.ttf`
- `C:\Windows\Fonts\simsun.ttc`

---

## 3. 输入数据与字段约定

脚本将 train/val/test 三个 CSV 合并后统一统计，仅保留 `label in {0,1}` 的样本。

### 3.1 Weibo

- 根目录：`<workspace-root>\weibo`
- 文件：`train_weibo.csv`、`val_weibo.csv`、`test_weibo.csv`
- 文本列：`text`
- 标签列：`label`
- 标签语义：`0=nonrumor`，`1=rumor`

### 3.2 Gossip

- 根目录：`<workspace-root>\gossip`
- 文件：`train_gossip.csv`、`val_gossip.csv`、`test_gossip.csv`
- 文本列：`text`
- 标签列：`label`
- 标签语义：`0=real`，`1=fake`

### 3.3 CFND

- 根目录：`<workspace-root>\..\CFND_dataset`
- 文件：`train_data_clean.csv`、`val_data.csv`、`test_data.csv`
- 文本列：`title`
- 标签列：`label`
- 标签语义：`0=real`，`1=fake`

---

## 4. 文本预处理流程（已实现）

对每条文本执行如下处理：

1. 空值处理：`None/NaN` 置为空字符串。
2. HTML 实体解码：`html.unescape` 连续执行两次，处理双重转义情况。
3. 特殊空白清理：替换 `\u200b` 和 `\xa0` 为空格。
4. 去 HTML 标签：正则 `<.*?>`。
5. 去残留 HTML 实体：正则 `&[#A-Za-z0-9]+;`。
6. URL 归一化：将 URL 替换为占位符 `URL`。
7. 空白归一化：多空格折叠并 `strip()`。

这一步已经用于当前可视化结果，因此 `nbsp`、`quot` 等噪声不会进入词频统计和词云。

---

## 5. 分词与过滤规则

### 5.1 英文数据（Gossip）

- 分词规则：`[A-Za-z][A-Za-z0-9_']+`
- 全部转小写
- 过滤：
  - 英文停用词（如 `the/and/is/...`）
  - 噪声词：`nbsp`、`quot`、`amp`、`lt`、`gt`、`ldquo`、`rdquo`、`apos`、`url`
  - 长度 `<=1` 的 token

### 5.2 中文数据（Weibo/CFND）

- 分词器：`jieba.lcut`
- 过滤：
  - 中文停用词（脚本内 `ZH_STOPWORDS`）
  - 上述噪声词集合
  - 纯数字 token
- 保留：
  - 含 CJK 字符的词（长度 `>=1`）
  - 合法英文词（满足英文正则，且非停用词）

---

## 6. 统计口径（重点）

`processed_text_dataset.csv` 及长度分布图中的关键口径如下：

- `raw_text`：原始文本（未清洗）
- `clean_text`：清洗后文本（按第 4 节规则）
- `tokens`：清洗后并过滤后的分词结果（空格拼接）
- `char_len`：`len(clean_text)`，即清洗后字符长度（Unicode 字符个数，不是字节数）
- `token_len`：`len(tokens)`，即最终保留 token 的数量

注意：

- 你在 `raw_text` 中仍可能看到 `&nbsp;`，这属于保留原始文本的正常现象。
- 词频统计、Top20、词云基于 `tokens`，不受 `raw_text` 噪声影响。

---

## 7. 输出目录与文件说明

每个数据集输出到：

- `C:\Users\gan\Desktop\code\viz\weibo`
- `C:\Users\gan\Desktop\code\viz\gossip`
- `C:\Users\gan\Desktop\code\viz\cfnd`

每个目录包含 `figures` 和 `data` 两个子目录。

### 7.1 figures

- `label_distribution.png`：标签分布柱状图
- `text_length_distribution.png`：字符长度/Token长度双子图
- `text_length_char_hist.png`：字符长度直方图
- `text_length_token_hist.png`：Token长度直方图
- `top20_words_comparison.png`：按类别对比的 Top20 高频词
- `wordcloud_label_0.png`：类别 0 词云
- `wordcloud_label_1.png`：类别 1 词云

### 7.2 data

- `label_distribution.csv`：标签计数与占比
- `text_length_records.csv`：每条样本的 `char_len/token_len`
- `text_length_hist_bins.csv`：长度直方图分箱统计
- `top20_words_by_label.csv`：每类 Top20 词及词频
- `word_frequencies_by_label.csv`：每类全量词频表
- `word_frequencies_top500_by_label.csv`：每类 Top500 词频
- `processed_text_dataset.csv`：每条样本的原文、清洗文、tokens 和长度
- `summary.json`：样本总量、标签分布、长度统计、输出路径

---

## 8. 当前一次生成结果（与你现有产物一致）

基于当前 `summary.json`：

- Weibo：`9527` 条（`0:4779`，`1:4748`）
- Gossip：`12840` 条（`0:2581`，`1:10259`）
- CFND：`26664` 条（`0:16394`，`1:10270`）

说明：

- CFND 总量为 `26664`，通常是因为某些来源记录被过滤（如标签非 0/1 或读入异常），这是脚本设计行为。

---

## 9. 复现命令

在项目目录 `C:\Users\gan\Desktop\dataset\社交媒体谣言检测数据集` 下执行：

```powershell
python C:\Users\gan\Desktop\code\viz\generate_raw_dataset_visualizations.py --workspace-root . --output-root C:\Users\gan\Desktop\code\viz
```

---

## 10. 论文中可直接使用的口径描述（建议）

可用描述模板：

“本文对原始文本进行统一清洗（HTML 实体解码、标签去除、URL 归一化和空白归一化），并分别统计字符长度与 token 长度分布。字符长度定义为清洗后文本的 Unicode 字符数，token 长度定义为清洗分词后保留 token 的数量。随后在各类别内统计高频词（Top20）并绘制词云，以分析真假新闻在词汇层面的差异。”

