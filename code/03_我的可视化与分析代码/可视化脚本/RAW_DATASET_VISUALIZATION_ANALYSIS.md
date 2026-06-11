# 原始数据集可视化结果分析

## 1. 分析说明

本文档基于可视化脚本生成的实际结果文件进行解读，分析对象为 Weibo、Gossip、CFND 三个数据集。  
统计口径与处理流程见：

- `C:\Users\gan\Desktop\code\viz\RAW_DATASET_VISUALIZATION_DOC.md`

本分析使用的核心数据文件包括：

- `C:\Users\gan\Desktop\code\viz\weibo\data\summary.json`
- `C:\Users\gan\Desktop\code\viz\gossip\data\summary.json`
- `C:\Users\gan\Desktop\code\viz\cfnd\data\summary.json`
- `C:\Users\gan\Desktop\code\viz\weibo\data\processed_text_dataset.csv`
- `C:\Users\gan\Desktop\code\viz\gossip\data\processed_text_dataset.csv`
- `C:\Users\gan\Desktop\code\viz\cfnd\data\processed_text_dataset.csv`
- `C:\Users\gan\Desktop\code\viz\weibo\data\top20_words_by_label.csv`
- `C:\Users\gan\Desktop\code\viz\gossip\data\top20_words_by_label.csv`
- `C:\Users\gan\Desktop\code\viz\cfnd\data\top20_words_by_label.csv`

---

## 2. 总体结论

1. Weibo 数据集类别最均衡，长度中等，几乎不存在截断问题，适合作为稳定训练与对比基线。
2. Gossip 数据集存在显著类别不平衡与超长文本长尾，若 `max_len=96`，多数样本会被截断，是模型性能波动的主要来源。
3. CFND 文本短、分布稳定、真假词汇差异较明显，适合开展可解释性分析和稳健性验证。

---

## 3. 类别分布分析

| 数据集 | 样本总数 | 类别0 | 类别1 | 多数类/少数类 |
|---|---:|---:|---:|---:|
| Weibo | 9527 | 4779 | 4748 | 1.007 |
| Gossip | 12840 | 2581 | 10259 | 3.975 |
| CFND | 26664 | 16394 | 10270 | 1.596 |

解读：

1. Weibo 接近 1:1，类别偏置影响较小。
2. Gossip 接近 1:4，若只看 accuracy 容易高估模型能力，必须同时报告 macro-F1 与分类别指标。
3. CFND 存在中等不平衡，但可通过加权损失或采样策略控制。

---

## 4. 文本长度分布分析

| 数据集 | char中位数 | char p95 | token中位数 | token p95 |
|---|---:|---:|---:|---:|
| Weibo | 128 | 156 | 51 | 67 |
| Gossip | 2056 | 10549 | 229 | 1141 |
| CFND | 20 | 30 | 9 | 13 |

补充指标：

| 数据集 | char尾部强度 p95/median | token尾部强度 p95/median |
|---|---:|---:|
| Weibo | 1.219 | 1.314 |
| Gossip | 5.131 | 4.983 |
| CFND | 1.500 | 1.444 |

解读：

1. Weibo 长度分布较集中，长尾弱。
2. Gossip 长尾极重，说明样本间文本长度差异非常大。
3. CFND 典型短文本特征，长度波动小。

---

## 5. 训练截断风险分析（以 max_len=96 为例）

| 数据集 | 截断率 token_len>96 |
|---|---:|
| Weibo | 0.0001 |
| Gossip | 0.8189 |
| CFND | 0.0000 |

分类别截断率：

- Weibo: label0=0.0002, label1=0.0000
- Gossip: label0=0.8415, label1=0.8132
- CFND: label0=0.0000, label1=0.0000

解读：

1. Weibo 和 CFND 在 `max_len=96` 下信息保留完整。
2. Gossip 在该设置下约 81.89% 样本被截断，模型主要看到“前文片段”，对长文语义理解不足，泛化压力较大。

---

## 6. 真假类别长度差异

### Weibo

- label0: char_mean=110.66, token_mean=45.53
- label1: char_mean=103.92, token_mean=41.51

结论：假新闻文本整体更短，常见“转发式/口号式”表达。

### Gossip

- label0: char_mean=3446.67, token_mean=374.99
- label1: char_mean=3407.26, token_mean=372.81

结论：真假长度分布相近，长度本身不是有效判别信号。

### CFND

- label0: char_mean=22.61, token_mean=9.64
- label1: char_mean=16.96, token_mean=7.73

结论：假新闻更短更凝练，真新闻更接近“报道标题”表达。

---

## 7. 高频词与词汇可分性分析

### 7.1 Top20 重叠度

| 数据集 | Top20重叠词数 | Jaccard |
|---|---:|---:|
| Weibo | 10 | 0.333 |
| Gossip | 18 | 0.818 |
| CFND | 7 | 0.212 |

解读：

1. Gossip 真伪类别高频词高度重叠，浅层词频特征区分度弱。
2. CFND 重叠最低，真假话题词差异更明显。
3. Weibo 处于中间水平，存在一定模板化谣言词簇。

### 7.2 典型词语现象

- Weibo 假新闻侧显著词：`爽歪歪`、`牛奶`、`肉毒`、`杆菌`、`旺仔` 等，反映“重复传播模板”特征。
- Gossip 真伪两类均大量出现通用叙事词：`her`、`his`、`but`、`their`、`who`，说明需依赖更深层语义建模。
- CFND 假新闻侧常见词：`新冠`、`病毒`、`疫苗`、`致癌`、`有毒`、`真的`；真新闻侧常见词：`发布`、`全国`、`同比`、`增长`、`专家`，语域差异明显。

---

## 8. 数据质量观察

1. 已处理 `nbsp`、`quot`、`amp` 等 HTML 实体噪声，词频统计与词云中不再出现该类噪声词。
2. Gossip 仍存在少量页面结构残词（如 `more_vert`、`bookmark_border`、`advertisement`），但总占比很低，对总体统计影响有限。
3. Weibo 假新闻重复率略高于真新闻（文本模板复用更明显），这一点与谣言传播机制相符。

---

## 9. 对建模与实验设计的启示

1. Weibo: 可采用较小 `max_len`，重点提升跨模态交互质量即可。
2. Gossip: 建议将 `max_len` 提升至 192 或 256，或采用分段编码策略；同时使用类别权重或采样平衡。
3. CFND: 当前设置较匹配，可重点关注模型可解释性与跨领域稳健性。
4. 三数据集联合比较时，建议统一报告：`macro-F1`、`AUC`、分类别 Precision/Recall/F1，避免仅看 accuracy。

---

## 10. 论文可直接使用的总结段

可写为：

“可视化分析表明，Weibo 数据集类别分布均衡且文本长度适中，在 `max_len=96` 下几乎无截断；CFND 数据集以短文本为主，真假词汇差异明显，具备较强词汇可分性。相比之下，Gossip 数据集呈现显著类别不平衡（约 1:4）与重长尾文本特征，在 `max_len=96` 设置下约 81.89% 样本发生截断，且真假类别 Top20 词高度重叠（Jaccard=0.818），说明其对模型的长文本语义建模与鲁棒性提出了更高要求。”

