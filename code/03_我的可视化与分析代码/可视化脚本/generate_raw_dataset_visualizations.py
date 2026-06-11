"""原始数据集可视化脚本：生成图表并导出对应统计数据。"""

import argparse
import html
import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import jieba
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from wordcloud import WordCloud


EN_STOPWORDS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "is",
    "are",
    "was",
    "were",
    "to",
    "of",
    "in",
    "on",
    "for",
    "with",
    "as",
    "at",
    "by",
    "from",
    "that",
    "this",
    "it",
    "its",
    "be",
    "been",
    "has",
    "have",
    "had",
    "will",
    "would",
    "can",
    "could",
    "should",
    "about",
    "into",
    "over",
    "after",
    "before",
    "up",
    "down",
    "out",
    "off",
    "not",
    "no",
    "you",
    "your",
    "he",
    "she",
    "they",
    "them",
    "we",
    "our",
    "i",
    "me",
    "my",
}

ZH_STOPWORDS = {
    "的",
    "了",
    "和",
    "是",
    "在",
    "就",
    "都",
    "而",
    "及",
    "与",
    "着",
    "或",
    "一个",
    "没有",
    "我们",
    "你们",
    "他们",
    "她们",
    "它们",
    "自己",
    "以及",
    "进行",
    "这个",
    "那个",
    "这种",
    "那种",
    "因为",
    "所以",
    "但是",
    "如果",
    "然后",
    "并且",
}

CJK_RE = re.compile(r"[\u4e00-\u9fff]")
EN_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_']+")
URL_RE = re.compile(r"https?://\S+|www\.\S+")
HTML_RE = re.compile(r"<.*?>")
HTML_ENTITY_RE = re.compile(r"&[#A-Za-z0-9]+;")

NOISE_TOKENS = {
    "nbsp",
    "quot",
    "amp",
    "lt",
    "gt",
    "ldquo",
    "rdquo",
    "apos",
    "url",
}


def clean_text(text: object) -> str:
    if text is None or (isinstance(text, float) and np.isnan(text)):
        value = ""
    else:
        value = str(text)
    # 先解码 HTML 实体，避免双重转义内容无法被后续规则清理。
    value = html.unescape(value)
    value = html.unescape(value)
    value = value.replace("\u200b", " ")
    value = value.replace("\xa0", " ")
    value = HTML_RE.sub(" ", value)
    value = HTML_ENTITY_RE.sub(" ", value)
    value = URL_RE.sub(" URL ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def tokenize_text(text: str, lang: str) -> List[str]:
    text = clean_text(text)
    if not text:
        return []

    if lang == "en":
        tokens = [tok.lower() for tok in EN_TOKEN_RE.findall(text.lower())]
        return [tok for tok in tokens if tok not in EN_STOPWORDS and tok not in NOISE_TOKENS and len(tok) > 1]

    # 中文/混合文本：使用 jieba 分词并结合英文词规则过滤
    raw_tokens = [tok.strip().lower() for tok in jieba.lcut(text) if tok.strip()]
    tokens: List[str] = []
    for tok in raw_tokens:
        if tok in ZH_STOPWORDS:
            continue
        if tok in NOISE_TOKENS:
            continue
        if tok.isdigit():
            continue
        if CJK_RE.search(tok):
            if len(tok) >= 1:
                tokens.append(tok)
            continue
        if EN_TOKEN_RE.fullmatch(tok) and tok not in EN_STOPWORDS and len(tok) > 1:
            tokens.append(tok)
    return tokens


def detect_chinese_font() -> str | None:
    candidates = [
        Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
        Path(r"C:\Windows\Fonts\simsun.ttc"),
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    return None


def configure_matplotlib_fonts() -> None:
    # 解决中文图表文字显示问题（坐标轴、标题、词云等）。
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def save_json(data: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def plot_label_distribution(df: pd.DataFrame, label_col: str, label_name_map: Dict[int, str], out_dir: Path) -> pd.DataFrame:
    counts = df[label_col].value_counts(dropna=False).sort_index()
    total = int(len(df))
    out = pd.DataFrame(
        {
            "label": counts.index.astype(int),
            "label_name": [label_name_map.get(int(x), f"class_{int(x)}") for x in counts.index],
            "count": counts.values.astype(int),
            "ratio": (counts.values / max(total, 1)).astype(float),
        }
    )

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    x = np.arange(len(out))
    bars = ax.bar(x, out["count"].values, color=["#4C72B0", "#DD8452"][: len(out)])
    ax.set_xticks(x)
    ax.set_xticklabels(out["label_name"].tolist())
    ax.set_ylabel("Count")
    ax.set_title("Label Distribution")
    ax.grid(axis="y", alpha=0.25)
    for i, b in enumerate(bars):
        cnt = int(out.iloc[i]["count"])
        ratio = float(out.iloc[i]["ratio"])
        ax.text(b.get_x() + b.get_width() / 2.0, b.get_height(), f"{cnt}\n({ratio:.1%})", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "label_distribution.png", dpi=180)
    plt.close(fig)
    return out


def plot_text_length_distribution(length_df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    bins_char = np.histogram_bin_edges(length_df["char_len"], bins=40)
    bins_token = np.histogram_bin_edges(length_df["token_len"], bins=40)
    char_hist, char_edges = np.histogram(length_df["char_len"], bins=bins_char)
    tok_hist, tok_edges = np.histogram(length_df["token_len"], bins=bins_token)

    hist_df = pd.DataFrame(
        {
            "char_bin_left": char_edges[:-1],
            "char_bin_right": char_edges[1:],
            "char_count": char_hist,
        }
    )
    tok_df = pd.DataFrame(
        {
            "token_bin_left": tok_edges[:-1],
            "token_bin_right": tok_edges[1:],
            "token_count": tok_hist,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    axes[0].hist(length_df["char_len"], bins=40, color="#4C72B0", alpha=0.85)
    axes[0].set_title("Text Length Distribution (Characters)")
    axes[0].set_xlabel("Character Length")
    axes[0].set_ylabel("Count")
    axes[0].grid(alpha=0.2)

    axes[1].hist(length_df["token_len"], bins=40, color="#DD8452", alpha=0.85)
    axes[1].set_title("Text Length Distribution (Tokens)")
    axes[1].set_xlabel("Token Length")
    axes[1].set_ylabel("Count")
    axes[1].grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_dir / "text_length_distribution.png", dpi=180)
    plt.close(fig)

    char_fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.hist(length_df["char_len"], bins=40, color="#4C72B0", alpha=0.9)
    ax.set_title("Text Length Distribution (Characters)")
    ax.set_xlabel("Character Length")
    ax.set_ylabel("Count")
    ax.grid(alpha=0.2)
    char_fig.tight_layout()
    char_fig.savefig(out_dir / "text_length_char_hist.png", dpi=180)
    plt.close(char_fig)

    tok_fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.hist(length_df["token_len"], bins=40, color="#DD8452", alpha=0.9)
    ax.set_title("Text Length Distribution (Tokens)")
    ax.set_xlabel("Token Length")
    ax.set_ylabel("Count")
    ax.grid(alpha=0.2)
    tok_fig.tight_layout()
    tok_fig.savefig(out_dir / "text_length_token_hist.png", dpi=180)
    plt.close(tok_fig)

    hist_merged = hist_df.merge(tok_df, left_index=True, right_index=True, how="outer")
    return hist_merged


def build_top_words(token_records: Iterable[Tuple[int, List[str]]], topn: int = 20) -> Tuple[Dict[int, Counter], pd.DataFrame]:
    counters: Dict[int, Counter] = {}
    for label, tokens in token_records:
        if label not in counters:
            counters[label] = Counter()
        counters[label].update(tokens)

    rows = []
    for label, counter in counters.items():
        for rank, (word, count) in enumerate(counter.most_common(topn), start=1):
            rows.append({"label": int(label), "rank": rank, "word": word, "count": int(count)})
    out_df = pd.DataFrame(rows).sort_values(["label", "rank"]).reset_index(drop=True)
    return counters, out_df


def plot_top20_words(top_df: pd.DataFrame, label_name_map: Dict[int, str], out_path: Path) -> None:
    labels = sorted(top_df["label"].unique().tolist())
    fig, axes = plt.subplots(1, len(labels), figsize=(7 * max(len(labels), 1), 7))
    if len(labels) == 1:
        axes = [axes]
    for ax, label in zip(axes, labels):
        sub = top_df[top_df["label"] == label].sort_values("count", ascending=True)
        ax.barh(sub["word"], sub["count"], color="#4C72B0")
        ax.set_title(f"Top 20 Words - {label_name_map.get(int(label), f'class_{label}')}")
        ax.set_xlabel("Count")
        ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def generate_wordcloud(counter: Counter, out_path: Path, lang: str) -> None:
    if not counter:
        return
    font_path = detect_chinese_font() if lang == "zh" else None
    wc = WordCloud(
        width=1600,
        height=1000,
        background_color="white",
        max_words=300,
        collocations=False,
        font_path=font_path,
    )
    wc.generate_from_frequencies(dict(counter))
    wc.to_file(str(out_path))


def prepare_dataset_frame(root: Path, files: Sequence[Tuple[str, str]], text_col: str, label_col: str) -> pd.DataFrame:
    frames = []
    for split, file_name in files:
        csv_path = root / file_name
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing csv: {csv_path}")
        df = pd.read_csv(csv_path, keep_default_na=False)
        if text_col not in df.columns:
            raise KeyError(f"{csv_path} missing text column `{text_col}`")
        if label_col not in df.columns:
            raise KeyError(f"{csv_path} missing label column `{label_col}`")
        out = pd.DataFrame(
            {
                "split": split,
                "text": df[text_col].astype(str),
                "label": pd.to_numeric(df[label_col], errors="coerce").fillna(-1).astype(int),
            }
        )
        frames.append(out)
    merged = pd.concat(frames, ignore_index=True)
    merged = merged[merged["label"].isin([0, 1])].reset_index(drop=True)
    return merged


def process_one_dataset(
    dataset_name: str,
    root: Path,
    files: Sequence[Tuple[str, str]],
    text_col: str,
    label_col: str,
    lang: str,
    label_name_map: Dict[int, str],
    output_root: Path,
) -> None:
    logging.info("[%s] Loading csv files from %s", dataset_name, root)
    df = prepare_dataset_frame(root, files, text_col, label_col)
    logging.info("[%s] Samples: %d", dataset_name, len(df))

    ds_dir = output_root / dataset_name
    fig_dir = ds_dir / "figures"
    data_dir = ds_dir / "data"
    fig_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    # 1) 标签分布图与对应统计表
    label_dist_df = plot_label_distribution(df, "label", label_name_map, fig_dir)
    label_dist_df.to_csv(data_dir / "label_distribution.csv", index=False, encoding="utf-8-sig")

    # 2) 文本清洗、分词、长度统计（字符长度与 token 长度）
    char_lens: List[int] = []
    token_lens: List[int] = []
    token_records: List[Tuple[int, List[str]]] = []
    rows: List[Dict[str, object]] = []
    processed_rows: List[Dict[str, object]] = []
    for idx, row in df.iterrows():
        raw_text = str(row["text"])
        text = clean_text(raw_text)
        tokens = tokenize_text(text, lang)
        char_len = len(text)
        token_len = len(tokens)
        lbl = int(row["label"])
        char_lens.append(char_len)
        token_lens.append(token_len)
        token_records.append((lbl, tokens))
        rows.append(
            {
                "row_id": idx,
                "split": row["split"],
                "label": lbl,
                "label_name": label_name_map.get(lbl, f"class_{lbl}"),
                "char_len": char_len,
                "token_len": token_len,
            }
        )
        processed_rows.append(
            {
                "row_id": idx,
                "split": row["split"],
                "label": lbl,
                "label_name": label_name_map.get(lbl, f"class_{lbl}"),
                "raw_text": raw_text,
                "clean_text": text,
                "tokens": " ".join(tokens),
                "char_len": char_len,
                "token_len": token_len,
            }
        )

    length_df = pd.DataFrame(rows)
    length_df.to_csv(data_dir / "text_length_records.csv", index=False, encoding="utf-8-sig")
    processed_df = pd.DataFrame(processed_rows)
    processed_df.to_csv(data_dir / "processed_text_dataset.csv", index=False, encoding="utf-8-sig")
    hist_df = plot_text_length_distribution(length_df, fig_dir)
    hist_df.to_csv(data_dir / "text_length_hist_bins.csv", index=False, encoding="utf-8-sig")

    # 3) 每类 Top20 高频词及词频导出
    counters, top20_df = build_top_words(token_records, topn=20)
    top20_df["label_name"] = top20_df["label"].map(lambda x: label_name_map.get(int(x), f"class_{int(x)}"))
    top20_df.to_csv(data_dir / "top20_words_by_label.csv", index=False, encoding="utf-8-sig")
    if not top20_df.empty:
        plot_top20_words(top20_df, label_name_map, fig_dir / "top20_words_comparison.png")

    # 4) 全量词频表与每类词云
    freq_rows = []
    for label, counter in sorted(counters.items()):
        label_name = label_name_map.get(int(label), f"class_{int(label)}")
        for word, count in counter.most_common():
            freq_rows.append({"label": int(label), "label_name": label_name, "word": word, "count": int(count)})
        # 每个类别各生成一张词云图
        generate_wordcloud(counter, fig_dir / f"wordcloud_label_{label}.png", lang=lang)
    freq_df = pd.DataFrame(freq_rows)
    if not freq_df.empty:
        freq_df.to_csv(data_dir / "word_frequencies_by_label.csv", index=False, encoding="utf-8-sig")
        freq_df.groupby("label").head(500).to_csv(data_dir / "word_frequencies_top500_by_label.csv", index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame(columns=["label", "label_name", "word", "count"]).to_csv(
            data_dir / "word_frequencies_by_label.csv",
            index=False,
            encoding="utf-8-sig",
        )

    summary = {
        "dataset": dataset_name,
        "root": str(root.resolve()),
        "samples": int(len(df)),
        "label_distribution": label_dist_df.to_dict(orient="records"),
        "text_length_stats": {
            "char_mean": float(np.mean(char_lens)) if char_lens else 0.0,
            "char_median": float(np.median(char_lens)) if char_lens else 0.0,
            "char_p95": float(np.percentile(char_lens, 95)) if char_lens else 0.0,
            "token_mean": float(np.mean(token_lens)) if token_lens else 0.0,
            "token_median": float(np.median(token_lens)) if token_lens else 0.0,
            "token_p95": float(np.percentile(token_lens, 95)) if token_lens else 0.0,
        },
        "outputs": {
            "figures_dir": str(fig_dir.resolve()),
            "data_dir": str(data_dir.resolve()),
        },
    }
    save_json(summary, data_dir / "summary.json")
    logging.info("[%s] Done. Outputs: %s", dataset_name, ds_dir.resolve())


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate raw dataset visualizations and export source data.")
    parser.add_argument("--workspace-root", type=str, default=".", help="Root of project workspace containing weibo/gossip folders.")
    parser.add_argument("--output-root", type=str, default=str(Path(__file__).resolve().parent), help="Directory to save figures and csv/json outputs.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    configure_matplotlib_fonts()
    workspace_root = Path(args.workspace_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    dataset_specs = [
        {
            "name": "weibo",
            "root": workspace_root / "weibo",
            "files": [("train", "train_weibo.csv"), ("val", "val_weibo.csv"), ("test", "test_weibo.csv")],
            "text_col": "text",
            "label_col": "label",
            "lang": "zh",
            "label_name_map": {0: "nonrumor", 1: "rumor"},
        },
        {
            "name": "gossip",
            "root": workspace_root / "gossip",
            "files": [("train", "train_gossip.csv"), ("val", "val_gossip.csv"), ("test", "test_gossip.csv")],
            "text_col": "text",
            "label_col": "label",
            "lang": "en",
            "label_name_map": {0: "real", 1: "fake"},
        },
        {
            "name": "cfnd",
            "root": workspace_root.parent / "CFND_dataset",
            "files": [("train", "train_data_clean.csv"), ("val", "val_data.csv"), ("test", "test_data.csv")],
            "text_col": "title",
            "label_col": "label",
            "lang": "zh",
            "label_name_map": {0: "real", 1: "fake"},
        },
    ]

    for spec in dataset_specs:
        process_one_dataset(
            dataset_name=spec["name"],
            root=spec["root"],
            files=spec["files"],
            text_col=spec["text_col"],
            label_col=spec["label_col"],
            lang=spec["lang"],
            label_name_map=spec["label_name_map"],
            output_root=output_root,
        )

    logging.info("All dataset visualizations finished.")


if __name__ == "__main__":
    main()
