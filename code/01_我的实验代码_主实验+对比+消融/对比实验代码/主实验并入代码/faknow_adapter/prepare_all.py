from __future__ import annotations

import argparse
import json
import logging
import importlib.util
import pickle
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import jieba
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.models import ResNet50_Weights, resnet50
from transformers import AutoModel, AutoTokenizer

DATASETS = ("cfnd", "gossip", "weibo")
ENCODINGS = ("utf-8-sig", "utf-8", "gb18030", "gbk", "latin1")
EN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
HTML_RE = re.compile(r"<.*?>")
URL_RE = re.compile(r"https?://\S+|www\.\S+")

DEFAULTS = {
    "cfnd_root": r"C:\Users\gan\Desktop\dataset\CFND_dataset",
    "gossip_root": r"C:\Users\gan\Desktop\dataset\社交媒体谣言检测数据集\gossip",
    "weibo_root": r"C:\Users\gan\Desktop\dataset\社交媒体谣言检测数据集\weibo",
    "gossip_image_root": r"C:\Users\gan\Desktop\dataset\AAAI_dataset",
    "text_model_zh": "hfl/chinese-bert-wwm-ext",
    "text_model_en": "bert-base-uncased",
}


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(obj: Any, path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_pickle(obj: Any, path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("wb") as f:
        pickle.dump(obj, f)


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_csv_auto(path: Path) -> pd.DataFrame:
    last = None
    for enc in ENCODINGS:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:  # noqa: PERF203
            last = e
    raise RuntimeError(f"cannot decode csv: {path}; last={last}")


def _load_module_from_file(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def find_helper_script(script_name: str) -> Path:
    base = Path(__file__).resolve().parent
    candidates = [
        base / script_name,
        base.parent / script_name,
        base.parent / "train" / script_name,
    ]
    for p in candidates:
        if p.exists():
            return p
    tried = "\n".join(str(x) for x in candidates)
    raise FileNotFoundError(f"missing helper script: {script_name}\ntried:\n{tried}")


def maybe_prepare_missing_csv(dataset: str, root: Path, cfg: argparse.Namespace) -> None:
    if not getattr(cfg, "auto_prepare_data", True) and not getattr(cfg, "force_prepare_data", False):
        return

    if dataset == "gossip":
        required = [root / "train_gossip.csv", root / "val_gossip.csv", root / "test_gossip.csv"]
        need_prepare = cfg.force_prepare_data or any(not p.exists() for p in required)
        if not need_prepare:
            return
        script = find_helper_script("prepare_gossip_data.py")
        logging.info("Preparing missing gossip csv files via %s ...", script)
        m = _load_module_from_file("prepare_gossip_data", script)
        prep_cfg = m.PreprocessConfig(
            dataset_root=str(Path(cfg.gossip_image_root).resolve()),
            output_dir=str(root),
            val_ratio=float(cfg.val_ratio),
            seed=int(cfg.seed),
        )
        m.run_preprocess(prep_cfg)
        return

    if dataset == "weibo":
        required = [root / "train_weibo.csv", root / "val_weibo.csv", root / "test_weibo.csv"]
        need_prepare = cfg.force_prepare_data or any(not p.exists() for p in required)
        if not need_prepare:
            return
        script = find_helper_script("prepare_weibo_data.py")
        logging.info("Preparing missing weibo csv files via %s ...", script)
        m = _load_module_from_file("prepare_weibo_data", script)
        prep_cfg = m.PreprocessConfig(
            dataset_root=str(root),
            output_dir=str(root),
            val_ratio=float(cfg.val_ratio),
            seed=int(cfg.seed),
        )
        m.run_preprocess(prep_cfg)
        return


def clean_text(v: Any) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        s = ""
    else:
        s = str(v)
    s = HTML_RE.sub(" ", s)
    s = URL_RE.sub(" URL ", s)
    s = s.replace("\u200b", " ").replace("\xa0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def norm_label(v: Any) -> int | None:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    s = str(v).strip().lower()
    if s in {"0", "real", "true", "nonrumor", "non_rumor", "legitimate"}:
        return 0
    if s in {"1", "fake", "false", "rumor"}:
        return 1
    try:
        x = int(float(s))
    except Exception:
        return None
    return x if x in (0, 1) else None


def make_placeholder(path: Path) -> Path:
    ensure_dir(path.parent)
    if not path.exists():
        Image.new("RGB", (224, 224), color=(255, 255, 255)).save(path, format="JPEG")
    return path


def resolve_image(rel: Any, roots: Sequence[Path], split: str, placeholder: Path) -> str:
    s = clean_text(rel).replace("\\", "/")
    if not s:
        return str(placeholder)
    p = Path(s)
    if p.is_absolute():
        return str(p if p.exists() else placeholder)
    cands: List[Path] = []
    for root in roots:
        cands.append(root / s)
        if "/" not in s and "\\" not in s:
            cands.append(root / "Images" / f"gossip_{split}" / s)
    for c in cands:
        if c.exists():
            return str(c.resolve())
    return str(placeholder)


def tokenize(text: str, lang: str) -> List[str]:
    t = clean_text(text)
    if not t:
        return []
    if lang == "zh":
        return [x.strip() for x in jieba.lcut(t) if x.strip()]
    return [x.lower() for x in EN_RE.findall(t.lower())]


def build_vocab(texts: Sequence[str], lang: str, min_freq: int) -> Dict[str, int]:
    cnt: Dict[str, int] = {}
    for t in texts:
        for w in tokenize(t, lang):
            cnt[w] = cnt.get(w, 0) + 1
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for w, c in cnt.items():
        if c >= min_freq:
            vocab[w] = len(vocab)
    return vocab


class TextEncoder:
    def __init__(self, model_name: str, device: str):
        self.model_name = model_name
        self.device = torch.device(device)
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.hidden = int(getattr(self.model.config, "hidden_size", 768))

    @torch.no_grad()
    def sentence(self, texts: Sequence[str], batch_size: int, max_len: int = 256) -> np.ndarray:
        out: List[np.ndarray] = []
        total_batches = max(1, (len(texts) + batch_size - 1) // batch_size)
        if texts:
            logging.info("TextEncoder.sentence start: samples=%d batches=%d model=%s", len(texts), total_batches, self.model_name)
        for bi, i in enumerate(range(0, len(texts), batch_size), start=1):
            batch = list(texts[i : i + batch_size])
            enc = self.tok(batch, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
            enc = {k: v.to(self.device) for k, v in enc.items()}
            h = self.model(**enc)
            if hasattr(h, "pooler_output") and h.pooler_output is not None:
                v = h.pooler_output
            else:
                hs = h.last_hidden_state
                m = enc["attention_mask"].unsqueeze(-1)
                v = (hs * m).sum(1) / m.sum(1).clamp(min=1)
            out.append(v.detach().cpu().numpy().astype(np.float32))
            if bi == 1 or bi == total_batches or bi % max(1, total_batches // 10) == 0:
                logging.info("TextEncoder.sentence progress: %d/%d batches", bi, total_batches)
        return np.concatenate(out, axis=0) if out else np.zeros((0, self.hidden), dtype=np.float32)

    @torch.no_grad()
    def sequence(self, texts: Sequence[str], seq_len: int, dim: int, batch_size: int) -> np.ndarray:
        arr = np.zeros((len(texts), seq_len, dim), dtype=np.float32)
        total_batches = max(1, (len(texts) + batch_size - 1) // batch_size)
        if texts:
            logging.info(
                "TextEncoder.sequence start: samples=%d batches=%d seq_len=%d dim=%d model=%s",
                len(texts),
                total_batches,
                seq_len,
                dim,
                self.model_name,
            )
        for bi, i in enumerate(range(0, len(texts), batch_size), start=1):
            batch = list(texts[i : i + batch_size])
            enc = self.tok(
                batch,
                padding="max_length",
                truncation=True,
                max_length=seq_len + 2,
                return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            hs = self.model(**enc).last_hidden_state[:, 1 : seq_len + 1, :]
            m = enc["attention_mask"][:, 1 : seq_len + 1].unsqueeze(-1)
            hs = (hs * m).detach().cpu().numpy().astype(np.float32)
            if hs.shape[-1] >= dim:
                hs = hs[:, :, :dim]
            else:
                pad = np.zeros((hs.shape[0], hs.shape[1], dim - hs.shape[-1]), dtype=np.float32)
                hs = np.concatenate([hs, pad], axis=-1)
            arr[i : i + len(batch)] = hs
            if bi == 1 or bi == total_batches or bi % max(1, total_batches // 10) == 0:
                logging.info("TextEncoder.sequence progress: %d/%d batches", bi, total_batches)
        return arr

    def word_vectors(self, vocab: Dict[str, int]) -> torch.Tensor:
        emb = self.model.get_input_embeddings().weight.detach().cpu()
        unk = self.tok.unk_token_id if self.tok.unk_token_id is not None else 0
        vecs = np.zeros((len(vocab), emb.shape[1]), dtype=np.float32)
        vecs[vocab["<UNK>"]] = emb[unk].numpy()
        cache: Dict[Tuple[int, ...], np.ndarray] = {}
        for w, idx in vocab.items():
            if w in {"<PAD>", "<UNK>"}:
                continue
            pcs = self.tok.tokenize(w) or [self.tok.unk_token]
            ids = tuple(int(x) for x in self.tok.convert_tokens_to_ids(pcs))
            if ids not in cache:
                cache[ids] = emb[list(ids)].mean(0).numpy().astype(np.float32)
            vecs[idx] = cache[ids]
        return torch.tensor(vecs, dtype=torch.float32)


class ImageEncoder:
    def __init__(self, device: str):
        self.device = torch.device(device)
        self.net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.net.fc = torch.nn.Identity()
        self.net.to(self.device).eval()
        self.tfms = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def _img(self, path: str) -> Image.Image:
        try:
            with Image.open(path) as img:
                return img.convert("RGB")
        except Exception:
            return Image.new("RGB", (224, 224), color=(255, 255, 255))

    @torch.no_grad()
    def encode(self, paths: Sequence[str], batch_size: int) -> np.ndarray:
        if not paths:
            return np.zeros((0, 2048), dtype=np.float32)
        out = np.zeros((len(paths), 2048), dtype=np.float32)
        total_batches = max(1, (len(paths) + batch_size - 1) // batch_size)
        logging.info("ImageEncoder.encode start: samples=%d batches=%d", len(paths), total_batches)
        for bi, i in enumerate(range(0, len(paths), batch_size), start=1):
            batch = paths[i : i + batch_size]
            x = torch.stack([self.tfms(self._img(p)) for p in batch]).to(self.device)
            out[i : i + len(batch)] = self.net(x).detach().cpu().numpy().astype(np.float32)
            if bi == 1 or bi == total_batches or bi % max(1, total_batches // 10) == 0:
                logging.info("ImageEncoder.encode progress: %d/%d batches", bi, total_batches)
        return out


def knn_graph(emb: np.ndarray, k: int, chunk: int = 512) -> Dict[str, List[int]]:
    if emb.size == 0:
        return {}
    x = torch.tensor(emb, dtype=torch.float32)
    x = F.normalize(x, p=2, dim=1)
    n = x.shape[0]
    k = max(1, min(k, max(1, n - 1)))
    nb: List[set[int]] = [set([i]) for i in range(n)]
    for st in range(0, n, chunk):
        ed = min(st + chunk, n)
        sim = x[st:ed] @ x.t()
        idx = torch.topk(sim, k=min(k + 1, n), dim=1).indices.cpu().numpy()
        for ii, arr in enumerate(idx):
            i = st + ii
            for j in arr.tolist():
                if j == i:
                    continue
                nb[i].add(j)
                nb[j].add(i)
    return {str(i): sorted(list(v)) for i, v in enumerate(nb)}


def read_dataset(dataset: str, cfg: argparse.Namespace, placeholder: Path) -> Dict[str, List[Dict[str, Any]]]:
    if dataset == "cfnd":
        root = Path(cfg.cfnd_root).resolve()
        files = {"train": "train_data_clean.csv", "val": "val_data.csv", "test": "test_data.csv"}
        txt_col, img_col, id_col = "title", "image", "num"
        roots = [root]
    elif dataset == "gossip":
        root = Path(cfg.gossip_root).resolve()
        maybe_prepare_missing_csv("gossip", root, cfg)
        files = {"train": "train_gossip.csv", "val": "val_gossip.csv", "test": "test_gossip.csv"}
        txt_col, img_col, id_col = "text", "image", "sample_id"
        roots = [root, Path(cfg.gossip_image_root).resolve()]
    elif dataset == "weibo":
        root = Path(cfg.weibo_root).resolve()
        maybe_prepare_missing_csv("weibo", root, cfg)
        files = {"train": "train_weibo.csv", "val": "val_weibo.csv", "test": "test_weibo.csv"}
        txt_col, img_col, id_col = "text", "image", "sample_id"
        roots = [root]
    else:
        raise ValueError(dataset)

    out: Dict[str, List[Dict[str, Any]]] = {}
    for split, fn in files.items():
        df = read_csv_auto(root / fn)
        rows: List[Dict[str, Any]] = []
        for i, row in df.iterrows():
            y = norm_label(row.get("label"))
            if y is None:
                continue
            sid = clean_text(row.get(id_col, "")) or f"{dataset}_{split}_{i}"
            text = clean_text(row.get(txt_col, row.get("content", row.get("origin_text", ""))))
            img = resolve_image(row.get(img_col, ""), roots, split, placeholder)
            raw_post = clean_text(row.get("post_id", row.get("raw_id", sid)))
            rows.append({"sample_id": sid, "text": text, "image_abs_path": img, "label": int(y), "raw_post_id": raw_post})
        out[split] = rows
    return out


def dump_standard_snapshot(out_dir: Path, samples: Dict[str, List[Dict[str, Any]]]) -> None:
    std = ensure_dir(out_dir / "standard")
    for split, rows in samples.items():
        save_json(rows, std / f"{split}.json")


def write_json_artifacts(out_dir: Path, samples: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    meta: Dict[str, Any] = {}

    for model in ("spotfake", "hmcan", "mcan", "eann"):
        mdir = ensure_dir(out_dir / model)
        meta[model] = {}
        for split, rows in samples.items():
            if model == "spotfake":
                recs = [{"post_text": r["text"], "image_id": r["image_abs_path"], "label": r["label"]} for r in rows]
            elif model == "eann":
                recs = [{"text": r["text"], "image": r["image_abs_path"], "domain": 0, "label": r["label"]} for r in rows]
            else:
                recs = [{"text": r["text"], "image": r["image_abs_path"], "label": r["label"]} for r in rows]
            p = mdir / f"{split}.json"
            save_json(recs, p)
            meta[model][split] = str(p)

    # mfan with contiguous post_id
    post_id_map: Dict[str, int] = {}
    next_id = 0
    for split in ("train", "val", "test"):
        for r in samples.get(split, []):
            k = r["raw_post_id"] or r["sample_id"]
            if k not in post_id_map:
                post_id_map[k] = next_id
                next_id += 1
    mdir = ensure_dir(out_dir / "mfan")
    meta["mfan"] = {}
    for split, rows in samples.items():
        recs = []
        for r in rows:
            k = r["raw_post_id"] or r["sample_id"]
            recs.append({"post_id": int(post_id_map[k]), "text": r["text"], "image": r["image_abs_path"], "label": r["label"]})
        p = mdir / f"{split}.json"
        save_json(recs, p)
        meta["mfan"][split] = str(p)
    save_json(post_id_map, mdir / "post_id_map.json")
    meta["mfan"]["post_id_map"] = str(mdir / "post_id_map.json")
    return meta


def write_stopwords(path: Path, lang: str) -> None:
    zh = ["的", "了", "和", "是", "在", "也", "就", "都", "与", "或", "及", "一个", "没有"]
    en = ["a", "an", "the", "and", "or", "is", "are", "to", "of", "in", "for", "on", "at", "with", "by"]
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        f.write("\n".join(zh if lang == "zh" else en))


def prepare_eann(out_dir: Path, samples: Dict[str, List[Dict[str, Any]]], lang: str, text_enc: TextEncoder, min_freq: int) -> Dict[str, Any]:
    eann_dir = ensure_dir(out_dir / "eann")
    vocab = build_vocab([r["text"] for r in samples["train"] + samples.get("val", [])], lang, min_freq=min_freq)
    vec = text_enc.word_vectors(vocab)
    save_pickle(vocab, eann_dir / "vocab.pkl")
    save_pickle(vec, eann_dir / "word_vectors.pkl")
    write_stopwords(eann_dir / "stop_words.txt", lang)
    return {
        "vocab_path": str(eann_dir / "vocab.pkl"),
        "word_vectors_path": str(eann_dir / "word_vectors.pkl"),
        "stop_words_path": str(eann_dir / "stop_words.txt"),
        "vocab_size": len(vocab),
        "vector_dim": int(vec.shape[1]),
    }


def prepare_mfan(
    out_dir: Path,
    samples: Dict[str, List[Dict[str, Any]]],
    lang: str,
    text_enc: TextEncoder,
    min_freq: int,
    knn_k: int,
    batch_size: int,
) -> Dict[str, Any]:
    mfan_dir = ensure_dir(out_dir / "mfan")
    with (mfan_dir / "post_id_map.json").open("r", encoding="utf-8") as f:
        post_map: Dict[str, int] = json.load(f)

    vocab = build_vocab([r["text"] for r in samples["train"] + samples.get("val", [])], lang, min_freq=min_freq)
    word_vec = text_enc.word_vectors(vocab)
    save_pickle(vocab, mfan_dir / "vocab.pkl")
    save_pickle(word_vec, mfan_dir / "word_vectors.pkl")

    node_texts = [""] * len(post_map)
    for split in ("train", "val", "test"):
        for r in samples.get(split, []):
            idx = int(post_map[r["raw_post_id"] or r["sample_id"]])
            if not node_texts[idx]:
                node_texts[idx] = r["text"]
    node_texts = [x if x else "empty" for x in node_texts]
    node_emb_np = text_enc.sentence(node_texts, batch_size=batch_size, max_len=256)
    node_emb = torch.tensor(node_emb_np, dtype=torch.float32)
    adj = knn_graph(node_emb_np, k=knn_k)
    save_pickle(node_emb, mfan_dir / "node_embedding.pkl")
    save_json(adj, mfan_dir / "adjacency.json")

    return {
        "vocab_path": str(mfan_dir / "vocab.pkl"),
        "word_vectors_path": str(mfan_dir / "word_vectors.pkl"),
        "node_embedding_path": str(mfan_dir / "node_embedding.pkl"),
        "adjacency_path": str(mfan_dir / "adjacency.json"),
        "node_num": int(node_emb_np.shape[0]),
        "node_dim": int(node_emb_np.shape[1]),
        "vocab_size": len(vocab),
        "vector_dim": int(word_vec.shape[1]),
    }


def build_safe_arrays(
    rows: Sequence[Dict[str, Any]],
    text_enc: TextEncoder,
    img_enc: ImageEncoder,
    batch_size: int,
    head_len: int,
    body_len: int,
    image_len: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    texts = [r["text"] for r in rows]
    labels = np.asarray([r["label"] for r in rows], dtype=np.float32)
    head = text_enc.sequence(texts, seq_len=head_len, dim=300, batch_size=batch_size)
    body = text_enc.sequence(texts, seq_len=body_len, dim=300, batch_size=batch_size)
    img = img_enc.encode([r["image_abs_path"] for r in rows], batch_size=batch_size)
    if img.shape[1] >= 300:
        img = img[:, :300]
    else:
        img = np.concatenate([img, np.zeros((img.shape[0], 300 - img.shape[1]), dtype=np.float32)], axis=1)
    image_seq = np.repeat(img[:, None, :], image_len, axis=1).astype(np.float32)
    return head.astype(np.float32), body.astype(np.float32), image_seq, labels


def write_safe_split(path: Path, head: np.ndarray, body: np.ndarray, image: np.ndarray, labels: np.ndarray) -> None:
    ensure_dir(path)
    np.save(path / "case_headline.npy", head.astype(np.float32))
    np.save(path / "case_body.npy", body.astype(np.float32))
    np.save(path / "case_image.npy", image.astype(np.float32))
    np.save(path / "case_y_fn_dim1.npy", labels.astype(np.float32))


def prepare_safe(out_dir: Path, samples: Dict[str, List[Dict[str, Any]]], text_enc: TextEncoder, img_enc: ImageEncoder, cfg: argparse.Namespace) -> Dict[str, Any]:
    sdir = ensure_dir(out_dir / "safe")
    train_rows = list(samples["train"]) + list(samples.get("val", []))
    test_rows = list(samples["test"])
    tr_h, tr_b, tr_i, tr_y = build_safe_arrays(
        train_rows,
        text_enc=text_enc,
        img_enc=img_enc,
        batch_size=cfg.batch_size,
        head_len=cfg.safe_head_len,
        body_len=cfg.safe_body_len,
        image_len=cfg.safe_image_len,
    )
    te_h, te_b, te_i, te_y = build_safe_arrays(
        test_rows,
        text_enc=text_enc,
        img_enc=img_enc,
        batch_size=cfg.batch_size,
        head_len=cfg.safe_head_len,
        body_len=cfg.safe_body_len,
        image_len=cfg.safe_image_len,
    )
    write_safe_split(sdir / "train", tr_h, tr_b, tr_i, tr_y)
    write_safe_split(sdir / "test", te_h, te_b, te_i, te_y)
    return {
        "train_dir": str(sdir / "train"),
        "test_dir": str(sdir / "test"),
        "train_shape": {"head": list(tr_h.shape), "body": list(tr_b.shape), "image": list(tr_i.shape), "label": list(tr_y.shape)},
        "test_shape": {"head": list(te_h.shape), "body": list(te_b.shape), "image": list(te_i.shape), "label": list(te_y.shape)},
    }


def build_cafe_arrays(
    rows: Sequence[Dict[str, Any]],
    text_enc: TextEncoder,
    img_enc: ImageEncoder,
    batch_size: int,
    text_len: int,
    text_dim: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    txt = text_enc.sequence([r["text"] for r in rows], seq_len=text_len, dim=text_dim, batch_size=batch_size)
    img = img_enc.encode([r["image_abs_path"] for r in rows], batch_size=batch_size)
    if img.shape[1] >= 512:
        img = img[:, :512]
    else:
        img = np.concatenate([img, np.zeros((img.shape[0], 512 - img.shape[1]), dtype=np.float32)], axis=1)
    y = np.asarray([r["label"] for r in rows], dtype=np.int64)
    return txt.astype(np.float32), img.astype(np.float32), y


def prepare_cafe(out_dir: Path, samples: Dict[str, List[Dict[str, Any]]], text_enc: TextEncoder, img_enc: ImageEncoder, cfg: argparse.Namespace) -> Dict[str, Any]:
    cdir = ensure_dir(out_dir / "cafe")
    train_rows = list(samples["train"]) + list(samples.get("val", []))
    test_rows = list(samples["test"])
    tr_t, tr_i, tr_y = build_cafe_arrays(
        train_rows,
        text_enc=text_enc,
        img_enc=img_enc,
        batch_size=cfg.batch_size,
        text_len=cfg.cafe_text_len,
        text_dim=cfg.cafe_text_dim,
    )
    te_t, te_i, te_y = build_cafe_arrays(
        test_rows,
        text_enc=text_enc,
        img_enc=img_enc,
        batch_size=cfg.batch_size,
        text_len=cfg.cafe_text_len,
        text_dim=cfg.cafe_text_dim,
    )
    # Keep both FaKnow's expected keys (`data`, `label`) and explicit contract keys.
    np.savez(cdir / "train_text_with_label.npz", data=tr_t, text_data=tr_t, label=tr_y)
    np.savez(cdir / "train_image_with_label.npz", data=tr_i, image_data=tr_i)
    np.savez(cdir / "test_text_with_label.npz", data=te_t, text_data=te_t, label=te_y)
    np.savez(cdir / "test_image_with_label.npz", data=te_i, image_data=te_i)
    return {
        "dataset_dir": str(cdir),
        "train_shape": {"text": list(tr_t.shape), "image": list(tr_i.shape), "label": list(tr_y.shape)},
        "test_shape": {"text": list(te_t.shape), "image": list(te_i.shape), "label": list(te_y.shape)},
    }


def split_stats(samples: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for split, rows in samples.items():
        labels = [r["label"] for r in rows]
        out[split] = {
            "count": len(rows),
            "label_0": int(sum(1 for x in labels if x == 0)),
            "label_1": int(sum(1 for x in labels if x == 1)),
            "image_exists": int(sum(1 for r in rows if Path(r["image_abs_path"]).exists())),
        }
    return out


def validate_images(samples: Dict[str, List[Dict[str, Any]]], n: int) -> Dict[str, int]:
    all_paths = [r["image_abs_path"] for rows in samples.values() for r in rows]
    if not all_paths:
        return {"checked": 0, "missing": 0}
    k = min(n, len(all_paths))
    picked = random.sample(all_paths, k=k)
    missing = sum(1 for p in picked if not Path(p).exists())
    return {"checked": k, "missing": int(missing)}


def text_model_for(dataset: str, cfg: argparse.Namespace) -> str:
    return cfg.text_model_en if dataset == "gossip" else cfg.text_model_zh


def lang_for(dataset: str) -> str:
    return "en" if dataset == "gossip" else "zh"


def prepare_one(
    dataset: str,
    cfg: argparse.Namespace,
    img_enc: ImageEncoder | None,
    placeholder: Path,
) -> Dict[str, Any]:
    ds_out = ensure_dir(Path(cfg.out).resolve() / dataset)
    logging.info("[%s] reading dataset ...", dataset)
    rows = read_dataset(dataset, cfg, placeholder)
    logging.info("[%s] writing standard/json artifacts ...", dataset)
    dump_standard_snapshot(ds_out, rows)
    json_meta = write_json_artifacts(ds_out, rows)
    if cfg.quick_prepare:
        meta = {
            "dataset": dataset,
            "quick_prepare": True,
            "split_stats": split_stats(rows),
            "image_check": validate_images(rows, cfg.validate_image_samples),
            "json_artifacts": json_meta,
            "eann": {"skipped": True, "reason": "quick_prepare"},
            "mfan": {"skipped": True, "reason": "quick_prepare"},
            "safe": {"skipped": True, "reason": "quick_prepare"},
            "cafe": {"skipped": True, "reason": "quick_prepare"},
        }
        save_json(meta, ds_out / "metadata.json")
        return meta

    if img_enc is None:
        raise RuntimeError("img_enc is required when quick_prepare=False")

    tmodel = text_model_for(dataset, cfg)
    lang = lang_for(dataset)
    tenc = TextEncoder(tmodel, device=cfg.device)
    logging.info("[%s] preparing EANN artifacts ...", dataset)
    eann_meta = prepare_eann(ds_out, rows, lang=lang, text_enc=tenc, min_freq=cfg.min_freq)
    logging.info("[%s] preparing MFAN artifacts ...", dataset)
    mfan_meta = prepare_mfan(
        ds_out,
        rows,
        lang=lang,
        text_enc=tenc,
        min_freq=cfg.min_freq,
        knn_k=cfg.mfan_knn,
        batch_size=cfg.batch_size,
    )
    logging.info("[%s] preparing SAFE artifacts ...", dataset)
    safe_meta = prepare_safe(ds_out, rows, text_enc=tenc, img_enc=img_enc, cfg=cfg)
    logging.info("[%s] preparing CAFE artifacts ...", dataset)
    cafe_meta = prepare_cafe(ds_out, rows, text_enc=tenc, img_enc=img_enc, cfg=cfg)
    meta = {
        "dataset": dataset,
        "language": lang,
        "text_model": tmodel,
        "split_stats": split_stats(rows),
        "image_check": validate_images(rows, cfg.validate_image_samples),
        "json_artifacts": json_meta,
        "eann": eann_meta,
        "mfan": mfan_meta,
        "safe": safe_meta,
        "cafe": cafe_meta,
    }
    save_json(meta, ds_out / "metadata.json")
    return meta


def prepare_datasets(cfg: argparse.Namespace) -> Dict[str, Any]:
    out_root = ensure_dir(Path(cfg.out).resolve())
    placeholder = make_placeholder(out_root / "_shared" / "missing_image.jpg")
    img_enc = None if cfg.quick_prepare else ImageEncoder(cfg.device)
    targets = list(DATASETS) if cfg.dataset == "all" else [cfg.dataset]
    report = {"config": vars(cfg).copy(), "datasets": {}}
    for ds in targets:
        logging.info("Preparing %s ...", ds)
        report["datasets"][ds] = prepare_one(ds, cfg, img_enc=img_enc, placeholder=placeholder)
    save_json(report, out_root / "prepare_report.json")
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare FaKnow artifacts for CFND/Gossip/Weibo.")
    p.add_argument("--dataset", choices=[*DATASETS, "all"], default="all")
    p.add_argument("--out", type=str, default=str((Path(__file__).resolve().parent / "artifacts").resolve()))

    p.add_argument("--cfnd-root", type=str, default=DEFAULTS["cfnd_root"])
    p.add_argument("--gossip-root", type=str, default=DEFAULTS["gossip_root"])
    p.add_argument("--weibo-root", type=str, default=DEFAULTS["weibo_root"])
    p.add_argument("--gossip-image-root", type=str, default=DEFAULTS["gossip_image_root"])
    p.add_argument("--auto-prepare-data", dest="auto_prepare_data", action="store_true")
    p.add_argument("--no-auto-prepare-data", dest="auto_prepare_data", action="store_false")
    p.add_argument("--force-prepare-data", action="store_true")
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.set_defaults(auto_prepare_data=True)

    p.add_argument("--text-model-zh", type=str, default=DEFAULTS["text_model_zh"])
    p.add_argument("--text-model-en", type=str, default=DEFAULTS["text_model_en"])
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--min-freq", type=int, default=2)
    p.add_argument("--mfan-knn", type=int, default=10)

    p.add_argument("--safe-head-len", type=int, default=32)
    p.add_argument("--safe-body-len", type=int, default=128)
    p.add_argument("--safe-image-len", type=int, default=32)
    p.add_argument("--cafe-text-len", type=int, default=30)
    p.add_argument("--cafe-text-dim", type=int, default=200)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--validate-image-samples", type=int, default=50)
    p.add_argument("--quick-prepare", action="store_true", help="Only build lightweight json artifacts; skip heavy feature extraction.")
    return p.parse_args()


def main() -> None:
    setup_logging()
    cfg = parse_args()
    set_seed(cfg.seed)
    logging.info("Prepare config: %s", json.dumps(vars(cfg), ensure_ascii=False, indent=2))
    report = prepare_datasets(cfg)
    logging.info("Done. datasets=%s", list(report["datasets"].keys()))
    logging.info("Artifacts root: %s", Path(cfg.out).resolve())


if __name__ == "__main__":
    main()
