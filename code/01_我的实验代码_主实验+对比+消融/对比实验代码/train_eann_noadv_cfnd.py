"""Common training core for 3 baselines: EANN(no-adv), SpotFake, MVAE."""

from __future__ import annotations

import json
import logging
import random
import re
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support, precision_score, recall_score, roc_auc_score
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import vgg19, VGG19_Weights
from tqdm import tqdm

try:
    import jieba
except Exception:
    jieba = None

try:
    from transformers import AutoModel, AutoTokenizer
except Exception:
    AutoModel = None
    AutoTokenizer = None

URL_RE = re.compile(r"https?://\\S+|www\\.\\S+")
HTML_RE = re.compile(r"<.*?>")
EN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_']+")


@dataclass
class BaselineConfig:
    model_type: str
    run_name: str
    seed: int
    dataset_root: str
    processed_dir: str
    train_csv: str
    val_csv: str
    test_csv: str
    text_column: str
    image_column: str
    label_column: str
    class_names: Tuple[str, str]
    lang: str
    save_root: str

    max_len: int = 96
    batch_size: int = 64
    num_workers: int = 8
    epochs: int = 25
    lr: float = 1e-4
    weight_decay: float = 1e-3
    grad_clip: float = 0.8
    early_stop_patience: int = 6
    dropout: float = 0.3
    min_freq: int = 2

    embed_dim: int = 128
    text_hidden_dim: int = 128
    image_dim: int = 128
    fusion_hidden_dim: int = 128

    bert_name: str = "bert-base-uncased"
    freeze_bert: bool = False

    latent_dim: int = 64
    lambda_text_recon: float = 0.2
    lambda_image_recon: float = 0.2
    beta_kl: float = 0.02


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def clean_text(x: object) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        s = ""
    else:
        s = str(x)
    s = HTML_RE.sub(" ", s)
    s = URL_RE.sub(" URL ", s)
    s = s.replace("\u200b", " ").replace("\xa0", " ")
    s = re.sub(r"\\s+", " ", s).strip()
    return s


def tokenize(s: str, lang: str) -> List[str]:
    s = clean_text(s)
    if not s:
        return []
    if lang == "en":
        return [t.lower() for t in EN_RE.findall(s.lower())]
    if jieba is None:
        return [c for c in s if not c.isspace()]
    return [t.strip() for t in jieba.lcut(s) if t.strip()]


def resolve_path(path_str: str, base: Path) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (base / p).resolve()


def build_vocab(df: pd.DataFrame, text_col: str, min_freq: int, lang: str) -> Dict[str, int]:
    cnt: Dict[str, int] = {}
    for t in df[text_col].astype(str).tolist():
        for w in tokenize(t, lang):
            cnt[w] = cnt.get(w, 0) + 1
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for w, c in cnt.items():
        if c >= min_freq:
            vocab[w] = len(vocab)
    return vocab


class MultiModalDataset(Dataset):
    def __init__(self, df: pd.DataFrame, cfg: BaselineConfig, root: Path, is_train: bool, vocab: Dict[str, int] | None):
        self.df = df.reset_index(drop=True).copy()
        self.cfg = cfg
        self.root = root
        self.vocab = vocab
        self.default_img = Image.new("RGB", (224, 224), "white")
        self.tfms = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop((224, 224)) if is_train else transforms.CenterCrop((224, 224)),
            transforms.RandomHorizontalFlip() if is_train else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self) -> int:
        return len(self.df)

    def _load_img(self, rel: str) -> Image.Image:
        rel = clean_text(rel).replace("\\\\", "/")
        if not rel:
            return self.default_img.copy()
        p = Path(rel)
        p = p if p.is_absolute() else (self.root / p)
        if not p.exists():
            return self.default_img.copy()
        try:
            img = Image.open(p)
            if img.mode == "P" and "transparency" in img.info:
                img = img.convert("RGBA")
            return img.convert("RGB")
        except Exception:
            return self.default_img.copy()

    def _encode(self, text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        assert self.vocab is not None
        ids = [self.vocab.get(w, self.vocab["<UNK>"]) for w in tokenize(text, self.cfg.lang)[: self.cfg.max_len]]
        valid = len(ids)
        pad = self.cfg.max_len - valid
        if pad > 0:
            ids.extend([self.vocab["<PAD>"]] * pad)
        mask = [1] * valid + [0] * pad
        return torch.tensor(ids, dtype=torch.long), torch.tensor(mask, dtype=torch.bool)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        text = clean_text(row[self.cfg.text_column])
        item = {
            "image": self.tfms(self._load_img(str(row.get(self.cfg.image_column, "")))),
            "targets": torch.tensor(int(row[self.cfg.label_column]), dtype=torch.long),
            "raw_text": text,
        }
        if self.cfg.model_type in {"eann_noadv", "mvae"}:
            ids, mask = self._encode(text)
            item["texts"] = ids
            item["text_mask"] = mask
        return item


class SpotFakeCollator:
    def __init__(self, tokenizer, max_len: int):
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __call__(self, batch):
        texts = [x["raw_text"] for x in batch]
        enc = self.tokenizer(texts, truncation=True, padding="max_length", max_length=self.max_len, return_tensors="pt")
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "image": torch.stack([x["image"] for x in batch]),
            "targets": torch.stack([x["targets"] for x in batch]),
        }


class TextCNN(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(emb_dim, 64, k) for k in (1, 2, 3, 5)])
        self.proj = nn.Sequential(nn.Dropout(dropout), nn.Linear(64 * 4, out_dim), nn.ReLU())

    def forward(self, texts, mask):
        x = self.emb(texts) * mask.unsqueeze(-1).float()
        x = x.transpose(1, 2)
        feats = [F.relu(c(x)).amax(dim=2) for c in self.convs]
        return self.proj(torch.cat(feats, dim=1))


class ImageEncoder(nn.Module):
    def __init__(self, out_dim: int, dropout: float):
        super().__init__()
        try:
            net = vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
        except Exception:
            net = vgg19(weights=None)
        # Keep a penultimate visual representation instead of final class logits.
        if isinstance(net.classifier, nn.Sequential):
            layers = list(net.classifier.children())
            if len(layers) >= 2 and isinstance(layers[-1], nn.Linear):
                net.classifier = nn.Sequential(*layers[:-1])
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            feat = net(dummy)
            if isinstance(feat, (tuple, list)):
                feat = feat[0]
            feat_dim = int(feat.reshape(1, -1).size(1))
        self.backbone = net
        self.proj = nn.Sequential(nn.Linear(feat_dim, out_dim), nn.ReLU(), nn.Dropout(dropout))

    def forward(self, image):
        return self.proj(self.backbone(image))


class EANNNoAdv(nn.Module):
    def __init__(self, vocab_size: int, cfg: BaselineConfig):
        super().__init__()
        self.t = TextCNN(vocab_size, cfg.embed_dim, cfg.text_hidden_dim, cfg.dropout)
        self.v = ImageEncoder(cfg.image_dim, cfg.dropout)
        self.cls = nn.Sequential(nn.Linear(cfg.text_hidden_dim + cfg.image_dim, cfg.fusion_hidden_dim), nn.ReLU(), nn.Dropout(cfg.dropout), nn.Linear(cfg.fusion_hidden_dim, 2))

    def forward(self, texts, text_mask, image):
        return self.cls(torch.cat([self.t(texts, text_mask), self.v(image)], dim=1))


class SpotFake(nn.Module):
    def __init__(self, cfg: BaselineConfig):
        super().__init__()
        if AutoModel is None:
            raise ImportError("transformers is required for SpotFake")
        self.bert = AutoModel.from_pretrained(cfg.bert_name)
        if cfg.freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad = False
        h = int(getattr(self.bert.config, "hidden_size", 768))
        self.tp = nn.Sequential(nn.Linear(h, cfg.text_hidden_dim), nn.ReLU(), nn.Dropout(cfg.dropout))
        self.v = ImageEncoder(cfg.image_dim, cfg.dropout)
        self.cls = nn.Sequential(nn.Linear(cfg.text_hidden_dim + cfg.image_dim, cfg.fusion_hidden_dim), nn.ReLU(), nn.Dropout(cfg.dropout), nn.Linear(cfg.fusion_hidden_dim, 2))

    def forward(self, input_ids, attention_mask, image):
        x = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        return self.cls(torch.cat([self.tp(x), self.v(image)], dim=1))


class MVAE(nn.Module):
    def __init__(self, vocab_size: int, cfg: BaselineConfig):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, cfg.embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(cfg.embed_dim, cfg.text_hidden_dim // 2, batch_first=True, bidirectional=True)
        self.v = ImageEncoder(cfg.image_dim, cfg.dropout)
        self.fuse = nn.Sequential(nn.Linear(cfg.text_hidden_dim + cfg.image_dim, cfg.fusion_hidden_dim), nn.ReLU(), nn.Dropout(cfg.dropout))
        self.mu = nn.Linear(cfg.fusion_hidden_dim, cfg.latent_dim)
        self.logvar = nn.Linear(cfg.fusion_hidden_dim, cfg.latent_dim)
        self.cls = nn.Sequential(nn.Linear(cfg.latent_dim, cfg.fusion_hidden_dim), nn.ReLU(), nn.Dropout(cfg.dropout), nn.Linear(cfg.fusion_hidden_dim, 2))
        self.tdec_h = nn.Linear(cfg.latent_dim, cfg.text_hidden_dim)
        self.tdec_out = nn.Linear(cfg.text_hidden_dim, vocab_size)
        self.idec = nn.Linear(cfg.latent_dim, cfg.image_dim)

    def _text_feat(self, texts, mask):
        out, _ = self.lstm(self.emb(texts))
        m = mask.unsqueeze(-1).float()
        return (out * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)

    def forward(self, texts, text_mask, image):
        tf = self._text_feat(texts, text_mask)
        vf = self.v(image)
        h = self.fuse(torch.cat([tf, vf], dim=1))
        mu, logvar = self.mu(h), self.logvar(h)
        std = torch.exp(0.5 * logvar)
        z = mu + torch.randn_like(std) * std
        logits = self.cls(z)
        th = torch.tanh(self.tdec_h(z)).unsqueeze(1).repeat(1, texts.size(1), 1)
        tlogits = self.tdec_out(th)
        irecon = self.idec(z)
        aux = {"mu": mu, "logvar": logvar, "text_logits": tlogits, "image_recon": irecon, "image_feat": vf}
        return logits, aux


def create_model(cfg: BaselineConfig, vocab_size: int | None):
    if cfg.model_type == "eann_noadv":
        return EANNNoAdv(vocab_size, cfg)
    if cfg.model_type == "spotfake":
        return SpotFake(cfg)
    if cfg.model_type == "mvae":
        return MVAE(vocab_size, cfg)
    raise ValueError(cfg.model_type)


def compute_loss(cfg: BaselineConfig, ce, logits, targets, batch, aux):
    base = ce(logits, targets)
    if cfg.model_type != "mvae":
        return base
    t_loss = F.cross_entropy(aux["text_logits"].reshape(-1, aux["text_logits"].size(-1)), batch["texts"].reshape(-1), ignore_index=0)
    i_loss = F.mse_loss(aux["image_recon"], aux["image_feat"])
    mu, logvar = aux["mu"], aux["logvar"]
    kl = torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
    return base + cfg.lambda_text_recon * t_loss + cfg.lambda_image_recon * i_loss + cfg.beta_kl * kl


def metrics_from_probs(y_true: List[int], y_prob: List[float], th: float = 0.5):
    yt = np.asarray(y_true, dtype=np.int64)
    yp = np.asarray(y_prob, dtype=np.float32)
    pred = (yp >= th).astype(np.int64)
    cm = confusion_matrix(yt, pred, labels=[0, 1]).astype(np.int64)
    p, r, f, s = precision_recall_fscore_support(yt, pred, labels=[0, 1], zero_division=0)
    out = {
        "acc": float(accuracy_score(yt, pred)),
        "precision": float(precision_score(yt, pred, zero_division=0)),
        "recall": float(recall_score(yt, pred, zero_division=0)),
        "f1": float(f1_score(yt, pred, zero_division=0)),
        "macro_f1": float(f1_score(yt, pred, average="macro", zero_division=0)),
        "confusion_matrix": cm.tolist(),
        "per_class": {
            "class_0": {"precision": float(p[0]), "recall": float(r[0]), "f1": float(f[0]), "support": int(s[0])},
            "class_1": {"precision": float(p[1]), "recall": float(r[1]), "f1": float(f[1]), "support": int(s[1])},
        },
    }
    try:
        out["auc"] = float(roc_auc_score(yt, yp))
    except Exception:
        out["auc"] = float("nan")
    return out


def move_to_device(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [move_to_device(v, device) for v in obj]
    if isinstance(obj, tuple):
        return tuple(move_to_device(v, device) for v in obj)
    return obj


def save_cm(cm_values, class_names, out_path: Path):
    cm = np.asarray(cm_values, dtype=np.int64)
    if cm.shape != (2, 2):
        return
    plt.figure(figsize=(5.2, 4.6))
    plt.imshow(cm, cmap="Blues")
    plt.colorbar()
    plt.xticks([0, 1], list(class_names))
    plt.yticks([0, 1], list(class_names))
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(int(cm[i, j])), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def save_curves(history, run_dir: Path):
    if not history["train_loss"]:
        return
    e = np.arange(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(e, history["train_loss"], label="Train Loss")
    plt.plot(e, history["val_loss"], label="Val Loss")
    plt.legend(); plt.grid(alpha=0.2); plt.tight_layout(); plt.savefig(run_dir / "loss_curve.png", dpi=160); plt.close()
    plt.figure(figsize=(8, 5))
    plt.plot(e, history["train_acc"], label="Train Acc")
    plt.plot(e, history["val_acc"], label="Val Acc")
    plt.plot(e, history["val_f1"], label="Val F1")
    plt.legend(); plt.grid(alpha=0.2); plt.tight_layout(); plt.savefig(run_dir / "metrics_curve.png", dpi=160); plt.close()


def run_experiment(cfg: BaselineConfig):
    set_seed(cfg.seed)
    script_dir = Path(__file__).resolve().parent
    save_root = resolve_path(cfg.save_root, script_dir)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = save_root / f"{cfg.run_name}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", handlers=[logging.FileHandler(run_dir / "train.log", encoding="utf-8"), logging.StreamHandler()], force=True)
    logging.info("Run dir: %s", run_dir)
    logging.info("Config: %s", json.dumps(asdict(cfg), ensure_ascii=False, indent=2))

    dataset_root = resolve_path(cfg.dataset_root, script_dir)
    processed_dir = resolve_path(cfg.processed_dir, script_dir)
    train_csv = resolve_path(cfg.train_csv, processed_dir)
    val_csv = resolve_path(cfg.val_csv, processed_dir)
    test_csv = resolve_path(cfg.test_csv, processed_dir)

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)
    for df in (train_df, val_df, test_df):
        df[cfg.text_column] = df[cfg.text_column].fillna("").astype(str)
        df[cfg.image_column] = df[cfg.image_column].fillna("").astype(str)
        df[cfg.label_column] = pd.to_numeric(df[cfg.label_column], errors="coerce").fillna(-1).astype(int)
        df.drop(df[~df[cfg.label_column].isin([0, 1])].index, inplace=True)
        df.reset_index(drop=True, inplace=True)

    vocab = None
    tokenizer = None
    if cfg.model_type in {"eann_noadv", "mvae"}:
        vocab = build_vocab(train_df, cfg.text_column, cfg.min_freq, cfg.lang)
        with open(run_dir / "vocab.json", "w", encoding="utf-8") as f:
            json.dump(vocab, f, ensure_ascii=False, indent=2)
        logging.info("Vocab size: %d", len(vocab))
    else:
        if AutoTokenizer is None:
            raise ImportError("transformers is required for SpotFake")
        tokenizer = AutoTokenizer.from_pretrained(cfg.bert_name)

    train_ds = MultiModalDataset(train_df, cfg, dataset_root, True, vocab)
    val_ds = MultiModalDataset(val_df, cfg, dataset_root, False, vocab)
    test_ds = MultiModalDataset(test_df, cfg, dataset_root, False, vocab)

    pin = torch.cuda.is_available()
    if cfg.model_type == "spotfake":
        collate = SpotFakeCollator(tokenizer, cfg.max_len)
        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0, pin_memory=pin, collate_fn=collate)
        val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0, pin_memory=pin, collate_fn=collate)
        test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0, pin_memory=pin, collate_fn=collate)
    else:
        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=pin)
        val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=pin)
        test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=pin)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    model = create_model(cfg, len(vocab) if vocab is not None else None).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info("Params | total=%.2fM | trainable=%.2fM", total_params / 1e6, trainable_params / 1e6)

    ce = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.lr, weight_decay=cfg.weight_decay)
    sch = CosineAnnealingLR(opt, T_max=cfg.epochs)
    scaler = GradScaler("cuda", enabled=amp)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_f1": []}
    best = -1.0
    best_epoch = 0
    best_state_dict = None
    stale = 0

    def forward_batch(batch):
        if cfg.model_type == "spotfake":
            logits = model(batch["input_ids"], batch["attention_mask"], batch["image"])
            return logits, None
        if cfg.model_type == "mvae":
            return model(batch["texts"], batch["text_mask"], batch["image"])
        logits = model(batch["texts"], batch["text_mask"], batch["image"])
        return logits, None

    for ep in range(1, cfg.epochs + 1):
        model.train()
        t_loss = 0.0
        t_correct = 0
        t_total = 0
        for b in tqdm(train_loader, desc="Train", leave=False):
            b = move_to_device(b, device)
            y = b["targets"]
            opt.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=amp):
                logits, aux = forward_batch(b)
                loss = compute_loss(cfg, ce, logits, y, b, aux)
            scaler.scale(loss).backward()
            if cfg.grad_clip > 0:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt)
            scaler.update()
            pred = logits.argmax(dim=1)
            t_loss += float(loss.item())
            t_correct += int((pred == y).sum().item())
            t_total += int(y.size(0))
        sch.step()

        model.eval()
        v_loss = 0.0
        y_true, y_prob = [], []
        with torch.no_grad():
            for b in val_loader:
                b = move_to_device(b, device)
                y = b["targets"]
                with autocast("cuda", enabled=amp):
                    logits, aux = forward_batch(b)
                    loss = compute_loss(cfg, ce, logits, y, b, aux)
                p = torch.softmax(logits, dim=1)[:, 1]
                v_loss += float(loss.item())
                y_true.extend(y.cpu().tolist())
                y_prob.extend(p.cpu().tolist())

        vm = metrics_from_probs(y_true, y_prob)
        train_loss = t_loss / max(len(train_loader), 1)
        train_acc = t_correct / max(t_total, 1)
        val_loss = v_loss / max(len(val_loader), 1)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(vm["acc"])
        history["val_f1"].append(vm["f1"])

        logging.info("Epoch %d/%d | train_loss=%.4f train_acc=%.4f | val_loss=%.4f val_acc=%.4f val_prec=%.4f val_recall=%.4f val_f1=%.4f val_macro_f1=%.4f val_auc=%.4f", ep, cfg.epochs, train_loss, train_acc, val_loss, vm["acc"], vm["precision"], vm["recall"], vm["f1"], vm["macro_f1"], vm["auc"])

        if vm["macro_f1"] > best:
            best = vm["macro_f1"]
            best_epoch = ep
            stale = 0
            # Keep best weights in memory only; do not write .pth files.
            best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if cfg.early_stop_patience > 0 and stale >= cfg.early_stop_patience:
            logging.info("Early stopping at epoch %d (best epoch=%d, best macro_f1=%.4f)", ep, best_epoch, best)
            break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    model.eval()
    te_loss = 0.0
    y_true, y_prob = [], []
    with torch.no_grad():
        for b in test_loader:
            b = move_to_device(b, device)
            y = b["targets"]
            with autocast("cuda", enabled=amp):
                logits, aux = forward_batch(b)
                loss = compute_loss(cfg, ce, logits, y, b, aux)
            p = torch.softmax(logits, dim=1)[:, 1]
            te_loss += float(loss.item())
            y_true.extend(y.cpu().tolist())
            y_prob.extend(p.cpu().tolist())

    tm = metrics_from_probs(y_true, y_prob)
    te_loss = te_loss / max(len(test_loader), 1)
    logging.info("Test | loss=%.4f acc=%.4f precision=%.4f recall=%.4f f1=%.4f macro_f1=%.4f auc=%.4f", te_loss, tm["acc"], tm["precision"], tm["recall"], tm["f1"], tm["macro_f1"], tm["auc"])
    cm = tm.get("confusion_matrix", [])
    if isinstance(cm, list) and len(cm) == 2 and len(cm[0]) == 2 and len(cm[1]) == 2:
        tn, fp = int(cm[0][0]), int(cm[0][1])
        fn, tp = int(cm[1][0]), int(cm[1][1])
        logging.info("Test confusion matrix | TN=%d FP=%d FN=%d TP=%d", tn, fp, fn, tp)
    c0 = tm.get("per_class", {}).get("class_0", {})
    c1 = tm.get("per_class", {}).get("class_1", {})
    logging.info(
        "Test class=%s | precision=%.4f recall=%.4f f1=%.4f support=%d",
        cfg.class_names[0],
        float(c0.get("precision", 0.0)),
        float(c0.get("recall", 0.0)),
        float(c0.get("f1", 0.0)),
        int(c0.get("support", 0)),
    )
    logging.info(
        "Test class=%s | precision=%.4f recall=%.4f f1=%.4f support=%d",
        cfg.class_names[1],
        float(c1.get("precision", 0.0)),
        float(c1.get("recall", 0.0)),
        float(c1.get("f1", 0.0)),
        int(c1.get("support", 0)),
    )

    save_cm(tm["confusion_matrix"], cfg.class_names, run_dir / "test_confusion_matrix.png")
    save_curves(history, run_dir)

    with open(run_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    with open(run_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump({"test_loss": te_loss, **tm}, f, ensure_ascii=False, indent=2)

    logging.info("Training complete. Best val macro_f1: %.4f (epoch=%d)", best, best_epoch)
    logging.info("Saved artifacts to: %s", run_dir.resolve())
import os

SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_ROOT = Path(os.getenv("DATASET_ROOT", str(SCRIPT_DIR)))
PROCESSED_DIR = Path(os.getenv("PROCESSED_DIR", str(DATASET_ROOT)))
SAVE_ROOT = Path(os.getenv("SAVE_ROOT", str(SCRIPT_DIR / "result" / "compare")))

CFG = BaselineConfig(
    model_type="eann_noadv",
    run_name="eann_noadv_cfnd",
    seed=42,
    dataset_root=str(DATASET_ROOT),
    processed_dir=str(PROCESSED_DIR),
    train_csv="train_data_clean.csv",
    val_csv="val_data.csv",
    test_csv="test_data.csv",
    text_column="title",
    image_column="image",
    label_column="label",
    class_names=("real", "fake"),
    lang="zh",
    save_root=str(SAVE_ROOT),
    max_len=42,
    batch_size=64,
    num_workers=8,
    epochs=25,
    lr=1e-4,
    weight_decay=1e-3,
    grad_clip=0.8,
    early_stop_patience=6,
    dropout=0.3,
    min_freq=2,
    embed_dim=128,
    text_hidden_dim=128,
    image_dim=128,
    fusion_hidden_dim=128,
    bert_name="hfl/chinese-bert-wwm-ext",
    freeze_bert=False,
    latent_dim=64,
    lambda_text_recon=0.2,
    lambda_image_recon=0.2,
    beta_kl=0.02,
)


if __name__ == "__main__":
    run_experiment(CFG)

