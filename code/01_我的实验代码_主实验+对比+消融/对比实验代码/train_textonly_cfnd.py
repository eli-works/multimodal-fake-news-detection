import json
import logging
import os
import random
import re
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    import jieba
except Exception:
    jieba = None


CFG = SimpleNamespace(
    run_name="textonly_cfnd",
    seed=42,
    dataset_root=os.getenv("DATASET_ROOT", "../CFND_dataset"),
    processed_dir=os.getenv("PROCESSED_DIR", os.getenv("DATASET_ROOT", "../CFND_dataset")),
    train_csv="train_data_clean.csv",
    val_csv="val_data.csv",
    test_csv="test_data.csv",
    text_column="title",
    label_column="label",
    class_names=("real", "fake"),
    lang="zh",
    save_root=os.getenv("SAVE_ROOT", "model"),
    min_freq=2,
    max_len=42,
    embed_dim=256,
    lstm_hidden_dim=128,
    lstm_layers=1,
    lstm_bidirectional=True,
    dropout=0.3,
    batch_size=64,
    num_workers=4,
    epochs=25,
    lr=1e-3,
    weight_decay=1e-4,
    grad_clip=1.0,
    early_stop_patience=6,
)


URL_RE = re.compile(r"https?://\S+|www\.\S+")
HTML_RE = re.compile(r"<.*?>")
EN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_']+")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def resolve_path(path_str: str, base_dir: Path) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (base_dir / p).resolve()


def clean_text(x: object) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        s = ""
    else:
        s = str(x)
    s = HTML_RE.sub(" ", s)
    s = URL_RE.sub(" URL ", s)
    s = s.replace("\u200b", " ").replace("\xa0", " ")
    s = re.sub(r"\s+", " ", s).strip()
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


def normalize_label(x: object) -> int:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return -1
    try:
        v = int(x)
        if v in (0, 1):
            return v
    except Exception:
        pass
    s = str(x).strip().lower()
    if s in {"0", "real", "true", "nonrumor", "non-rumor"}:
        return 0
    if s in {"1", "fake", "false", "rumor"}:
        return 1
    return -1


def build_vocab(texts: List[str], min_freq: int, lang: str) -> Dict[str, int]:
    cnt: Dict[str, int] = {}
    for t in texts:
        for w in tokenize(t, lang):
            cnt[w] = cnt.get(w, 0) + 1
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for w, c in cnt.items():
        if c >= min_freq:
            vocab[w] = len(vocab)
    return vocab


def encode_text(text: str, vocab: Dict[str, int], max_len: int, lang: str) -> Tuple[torch.Tensor, torch.Tensor]:
    tokens = tokenize(text, lang)[:max_len]
    ids = [vocab.get(t, vocab["<UNK>"]) for t in tokens]
    valid = len(ids)
    if valid < max_len:
        ids += [vocab["<PAD>"]] * (max_len - valid)
    mask = [1] * valid + [0] * (max_len - valid)
    return torch.tensor(ids, dtype=torch.long), torch.tensor(mask, dtype=torch.bool)


class TextDataset(Dataset):
    def __init__(self, df: pd.DataFrame, cfg: SimpleNamespace, vocab: Dict[str, int]):
        self.texts = df[cfg.text_column].astype(str).tolist()
        self.labels = df[cfg.label_column].astype(int).tolist()
        self.vocab = vocab
        self.max_len = int(cfg.max_len)
        self.lang = cfg.lang

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        ids, mask = encode_text(self.texts[idx], self.vocab, self.max_len, self.lang)
        return {
            "texts": ids,
            "text_mask": mask,
            "targets": torch.tensor(self.labels[idx], dtype=torch.long),
        }


class TextBiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size: int, cfg: SimpleNamespace):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, cfg.embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(cfg.dropout)
        self.lstm = nn.LSTM(
            input_size=cfg.embed_dim,
            hidden_size=cfg.lstm_hidden_dim,
            num_layers=cfg.lstm_layers,
            batch_first=True,
            bidirectional=cfg.lstm_bidirectional,
            dropout=cfg.dropout if cfg.lstm_layers > 1 else 0.0,
        )
        lstm_out = cfg.lstm_hidden_dim * (2 if cfg.lstm_bidirectional else 1)
        self.cls = nn.Sequential(
            nn.Linear(lstm_out, 128),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(128, 2),
        )

    def forward(self, texts: torch.Tensor, text_mask: torch.Tensor) -> torch.Tensor:
        x = self.dropout(self.emb(texts))
        out, _ = self.lstm(x)
        m = text_mask.unsqueeze(-1).float()
        pooled = (out * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
        return self.cls(pooled)


def metrics_from_probs(y_true: List[int], y_prob: List[float], th: float = 0.5) -> Dict[str, object]:
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


def save_cm(cm_values, class_names: Tuple[str, str], out_path: Path) -> None:
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


def save_curves(history: Dict[str, List[float]], run_dir: Path) -> None:
    if not history["train_loss"]:
        return
    e = np.arange(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(e, history["train_loss"], label="Train Loss")
    plt.plot(e, history["val_loss"], label="Val Loss")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(run_dir / "loss_curve.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(e, history["train_acc"], label="Train Acc")
    plt.plot(e, history["val_acc"], label="Val Acc")
    plt.plot(e, history["val_f1"], label="Val F1")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(run_dir / "metrics_curve.png", dpi=160)
    plt.close()


def move_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def load_split_df(csv_path: Path, cfg: SimpleNamespace) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if cfg.text_column not in df.columns:
        raise KeyError(f"Missing text column '{cfg.text_column}' in {csv_path}")
    if cfg.label_column not in df.columns:
        raise KeyError(f"Missing label column '{cfg.label_column}' in {csv_path}")
    df[cfg.text_column] = df[cfg.text_column].fillna("").astype(str).map(clean_text)
    df[cfg.label_column] = df[cfg.label_column].map(normalize_label).astype(int)
    df = df[df[cfg.label_column].isin([0, 1])].reset_index(drop=True)
    return df


def main() -> None:
    set_seed(int(CFG.seed))

    script_dir = Path(__file__).resolve().parent
    save_root = resolve_path(CFG.save_root, script_dir)
    run_dir = save_root / f"{CFG.run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(run_dir / "train.log", encoding="utf-8"), logging.StreamHandler()],
        force=True,
    )
    logging.info("Run dir: %s", run_dir)
    logging.info("Config: %s", json.dumps(vars(CFG), ensure_ascii=False, indent=2))

    processed_dir = resolve_path(CFG.processed_dir, script_dir)
    train_csv = resolve_path(CFG.train_csv, processed_dir)
    val_csv = resolve_path(CFG.val_csv, processed_dir)
    test_csv = resolve_path(CFG.test_csv, processed_dir)

    train_df = load_split_df(train_csv, CFG)
    val_df = load_split_df(val_csv, CFG)
    test_df = load_split_df(test_csv, CFG)
    logging.info("Data | train=%d val=%d test=%d", len(train_df), len(val_df), len(test_df))

    vocab = build_vocab(train_df[CFG.text_column].tolist(), int(CFG.min_freq), CFG.lang)
    with open(run_dir / "vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    logging.info("Vocab size: %d", len(vocab))

    train_ds = TextDataset(train_df, CFG, vocab)
    val_ds = TextDataset(val_df, CFG, vocab)
    test_ds = TextDataset(test_df, CFG, vocab)

    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=CFG.batch_size, shuffle=True, num_workers=CFG.num_workers, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=CFG.batch_size, shuffle=False, num_workers=CFG.num_workers, pin_memory=pin)
    test_loader = DataLoader(test_ds, batch_size=CFG.batch_size, shuffle=False, num_workers=CFG.num_workers, pin_memory=pin)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"

    model = TextBiLSTMClassifier(len(vocab), CFG).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info("Device: %s | AMP: %s", device, amp)
    logging.info("Params | total=%.2fM | trainable=%.2fM", total_params / 1e6, trainable_params / 1e6)

    ce = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
    sch = CosineAnnealingLR(opt, T_max=CFG.epochs)
    scaler = GradScaler("cuda", enabled=amp)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_f1": []}
    best = -1.0
    best_epoch = 0
    best_state_dict = None
    stale = 0

    for ep in range(1, CFG.epochs + 1):
        model.train()
        t_loss = 0.0
        t_correct = 0
        t_total = 0
        for b in tqdm(train_loader, desc="Train", leave=False):
            b = move_to_device(b, device)
            y = b["targets"]
            opt.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=amp):
                logits = model(b["texts"], b["text_mask"])
                loss = ce(logits, y)
            scaler.scale(loss).backward()
            if CFG.grad_clip > 0:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), float(CFG.grad_clip))
            scaler.step(opt)
            scaler.update()
            pred = logits.argmax(dim=1)
            t_loss += float(loss.item())
            t_correct += int((pred == y).sum().item())
            t_total += int(y.size(0))
        sch.step()

        model.eval()
        v_loss = 0.0
        y_true: List[int] = []
        y_prob: List[float] = []
        with torch.no_grad():
            for b in val_loader:
                b = move_to_device(b, device)
                y = b["targets"]
                with autocast("cuda", enabled=amp):
                    logits = model(b["texts"], b["text_mask"])
                    loss = ce(logits, y)
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

        logging.info(
            "Epoch %d/%d | train_loss=%.4f train_acc=%.4f | val_loss=%.4f val_acc=%.4f val_prec=%.4f val_recall=%.4f val_f1=%.4f val_macro_f1=%.4f val_auc=%.4f",
            ep, CFG.epochs, train_loss, train_acc, val_loss, vm["acc"], vm["precision"], vm["recall"], vm["f1"], vm["macro_f1"], vm["auc"]
        )

        if vm["macro_f1"] > best:
            best = vm["macro_f1"]
            best_epoch = ep
            stale = 0
            best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1

        if CFG.early_stop_patience > 0 and stale >= CFG.early_stop_patience:
            logging.info("Early stopping at epoch %d (best epoch=%d, best macro_f1=%.4f)", ep, best_epoch, best)
            break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    model.eval()
    te_loss = 0.0
    y_true = []
    y_prob = []
    with torch.no_grad():
        for b in test_loader:
            b = move_to_device(b, device)
            y = b["targets"]
            with autocast("cuda", enabled=amp):
                logits = model(b["texts"], b["text_mask"])
                loss = ce(logits, y)
            p = torch.softmax(logits, dim=1)[:, 1]
            te_loss += float(loss.item())
            y_true.extend(y.cpu().tolist())
            y_prob.extend(p.cpu().tolist())

    tm = metrics_from_probs(y_true, y_prob)
    te_loss = te_loss / max(len(test_loader), 1)
    logging.info(
        "Test | loss=%.4f acc=%.4f precision=%.4f recall=%.4f f1=%.4f macro_f1=%.4f auc=%.4f",
        te_loss, tm["acc"], tm["precision"], tm["recall"], tm["f1"], tm["macro_f1"], tm["auc"]
    )
    cm = tm.get("confusion_matrix", [])
    if isinstance(cm, list) and len(cm) == 2 and len(cm[0]) == 2 and len(cm[1]) == 2:
        tn, fp = int(cm[0][0]), int(cm[0][1])
        fn, tp = int(cm[1][0]), int(cm[1][1])
        logging.info("Test confusion matrix | TN=%d FP=%d FN=%d TP=%d", tn, fp, fn, tp)
    c0 = tm.get("per_class", {}).get("class_0", {})
    c1 = tm.get("per_class", {}).get("class_1", {})
    logging.info(
        "Test class=%s | precision=%.4f recall=%.4f f1=%.4f support=%d",
        CFG.class_names[0], float(c0.get("precision", 0.0)), float(c0.get("recall", 0.0)), float(c0.get("f1", 0.0)), int(c0.get("support", 0))
    )
    logging.info(
        "Test class=%s | precision=%.4f recall=%.4f f1=%.4f support=%d",
        CFG.class_names[1], float(c1.get("precision", 0.0)), float(c1.get("recall", 0.0)), float(c1.get("f1", 0.0)), int(c1.get("support", 0))
    )

    save_cm(tm["confusion_matrix"], CFG.class_names, run_dir / "test_confusion_matrix.png")
    save_curves(history, run_dir)

    with open(run_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    with open(run_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump({"test_loss": te_loss, **tm}, f, ensure_ascii=False, indent=2)

    logging.info("Training complete. Best val macro_f1: %.4f (epoch=%d)", best, best_epoch)
    logging.info("Saved artifacts to: %s", run_dir.resolve())


if __name__ == "__main__":
    main()

