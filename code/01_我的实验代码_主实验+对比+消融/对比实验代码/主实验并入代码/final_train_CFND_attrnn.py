"""CFND 数据集训练脚本（BiCrossAttn 版本）。"""

import json
import logging
import math
import os
import pickle
import random
import re
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Sequence, Tuple

import jieba
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
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
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import VGG19_Weights, vgg19
from tqdm import tqdm

jieba.setLogLevel(logging.INFO)


# 直接在这里修改训练配置，无需命令行参数。
CFG = SimpleNamespace(
    seed=42,
    dataset_root="../CFND_dataset",
    train_csv="train_data_clean.csv",
    val_csv="val_data.csv",
    test_csv="test_data.csv",
    text_columns=("title",),
    image_column="image",
    label_column="label",
    class_names=("real", "fake"),
    save_root="model",
    min_freq=2,
    max_len=42,
    embed_dim=192,
    lstm_hidden_dim=128,
    lstm_layers=1,
    model_dim=256,
    att_hidden_dim=256,
    fusion_hidden_dim=128,
    dropout=0.3,
    batch_size=16,
    num_workers=8,
    epochs=30,
    lr=1e-4,
    backbone_lr=1e-5,
    weight_decay=1e-3,
    grad_clip=0.8,
    early_stop_patience=5,
    use_pretrained_vgg=True,
    freeze_vgg_features=False,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 为现代 NVIDIA GPU 优先开启吞吐优化（与严格确定性做权衡）。
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")


def create_run_dir(save_root: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(save_root) / f"run_cfnd_att_rnn_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def clean_text(text: object) -> str:
    if text is None or (isinstance(text, float) and np.isnan(text)):
        txt = ""
    else:
        txt = str(text)
    txt = re.sub(r"<.*?>", " ", txt)
    txt = re.sub(r"https?://\\S+|www\\.\\S+", " URL ", txt)
    txt = re.sub(r"\\s+", " ", txt).strip()
    return txt


def resolve_path(path_str: str, base_dir: Path) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def combine_text(row: pd.Series, fields: Sequence[str]) -> str:
    parts = [clean_text(row.get(col, "")) for col in fields]
    return " ".join(part for part in parts if part)


def build_vocab(df: pd.DataFrame, text_fields: Sequence[str], min_freq: int) -> Dict[str, int]:
    word_count: Dict[str, int] = {}
    for _, row in df.iterrows():
        text = combine_text(row, text_fields)
        for token in jieba.lcut(text):
            token = token.strip()
            if token:
                word_count[token] = word_count.get(token, 0) + 1

    vocab = {"<PAD>": 0, "<UNK>": 1}
    for token, count in word_count.items():
        if count >= min_freq:
            vocab[token] = len(vocab)
    return vocab


class NewsDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        vocab: Dict[str, int],
        max_len: int,
        dataset_root: Path,
        text_fields: Sequence[str],
        image_column: str,
        label_column: str,
        is_train: bool,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.vocab = vocab
        self.max_len = max_len
        self.dataset_root = dataset_root
        self.text_fields = tuple(text_fields)
        self.image_column = image_column
        self.label_column = label_column
        self.default_image = Image.new("RGB", (224, 224), color="white")
        if is_train:
            self.tfms = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomCrop((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(8),
                transforms.ColorJitter(
                    brightness=0.3,
                    contrast=0.3,
                    saturation=0.2,
                    hue=0.05,
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                transforms.RandomErasing(
                    p=0.25,
                    scale=(0.02, 0.15),
                    ratio=(0.3, 3.3),
                    value="random",
                ),
            ])
        else:
            self.tfms = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.CenterCrop((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    def __len__(self) -> int:
        return len(self.df)

    def _encode_text(self, text: str) -> Tuple[List[int], List[int]]:
        tokens = [t.strip() for t in jieba.lcut(text) if t.strip()]
        ids = [self.vocab.get(t, self.vocab["<UNK>"]) for t in tokens[: self.max_len]]
        valid = len(ids)
        pad = self.max_len - valid
        if pad > 0:
            ids.extend([self.vocab["<PAD>"]] * pad)
        mask = [1] * valid + [0] * pad
        return ids, mask

    def _load_image(self, image_path_value: object) -> Image.Image:
        rel = clean_text(image_path_value).replace("\\", "/")
        if not rel:
            return self.default_image.copy()
        path = self.dataset_root / rel
        if not path.exists():
            return self.default_image.copy()
        try:
            img = Image.open(path)
            # 调色板图像若包含透明信息，先转 RGBA 再转 RGB，避免颜色异常。
            if img.mode == "P" and "transparency" in img.info:
                img = img.convert("RGBA")
            return img.convert("RGB")
        except Exception:
            return self.default_image.copy()

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        text = combine_text(row, self.text_fields)
        token_ids, token_mask = self._encode_text(text)
        image = self._load_image(row.get(self.image_column, ""))
        return {
            "texts": torch.tensor(token_ids, dtype=torch.long),
            "text_mask": torch.tensor(token_mask, dtype=torch.bool),
            "image": self.tfms(image),
            "targets": torch.tensor(int(row[self.label_column]), dtype=torch.long),
        }


class AttRNNTextEncoder(nn.Module):
    def __init__(self, vocab_size: int, cfg) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, cfg.embed_dim, padding_idx=0)
        self.embed_dropout = nn.Dropout(cfg.dropout)
        self.lstm = nn.LSTM(
            cfg.embed_dim,
            cfg.lstm_hidden_dim,
            num_layers=cfg.lstm_layers,
            bidirectional=False,
            batch_first=True,
            dropout=cfg.dropout if cfg.lstm_layers > 1 else 0.0,
        )
        self.proj = nn.Linear(cfg.lstm_hidden_dim, cfg.model_dim)
        self.norm = nn.LayerNorm(cfg.model_dim)

    def forward(self, texts: torch.Tensor, text_mask: torch.Tensor) -> torch.Tensor:
        x = self.embedding(texts)
        x = self.embed_dropout(x)
        lengths = text_mask.sum(dim=1).clamp(min=1).cpu()
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        x, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=texts.size(1))
        return self.norm(self.proj(x))


class VGGImageEncoder(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        if cfg.use_pretrained_vgg:
            try:
                backbone = vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
            except Exception as e:
                logging.warning("Cannot load pretrained VGG19 weights: %s", e)
                backbone = vgg19(weights=None)
        else:
            backbone = vgg19(weights=None)

        self.net = backbone.features
        self.avgpool = backbone.avgpool
        if bool(getattr(cfg, "freeze_vgg_features", False)):
            for param in self.net.parameters():
                param.requires_grad = False
            logging.info("VGG19 feature extractor frozen.")

        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.dropout),
            nn.Linear(512, cfg.model_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        feat = self.net(images)
        feat = self.avgpool(feat)
        return self.proj(feat)


class VisualAttentionFusion(nn.Module):
    def __init__(self, dim: int, att_hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.att_mlp = nn.Sequential(
            nn.Linear(dim, att_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(att_hidden_dim, dim),
        )
        self.text_out_norm = nn.LayerNorm(dim)
        self.image_out_norm = nn.LayerNorm(dim)

    def forward(
        self,
        text_tokens: torch.Tensor,
        text_mask: torch.Tensor,
        image_vec: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mask = text_mask.unsqueeze(-1).float()
        lengths = mask.sum(dim=1).clamp(min=1.0)
        text_vec = (text_tokens * mask).sum(dim=1) / lengths
        att_scores = torch.softmax(self.att_mlp(text_tokens), dim=-1)
        attended_image = (att_scores * image_vec.unsqueeze(1) * mask).sum(dim=1) / lengths
        return self.text_out_norm(text_vec), self.image_out_norm(attended_image)


class MultiModalModel(nn.Module):
    def __init__(self, vocab_size: int, cfg) -> None:
        super().__init__()
        self.text_encoder = AttRNNTextEncoder(vocab_size, cfg)
        self.image_encoder = VGGImageEncoder(cfg)
        self.fusion = VisualAttentionFusion(
            dim=cfg.model_dim,
            att_hidden_dim=cfg.att_hidden_dim,
            dropout=cfg.dropout,
        )
        self.cls = nn.Sequential(
            nn.Linear(cfg.model_dim * 2, cfg.fusion_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.fusion_hidden_dim, 2),
        )

    def forward(self, texts: torch.Tensor, text_mask: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
        text_tokens = self.text_encoder(texts, text_mask)
        image_vec = self.image_encoder(images)
        text_vec, image_att = self.fusion(text_tokens, text_mask, image_vec)
        fused = torch.cat([text_vec, image_att], dim=1)
        return self.cls(fused)

def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    amp: bool,
) -> Tuple[float, List[int], List[float]]:
    model.eval()
    loss_sum = 0.0
    y_true: List[int] = []
    y_prob: List[float] = []

    with torch.no_grad():
        for batch in loader:
            texts = batch["texts"].to(device)
            text_mask = batch["text_mask"].to(device)
            images = batch["image"].to(device)
            targets = batch["targets"].to(device)

            with autocast("cuda", enabled=amp):
                logits = model(texts, text_mask, images)
                loss = criterion(logits, targets)

            probs = torch.softmax(logits, dim=1)[:, 1]
            loss_sum += float(loss.item())
            y_true.extend(targets.cpu().tolist())
            y_prob.extend(probs.cpu().tolist())

    return loss_sum / max(len(loader), 1), y_true, y_prob


def compute_metrics_from_probs(y_true: List[int], y_prob: List[float], threshold: float = 0.5) -> Dict[str, object]:
    if len(y_true) == 0:
        return {
            "acc": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "macro_f1": 0.0,
            "auc": float("nan"),
            "confusion_matrix": [[0, 0], [0, 0]],
            "per_class": {
                "class_0": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0},
                "class_1": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0},
            },
        }

    y_true_arr = np.asarray(y_true, dtype=np.int64)
    y_prob_arr = np.asarray(y_prob, dtype=np.float32)
    y_pred_arr = (y_prob_arr >= float(threshold)).astype(np.int64)
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1]).astype(np.int64)
    cls_prec, cls_rec, cls_f1, cls_sup = precision_recall_fscore_support(
        y_true_arr,
        y_pred_arr,
        labels=[0, 1],
        zero_division=0,
    )

    metrics = {
        "acc": float(accuracy_score(y_true_arr, y_pred_arr)),
        "precision": float(precision_score(y_true_arr, y_pred_arr, zero_division=0)),
        "recall": float(recall_score(y_true_arr, y_pred_arr, zero_division=0)),
        "f1": float(f1_score(y_true_arr, y_pred_arr, zero_division=0)),
        "macro_f1": float(f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)),
        "confusion_matrix": cm.tolist(),
        "per_class": {
            "class_0": {
                "precision": float(cls_prec[0]),
                "recall": float(cls_rec[0]),
                "f1": float(cls_f1[0]),
                "support": int(cls_sup[0]),
            },
            "class_1": {
                "precision": float(cls_prec[1]),
                "recall": float(cls_rec[1]),
                "f1": float(cls_f1[1]),
                "support": int(cls_sup[1]),
            },
        },
    }
    try:
        metrics["auc"] = float(roc_auc_score(y_true_arr, y_prob_arr))
    except ValueError:
        metrics["auc"] = float("nan")
    return metrics


def save_confusion_matrix_plot(
    cm_values: Sequence[Sequence[int]],
    class_names: Sequence[str],
    out_path: Path,
) -> None:
    cm = np.asarray(cm_values, dtype=np.int64)
    if cm.shape != (2, 2):
        return

    labels = list(class_names) if len(class_names) == 2 else ["class_0", "class_1"]
    plt.figure(figsize=(5.6, 4.8))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    ticks = np.arange(2)
    plt.xticks(ticks, labels)
    plt.yticks(ticks, labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")

    thresh = cm.max() / 2.0 if cm.size else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                str(int(cm[i, j])),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def log_per_class_metrics(split_name: str, metrics: Dict[str, object], class_names: Sequence[str]) -> None:
    cm = np.asarray(metrics.get("confusion_matrix", [[0, 0], [0, 0]]), dtype=np.int64)
    if cm.shape == (2, 2):
        logging.info(
            "%s confusion matrix | TN=%d FP=%d FN=%d TP=%d",
            split_name,
            int(cm[0, 0]),
            int(cm[0, 1]),
            int(cm[1, 0]),
            int(cm[1, 1]),
        )

    labels = list(class_names) if len(class_names) == 2 else ["class_0", "class_1"]
    per_class = metrics.get("per_class", {})
    for idx, label_name in enumerate(labels):
        cls = per_class.get(f"class_{idx}", {})
        logging.info(
            "%s class=%s | precision=%.4f recall=%.4f f1=%.4f support=%d",
            split_name,
            label_name,
            float(cls.get("precision", 0.0)),
            float(cls.get("recall", 0.0)),
            float(cls.get("f1", 0.0)),
            int(cls.get("support", 0)),
        )

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    criterion: nn.Module,
    scaler: GradScaler,
    device: torch.device,
    amp: bool,
    grad_clip: float,
):
    model.train()
    loss_sum = 0.0
    correct = 0
    total = 0

    for batch in tqdm(loader, desc="Train", leave=False):
        texts = batch["texts"].to(device)
        text_mask = batch["text_mask"].to(device)
        images = batch["image"].to(device)
        targets = batch["targets"].to(device)

        optimizer.zero_grad(set_to_none=True)
        with autocast("cuda", enabled=amp):
            logits = model(texts, text_mask, images)
            loss = criterion(logits, targets)

        scaler.scale(loss).backward()
        if grad_clip > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        preds = logits.argmax(dim=1)
        loss_sum += float(loss.item())
        correct += int((preds == targets).sum().item())
        total += int(targets.size(0))

    scheduler.step()
    return loss_sum / max(len(loader), 1), correct / max(total, 1)


def save_training_plots(history: Dict[str, List[float]], run_dir: Path) -> None:
    if not history["train_loss"]:
        return

    epochs = np.arange(1, len(history["train_loss"]) + 1)

    # 训练/验证损失曲线
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(run_dir / "loss_curve.png", dpi=160)
    plt.close()

    # 训练/验证精度曲线（辅助判断收敛与泛化）
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_acc"], label="Train Acc")
    plt.plot(epochs, history["val_acc"], label="Val Acc")
    plt.plot(epochs, history["val_f1"], label="Val F1")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("Accuracy/F1 Curve")
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(run_dir / "metrics_curve.png", dpi=160)
    plt.close()


def main() -> None:
    cfg = CFG
    set_seed(cfg.seed)
    class_names = tuple(getattr(cfg, "class_names", ("real", "fake")))

    script_dir = Path(__file__).resolve().parent
    dataset_root = resolve_path(cfg.dataset_root, script_dir)
    train_csv_path = resolve_path(cfg.train_csv, dataset_root)
    val_csv_path = resolve_path(cfg.val_csv, dataset_root)
    test_csv_path = resolve_path(cfg.test_csv, dataset_root)
    save_root = resolve_path(cfg.save_root, script_dir)

    run_dir = create_run_dir(str(save_root))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(run_dir / "train.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    logging.info("Run dir: %s", run_dir)
    logging.info("Config: %s", json.dumps(vars(cfg), ensure_ascii=False, indent=2))
    logging.info("Dataset root: %s", dataset_root)
    logging.info("Train CSV: %s", train_csv_path)
    logging.info("Val CSV: %s", val_csv_path)
    logging.info("Test CSV: %s", test_csv_path)
    logging.info("Label convention: 0=real, 1=fake (kept unchanged).")
    logging.info("Model: att-RNN baseline with LSTM text encoder, VGG19 image encoder and visual attention fusion.")

    train_df = pd.read_csv(train_csv_path)
    val_df = pd.read_csv(val_csv_path)
    test_df = pd.read_csv(test_csv_path)
    vocab = build_vocab(train_df, cfg.text_columns, cfg.min_freq)
    logging.info("Vocab size: %d", len(vocab))

    with open(run_dir / "vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)

    train_ds = NewsDataset(
        train_df,
        vocab,
        cfg.max_len,
        dataset_root=dataset_root,
        text_fields=cfg.text_columns,
        image_column=cfg.image_column,
        label_column=cfg.label_column,
        is_train=True,
    )
    val_ds = NewsDataset(
        val_df,
        vocab,
        cfg.max_len,
        dataset_root=dataset_root,
        text_fields=cfg.text_columns,
        image_column=cfg.image_column,
        label_column=cfg.label_column,
        is_train=False,
    )
    test_ds = NewsDataset(
        test_df,
        vocab,
        cfg.max_len,
        dataset_root=dataset_root,
        text_fields=cfg.text_columns,
        image_column=cfg.image_column,
        label_column=cfg.label_column,
        is_train=False,
    )

    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=pin)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=pin)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    logging.info("Device: %s | AMP: %s", device, amp)

    model = MultiModalModel(vocab_size=len(vocab), cfg=cfg).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info("Params | total=%.2fM | trainable=%.2fM", total_params / 1e6, trainable_params / 1e6)

    criterion = nn.CrossEntropyLoss()
    backbone_lr = float(getattr(cfg, "backbone_lr", cfg.lr))
    backbone_params = [p for p in model.image_encoder.net.parameters() if p.requires_grad]
    backbone_param_ids = {id(p) for p in backbone_params}
    other_params = [p for p in model.parameters() if p.requires_grad and id(p) not in backbone_param_ids]

    optim_groups = []
    if backbone_params:
        optim_groups.append({"params": backbone_params, "lr": backbone_lr})
    if other_params:
        optim_groups.append({"params": other_params, "lr": cfg.lr})
    if not optim_groups:
        raise RuntimeError("No trainable parameters found for optimizer.")

    optimizer = torch.optim.AdamW(
        optim_groups,
        weight_decay=cfg.weight_decay,
    )
    logging.info(
        "Optimizer groups | backbone_lr=%.2e params=%.2fM | other_lr=%.2e params=%.2fM",
        backbone_lr,
        sum(p.numel() for p in backbone_params) / 1e6,
        cfg.lr,
        sum(p.numel() for p in other_params) / 1e6,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    scaler = GradScaler("cuda", enabled=amp)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_f1": []}
    best_f1 = -1.0
    best_epoch = 0
    patience = int(getattr(cfg, "early_stop_patience", 0))
    no_improve_epochs = 0

    for epoch in range(1, cfg.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            criterion,
            scaler,
            device,
            amp,
            cfg.grad_clip,
        )
        val_loss, val_labels, val_probs = evaluate(model, val_loader, criterion, device, amp)
        val_metrics = compute_metrics_from_probs(val_labels, val_probs)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_metrics["acc"])
        history["val_f1"].append(val_metrics["f1"])

        logging.info(
            "Epoch %d/%d | train_loss=%.4f train_acc=%.4f | val_loss=%.4f val_acc=%.4f val_prec=%.4f val_recall=%.4f val_f1=%.4f val_auc=%.4f",
            epoch,
            cfg.epochs,
            train_loss,
            train_acc,
            val_loss,
            val_metrics["acc"],
            val_metrics["precision"],
            val_metrics["recall"],
            val_metrics["f1"],
            val_metrics["auc"],
        )

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            best_epoch = epoch
            no_improve_epochs = 0
            torch.save(
                {
                    "epoch": epoch,
                    "best_f1": best_f1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "config": vars(cfg),
                    "vocab_size": len(vocab),
                },
                run_dir / "best_model.pth",
            )
        else:
            no_improve_epochs += 1

        if patience > 0 and no_improve_epochs >= patience:
            logging.info(
                "Early stopping at epoch %d (best epoch=%d, best val F1=%.4f).",
                epoch,
                best_epoch,
                best_f1,
            )
            break

    best_ckpt_path = run_dir / "best_model.pth"
    if best_ckpt_path.exists():
        try:
            ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=True)
        except TypeError:
            ckpt = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        logging.info("Loaded best checkpoint from epoch %s for test evaluation.", ckpt.get("epoch", "unknown"))

    test_loss, test_labels, test_probs = evaluate(model, test_loader, criterion, device, amp)
    test_metrics = compute_metrics_from_probs(test_labels, test_probs)
    logging.info(
        "Test | loss=%.4f acc=%.4f precision=%.4f recall=%.4f f1=%.4f macro_f1=%.4f auc=%.4f",
        test_loss,
        test_metrics["acc"],
        test_metrics["precision"],
        test_metrics["recall"],
        test_metrics["f1"],
        test_metrics["macro_f1"],
        test_metrics["auc"],
    )
    log_per_class_metrics("Test", test_metrics, class_names)
    save_confusion_matrix_plot(
        test_metrics["confusion_matrix"],
        class_names,
        run_dir / "test_confusion_matrix.png",
    )

    torch.save(model.state_dict(), run_dir / "final_model.pth")
    with open(run_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    test_metrics_payload = {
        "test_loss": test_loss,
        **test_metrics,
    }
    with open(run_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(test_metrics_payload, f, ensure_ascii=False, indent=2)
    save_training_plots(history, run_dir)

    logging.info("Training complete. Best val F1: %.4f", best_f1)
    logging.info("Saved artifacts to: %s", run_dir.resolve())


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUTF8", "1")
    main()
