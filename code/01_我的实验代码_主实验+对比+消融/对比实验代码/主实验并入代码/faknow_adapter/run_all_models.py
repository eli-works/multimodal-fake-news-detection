from __future__ import annotations

import argparse
import contextlib
import inspect
import json
import logging
import os
import pickle
import random
import re
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import jieba
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import transforms

try:
    from prepare_all import DEFAULTS as PREP_DEFAULTS
    from prepare_all import prepare_datasets
except ImportError:
    from .prepare_all import DEFAULTS as PREP_DEFAULTS
    from .prepare_all import prepare_datasets

DATASETS = ("cfnd", "gossip", "weibo")
MODELS = ("spotfake", "hmcan", "mcan", "eann", "mfan", "safe", "cafe")
WEIGHT_SUFFIXES = (".pth", ".pt", ".ckpt")
DEFAULT_METRICS = ["accuracy", "precision", "recall", "f1"]
EN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(obj: Any, path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_plain(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_plain(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_plain(x) for x in obj]
    if isinstance(obj, tuple):
        return [to_plain(x) for x in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if torch.is_tensor(obj):
        if obj.numel() == 1:
            return obj.item()
        return obj.detach().cpu().tolist()
    return obj


@contextlib.contextmanager
def pushd(path: Path):
    old = Path.cwd()
    ensure_dir(path)
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def remove_weight_files(root: Path) -> List[str]:
    removed: List[str] = []
    if not root.exists():
        return removed
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in WEIGHT_SUFFIXES:
            try:
                p.unlink()
                removed.append(str(p))
            except Exception:
                logging.warning("Failed to delete weight file: %s", p)
    return removed


def parse_metrics(text: str) -> Optional[List[str]]:
    values = [x.strip() for x in text.split(",") if x.strip()]
    if not values:
        return None
    return values


def language_for(dataset: str) -> str:
    return "en" if dataset == "gossip" else "zh"


def bert_for_dataset(dataset: str, cfg: argparse.Namespace) -> str:
    return cfg.bert_en if dataset == "gossip" else cfg.bert_zh


def infer_targets(dataset: str, model: str) -> Tuple[List[str], List[str]]:
    target_datasets = list(DATASETS) if dataset == "all" else [dataset]
    target_models = list(MODELS) if model == "all" else [model]
    return target_datasets, target_models


def ensure_artifacts_for_model(dataset_dir: Path, model: str) -> None:
    checks: List[Path] = []
    if model in {"spotfake", "hmcan", "mcan", "eann", "mfan"}:
        checks.extend([dataset_dir / model / "train.json", dataset_dir / model / "test.json"])
    if model == "eann":
        checks.extend(
            [
                dataset_dir / "eann" / "vocab.pkl",
                dataset_dir / "eann" / "word_vectors.pkl",
                dataset_dir / "eann" / "stop_words.txt",
            ]
        )
    if model == "mfan":
        checks.extend(
            [
                dataset_dir / "mfan" / "vocab.pkl",
                dataset_dir / "mfan" / "word_vectors.pkl",
                dataset_dir / "mfan" / "node_embedding.pkl",
                dataset_dir / "mfan" / "adjacency.json",
            ]
        )
    if model == "safe":
        checks.extend(
            [
                dataset_dir / "safe" / "train" / "case_headline.npy",
                dataset_dir / "safe" / "train" / "case_body.npy",
                dataset_dir / "safe" / "train" / "case_image.npy",
                dataset_dir / "safe" / "train" / "case_y_fn_dim1.npy",
                dataset_dir / "safe" / "test" / "case_headline.npy",
                dataset_dir / "safe" / "test" / "case_body.npy",
                dataset_dir / "safe" / "test" / "case_image.npy",
                dataset_dir / "safe" / "test" / "case_y_fn_dim1.npy",
            ]
        )
    if model == "cafe":
        checks.extend(
            [
                dataset_dir / "cafe" / "train_text_with_label.npz",
                dataset_dir / "cafe" / "train_image_with_label.npz",
                dataset_dir / "cafe" / "test_text_with_label.npz",
                dataset_dir / "cafe" / "test_image_with_label.npz",
            ]
        )
    missing = [str(p) for p in checks if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing artifacts for model=%s dataset=%s:\n%s"
            % (model, dataset_dir.name, "\n".join(missing))
        )


def prepare_stage(cfg: argparse.Namespace, dataset: str) -> Dict[str, Any]:
    prepare_cfg = argparse.Namespace(
        dataset=dataset,
        out=str(Path(cfg.artifacts_dir).resolve()),
        cfnd_root=cfg.cfnd_root,
        gossip_root=cfg.gossip_root,
        weibo_root=cfg.weibo_root,
        gossip_image_root=cfg.gossip_image_root,
        auto_prepare_data=cfg.auto_prepare_data,
        force_prepare_data=cfg.force_prepare_data,
        val_ratio=cfg.val_ratio,
        text_model_zh=cfg.prepare_text_model_zh,
        text_model_en=cfg.prepare_text_model_en,
        device=cfg.prepare_device or cfg.device,
        batch_size=cfg.prepare_batch_size,
        min_freq=cfg.min_freq,
        mfan_knn=cfg.mfan_knn,
        safe_head_len=cfg.safe_head_len,
        safe_body_len=cfg.safe_body_len,
        safe_image_len=cfg.safe_image_len,
        cafe_text_len=cfg.cafe_text_len,
        cafe_text_dim=cfg.cafe_text_dim,
        seed=cfg.seed,
        validate_image_samples=cfg.validate_image_samples,
        quick_prepare=cfg.quick_prepare,
    )
    return prepare_datasets(prepare_cfg)


def _clean_text(text: str) -> str:
    t = str(text or "").strip().lower()
    t = re.sub(r"\s+", " ", t)
    return t


class TokenizerEANNFixed:
    def __init__(self, vocab: Dict[str, int], max_len=255, stop_words: Optional[Sequence[str]] = None, language="zh"):
        assert language in {"zh", "en"}
        self.vocab = vocab
        self.max_len = int(max_len)
        self.stop_words = set(stop_words or [])
        self.language = language
        self.unk = int(vocab.get("<UNK>", 1))

    def _tokens_zh(self, text: str) -> List[str]:
        cleaned = re.sub(
            u"[锛屻€?:,.锛泑-鈥溾€濃€斺€擾/nbsp+&;@銆併€娿€嬶綖锛堬級())#O锛侊細銆愩€慮",
            "",
            text,
        ).strip().lower()
        return [w for w in jieba.cut_for_search(cleaned) if w and w not in self.stop_words]

    def _tokens_en(self, text: str) -> List[str]:
        cleaned = re.sub(r"[^a-z0-9\s']", " ", _clean_text(text))
        return [w for w in EN_RE.findall(cleaned) if w and w not in self.stop_words]

    def __call__(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        token_ids: List[List[int]] = []
        masks: List[torch.Tensor] = []
        for text in texts:
            words = self._tokens_zh(text) if self.language == "zh" else self._tokens_en(text)
            ids = [int(self.vocab.get(word, self.unk)) for word in words]
            real_len = min(len(ids), self.max_len)
            if len(ids) < self.max_len:
                ids = ids + [0] * (self.max_len - len(ids))
            else:
                ids = ids[: self.max_len]
            mask = torch.zeros(self.max_len, dtype=torch.float32)
            mask[:real_len] = 1.0
            token_ids.append(ids)
            masks.append(mask)
        return {"token_id": torch.tensor(token_ids, dtype=torch.long), "mask": torch.stack(masks)}


class TokenizerMFANFixed:
    def __init__(self, vocab: Dict[str, int], max_len=50, language="zh", stop_words: Optional[Sequence[str]] = None):
        assert language in {"zh", "en"}
        self.vocab = vocab
        self.max_len = int(max_len)
        self.language = language
        self.stop_words = set(stop_words or [])
        self.unk = int(vocab.get("<UNK>", 1))

    def __call__(self, texts: List[str]) -> torch.Tensor:
        token_ids: List[List[int]] = []
        for text in texts:
            cleaned = re.sub(r"\s+", " ", str(text or "")).strip().lower()
            if self.language == "zh":
                words = [w for w in jieba.cut(cleaned) if w and w not in self.stop_words]
            else:
                words = [w for w in EN_RE.findall(cleaned) if w and w not in self.stop_words]
            ids = [int(self.vocab.get(word, self.unk)) for word in words]
            if len(ids) < self.max_len:
                ids = [0] * (self.max_len - len(ids)) + ids
            else:
                ids = ids[: self.max_len]
            token_ids.append(ids)
        return torch.tensor(token_ids, dtype=torch.long)


def text_preprocessing_spotfake(texts: List[str]) -> List[str]:
    processed = []
    for text in texts:
        text = re.sub(r"(@.*?)[\s]", " ", str(text))
        text = re.sub(r"&amp;", "&", text)
        text = re.sub(r"\s+", " ", text).strip()
        processed.append(text)
    return processed


def text_preprocessing_mcan(texts: List[str]) -> List[str]:
    reg = r"(http|https)((\W+)(\w+)(\W+)(\w*)(\W+)(\w*)|(\W+)(\w+)(\W+)|(\W+))"
    return [re.sub(reg, "", str(text)) for text in texts]


def _load_rgb(path: str) -> Image.Image:
    with open(path, "rb") as f:
        return Image.open(f).convert("RGB")


def transform_spotfake_local(path: str) -> torch.Tensor:
    img = _load_rgb(path)
    trans = transforms.Compose(
        [
            transforms.Resize(size=(224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return trans(img)


def transform_hmcan_local(path: str) -> torch.Tensor:
    img = _load_rgb(path)
    trans = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return trans(img)


def transform_eann_local(path: str) -> torch.Tensor:
    return transform_hmcan_local(path)


def process_dct_mcan_local(img_tensor: torch.Tensor) -> torch.Tensor:
    from scipy.fftpack import dct, fft

    img = img_tensor.numpy()
    height = img.shape[1]
    width = img.shape[2]
    n = 8
    step = int(height / n)

    dct_img = np.zeros((1, n * n, step * step, 1), dtype=np.float32)
    fft_img = np.zeros((1, n * n, step * step, 1), dtype=np.float32)

    i = 0
    for row in np.arange(0, height, step):
        for col in np.arange(0, width, step):
            block = np.array(img[:, row : (row + step), col : (col + step)], dtype=np.float32)
            block1 = block.reshape((-1, step * step, 1))
            dct_img[:, i, :, :] = dct(block1)
            i += 1

    fft_img[:, :, :, :] = fft(dct_img[:, :, :, :]).real
    fft_img = torch.from_numpy(fft_img).float()
    new_img = F.interpolate(fft_img, size=[250, 1])
    return new_img.squeeze(0).squeeze(-1)


def transform_mcan_local(path: str) -> Dict[str, torch.Tensor]:
    with open(path, "rb") as f:
        img = Image.open(f)
        transform_img = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
        vgg_feature = transform_img(img.convert("RGB"))
        dct_feature = process_dct_mcan_local(transform_img(img.convert("L")))
    return {"vgg": vgg_feature, "dct": dct_feature}


def get_optimizer_mcan_local(
    model,
    lr=0.0001,
    weight_decay=0.15,
    bert_lr=1e-5,
    vgg_lr=1e-5,
    dtc_lr=1e-5,
    fusion_lr=1e-2,
    linear_lr=1e-2,
    classifier_lr=1e-2,
):
    no_decay = [
        "bias",
        "gamma",
        "beta",
        "LayerNorm.weight",
        "bn_text.weight",
        "bn_dct.weight",
        "bn_1.weight",
    ]

    bert_params = list(model.bert.named_parameters())
    vgg_params = list(model.vgg.named_parameters())
    dtc_params = list(model.dct_img.named_parameters())
    fusion_params = list(model.fusion_layers.named_parameters())
    linear_params = (
        list(model.linear_text.named_parameters())
        + list(model.linear_vgg.named_parameters())
        + list(model.linear_dct.named_parameters())
    )
    classifier_params = list(model.linear1.named_parameters()) + list(model.linear2.named_parameters())

    optimizer_grouped_parameters = [
        {"params": [p for n, p in bert_params if not any(nd in n for nd in no_decay)], "weight_decay": weight_decay, "lr": bert_lr},
        {"params": [p for n, p in bert_params if any(nd in n for nd in no_decay)], "weight_decay": 0.0, "lr": bert_lr},
        {"params": [p for n, p in vgg_params if not any(nd in n for nd in no_decay)], "weight_decay": weight_decay, "lr": vgg_lr},
        {"params": [p for n, p in vgg_params if any(nd in n for nd in no_decay)], "weight_decay": 0.0, "lr": vgg_lr},
        {"params": [p for n, p in dtc_params if not any(nd in n for nd in no_decay)], "weight_decay": weight_decay, "lr": dtc_lr},
        {"params": [p for n, p in dtc_params if any(nd in n for nd in no_decay)], "weight_decay": 0.0, "lr": dtc_lr},
        {"params": [p for n, p in fusion_params if not any(nd in n for nd in no_decay)], "weight_decay": weight_decay, "lr": fusion_lr},
        {"params": [p for n, p in fusion_params if any(nd in n for nd in no_decay)], "weight_decay": 0.0, "lr": fusion_lr},
        {"params": [p for n, p in linear_params if not any(nd in n for nd in no_decay)], "weight_decay": weight_decay, "lr": linear_lr},
        {"params": [p for n, p in linear_params if any(nd in n for nd in no_decay)], "weight_decay": 0.0, "lr": linear_lr},
        {"params": [p for n, p in classifier_params if not any(nd in n for nd in no_decay)], "weight_decay": weight_decay, "lr": classifier_lr},
        {"params": [p for n, p in classifier_params if any(nd in n for nd in no_decay)], "weight_decay": 0.0, "lr": classifier_lr},
    ]
    return torch.optim.AdamW(optimizer_grouped_parameters, lr=lr, weight_decay=weight_decay)


def get_scheduler_mcan_local(batch_num: int, epoch_num: int, optimizer: torch.optim.Optimizer, warm_up_percentage=0.1):
    from transformers import get_linear_schedule_with_warmup

    total_steps = batch_num * epoch_num
    return get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=round(total_steps * warm_up_percentage),
        num_training_steps=total_steps,
    )


def train_spotfake(dataset: str, dataset_dir: Path, cfg: argparse.Namespace) -> Dict[str, Any]:
    from torch import nn
    from torch.utils.data import DataLoader

    from faknow.data.dataset.multi_modal import MultiModalDataset
    from faknow.data.process.text_process import TokenizerFromPreTrained
    from faknow.evaluate.evaluator import Evaluator
    from faknow.model.content_based.multi_modal.spotfake import SpotFake
    from faknow.train.trainer import BaseTrainer

    model_dir = dataset_dir / "spotfake"
    train_path = str(model_dir / "train.json")
    val_path = str(model_dir / "val.json")
    test_path = str(model_dir / "test.json")

    tokenizer = TokenizerFromPreTrained(cfg.spotfake_max_len, bert_for_dataset(dataset, cfg), text_preprocessing_spotfake)
    train_set = MultiModalDataset(train_path, ["post_text"], tokenizer, ["image_id"], transform_spotfake_local)
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    val_loader = None
    if Path(val_path).exists():
        val_set = MultiModalDataset(val_path, ["post_text"], tokenizer, ["image_id"], transform_spotfake_local)
        val_loader = DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    model = SpotFake(
        loss_func=nn.BCELoss(),
        pre_trained_bert_name=bert_for_dataset(dataset, cfg),
        fine_tune_text_module=cfg.spotfake_finetune_text,
        fine_tune_vis_module=cfg.spotfake_finetune_vision,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.spotfake_lr)
    evaluator = Evaluator(cfg.metrics)
    trainer = BaseTrainer(model=model, evaluator=evaluator, optimizer=optimizer, device=cfg.device)
    trainer.fit(train_loader=train_loader, num_epochs=cfg.epochs, validate_loader=val_loader, save_best=None)

    result: Dict[str, Any] = {"train_size": len(train_set), "val_size": len(val_loader.dataset) if val_loader else 0}
    if Path(test_path).exists():
        test_set = MultiModalDataset(test_path, ["post_text"], tokenizer, ["image_id"], transform_spotfake_local)
        test_loader = DataLoader(test_set, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
        result["test_size"] = len(test_set)
        result["test"] = to_plain(trainer.evaluate(test_loader))
    return result


def train_hmcan(dataset: str, dataset_dir: Path, cfg: argparse.Namespace) -> Dict[str, Any]:
    from torch.utils.data import DataLoader

    from faknow.data.dataset.multi_modal import MultiModalDataset
    from faknow.data.process.text_process import TokenizerFromPreTrained
    from faknow.evaluate.evaluator import Evaluator
    from faknow.model.content_based.multi_modal.hmcan import HMCAN
    from faknow.train.trainer import BaseTrainer

    model_dir = dataset_dir / "hmcan"
    train_path = str(model_dir / "train.json")
    val_path = str(model_dir / "val.json")
    test_path = str(model_dir / "test.json")

    bert_name = bert_for_dataset(dataset, cfg)
    tokenizer = TokenizerFromPreTrained(cfg.hmcan_max_len, bert_name)
    train_set = MultiModalDataset(train_path, ["text"], tokenizer, ["image"], transform_hmcan_local)
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    val_loader = None
    if Path(val_path).exists():
        val_set = MultiModalDataset(val_path, ["text"], tokenizer, ["image"], transform_hmcan_local)
        val_loader = DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    hmcan_kwargs = dict(
        left_num_layers=cfg.hmcan_left_layers,
        left_num_heads=cfg.hmcan_left_heads,
        dropout=cfg.hmcan_dropout,
        right_num_layers=cfg.hmcan_right_layers,
        right_num_heads=cfg.hmcan_right_heads,
        alpha=cfg.hmcan_alpha,
    )
    sig = inspect.signature(HMCAN.__init__)
    for bert_key in ("bert", "bert_name", "pre_trained_bert_name", "pretrained_bert_name"):
        if bert_key in sig.parameters:
            hmcan_kwargs[bert_key] = bert_name
            break
    else:
        logging.warning(
            "Current faknow HMCAN.__init__ has no explicit bert argument; using library default text encoder."
        )

    model = HMCAN(**hmcan_kwargs)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.hmcan_lr)
    evaluator = Evaluator(cfg.metrics)
    trainer = BaseTrainer(model=model, evaluator=evaluator, optimizer=optimizer, device=cfg.device)
    trainer.fit(train_loader=train_loader, num_epochs=cfg.epochs, validate_loader=val_loader, save_best=None)

    result: Dict[str, Any] = {"train_size": len(train_set), "val_size": len(val_loader.dataset) if val_loader else 0}
    if Path(test_path).exists():
        test_set = MultiModalDataset(test_path, ["text"], tokenizer, ["image"], transform_hmcan_local)
        test_loader = DataLoader(test_set, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
        result["test_size"] = len(test_set)
        result["test"] = to_plain(trainer.evaluate(test_loader))
    return result


def train_mcan(dataset: str, dataset_dir: Path, cfg: argparse.Namespace) -> Dict[str, Any]:
    from torch.utils.data import DataLoader

    from faknow.data.dataset.multi_modal import MultiModalDataset
    from faknow.data.process.text_process import TokenizerFromPreTrained
    from faknow.evaluate.evaluator import Evaluator
    from faknow.model.content_based.multi_modal.mcan import MCAN
    from faknow.train.trainer import BaseTrainer

    model_dir = dataset_dir / "mcan"
    train_path = str(model_dir / "train.json")
    val_path = str(model_dir / "val.json")
    test_path = str(model_dir / "test.json")

    bert_name = bert_for_dataset(dataset, cfg)
    tokenizer = TokenizerFromPreTrained(cfg.mcan_max_len, bert_name, text_preprocessing_mcan)
    train_set = MultiModalDataset(train_path, ["text"], tokenizer, ["image"], transform_mcan_local)
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    val_loader = None
    if Path(val_path).exists():
        val_set = MultiModalDataset(val_path, ["text"], tokenizer, ["image"], transform_mcan_local)
        val_loader = DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    model = MCAN(bert_name)
    optimizer = get_optimizer_mcan_local(model, lr=cfg.mcan_lr, weight_decay=cfg.mcan_weight_decay)
    scheduler = get_scheduler_mcan_local(len(train_loader), cfg.epochs, optimizer, warm_up_percentage=cfg.mcan_warmup_ratio)
    evaluator = Evaluator(cfg.metrics)
    trainer = BaseTrainer(
        model=model,
        evaluator=evaluator,
        optimizer=optimizer,
        scheduler=scheduler,
        clip_grad_norm={"max_norm": 1.0},
        device=cfg.device,
    )
    trainer.fit(train_loader=train_loader, num_epochs=cfg.epochs, validate_loader=val_loader, save_best=None)

    result: Dict[str, Any] = {"train_size": len(train_set), "val_size": len(val_loader.dataset) if val_loader else 0}
    if Path(test_path).exists():
        test_set = MultiModalDataset(test_path, ["text"], tokenizer, ["image"], transform_mcan_local)
        test_loader = DataLoader(test_set, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
        result["test_size"] = len(test_set)
        result["test"] = to_plain(trainer.evaluate(test_loader))
    return result


def train_eann(dataset: str, dataset_dir: Path, cfg: argparse.Namespace) -> Dict[str, Any]:
    from torch.utils.data import DataLoader

    from faknow.data.dataset.multi_modal import MultiModalDataset
    from faknow.evaluate.evaluator import Evaluator
    from faknow.model.content_based.multi_modal.eann import EANN
    from faknow.train.trainer import BaseTrainer

    model_dir = dataset_dir / "eann"
    train_path = str(model_dir / "train.json")
    val_path = str(model_dir / "val.json")
    test_path = str(model_dir / "test.json")
    vocab_path = model_dir / "vocab.pkl"
    word_vec_path = model_dir / "word_vectors.pkl"
    stop_words_path = model_dir / "stop_words.txt"

    vocab = load_pickle(vocab_path)
    word_vectors = load_pickle(word_vec_path)
    if not torch.is_tensor(word_vectors):
        word_vectors = torch.tensor(word_vectors, dtype=torch.float32)
    stop_words = [x.strip() for x in stop_words_path.read_text(encoding="utf-8").splitlines() if x.strip()]
    language = language_for(dataset)
    tokenizer = TokenizerEANNFixed(vocab, max_len=cfg.eann_max_len, stop_words=stop_words, language=language)

    train_set = MultiModalDataset(train_path, ["text"], tokenizer, ["image"], transform_eann_local)
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    val_loader = None
    if Path(val_path).exists():
        val_set = MultiModalDataset(val_path, ["text"], tokenizer, ["image"], transform_eann_local)
        val_loader = DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    event_num = int(cfg.eann_event_num)
    if event_num <= 0:
        train_rows = load_json(Path(train_path))
        max_domain = 0
        for row in train_rows:
            max_domain = max(max_domain, int(row.get("domain", 0)))
        event_num = max_domain + 1
    model = EANN(event_num=event_num, embed_weight=word_vectors)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.eann_lr)
    evaluator = Evaluator(cfg.metrics)
    trainer = BaseTrainer(model=model, evaluator=evaluator, optimizer=optimizer, device=cfg.device)
    trainer.fit(train_loader=train_loader, num_epochs=cfg.epochs, validate_loader=val_loader, save_best=None)

    result: Dict[str, Any] = {
        "train_size": len(train_set),
        "val_size": len(val_loader.dataset) if val_loader else 0,
        "event_num": event_num,
    }
    if Path(test_path).exists():
        test_set = MultiModalDataset(test_path, ["text"], tokenizer, ["image"], transform_eann_local)
        test_loader = DataLoader(test_set, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
        result["test_size"] = len(test_set)
        result["test"] = to_plain(trainer.evaluate(test_loader))
    return result


def _load_adj_matrix(path: Path, node_num: int) -> torch.Tensor:
    adj = load_json(path)
    mat = torch.zeros((node_num, node_num), dtype=torch.float32)
    for src, dsts in adj.items():
        i = int(src)
        if not dsts:
            continue
        mat[i, torch.tensor(dsts, dtype=torch.long)] = 1.0
    return mat


def train_mfan(dataset: str, dataset_dir: Path, cfg: argparse.Namespace) -> Dict[str, Any]:
    from torch.utils.data import DataLoader

    from faknow.data.dataset.multi_modal import MultiModalDataset
    from faknow.evaluate.evaluator import Evaluator
    from faknow.model.content_based.multi_modal.mfan import MFAN
    from faknow.train.trainer import BaseTrainer
    from faknow.utils.pgd import PGD

    model_dir = dataset_dir / "mfan"
    train_path = str(model_dir / "train.json")
    val_path = str(model_dir / "val.json")
    test_path = str(model_dir / "test.json")
    vocab_path = model_dir / "vocab.pkl"
    word_vec_path = model_dir / "word_vectors.pkl"
    node_embedding_path = model_dir / "node_embedding.pkl"
    adjacency_path = model_dir / "adjacency.json"

    vocab = load_pickle(vocab_path)
    word_vectors = load_pickle(word_vec_path)
    node_embedding = load_pickle(node_embedding_path)
    if not torch.is_tensor(word_vectors):
        word_vectors = torch.tensor(word_vectors, dtype=torch.float32)
    if not torch.is_tensor(node_embedding):
        node_embedding = torch.tensor(node_embedding, dtype=torch.float32)
    node_num = int(node_embedding.shape[0])
    adj_matrix = _load_adj_matrix(adjacency_path, node_num=node_num)

    language = language_for(dataset)
    tokenizer = TokenizerMFANFixed(vocab, max_len=cfg.mfan_max_len, language=language)
    train_set = MultiModalDataset(train_path, ["text"], tokenizer, ["image"], transform_hmcan_local)
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    val_loader = None
    if Path(val_path).exists():
        val_set = MultiModalDataset(val_path, ["text"], tokenizer, ["image"], transform_hmcan_local)
        val_loader = DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    class MFANTrainerFixed(BaseTrainer):
        def _train_epoch(self, loader, epoch: int) -> Dict[str, float]:
            import torch.nn.functional as F

            self.model.train()
            pgd_word = PGD(self.model, emb_name="word_embedding", epsilon=6, alpha=1.8)
            losses: Optional[Dict[str, torch.Tensor]] = None

            for batch_data in loader:
                batch_data = self._move_data_to_device(batch_data)
                losses = self.model.calculate_loss(batch_data)
                loss_defense = losses["total_loss"]
                self.optimizer.zero_grad()
                loss_defense.backward()

                k = 3
                pgd_word.backup_grad()
                for t in range(k):
                    pgd_word.attack(is_first_attack=(t == 0))
                    if t != k - 1:
                        self.model.zero_grad()
                    else:
                        pgd_word.restore_grad()
                    y_pred = self.model.predict(batch_data)
                    loss_adv = F.cross_entropy(y_pred, batch_data["label"])
                    loss_adv.backward()
                pgd_word.restore()
                self.optimizer.step()

            if losses is None:
                return {"total_loss": 0.0, "class_loss": 0.0, "dis_loss": 0.0}
            return {k: float(v.item()) for k, v in losses.items()}

    model = MFAN(word_vectors, node_num=node_num, node_embedding=node_embedding, adj_matrix=adj_matrix)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.mfan_lr)
    evaluator = Evaluator(cfg.metrics)
    trainer = MFANTrainerFixed(model=model, evaluator=evaluator, optimizer=optimizer, device=cfg.device)
    trainer.fit(train_loader=train_loader, num_epochs=cfg.epochs, validate_loader=val_loader, save_best=None)

    result: Dict[str, Any] = {
        "train_size": len(train_set),
        "val_size": len(val_loader.dataset) if val_loader else 0,
        "node_num": node_num,
        "node_dim": int(node_embedding.shape[1]),
    }
    if Path(test_path).exists():
        test_set = MultiModalDataset(test_path, ["text"], tokenizer, ["image"], transform_hmcan_local)
        test_loader = DataLoader(test_set, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
        result["test_size"] = len(test_set)
        result["test"] = to_plain(trainer.evaluate(test_loader))
    return result


class SAFENumpyDatasetCompat(torch.utils.data.Dataset):
    def __init__(self, root_dir: str):
        super().__init__()
        root = Path(root_dir)
        self.x_heads = np.load(root / "case_headline.npy", allow_pickle=True).astype(np.float32)
        self.x_bodies = np.load(root / "case_body.npy", allow_pickle=True).astype(np.float32)
        self.x_images = np.load(root / "case_image.npy", allow_pickle=True).astype(np.float32)
        self.y = np.load(root / "case_y_fn_dim1.npy").astype(np.float32)
        assert self.x_heads.shape[0] == self.x_bodies.shape[0] == self.x_images.shape[0] == self.y.shape[0]

    def __len__(self):
        return int(self.x_heads.shape[0])

    def __getitem__(self, index: int):
        return {
            "head": self.x_heads[index],
            "body": self.x_bodies[index],
            "image": self.x_images[index],
            "label": self.y[index],
        }


def train_safe(dataset_dir: Path, cfg: argparse.Namespace) -> Dict[str, Any]:
    from torch.utils.data import DataLoader

    from faknow.evaluate.evaluator import Evaluator
    from faknow.model.content_based.multi_modal.safe import SAFE
    from faknow.train.trainer import BaseTrainer

    safe_dir = dataset_dir / "safe"
    train_dir = str(safe_dir / "train")
    test_dir = str(safe_dir / "test")

    train_set = SAFENumpyDatasetCompat(train_dir)
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    model = SAFE(
        embedding_size=cfg.safe_embedding_size,
        conv_in_size=cfg.safe_conv_in_size,
        filter_num=cfg.safe_filter_num,
        cnn_out_size=cfg.safe_cnn_out_size,
        dropout=cfg.safe_dropout,
        loss_weights=None,
    )
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.safe_lr)
    evaluator = Evaluator(cfg.metrics)
    trainer = BaseTrainer(model=model, evaluator=evaluator, optimizer=optimizer, device=cfg.device)
    trainer.fit(train_loader=train_loader, num_epochs=cfg.epochs, validate_loader=None, save_best=None)

    result: Dict[str, Any] = {"train_size": len(train_set)}
    if Path(test_dir).exists():
        test_set = SAFENumpyDatasetCompat(test_dir)
        test_loader = DataLoader(test_set, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
        result["test_size"] = len(test_set)
        result["test"] = to_plain(trainer.evaluate(test_loader))
    return result


def train_cafe(dataset_dir: Path, cfg: argparse.Namespace) -> Dict[str, Any]:
    from torch.utils.data import DataLoader

    from faknow.data.dataset.cafe_dataset import CafeDataset
    from faknow.evaluate.evaluator import Evaluator
    from faknow.model.content_based.multi_modal.cafe import CAFE
    from faknow.train.cafe_trainer import CafeTrainer

    cafe_dir = dataset_dir / "cafe"
    train_set = CafeDataset(str(cafe_dir / "train_text_with_label.npz"), str(cafe_dir / "train_image_with_label.npz"))
    test_set = CafeDataset(str(cafe_dir / "test_text_with_label.npz"), str(cafe_dir / "test_image_with_label.npz"))
    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=cfg.cafe_drop_last,
        num_workers=cfg.num_workers,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=cfg.batch_size,
        shuffle=False,
        drop_last=cfg.cafe_drop_last,
        num_workers=cfg.num_workers,
    )

    model = CAFE()
    optim_task_similarity = torch.optim.Adam(
        model.similarity_module.parameters(),
        lr=cfg.cafe_lr,
        weight_decay=cfg.cafe_weight_decay,
    )
    sim_params_id = list(map(id, model.similarity_module.parameters()))
    base_params = filter(lambda p: id(p) not in sim_params_id, model.parameters())
    optim_task_detection = torch.optim.Adam(base_params, lr=cfg.cafe_lr, weight_decay=cfg.cafe_weight_decay)
    evaluator = Evaluator(cfg.metrics)
    trainer = CafeTrainer(model, evaluator, optim_task_detection, optim_task_similarity, device=cfg.device)
    trainer.fit(train_loader=train_loader, num_epochs=cfg.epochs, validate_loader=None, save_best=None)

    result: Dict[str, Any] = {"train_size": len(train_set), "drop_last": bool(cfg.cafe_drop_last)}
    if len(test_loader) > 0:
        result["test_size"] = len(test_set)
        result["test"] = to_plain(trainer.evaluate(test_loader))
    else:
        result["test_size"] = len(test_set)
        result["test"] = {"warning": "test loader is empty because drop_last=True and dataset smaller than batch size"}
    return result


def train_one_model(dataset: str, model: str, dataset_dir: Path, cfg: argparse.Namespace) -> Dict[str, Any]:
    if model == "spotfake":
        return train_spotfake(dataset, dataset_dir, cfg)
    if model == "hmcan":
        return train_hmcan(dataset, dataset_dir, cfg)
    if model == "mcan":
        return train_mcan(dataset, dataset_dir, cfg)
    if model == "eann":
        return train_eann(dataset, dataset_dir, cfg)
    if model == "mfan":
        return train_mfan(dataset, dataset_dir, cfg)
    if model == "safe":
        return train_safe(dataset_dir, cfg)
    if model == "cafe":
        return train_cafe(dataset_dir, cfg)
    raise ValueError(f"Unsupported model: {model}")


def ensure_faknow_available() -> None:
    try:
        import faknow  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "faknow is required for training. Install with: pip install faknow==0.0.4"
        ) from e


def run_pipeline(cfg: argparse.Namespace) -> Dict[str, Any]:
    set_seed(cfg.seed)
    datasets, models = infer_targets(cfg.dataset, cfg.model)
    artifacts_root = Path(cfg.artifacts_dir).resolve()
    runs_root = ensure_dir(Path(cfg.runs_dir).resolve())
    report: Dict[str, Any] = {
        "config": to_plain(vars(cfg)),
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "prepared": {},
        "runs": [],
        "errors": [],
    }

    need_prepare = cfg.prepare_only or cfg.all_steps
    need_train = cfg.train_only or cfg.all_steps

    if need_prepare:
        for dataset in datasets:
            logging.info("Preparing dataset artifacts: %s", dataset)
            prep_result = prepare_stage(cfg, dataset)
            report["prepared"][dataset] = {
                "artifact_root": str(artifacts_root / dataset),
                "report_keys": list(prep_result.get("datasets", {}).keys()),
            }

    if not need_train:
        report["finished_at"] = datetime.now().isoformat(timespec="seconds")
        return report

    if cfg.quick_prepare:
        raise RuntimeError(
            "quick_prepare only generates lightweight json artifacts. "
            "Use --no-quick-prepare (default) before training EANN/MFAN/SAFE/CAFE."
        )

    ensure_faknow_available()

    for dataset in datasets:
        dataset_dir = artifacts_root / dataset
        if not dataset_dir.exists():
            raise FileNotFoundError(
                f"Artifact directory does not exist: {dataset_dir}. "
                "Run with --prepare-only or --all first."
            )
        for model in models:
            try:
                ensure_artifacts_for_model(dataset_dir, model)
                run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
                run_dir = ensure_dir(runs_root / dataset / model / run_id)
                snapshot = {
                    "dataset": dataset,
                    "model": model,
                    "run_id": run_id,
                    "artifact_dataset_dir": str(dataset_dir),
                    "device": cfg.device,
                    "epochs": cfg.epochs,
                    "batch_size": cfg.batch_size,
                }
                save_json(snapshot, run_dir / "config_snapshot.json")
                logging.info("Training start dataset=%s model=%s", dataset, model)
                with pushd(run_dir):
                    metrics = train_one_model(dataset, model, dataset_dir, cfg)
                removed = remove_weight_files(run_dir)
                run_result = {
                    "dataset": dataset,
                    "model": model,
                    "run_dir": str(run_dir),
                    "metrics": to_plain(metrics),
                    "removed_weight_files": removed,
                }
                save_json(run_result, run_dir / "metrics.json")
                report["runs"].append(run_result)
                logging.info("Training done dataset=%s model=%s run_dir=%s", dataset, model, run_dir)
            except Exception as e:  # noqa: PERF203
                err = {
                    "dataset": dataset,
                    "model": model,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                }
                report["errors"].append(err)
                logging.error("Training failed dataset=%s model=%s: %s", dataset, model, e)
                if not cfg.continue_on_error:
                    raise

    report["finished_at"] = datetime.now().isoformat(timespec="seconds")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified FaKnow adapter runner for CFND/Gossip/Weibo (SpotFake/HMCAN/MCAN/EANN/MFAN/SAFE/CAFE)."
    )
    parser.add_argument("--dataset", choices=[*DATASETS, "all"], default="all")
    parser.add_argument("--model", choices=[*MODELS, "all"], default="all")

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare-only", action="store_true", help="Only prepare artifacts.")
    mode.add_argument("--train-only", action="store_true", help="Only train from existing artifacts.")
    mode.add_argument("--all", dest="all_steps", action="store_true", help="Prepare then train.")

    parser.add_argument(
        "--artifacts-dir",
        type=str,
        default=str((Path(__file__).resolve().parent / "artifacts").resolve()),
    )
    parser.add_argument(
        "--runs-dir",
        type=str,
        default=str((Path(__file__).resolve().parent / "runs").resolve()),
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--metrics", type=str, default="accuracy,precision,recall,f1")
    parser.add_argument("--quick-prepare", action="store_true", help="Only prepare lightweight json artifacts (fast smoke-test mode).")

    parser.add_argument("--bert-zh", type=str, default="bert-base-chinese")
    parser.add_argument("--bert-en", type=str, default="bert-base-uncased")

    parser.add_argument("--spotfake-max-len", type=int, default=500)
    parser.add_argument("--spotfake-lr", type=float, default=3e-5)
    parser.add_argument("--spotfake-finetune-text", action="store_true")
    parser.add_argument("--spotfake-finetune-vision", action="store_true")

    parser.add_argument("--hmcan-max-len", type=int, default=20)
    parser.add_argument("--hmcan-lr", type=float, default=1e-3)
    parser.add_argument("--hmcan-left-layers", type=int, default=2)
    parser.add_argument("--hmcan-left-heads", type=int, default=12)
    parser.add_argument("--hmcan-right-layers", type=int, default=2)
    parser.add_argument("--hmcan-right-heads", type=int, default=12)
    parser.add_argument("--hmcan-dropout", type=float, default=0.1)
    parser.add_argument("--hmcan-alpha", type=float, default=0.7)

    parser.add_argument("--mcan-max-len", type=int, default=255)
    parser.add_argument("--mcan-lr", type=float, default=1e-4)
    parser.add_argument("--mcan-weight-decay", type=float, default=0.15)
    parser.add_argument("--mcan-warmup-ratio", type=float, default=0.1)

    parser.add_argument("--eann-max-len", type=int, default=255)
    parser.add_argument("--eann-lr", type=float, default=1e-3)
    parser.add_argument("--eann-event-num", type=int, default=1, help="Set to <=0 to infer from train.json domains.")

    parser.add_argument("--mfan-max-len", type=int, default=50)
    parser.add_argument("--mfan-lr", type=float, default=2e-3)

    parser.add_argument("--safe-lr", type=float, default=2.5e-4)
    parser.add_argument("--safe-embedding-size", type=int, default=300)
    parser.add_argument("--safe-conv-in-size", type=int, default=32)
    parser.add_argument("--safe-filter-num", type=int, default=128)
    parser.add_argument("--safe-cnn-out-size", type=int, default=200)
    parser.add_argument("--safe-dropout", type=float, default=0.0)

    parser.add_argument("--cafe-lr", type=float, default=1e-3)
    parser.add_argument("--cafe-weight-decay", type=float, default=0.0)
    parser.add_argument("--cafe-drop-last", dest="cafe_drop_last", action="store_true")
    parser.add_argument("--cafe-keep-last", dest="cafe_drop_last", action="store_false")

    parser.add_argument("--cfnd-root", type=str, default=PREP_DEFAULTS["cfnd_root"])
    parser.add_argument("--gossip-root", type=str, default=PREP_DEFAULTS["gossip_root"])
    parser.add_argument("--weibo-root", type=str, default=PREP_DEFAULTS["weibo_root"])
    parser.add_argument("--gossip-image-root", type=str, default=PREP_DEFAULTS["gossip_image_root"])
    parser.add_argument("--auto-prepare-data", dest="auto_prepare_data", action="store_true")
    parser.add_argument("--no-auto-prepare-data", dest="auto_prepare_data", action="store_false")
    parser.add_argument("--force-prepare-data", action="store_true")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--prepare-text-model-zh", type=str, default=PREP_DEFAULTS["text_model_zh"])
    parser.add_argument("--prepare-text-model-en", type=str, default=PREP_DEFAULTS["text_model_en"])
    parser.add_argument("--prepare-device", type=str, default=None)
    parser.add_argument("--prepare-batch-size", type=int, default=16)
    parser.add_argument("--min-freq", type=int, default=2)
    parser.add_argument("--mfan-knn", type=int, default=10)
    parser.add_argument("--safe-head-len", type=int, default=32)
    parser.add_argument("--safe-body-len", type=int, default=128)
    parser.add_argument("--safe-image-len", type=int, default=32)
    parser.add_argument("--cafe-text-len", type=int, default=30)
    parser.add_argument("--cafe-text-dim", type=int, default=200)
    parser.add_argument("--validate-image-samples", type=int, default=50)

    parser.set_defaults(auto_prepare_data=True, cafe_drop_last=True)
    cfg = parser.parse_args()
    if not cfg.prepare_only and not cfg.train_only and not cfg.all_steps:
        cfg.all_steps = True
    cfg.metrics = parse_metrics(cfg.metrics) or DEFAULT_METRICS
    return cfg


def main() -> None:
    setup_logging()
    cfg = parse_args()
    logging.info("Run config: %s", json.dumps(to_plain(vars(cfg)), ensure_ascii=False, indent=2))
    report = run_pipeline(cfg)
    runs_root = ensure_dir(Path(cfg.runs_dir).resolve())
    report_path = runs_root / "run_report.json"
    save_json(report, report_path)
    logging.info("Run report saved: %s", report_path)
    if report.get("errors"):
        raise RuntimeError(f"Finished with {len(report['errors'])} errors. Check {report_path}")


if __name__ == "__main__":
    main()
