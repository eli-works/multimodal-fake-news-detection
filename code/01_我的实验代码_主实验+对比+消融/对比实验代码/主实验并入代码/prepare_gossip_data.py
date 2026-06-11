import argparse
import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass
class PreprocessConfig:
    dataset_root: str = "../AAAI_dataset"
    output_dir: str = "gossip"
    val_ratio: float = 0.1
    seed: int = 42
    train_raw_csv: str = "gossip_train.csv"
    test_raw_csv: str = "gossip_test.csv"
    text_column_raw: str = "content"
    image_column_raw: str = "image"
    label_column_raw: str = "label"


def normalize_null(value: object) -> str:
    text = str(value).strip()
    if not text or text.lower() == "null":
        return ""
    return text


def clean_text(value: object) -> str:
    text = normalize_null(value)
    text = text.replace("\u200b", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_label(raw: object) -> int:
    text = normalize_null(raw).lower()
    if text in ("0", "1"):
        return int(text)
    if text in ("false", "fake", "rumor"):
        return 1
    if text in ("true", "real", "nonrumor", "non_rumor"):
        return 0
    raise ValueError(f"Unexpected label value: {raw}")


def summarize_split(df: pd.DataFrame, split_name: str) -> Dict[str, object]:
    if df.empty:
        return {
            "split": split_name,
            "samples": 0,
            "label_0": 0,
            "label_1": 0,
            "with_local_image": 0,
            "missing_local_image": 0,
        }

    total = int(len(df))
    label_0 = int((df["label"] == 0).sum())
    label_1 = int((df["label"] == 1).sum())
    with_local = int(df["image"].astype(str).str.len().gt(0).sum())
    return {
        "split": split_name,
        "samples": total,
        "label_0": label_0,
        "label_1": label_1,
        "with_local_image": with_local,
        "missing_local_image": total - with_local,
    }


def build_records(
    df: pd.DataFrame,
    split_source: str,
    dataset_root: Path,
    cfg: PreprocessConfig,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    records = []
    with_local_images = 0

    image_subdir = f"Images/gossip_{split_source}"
    for row_idx, row in df.iterrows():
        text = clean_text(row.get(cfg.text_column_raw, ""))
        image_name = clean_text(row.get(cfg.image_column_raw, ""))
        label = normalize_label(row.get(cfg.label_column_raw, ""))
        raw_id = clean_text(row.get("Unnamed: 0", row_idx))

        rel_image = f"{image_subdir}/{image_name}" if image_name else ""
        abs_image = dataset_root / rel_image if rel_image else None
        if abs_image is not None and abs_image.exists():
            with_local_images += 1
        else:
            rel_image = ""

        records.append(
            {
                "sample_id": f"gossip_{split_source}_{row_idx}_{raw_id}",
                "raw_id": raw_id,
                "text": text,
                "image": rel_image,
                "label": label,
                "split_source": split_source,
            }
        )

    out_df = pd.DataFrame(records)
    stats = {
        "split_source": split_source,
        "records": int(len(out_df)),
        "with_local_images": int(with_local_images),
        "missing_local_images": int(len(out_df) - with_local_images),
    }
    return out_df, stats


def run_preprocess(cfg: PreprocessConfig) -> Dict[str, str]:
    dataset_root = Path(cfg.dataset_root).resolve()
    output_dir = Path(cfg.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_raw_path = dataset_root / cfg.train_raw_csv
    test_raw_path = dataset_root / cfg.test_raw_csv
    if not train_raw_path.exists():
        raise FileNotFoundError(f"Missing train raw csv: {train_raw_path}")
    if not test_raw_path.exists():
        raise FileNotFoundError(f"Missing test raw csv: {test_raw_path}")

    train_raw_df = pd.read_csv(train_raw_path, keep_default_na=False)
    test_raw_df = pd.read_csv(test_raw_path, keep_default_na=False)
    for col in (cfg.text_column_raw, cfg.image_column_raw, cfg.label_column_raw):
        if col not in train_raw_df.columns:
            raise KeyError(f"Missing required column `{col}` in {train_raw_path}")
        if col not in test_raw_df.columns:
            raise KeyError(f"Missing required column `{col}` in {test_raw_path}")

    train_full_df, train_stats = build_records(train_raw_df, "train", dataset_root, cfg)
    test_df, test_stats = build_records(test_raw_df, "test", dataset_root, cfg)
    if train_full_df.empty or test_df.empty:
        raise RuntimeError("Parsed gossip dataset is empty. Please check raw csv files.")

    train_df, val_df = train_test_split(
        train_full_df,
        test_size=cfg.val_ratio,
        random_state=cfg.seed,
        shuffle=True,
        stratify=train_full_df["label"],
    )

    train_df = train_df.sample(frac=1.0, random_state=cfg.seed).reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    train_full_df = train_full_df.reset_index(drop=True)

    train_path = output_dir / "train_gossip.csv"
    val_path = output_dir / "val_gossip.csv"
    test_path = output_dir / "test_gossip.csv"
    train_full_path = output_dir / "train_full_gossip.csv"
    stats_path = output_dir / "gossip_data_stats.json"

    train_df.to_csv(train_path, index=False, encoding="utf-8-sig")
    val_df.to_csv(val_path, index=False, encoding="utf-8-sig")
    test_df.to_csv(test_path, index=False, encoding="utf-8-sig")
    train_full_df.to_csv(train_full_path, index=False, encoding="utf-8-sig")

    stats_payload = {
        "config": asdict(cfg),
        "dataset_root": str(dataset_root),
        "output_dir": str(output_dir),
        "raw_files": {
            "train_raw_csv": str(train_raw_path),
            "test_raw_csv": str(test_raw_path),
        },
        "raw_parse_stats": {
            "train_raw": train_stats,
            "test_raw": test_stats,
        },
        "split_summary": {
            "train": summarize_split(train_df, "train"),
            "val": summarize_split(val_df, "val"),
            "test": summarize_split(test_df, "test"),
            "train_full": summarize_split(train_full_df, "train_full"),
        },
        "output_files": {
            "train_csv": str(train_path),
            "val_csv": str(val_path),
            "test_csv": str(test_path),
            "train_full_csv": str(train_full_path),
        },
        "label_note": "Labels keep source semantics (0/1) without remapping.",
    }
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats_payload, f, ensure_ascii=False, indent=2)

    logging.info("Saved train csv: %s (%d rows)", train_path, len(train_df))
    logging.info("Saved val csv: %s (%d rows)", val_path, len(val_df))
    logging.info("Saved test csv: %s (%d rows)", test_path, len(test_df))
    logging.info("Saved stats json: %s", stats_path)
    logging.info(
        "Train image hit-rate: %.2f%%",
        100.0 * summarize_split(train_df, "train")["with_local_image"] / max(len(train_df), 1),
    )
    logging.info(
        "Val image hit-rate: %.2f%%",
        100.0 * summarize_split(val_df, "val")["with_local_image"] / max(len(val_df), 1),
    )
    logging.info(
        "Test image hit-rate: %.2f%%",
        100.0 * summarize_split(test_df, "test")["with_local_image"] / max(len(test_df), 1),
    )

    return {
        "train_csv": str(train_path),
        "val_csv": str(val_path),
        "test_csv": str(test_path),
        "stats_json": str(stats_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare AAAI Gossip dataset for multimodal training.")
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="../AAAI_dataset",
        help="Root directory containing gossip_train.csv, gossip_test.csv and Images/",
    )
    parser.add_argument("--output-dir", type=str, default="gossip", help="Directory to save generated CSV and stats.")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio from original train split.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for data split.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    cfg = PreprocessConfig(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    run_preprocess(cfg)


if __name__ == "__main__":
    main()
