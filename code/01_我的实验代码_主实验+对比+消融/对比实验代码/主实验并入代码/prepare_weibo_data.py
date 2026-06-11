import argparse
import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
from urllib.parse import urlparse

import pandas as pd
from sklearn.model_selection import train_test_split


META_COLUMNS: Sequence[str] = (
    "post_id",
    "user_name",
    "post_url",
    "verify_desc",
    "publish_time",
    "is_original",
    "repost_count",
    "comment_count",
    "like_count",
    "user_id",
    "is_mentioned",
    "follow_count",
    "follower_count",
    "status_count",
    "source",
)

RAW_SPECS: Sequence[Tuple[str, int, str, str]] = (
    ("train_rumor.txt", 1, "train", "rumor_images"),
    ("train_nonrumor.txt", 0, "train", "nonrumor_images"),
    ("test_rumor.txt", 1, "test", "rumor_images"),
    ("test_nonrumor.txt", 0, "test", "nonrumor_images"),
)

ENCODINGS: Sequence[str] = ("utf-8-sig", "utf-8", "gb18030", "gbk")


@dataclass
class PreprocessConfig:
    dataset_root: str = "weibo"
    output_dir: str = "weibo"
    val_ratio: float = 0.1
    seed: int = 42


def normalize_null(value: object) -> str:
    text = str(value).strip()
    if not text or text.lower() == "null":
        return ""
    return text


def clean_text(text: object) -> str:
    value = normalize_null(text)
    value = value.replace("\u200b", " ")
    value = re.sub(r"\s+", " ", value).strip()
    return value


def read_lines_with_fallback(path: Path) -> Tuple[List[str], str]:
    last_error = None
    for enc in ENCODINGS:
        try:
            with path.open("r", encoding=enc) as f:
                lines = [line.rstrip("\r\n") for line in f]
            return lines, enc
        except UnicodeDecodeError as err:
            last_error = err
    raise UnicodeDecodeError(
        "unknown",
        b"",
        0,
        1,
        f"Unable to decode {path} with encodings={list(ENCODINGS)}. Last error: {last_error}",
    )


def split_meta_fields(meta_line: str) -> Dict[str, str]:
    fields = meta_line.split("|")
    if len(fields) < len(META_COLUMNS):
        fields.extend([""] * (len(META_COLUMNS) - len(fields)))
    elif len(fields) > len(META_COLUMNS):
        fields = fields[: len(META_COLUMNS) - 1] + ["|".join(fields[len(META_COLUMNS) - 1 :])]
    return {name: normalize_null(value) for name, value in zip(META_COLUMNS, fields)}


def build_image_maps(dataset_root: Path) -> Dict[str, Dict[str, str]]:
    maps: Dict[str, Dict[str, str]] = {}
    for folder in ("rumor_images", "nonrumor_images"):
        folder_path = dataset_root / folder
        image_map: Dict[str, str] = {}
        for image_path in folder_path.iterdir():
            if image_path.is_file():
                image_map[image_path.name.lower()] = f"{folder}/{image_path.name}"
        maps[folder] = image_map
    return maps


def parse_image_urls(raw_line: str) -> List[str]:
    urls = [normalize_null(part) for part in raw_line.split("|")]
    return [url for url in urls if url]


def url_to_filename(url: str) -> str:
    parsed = urlparse(url)
    path = parsed.path if parsed.path else url
    filename = Path(path).name
    return normalize_null(filename)


def select_local_image(
    image_urls: Sequence[str],
    preferred_folder: str,
    image_maps: Dict[str, Dict[str, str]],
) -> Tuple[str, str]:
    primary = image_maps.get(preferred_folder, {})
    secondary_folder = "nonrumor_images" if preferred_folder == "rumor_images" else "rumor_images"
    secondary = image_maps.get(secondary_folder, {})

    for url in image_urls:
        filename = url_to_filename(url)
        if not filename:
            continue
        key = filename.lower()
        if key in primary:
            return primary[key], url
        if key in secondary:
            return secondary[key], url
    return "", ""


def parse_raw_file(
    path: Path,
    label: int,
    split_source: str,
    preferred_folder: str,
    image_maps: Dict[str, Dict[str, str]],
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    lines, encoding = read_lines_with_fallback(path)
    if len(lines) % 3 != 0:
        raise ValueError(f"{path} has {len(lines)} lines, cannot be grouped by 3.")

    records: List[Dict[str, object]] = []
    with_urls = 0
    with_local_images = 0

    for i in range(0, len(lines), 3):
        meta_line = lines[i]
        image_line = lines[i + 1]
        text_line = lines[i + 2]

        meta = split_meta_fields(meta_line)
        image_urls = parse_image_urls(image_line)
        local_image, selected_image_url = select_local_image(image_urls, preferred_folder, image_maps)

        if image_urls:
            with_urls += 1
        if local_image:
            with_local_images += 1

        post_id = meta["post_id"] or f"{path.stem}_{i // 3}"
        text = clean_text(text_line)
        if not text:
            # If post body is unexpectedly empty, fall back to metadata user name.
            text = clean_text(meta.get("user_name", ""))

        record = {
            "sample_id": f"{path.stem}_{i // 3}_{post_id}",
            "post_id": post_id,
            "text": text,
            "image": local_image,
            "label": int(label),
            "split_source": split_source,
            "source_file": path.name,
            "publish_time": meta.get("publish_time", ""),
            "user_id": meta.get("user_id", ""),
            "user_name": meta.get("user_name", ""),
            "repost_count": meta.get("repost_count", ""),
            "comment_count": meta.get("comment_count", ""),
            "like_count": meta.get("like_count", ""),
            "raw_image_urls": "|".join(image_urls),
            "selected_image_url": selected_image_url,
            "image_url_count": len(image_urls),
            "local_image_found": int(bool(local_image)),
        }
        records.append(record)

    stats = {
        "file": path.name,
        "encoding": encoding,
        "records": len(records),
        "with_image_urls": with_urls,
        "with_local_images": with_local_images,
        "missing_local_images": len(records) - with_local_images,
    }
    return records, stats


def summarize_split(df: pd.DataFrame, split_name: str) -> Dict[str, object]:
    if df.empty:
        return {
            "split": split_name,
            "samples": 0,
            "rumor": 0,
            "nonrumor": 0,
            "with_local_image": 0,
            "missing_local_image": 0,
        }

    with_local = int(df["image"].astype(str).str.len().gt(0).sum())
    total = int(len(df))
    rumor = int((df["label"] == 1).sum())
    nonrumor = int((df["label"] == 0).sum())
    return {
        "split": split_name,
        "samples": total,
        "rumor": rumor,
        "nonrumor": nonrumor,
        "with_local_image": with_local,
        "missing_local_image": total - with_local,
    }


def run_preprocess(cfg: PreprocessConfig) -> Dict[str, str]:
    dataset_root = Path(cfg.dataset_root).resolve()
    output_dir = Path(cfg.output_dir).resolve()
    tweets_dir = dataset_root / "tweets"
    if not tweets_dir.exists():
        raise FileNotFoundError(f"Cannot find tweets directory: {tweets_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    image_maps = build_image_maps(dataset_root)

    train_records: List[Dict[str, object]] = []
    test_records: List[Dict[str, object]] = []
    per_file_stats: List[Dict[str, object]] = []

    for filename, label, split_source, preferred_folder in RAW_SPECS:
        file_path = tweets_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Missing required file: {file_path}")
        parsed_records, stats = parse_raw_file(
            file_path,
            label=label,
            split_source=split_source,
            preferred_folder=preferred_folder,
            image_maps=image_maps,
        )
        per_file_stats.append(stats)
        if split_source == "train":
            train_records.extend(parsed_records)
        else:
            test_records.extend(parsed_records)

    train_full_df = pd.DataFrame(train_records)
    test_df = pd.DataFrame(test_records)
    if train_full_df.empty or test_df.empty:
        raise RuntimeError("Parsed dataset is empty. Please check source files.")

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

    train_path = output_dir / "train_weibo.csv"
    val_path = output_dir / "val_weibo.csv"
    test_path = output_dir / "test_weibo.csv"
    full_train_path = output_dir / "train_full_weibo.csv"
    stats_path = output_dir / "weibo_data_stats.json"

    train_df.to_csv(train_path, index=False, encoding="utf-8-sig")
    val_df.to_csv(val_path, index=False, encoding="utf-8-sig")
    test_df.to_csv(test_path, index=False, encoding="utf-8-sig")
    train_full_df.to_csv(full_train_path, index=False, encoding="utf-8-sig")

    stats_payload = {
        "config": asdict(cfg),
        "dataset_root": str(dataset_root),
        "output_dir": str(output_dir),
        "file_level_stats": per_file_stats,
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
            "train_full_csv": str(full_train_path),
        },
    }
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats_payload, f, ensure_ascii=False, indent=2)

    logging.info("Saved train csv: %s (%d rows)", train_path, len(train_df))
    logging.info("Saved val csv: %s (%d rows)", val_path, len(val_df))
    logging.info("Saved test csv: %s (%d rows)", test_path, len(test_df))
    logging.info("Saved stats json: %s", stats_path)
    logging.info("Train image hit-rate: %.2f%%", 100.0 * summarize_split(train_df, "train")["with_local_image"] / max(len(train_df), 1))
    logging.info("Val image hit-rate: %.2f%%", 100.0 * summarize_split(val_df, "val")["with_local_image"] / max(len(val_df), 1))
    logging.info("Test image hit-rate: %.2f%%", 100.0 * summarize_split(test_df, "test")["with_local_image"] / max(len(test_df), 1))

    return {
        "train_csv": str(train_path),
        "val_csv": str(val_path),
        "test_csv": str(test_path),
        "stats_json": str(stats_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Weibo rumor dataset for multimodal training.")
    parser.add_argument("--dataset-root", type=str, default="weibo", help="Root directory containing tweets/ and image folders.")
    parser.add_argument("--output-dir", type=str, default="weibo", help="Directory to save generated CSV and stats.")
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
