"""将多个数据集的 label_distribution.png 按横向或纵向拼接。"""

import argparse
from pathlib import Path
from typing import List, Tuple

from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge label_distribution.png from weibo/gossip/cfnd.")
    parser.add_argument("--viz-root", type=str, default="C:/Users/gan/Desktop/code/viz", help="Root directory containing dataset folders.")
    parser.add_argument("--datasets", nargs="+", default=["weibo", "gossip", "cfnd"], help="Dataset folder names in order.")
    parser.add_argument("--orientation", choices=["horizontal", "vertical"], default="horizontal", help="Merge orientation.")
    parser.add_argument("--gap", type=int, default=24, help="Gap between panels in pixels.")
    parser.add_argument("--padding", type=int, default=24, help="Canvas padding in pixels.")
    parser.add_argument("--title-height", type=int, default=44, help="Title area height for each panel in pixels.")
    parser.add_argument(
        "--output",
        type=str,
        default="C:/Users/gan/Desktop/code/viz/label_distribution_combined.png",
        help="Output image path.",
    )
    return parser.parse_args()


def try_load_font(size: int = 22) -> ImageFont.ImageFont:
    candidates = [
        Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
        Path(r"C:\Windows\Fonts\arial.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def load_panels(viz_root: Path, datasets: List[str]) -> List[Tuple[str, Image.Image]]:
    panels: List[Tuple[str, Image.Image]] = []
    for ds in datasets:
        img_path = viz_root / ds / "figures" / "label_distribution.png"
        if not img_path.exists():
            raise FileNotFoundError(f"Missing image: {img_path}")
        panels.append((ds, Image.open(img_path).convert("RGB")))
    return panels


def merge_panels(
    panels: List[Tuple[str, Image.Image]],
    orientation: str,
    gap: int,
    padding: int,
    title_height: int,
) -> Image.Image:
    widths = [img.width for _, img in panels]
    heights = [img.height for _, img in panels]

    if orientation == "horizontal":
        canvas_w = padding * 2 + sum(widths) + gap * (len(panels) - 1)
        canvas_h = padding * 2 + max(h + title_height for h in heights)
    else:
        canvas_w = padding * 2 + max(widths)
        canvas_h = padding * 2 + sum(h + title_height for h in heights) + gap * (len(panels) - 1)

    canvas = Image.new("RGB", (canvas_w, canvas_h), color="white")
    draw = ImageDraw.Draw(canvas)
    font = try_load_font(size=22)

    x = padding
    y = padding
    for name, img in panels:
        title = name.upper()
        title_box = draw.textbbox((0, 0), title, font=font)
        title_w = title_box[2] - title_box[0]
        if orientation == "horizontal":
            title_x = x + (img.width - title_w) // 2
            draw.text((title_x, y), title, fill=(20, 20, 20), font=font)
            canvas.paste(img, (x, y + title_height))
            x += img.width + gap
        else:
            title_x = padding + (canvas_w - 2 * padding - title_w) // 2
            draw.text((title_x, y), title, fill=(20, 20, 20), font=font)
            paste_x = padding + (canvas_w - 2 * padding - img.width) // 2
            canvas.paste(img, (paste_x, y + title_height))
            y += img.height + title_height + gap

    return canvas


def main() -> None:
    args = parse_args()
    viz_root = Path(args.viz_root).resolve()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    panels = load_panels(viz_root, args.datasets)
    merged = merge_panels(
        panels=panels,
        orientation=args.orientation,
        gap=int(args.gap),
        padding=int(args.padding),
        title_height=int(args.title_height),
    )
    merged.save(output, format="PNG")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()

