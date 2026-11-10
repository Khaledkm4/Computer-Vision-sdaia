"""
Step 1 — Data setup.

1. Download the Kaggle dataset to ``config.RAW_DIR`` (on the SSD).
2. Auto-generate YOLO bounding boxes.
   The source dataset is *classification-only* (one date per image, white
   background, ring-light). Because the background is controlled, a classical
   OpenCV foreground segmentation gives tight, reliable boxes. Images whose
   segmentation looks wrong are excluded and listed in a review CSV.
3. Stratified train/val/test split, resize, write YOLO labels + data.yaml.
4. Save a QA grid so you can eyeball the auto-labels before training.

Run:  python -m date_vision.data_setup [--skip-download] [--force]
"""
from __future__ import annotations

import argparse
import csv
import random
import shutil
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

from date_vision import config

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
SPLITS = ("train", "val", "test")


# ──────────────────────────────────────────────────────────────────────────────
# Download
# ──────────────────────────────────────────────────────────────────────────────
def download_dataset(force: bool = False) -> Path:
    """Download + unzip the Kaggle dataset into RAW_DIR on the SSD.

    Auth: place kaggle.json in ``<SSD>/.kaggle/`` or ``~/.kaggle/``, or set
    KAGGLE_USERNAME / KAGGLE_KEY. Manual alternative: download the zip from the
    Kaggle page and extract it into ``config.RAW_DIR``.
    """
    if not force and any(p.suffix.lower() in IMG_EXTS for p in config.RAW_DIR.rglob("*")):
        print(f"[data] Raw images already present in {config.RAW_DIR} — skipping download.")
        return config.RAW_DIR

    from kaggle.api.kaggle_api_extended import KaggleApi  # import here: no auth at module import

    api = KaggleApi()
    api.authenticate()
    print(f"[data] Downloading {config.KAGGLE_DATASET} (~3.4 GB) -> {config.RAW_DIR}")
    api.dataset_download_files(config.KAGGLE_DATASET, path=str(config.RAW_DIR), unzip=True, quiet=False)
    return config.RAW_DIR


# ──────────────────────────────────────────────────────────────────────────────
# Auto-labelling (classical CV on a controlled white background)
# ──────────────────────────────────────────────────────────────────────────────
def segment_fruit(img_bgr: np.ndarray, work_side: int = 512) -> tuple[np.ndarray, tuple[int, int, int, int] | None]:
    """Return (full-res binary mask, xyxy box) of the dominant fruit, or (mask, None) on failure.

    Dates are dark/saturated; the background is bright and unsaturated.
    Otsu on inverted grey + a saturation threshold separates them robustly.
    """
    h, w = img_bgr.shape[:2]
    s = min(1.0, work_side / max(h, w))
    small = cv2.resize(img_bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img_bgr
    sh, sw = small.shape[:2]

    gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    _, dark = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    sat = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)[..., 1]
    _, saturated = cv2.threshold(sat, 60, 255, cv2.THRESH_BINARY)
    mask = cv2.bitwise_or(dark, saturated)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)    # remove dust / noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=3)   # fill flash highlights

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros((h, w), np.uint8), None

    def score(c: np.ndarray) -> float:
        x, y, cw, ch = cv2.boundingRect(c)
        touches_border = x <= 1 or y <= 1 or x + cw >= sw - 1 or y + ch >= sh - 1
        return cv2.contourArea(c) * (0.2 if touches_border else 1.0)   # down-weight vignette corners

    best = max(contours, key=score)
    clean = np.zeros((sh, sw), np.uint8)
    cv2.drawContours(clean, [best], -1, 255, thickness=cv2.FILLED)     # filled -> no holes

    area_ratio = cv2.contourArea(best) / float(sh * sw)
    x, y, cw, ch = cv2.boundingRect(best)
    box_ratio = (cw * ch) / float(sh * sw)
    full_mask = cv2.resize(clean, (w, h), interpolation=cv2.INTER_NEAREST)

    if area_ratio < 0.005 or box_ratio > 0.95:           # implausible -> flag for review
        return full_mask, None

    pad = 0.02                                          # 2% padding around the fruit
    x1 = max(0, int((x - pad * sw) / s))
    y1 = max(0, int((y - pad * sh) / s))
    x2 = min(w, int((x + cw + pad * sw) / s))
    y2 = min(h, int((y + ch + pad * sh) / s))
    return full_mask, (x1, y1, x2, y2)


def resize_max_side(img: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    h, w = img.shape[:2]
    s = min(1.0, max_side / max(h, w))
    if s < 1:
        img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    return img, s


def to_yolo_line(cls_id: int, box: tuple[int, int, int, int], w: int, h: int) -> str:
    x1, y1, x2, y2 = box
    return f"{cls_id} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}"


# ──────────────────────────────────────────────────────────────────────────────
# Dataset discovery / split
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class Sample:
    src: Path
    rel: str          # path relative to RAW_DIR (posix) — key for quality CSV
    cls_name: str
    split: str = ""
    dst_stem: str = ""


def normalize_name(name: str) -> str:
    return "_".join(name.strip().split())   # "Nabtat Ali" -> "Nabtat_Ali"


def load_quality_map(csv_path: Path) -> dict[str, str] | None:
    if not csv_path.exists():
        return None
    with csv_path.open(newline="", encoding="utf-8") as f:
        return {row["image"].replace("\\", "/"): row["quality"].strip().lower() for row in csv.DictReader(f)}


def discover_samples(raw_dir: Path) -> list[Sample]:
    """Class = immediate parent folder of each image (Kaggle layout: <Variety>/<img>.JPG)."""
    quality = load_quality_map(config.QUALITY_CSV)
    if quality:
        print(f"[data] Quality labels found ({len(quality)} rows) -> classes = <Variety>_<quality>")

    samples = []
    for p in sorted(raw_dir.rglob("*")):
        if p.suffix.lower() not in IMG_EXTS or not p.is_file():
            continue
        rel = p.relative_to(raw_dir).as_posix()
        variety = normalize_name(p.parent.name)
        if quality is not None:
            q = quality.get(rel)
            if q is None:
                continue                  # only use images that carry a quality label
            variety = f"{variety}_{q}"
        samples.append(Sample(src=p, rel=rel, cls_name=variety))
    if not samples:
        raise FileNotFoundError(f"No images found under {raw_dir}. Did the download/unzip succeed?")
    return samples


def stratified_split(samples: list[Sample], ratios=config.SPLIT_RATIOS, seed=config.SEED) -> None:
    """Assign split per class so every variety is represented in val/test."""
    rng = random.Random(seed)
    by_cls: dict[str, list[Sample]] = defaultdict(list)
    for s in samples:
        by_cls[s.cls_name].append(s)
    for cls_name, items in by_cls.items():
        rng.shuffle(items)
        n = len(items)
        n_train, n_val = int(round(n * ratios[0])), int(round(n * ratios[1]))
        for i, s in enumerate(items):
            s.split = "train" if i < n_train else "val" if i < n_train + n_val else "test"
            s.dst_stem = f"{cls_name}_{i:04d}"


# ──────────────────────────────────────────────────────────────────────────────
# Build YOLO dataset
# ──────────────────────────────────────────────────────────────────────────────
def _process(sample: Sample, cls_id: int) -> dict:
    img = cv2.imread(str(sample.src), cv2.IMREAD_COLOR)   # OpenCV applies EXIF orientation
    if img is None:
        return {"ok": False, "reason": "unreadable", "sample": sample}
    img, _ = resize_max_side(img, config.MAX_SIDE)
    _, box = segment_fruit(img)
    if box is None:
        return {"ok": False, "reason": "segmentation_failed", "sample": sample}

    h, w = img.shape[:2]
    img_out = config.YOLO_DATA_DIR / "images" / sample.split / f"{sample.dst_stem}.jpg"
    lbl_out = config.YOLO_DATA_DIR / "labels" / sample.split / f"{sample.dst_stem}.txt"
    cv2.imwrite(str(img_out), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    lbl_out.write_text(to_yolo_line(cls_id, box, w, h) + "\n")
    return {"ok": True, "sample": sample, "box": box, "img": img_out}


def build_yolo_dataset(force: bool = False) -> Path:
    if config.DATA_YAML.exists() and not force:
        print(f"[data] {config.DATA_YAML} exists — skipping build (use --force to rebuild).")
        return config.DATA_YAML

    for sub in ("images", "labels"):
        shutil.rmtree(config.YOLO_DATA_DIR / sub, ignore_errors=True)
        for split in SPLITS:
            (config.YOLO_DATA_DIR / sub / split).mkdir(parents=True, exist_ok=True)

    samples = discover_samples(config.RAW_DIR)
    class_names = sorted({s.cls_name for s in samples})
    cls_to_id = {c: i for i, c in enumerate(class_names)}
    stratified_split(samples)
    print(f"[data] {len(samples)} images, {len(class_names)} classes: {class_names}")

    # cv2 releases the GIL during decode/resize -> threads give a real speed-up.
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda s: _process(s, cls_to_id[s.cls_name]), samples))

    ok = [r for r in results if r["ok"]]
    bad = [r for r in results if not r["ok"]]

    with config.MANIFEST_CSV.open("w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["source", "class", "split", "image", "x1", "y1", "x2", "y2"])
        for r in ok:
            s = r["sample"]
            wr.writerow([s.rel, s.cls_name, s.split, r["img"].name, *r["box"]])

    review_csv = config.QA_DIR / "needs_review.csv"
    with review_csv.open("w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["source", "class", "reason"])
        for r in bad:
            wr.writerow([r["sample"].rel, r["sample"].cls_name, r["reason"]])

    counts = Counter((r["sample"].cls_name, r["sample"].split) for r in ok)
    print(f"\n{'class':<22}{'train':>7}{'val':>7}{'test':>7}")
    for c in class_names:
        print(f"{c:<22}{counts[(c, 'train')]:>7}{counts[(c, 'val')]:>7}{counts[(c, 'test')]:>7}")
    print(f"[data] Labelled {len(ok)} / {len(results)}; {len(bad)} excluded -> {review_csv}")

    data_cfg = {
        "path": config.YOLO_DATA_DIR.as_posix(),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {i: c for i, c in enumerate(class_names)},
    }
    config.DATA_YAML.write_text(yaml.safe_dump(data_cfg, sort_keys=False))
    print(f"[data] Wrote {config.DATA_YAML}")
    save_qa_grid()
    return config.DATA_YAML


def save_qa_grid(n: int = 16, tile: int = 320, split: str = "train") -> Path:
    """Draw auto-generated boxes on random images -> quick visual sanity check."""
    names = yaml.safe_load(config.DATA_YAML.read_text())["names"]
    imgs = sorted((config.YOLO_DATA_DIR / "images" / split).glob("*.jpg"))
    picks = random.Random(config.SEED).sample(imgs, min(n, len(imgs)))
    tiles = []
    for p in picks:
        img = cv2.imread(str(p))
        h, w = img.shape[:2]
        lbl = config.YOLO_DATA_DIR / "labels" / split / f"{p.stem}.txt"
        for line in lbl.read_text().splitlines():
            c, cx, cy, bw, bh = line.split()
            cx, cy, bw, bh = float(cx) * w, float(cy) * h, float(bw) * w, float(bh) * h
            cv2.rectangle(img, (int(cx - bw / 2), int(cy - bh / 2)), (int(cx + bw / 2), int(cy + bh / 2)), (0, 200, 0), 6)
            cv2.putText(img, names[int(c)], (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
        tiles.append(cv2.resize(img, (tile, tile)))
    while len(tiles) % 4:
        tiles.append(np.full((tile, tile, 3), 255, np.uint8))
    grid = np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)])
    out = config.QA_DIR / "autolabel_qa_grid.jpg"
    cv2.imwrite(str(out), grid)
    print(f"[data] QA grid -> {out}")
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-download", action="store_true", help="use images already in RAW_DIR")
    ap.add_argument("--force", action="store_true", help="rebuild the YOLO dataset (labels/splits)")
    ap.add_argument("--redownload", action="store_true", help="download from Kaggle even if raw images exist")
    args = ap.parse_args(argv)

    if not args.skip_download:
        download_dataset(force=args.redownload)
    build_yolo_dataset(force=args.force)


if __name__ == "__main__":
    main()
