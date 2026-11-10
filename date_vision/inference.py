"""
Step 4 — Inference & visualisation.

* ``test``   : predict on the held-out test split, compare each prediction with
               ground truth (IoU >= 0.5, class match) and file every annotated
               image under  <SSD>/showcase/test/{true_positive, misclassified, missed, false_positive}/
* ``stress`` : the source dataset has ONE date per image, so overlapping fruit
               never occurs naturally. We composite 3-6 masked test-set fruits
               with heavy overlap onto a white canvas (with exact GT boxes) to
               reproduce the classic sorting-line failure mode.
* ``stream`` : real-time demo on a webcam index / video file / RTSP URL
               (works with the .pt OR the exported .onnx model).

Green boxes = ground truth, coloured boxes = model predictions.

Run:
    python -m date_vision.inference --mode test
    python -m date_vision.inference --mode stress --n-stress 40
    python -m date_vision.inference --mode stream --source 0 --weights <SSD>/exports/yolov8s_dates.onnx
"""
from __future__ import annotations

import argparse
import csv
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

from date_vision import config
from date_vision.data_setup import segment_fruit
from ultralytics import YOLO

GT_COLOR = (0, 200, 0)


# ──────────────────────────────────────────────────────────────────────────────
# Ground truth & matching
# ──────────────────────────────────────────────────────────────────────────────
def label_path_for(img_path: Path) -> Path:
    """<root>/images/<split>/x.jpg  ->  <root>/labels/<split>/x.txt"""
    return img_path.parents[2] / "labels" / img_path.parent.name / f"{img_path.stem}.txt"


def load_gt(img_path: Path, w: int, h: int) -> list[tuple[int, np.ndarray]]:
    lp = label_path_for(img_path)
    gts = []
    if lp.exists():
        for line in lp.read_text().splitlines():
            c, cx, cy, bw, bh = map(float, line.split())
            gts.append((int(c), np.array([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])))
    return gts


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(br - tl, 0, None), axis=2)
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])  # noqa: E731
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def classify_outcome(pred_boxes, pred_cls, gts, iou_thr=config.MATCH_IOU) -> tuple[str, dict]:
    """Greedy one-to-one matching. Returns (outcome, counts)."""
    gt_boxes = np.array([g[1] for g in gts]).reshape(-1, 4)
    gt_cls = np.array([g[0] for g in gts], dtype=int)
    ious = iou_matrix(pred_boxes, gt_boxes)

    tp = mis = 0
    used_p, used_g = set(), set()
    for pi, gi in sorted(np.argwhere(ious >= iou_thr), key=lambda ij: -ious[ij[0], ij[1]]):
        if pi in used_p or gi in used_g:
            continue
        used_p.add(pi)
        used_g.add(gi)
        if pred_cls[pi] == gt_cls[gi]:
            tp += 1
        else:
            mis += 1
    fn = len(gts) - len(used_g)
    fp = len(pred_boxes) - len(used_p)
    counts = {"tp": tp, "misclassified": mis, "fn": fn, "fp": fp, "n_gt": len(gts)}

    if mis:
        outcome = "misclassified"
    elif fn:
        outcome = "missed"            # typical for occluded / merged fruit
    elif fp:
        outcome = "false_positive"    # duplicate or phantom box
    else:
        outcome = "true_positive"
    return outcome, counts


def render(result, gts, names, outcome, counts) -> np.ndarray:
    img = result.plot(line_width=3)                       # predictions (BGR)
    for c, b in gts:
        x1, y1, x2, y2 = b.astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), GT_COLOR, 2)
        cv2.putText(img, f"GT:{names[c]}", (x1 + 4, y2 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, GT_COLOR, 2)
    banner = f"{outcome.upper()}  TP={counts['tp']} MIS={counts['misclassified']} FN={counts['fn']} FP={counts['fp']}"
    color = (0, 160, 0) if outcome == "true_positive" else (0, 0, 220)
    cv2.rectangle(img, (0, 0), (img.shape[1], 40), color, -1)
    cv2.putText(img, banner, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return img


# ──────────────────────────────────────────────────────────────────────────────
# Batch analysis
# ──────────────────────────────────────────────────────────────────────────────
def analyse(model: YOLO, images: list[Path], tag: str, max_per_bucket: int = 25) -> Path:
    out_root = config.SHOWCASE_DIR / tag
    shutil.rmtree(out_root, ignore_errors=True)
    names = model.names
    rows, bucket_counts = [], {}

    # Ultralytics also saves its own annotated copies to <SSD>/runs/predict/<tag>/
    results = model.predict(
        source=[str(p) for p in images], stream=True, imgsz=config.IMGSZ,
        conf=config.CONF_THRES, iou=config.NMS_IOU, save=True, save_txt=True, save_conf=True,
        project=str(config.PREDICT_PROJECT), name=tag, exist_ok=True, verbose=False,
    )
    for img_path, r in zip(images, results):
        h, w = r.orig_shape
        gts = load_gt(img_path, w, h)
        pb = r.boxes.xyxy.cpu().numpy()
        pc = r.boxes.cls.cpu().numpy().astype(int)
        ps = r.boxes.conf.cpu().numpy()
        outcome, counts = classify_outcome(pb, pc, gts)

        n = bucket_counts.get(outcome, 0)
        if n < max_per_bucket:
            dst = out_root / outcome / f"{img_path.stem}.jpg"
            dst.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(dst), render(r, gts, names, outcome, counts))
        bucket_counts[outcome] = n + 1
        rows.append({"image": img_path.name, "outcome": outcome, **counts,
                     "pred": ";".join(f"{names[c]}:{s:.2f}" for c, s in zip(pc, ps))})

    if not rows:
        raise SystemExit(f"[{tag}] No images to analyse.")
    report = out_root / "report.csv"
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    print(f"\n[{tag}] {len(images)} images -> " + ", ".join(f"{k}: {v}" for k, v in sorted(bucket_counts.items())))
    print(f"[{tag}] Showcase images -> {out_root}")
    return out_root


# ──────────────────────────────────────────────────────────────────────────────
# Synthetic overlapping-fruit stress set
# ──────────────────────────────────────────────────────────────────────────────
def build_overlap_stress_set(n_images: int = 30, canvas: int = 1280, seed: int = config.SEED) -> list[Path]:
    rng = random.Random(seed)
    test_imgs = sorted((config.YOLO_DATA_DIR / "images" / "test").glob("*.jpg"))
    if not test_imgs:
        raise SystemExit("No test images — run data_setup first.")

    img_dir = config.STRESS_DIR / "images" / "stress"
    lbl_dir = config.STRESS_DIR / "labels" / "stress"
    for d in (img_dir, lbl_dir):
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True, exist_ok=True)

    cutouts: dict[Path, tuple] = {}

    def cutout(p: Path):
        if p not in cutouts:                       # (crop, mask, class) — cached
            img = cv2.imread(str(p))
            h, w = img.shape[:2]
            cls = int(label_path_for(p).read_text().split()[0])
            mask, box = segment_fruit(img)
            x1, y1, x2, y2 = box or (0, 0, w, h)
            cutouts[p] = (img[y1:y2, x1:x2], mask[y1:y2, x1:x2], cls)
        return cutouts[p]

    out_paths = []
    for i in range(n_images):
        bg = np.full((canvas, canvas, 3), 245, np.uint8)
        labels, prev = [], None
        for p in rng.sample(test_imgs, rng.randint(3, 6)):
            crop, mask, cls = cutout(p)
            if not mask.any():
                continue                            # segmentation failed for this fruit
            s = rng.uniform(280, 420) / max(crop.shape[:2])
            crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_NEAREST)
            ch, cw = crop.shape[:2]
            if prev is None:
                cx, cy = rng.randint(cw, canvas - cw), rng.randint(ch, canvas - ch)
            else:                                   # place next to previous fruit -> 20-55% overlap
                ang = rng.uniform(0, 2 * np.pi)
                d = rng.uniform(0.45, 0.8) * (cw + prev[2]) / 2
                cx, cy = int(prev[0] + d * np.cos(ang)), int(prev[1] + d * np.sin(ang))
            x1 = int(np.clip(cx - cw // 2, 0, canvas - cw))
            y1 = int(np.clip(cy - ch // 2, 0, canvas - ch))
            roi = bg[y1:y1 + ch, x1:x1 + cw]
            roi[mask > 0] = crop[mask > 0]          # paste fruit pixels only (later fruit occludes earlier)
            ys, xs = np.nonzero(mask)
            bx1, by1, bx2, by2 = x1 + xs.min(), y1 + ys.min(), x1 + xs.max(), y1 + ys.max()
            labels.append(f"{cls} {(bx1 + bx2) / 2 / canvas:.6f} {(by1 + by2) / 2 / canvas:.6f} "
                          f"{(bx2 - bx1) / canvas:.6f} {(by2 - by1) / canvas:.6f}")
            prev = (x1 + cw // 2, y1 + ch // 2, cw)
        out = img_dir / f"overlap_{i:03d}.jpg"
        cv2.imwrite(str(out), bg)
        (lbl_dir / f"{out.stem}.txt").write_text("\n".join(labels) + "\n")
        out_paths.append(out)
    print(f"[stress] Built {len(out_paths)} overlapping-fruit images -> {img_dir}")
    return out_paths


# ──────────────────────────────────────────────────────────────────────────────
# Real-time stream (webcam / video / RTSP)
# ──────────────────────────────────────────────────────────────────────────────
def run_stream(model: YOLO, source: str) -> None:
    src = int(source) if source.isdigit() else source
    for _ in model.predict(source=src, stream=True, show=True, imgsz=config.IMGSZ,
                           conf=config.CONF_THRES, iou=config.NMS_IOU, save=True,
                           project=str(config.PREDICT_PROJECT), name="stream", exist_ok=True, verbose=False):
        pass  # hook for a PLC / sorting actuator: inspect r.boxes here


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["test", "stress", "both", "stream"], default="both")
    ap.add_argument("--weights", default=None, help=".pt or .onnx (default: best.pt of RUN_NAME)")
    ap.add_argument("--n-stress", type=int, default=30)
    ap.add_argument("--source", default="0", help="stream mode: webcam index, video path or URL")
    a = ap.parse_args(argv)

    config.configure_ultralytics()
    weights = a.weights or str(config.best_weights())
    model = YOLO(weights, task="detect")   # ONNX exports embed class names in their metadata

    if a.mode == "stream":
        run_stream(model, a.source)
        return
    if a.mode in ("test", "both"):
        analyse(model, sorted((config.YOLO_DATA_DIR / "images" / "test").glob("*.jpg")), "test")
    if a.mode in ("stress", "both"):
        analyse(model, build_overlap_stress_set(a.n_stress), "stress_overlap")


if __name__ == "__main__":
    main()
