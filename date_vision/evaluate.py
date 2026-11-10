"""
Step 3 — Evaluate the trained detector: mAP50, mAP50-95, per-class AP and the
confusion matrix.

Outputs -> <SSD>/runs/eval/<RUN_NAME>_<split>/
    confusion_matrix.png, confusion_matrix_normalized.png   (Ultralytics plots)
    PR_curve.png, F1_curve.png, P_curve.png, R_curve.png
    confusion_matrix.csv   (rows = predicted, cols = ground truth, incl. background)
    metrics.json           (overall + per-class)

Run:  python -m date_vision.evaluate [--weights path/to/best.pt] [--split test]
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from date_vision import config
from ultralytics import YOLO


def evaluate(weights: Path | str | None = None, split: str = "test", device: str | None = None) -> dict:
    config.configure_ultralytics()
    weights = Path(weights or config.best_weights())
    model = YOLO(str(weights))

    metrics = model.val(
        data=str(config.DATA_YAML),
        split=split,                 # evaluate on the held-out TEST split, not val
        imgsz=config.IMGSZ,
        batch=16,
        conf=0.001,                  # low conf + standard NMS IoU = the COCO-style mAP protocol
        iou=0.6,
        plots=True,                  # saves confusion matrices + PR/F1 curves
        device=device,
        project=str(config.EVAL_PROJECT),
        name=f"{weights.parent.parent.name}_{split}",
        exist_ok=True,
    )
    save_dir = Path(metrics.save_dir)
    names = model.names

    summary = {
        "weights": str(weights),
        "split": split,
        "mAP50": float(metrics.box.map50),
        "mAP50-95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "speed_ms_per_img": metrics.speed,
        "per_class": {},
    }
    for i, c in enumerate(metrics.box.ap_class_index):
        p, r, ap50, ap = metrics.box.class_result(i)
        summary["per_class"][names[int(c)]] = {
            "precision": float(p), "recall": float(r), "AP50": float(ap50), "AP50-95": float(ap)}

    # Confusion matrix as CSV (Ultralytics convention: matrix[predicted, true]).
    cm = getattr(metrics, "confusion_matrix", None)
    if cm is not None:
        m = np.asarray(cm.matrix, dtype=int)
        labels = [names[i] for i in range(len(names))] + ["background"]
        with (save_dir / "confusion_matrix.csv").open("w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["pred \\ true", *labels])
            for lab, row in zip(labels, m):
                wr.writerow([lab, *row.tolist()])
        nc = len(names)
        core = m[:nc, :nc]
        # Of the fruits that were detected, how often was the variety correct?
        summary["variety_accuracy_on_detected"] = float(np.trace(core) / max(core.sum(), 1))

    (save_dir / "metrics.json").write_text(json.dumps(summary, indent=2))

    print(f"\n{'=' * 60}\n Evaluation on '{split}'  ({weights.name})\n{'=' * 60}")
    print(f" mAP50     : {summary['mAP50']:.4f}")
    print(f" mAP50-95  : {summary['mAP50-95']:.4f}")
    print(f" Precision : {summary['precision']:.4f}   Recall: {summary['recall']:.4f}")
    if "variety_accuracy_on_detected" in summary:
        print(f" Variety accuracy (detected fruits): {summary['variety_accuracy_on_detected']:.4f}")
    print(f"\n {'class':<22}{'P':>8}{'R':>8}{'AP50':>8}{'AP50-95':>9}")
    for c, v in summary["per_class"].items():
        print(f" {c:<22}{v['precision']:>8.3f}{v['recall']:>8.3f}{v['AP50']:>8.3f}{v['AP50-95']:>9.3f}")
    print(f"\n Artifacts (incl. confusion_matrix.png) -> {save_dir}")
    return summary


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--device", default=None)
    a = ap.parse_args(argv)
    evaluate(a.weights, a.split, a.device)


if __name__ == "__main__":
    main()
