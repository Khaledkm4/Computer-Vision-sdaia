"""
Step 2 — Fine-tune a COCO-pretrained YOLOv8 detector on the date dataset.

All outputs (checkpoints, curves, args.yaml) go to:
    <SSD>/runs/train/<RUN_NAME>/weights/{best,last}.pt

Run:  python -m date_vision.train [--model yolov8s.pt] [--epochs 100] [--batch 16] [--resume]
"""
from __future__ import annotations

import argparse
import contextlib
from pathlib import Path

from date_vision import config  # noqa: F401  (bootstraps SSD paths before ultralytics import)
from ultralytics import YOLO


def train(model_name: str = config.BASE_MODEL, run_name: str = config.RUN_NAME,
          device: str | None = None, resume: bool = False, **overrides) -> Path:
    config.configure_ultralytics()
    if not config.DATA_YAML.exists():
        raise SystemExit(f"{config.DATA_YAML} not found — run `python -m date_vision.data_setup` first.")

    if resume:
        last = config.last_weights(run_name)
        print(f"[train] Resuming from {last}")
        YOLO(str(last)).train(resume=True)
        return config.best_weights(run_name)

    hyp = {**config.TRAIN_HYP, **{k: v for k, v in overrides.items() if v is not None}}

    # Pretrained weights are downloaded straight to <SSD>/weights/.
    model = YOLO(str(config.WEIGHTS_DIR / model_name))

    # Ultralytics' AMP self-check may download a tiny helper model into the CWD;
    # running from the SSD weights folder keeps every byte on the SSD.
    with contextlib.chdir(config.WEIGHTS_DIR):
        model.train(
            data=str(config.DATA_YAML),
            project=str(config.TRAIN_PROJECT),   # <SSD>/runs/train
            name=run_name,                       # <SSD>/runs/train/<run_name>
            exist_ok=True,                       # overwrite same run name -> stable best.pt path
            device=device,                       # None = auto (GPU 0 if available, else CPU)
            **hyp,
        )

    best = config.best_weights(run_name)
    print(f"[train] Best checkpoint: {best}")
    return best


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=config.BASE_MODEL, help="yolov8n/s/m/l/x.pt")
    ap.add_argument("--name", default=config.RUN_NAME)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--batch", type=int)
    ap.add_argument("--imgsz", type=int)
    ap.add_argument("--device", default=None, help="'0', '0,1', 'cpu' (default: auto)")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args(argv)
    train(a.model, a.name, a.device, a.resume, epochs=a.epochs, batch=a.batch, imgsz=a.imgsz)


if __name__ == "__main__":   # required on Windows: dataloader workers use spawn()
    main()
