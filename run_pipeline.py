"""
End-to-end runner for the Date-Vision pipeline.

    python run_pipeline.py                         # all steps
    python run_pipeline.py --steps train eval      # selected steps
    python run_pipeline.py --epochs 50 --model yolov8n.pt

Every artifact is written under config.SSD_PATH (env var DATE_VISION_SSD).
"""
from __future__ import annotations

import argparse

from date_vision import config

STEPS = ["data", "train", "eval", "infer", "export"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", nargs="+", choices=STEPS, default=STEPS)
    ap.add_argument("--model", default=config.BASE_MODEL)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--batch", type=int)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    print(f"SSD root: {config.SSD_PATH}")

    if "data" in a.steps:
        from date_vision import data_setup
        data_setup.main([])
    if "train" in a.steps:
        from date_vision.train import train
        train(a.model, device=a.device, epochs=a.epochs, batch=a.batch)
    if "eval" in a.steps:
        from date_vision.evaluate import evaluate
        evaluate(split="test", device=a.device)
    if "infer" in a.steps:
        from date_vision import inference
        inference.main(["--mode", "both"])
    if "export" in a.steps:
        from date_vision.export_onnx import export_onnx
        export_onnx(device=a.device)


if __name__ == "__main__":   # required on Windows (multiprocessing spawn)
    main()
