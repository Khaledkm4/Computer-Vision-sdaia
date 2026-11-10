"""
Step 5 — Deployment prep: export best.pt -> ONNX, verify it, and benchmark it.

Output: <SSD>/exports/<RUN_NAME>.onnx  +  export_info.json

Run:  python -m date_vision.export_onnx [--weights best.pt] [--dynamic] [--half] [--device 0]
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np

from date_vision import config
from ultralytics import YOLO


def export_onnx(weights: Path | str | None = None, imgsz: int = config.IMGSZ, dynamic: bool = False,
                half: bool = False, opset: int | None = None, device: str | None = None) -> Path:
    config.configure_ultralytics()
    weights = Path(weights or config.best_weights())
    model = YOLO(str(weights))

    # Ultralytics writes the .onnx next to the .pt (already on the SSD) ...
    exported = Path(model.export(
        format="onnx",
        imgsz=imgsz,
        dynamic=dynamic,     # True -> variable batch/resolution (slightly slower, more flexible)
        simplify=True,       # graph simplification (onnxslim) -> fewer nodes, faster
        half=half,           # FP16 — only valid when exporting on a CUDA device
        opset=opset,         # None = latest supported opset
        device=device,
    ))
    # ... then we copy it into the dedicated exports folder on the SSD.
    target = config.EXPORT_DIR / f"{weights.parent.parent.name}.onnx"
    shutil.copy2(exported, target)

    info = {"onnx": str(target), "source_weights": str(weights), "imgsz": imgsz,
            "dynamic": dynamic, "half": half, "names": model.names,
            "size_mb": round(target.stat().st_size / 1e6, 2)}
    info.update(verify_onnx(target, imgsz))
    info.update(benchmark(weights, target, imgsz))
    (config.EXPORT_DIR / "export_info.json").write_text(json.dumps(info, indent=2))
    print(f"[export] ONNX model -> {target}\n[export] {json.dumps({k: info[k] for k in info if k != 'names'}, indent=2)}")
    return target


def verify_onnx(onnx_path: Path, imgsz: int) -> dict:
    """Run one dummy forward pass in raw ONNX Runtime (what a production service would use)."""
    import onnxruntime as ort

    providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in ort.get_available_providers()]
    sess = ort.InferenceSession(str(onnx_path), providers=providers)
    inp = sess.get_inputs()[0]
    dtype = np.float16 if "float16" in inp.type else np.float32
    out = sess.run(None, {inp.name: np.random.rand(1, 3, imgsz, imgsz).astype(dtype)})
    print(f"[export] ORT OK — input {inp.name}{inp.shape} -> output {out[0].shape} via {sess.get_providers()[0]}")
    return {"ort_input": inp.name, "ort_output_shape": list(out[0].shape), "ort_provider": sess.get_providers()[0]}


def benchmark(pt_path: Path, onnx_path: Path, imgsz: int, n: int = 50) -> dict:
    """End-to-end latency (pre-process + inference + NMS) on a real test image."""
    test_imgs = sorted((config.YOLO_DATA_DIR / "images" / "test").glob("*.jpg"))
    if not test_imgs:
        return {}
    img = str(test_imgs[0])
    out = {}
    for tag, path in (("pt", pt_path), ("onnx", onnx_path)):
        m = YOLO(str(path), task="detect")
        for _ in range(5):                                   # warm-up
            m.predict(img, imgsz=imgsz, verbose=False)
        t0 = time.perf_counter()
        for _ in range(n):
            m.predict(img, imgsz=imgsz, verbose=False)
        out[f"latency_ms_{tag}"] = round((time.perf_counter() - t0) / n * 1000, 2)
    out["fps_onnx"] = round(1000 / out["latency_ms_onnx"], 1)
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--imgsz", type=int, default=config.IMGSZ)
    ap.add_argument("--dynamic", action="store_true")
    ap.add_argument("--half", action="store_true", help="FP16 (requires --device 0)")
    ap.add_argument("--opset", type=int, default=None)
    ap.add_argument("--device", default=None)
    a = ap.parse_args(argv)
    export_onnx(a.weights, a.imgsz, a.dynamic, a.half, a.opset, a.device)


if __name__ == "__main__":
    main()
