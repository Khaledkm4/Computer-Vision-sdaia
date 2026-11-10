"""
Central configuration for the Date-Vision pipeline.

EVERY path the pipeline reads from or writes to is derived from ``SSD_PATH``.
Override it without editing code:

    PowerShell:  $env:DATE_VISION_SSD = "E:/Date_Vision_Project"
    bash:        export DATE_VISION_SSD=/mnt/ssd/Date_Vision_Project
"""
from __future__ import annotations

import os
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# 1. External SSD root  (the only value you normally need to change)
# ──────────────────────────────────────────────────────────────────────────────
SSD_PATH: Path = Path(os.environ.get("DATE_VISION_SSD", "D:/Date_Vision_Project")).resolve()

# ──────────────────────────────────────────────────────────────────────────────
# 2. Directory layout on the SSD
# ──────────────────────────────────────────────────────────────────────────────
DATA_DIR = SSD_PATH / "data"
RAW_DIR = DATA_DIR / "raw"                    # untouched Kaggle download
YOLO_DATA_DIR = DATA_DIR / "yolo"             # images/{train,val,test} + labels/{...}
STRESS_DIR = DATA_DIR / "stress_overlap"      # synthetic overlapping-fruit test set
QA_DIR = DATA_DIR / "qa"                      # auto-label QA grid + review CSV
DATA_YAML = YOLO_DATA_DIR / "date_fruits.yaml"
MANIFEST_CSV = YOLO_DATA_DIR / "manifest.csv"

# Optional: per-image quality labels -> classes become "<Variety>_<quality>".
# CSV columns: image,quality   (image = path relative to RAW_DIR, e.g. "Ajwa/Ajwa Date (1).JPG")
QUALITY_CSV = DATA_DIR / "annotations" / "quality_labels.csv"

WEIGHTS_DIR = SSD_PATH / "weights"            # pretrained checkpoints (yolov8*.pt)
RUNS_DIR = SSD_PATH / "runs"
TRAIN_PROJECT = RUNS_DIR / "train"            # -> model.train(project=..., name=RUN_NAME)
EVAL_PROJECT = RUNS_DIR / "eval"
PREDICT_PROJECT = RUNS_DIR / "predict"
EXPORT_DIR = SSD_PATH / "exports"             # ONNX + export metadata
SHOWCASE_DIR = SSD_PATH / "showcase"          # curated TP / failure images
CACHE_DIR = SSD_PATH / "cache"                # Ultralytics settings, torch hub, kaggle

KAGGLE_CONFIG_DIR = SSD_PATH / ".kaggle"      # put kaggle.json here (optional)

# ──────────────────────────────────────────────────────────────────────────────
# 3. Dataset / experiment settings
# ──────────────────────────────────────────────────────────────────────────────
KAGGLE_DATASET = "wadhasnalhamdan/date-fruit-image-dataset-in-controlled-environment"
SEED = 42
SPLIT_RATIOS = (0.70, 0.15, 0.15)             # train / val / test (stratified per class)
MAX_SIDE = 1280                               # downscale 5184px DSLR frames -> faster I/O, less SSD space
IMGSZ = 640                                   # network input size

BASE_MODEL = "yolov8s.pt"                     # n/s/m/l/x — 's' is a good speed/accuracy trade-off
RUN_NAME = "yolov8s_dates"

# ──────────────────────────────────────────────────────────────────────────────
# 4. Training hyper-parameters
#    The augmentation block below is WHERE factory-floor conditions are simulated.
#    Ultralytics applies these on-the-fly inside its dataloader (no extra code).
# ──────────────────────────────────────────────────────────────────────────────
TRAIN_HYP: dict = dict(
    # --- schedule / optimiser ---
    epochs=100,
    imgsz=IMGSZ,
    batch=16,               # use -1 for Ultralytics AutoBatch (fits ~60% of GPU memory)
    patience=20,            # early stopping on val fitness
    optimizer="AdamW",
    lr0=1e-3,
    lrf=0.01,               # final LR = lr0 * lrf
    cos_lr=True,
    momentum=0.937,
    weight_decay=5e-4,
    warmup_epochs=3,
    close_mosaic=10,        # disable mosaic for the last N epochs -> stabilises box regression
    seed=SEED,
    deterministic=True,
    workers=4,              # Windows: keep modest; dataloader uses spawn()
    cache=False,            # "ram" speeds up epochs if you have >8 GB free RAM
    amp=True,
    plots=True,

    # --- AUGMENTATION: simulate a sorting line / factory environment ---
    hsv_h=0.010,            # hue: keep SMALL — colour is a key cue separating varieties (e.g. Rutab vs Sokari)
    hsv_s=0.50,             # saturation drift (camera white-balance changes)
    hsv_v=0.40,             # BRIGHTNESS: flickering / ageing LED lighting over the conveyor
    degrees=30.0,           # ROTATION: fruit lands at arbitrary angles on the belt
    translate=0.10,         # fruit not centred under the camera
    scale=0.50,             # camera-height / fruit-size variation
    shear=2.0,
    perspective=0.0005,     # slight camera tilt
    flipud=0.5,
    fliplr=0.5,
    mosaic=1.0,             # stitches 4 images -> MULTIPLE fruits per frame (dataset has only 1/frame)
    mixup=0.10,             # semi-transparent overlap -> some robustness to touching fruit
    copy_paste=0.0,         # needs segmentation masks; not used for box-only labels
    erasing=0.0,
)

# Inference thresholds
CONF_THRES = 0.25
NMS_IOU = 0.50
MATCH_IOU = 0.50            # IoU for counting a prediction as a True Positive


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def best_weights(run_name: str = RUN_NAME) -> Path:
    return TRAIN_PROJECT / run_name / "weights" / "best.pt"


def last_weights(run_name: str = RUN_NAME) -> Path:
    return TRAIN_PROJECT / run_name / "weights" / "last.pt"


def bootstrap() -> None:
    """Create SSD folders and redirect library caches to the SSD.

    Must run BEFORE ``import ultralytics`` (it reads YOLO_CONFIG_DIR at import time).
    It is executed automatically by ``date_vision/__init__.py``.
    """
    for d in (RAW_DIR, YOLO_DATA_DIR, STRESS_DIR, QA_DIR, WEIGHTS_DIR, TRAIN_PROJECT,
              EVAL_PROJECT, PREDICT_PROJECT, EXPORT_DIR, SHOWCASE_DIR, CACHE_DIR):
        d.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("YOLO_CONFIG_DIR", str(CACHE_DIR / "ultralytics"))  # settings.json, fonts
    os.environ.setdefault("TORCH_HOME", str(CACHE_DIR / "torch"))
    os.environ.setdefault("MPLCONFIGDIR", str(CACHE_DIR / "matplotlib"))
    if (KAGGLE_CONFIG_DIR / "kaggle.json").exists():
        os.environ.setdefault("KAGGLE_CONFIG_DIR", str(KAGGLE_CONFIG_DIR))


def configure_ultralytics() -> None:
    """Point Ultralytics' global default dirs at the SSD (persisted in its settings.json)."""
    from ultralytics import settings

    settings.update({
        "datasets_dir": str(DATA_DIR),
        "weights_dir": str(WEIGHTS_DIR),
        "runs_dir": str(RUNS_DIR),
    })
