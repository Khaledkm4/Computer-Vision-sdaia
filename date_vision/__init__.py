"""Date-Vision: YOLOv8 detection & sorting of date fruits.

Importing the package redirects all caches/outputs to the external SSD
before any submodule imports Ultralytics.
"""
from . import config

config.bootstrap()

__all__ = ["config"]
