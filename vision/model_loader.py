"""
Downloads and caches the MediaPipe Pose Landmarker model file. Split out of
pose.py so the "do I have the model on disk yet" concern never mixes with
the "how do I run it" concern.
"""

from __future__ import annotations

import os
import urllib.request

from . import config


def ensure_model(model_path: str = config.MODEL_PATH, variant: str = "full") -> str:
    """Downloads the pose_landmarker .task file if it isn't already on disk.

    Parameters
    ----------
    model_path: where to save/look for the model file.
    variant: "lite" (fastest, least accurate), "full" (recommended default),
        or "heavy" (most accurate, slowest).

    Returns the path to the model file.
    """
    if os.path.exists(model_path):
        return model_path

    if variant not in config.MODEL_URLS:
        raise ValueError(f"Unknown model variant '{variant}'. Choose one of {list(config.MODEL_URLS)}.")

    url = config.MODEL_URLS[variant]
    print(f"Downloading pose landmarker model ({variant}) to {model_path} ...")
    urllib.request.urlretrieve(url, model_path)
    size_mb = os.path.getsize(model_path) / 1e6
    print(f"Done: {model_path} ({size_mb:.1f} MB)")
    return model_path
