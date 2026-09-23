"""
Part 1 — Pose Extraction

Turns a single video frame into 33 body-joint landmarks, in two coordinate
systems: pixel space (for drawing on screen) and real-world meters (for
depth-aware rotation math later on).
"""

from __future__ import annotations

import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

from . import config


def create_landmarker(model_path: str = config.MODEL_PATH):
    """Builds a MediaPipe PoseLandmarker configured for video-mode inference."""
    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO,
        min_pose_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
    )
    return vision.PoseLandmarker.create_from_options(options)


def extract_landmarks(landmarker, frame_bgr, timestamp_ms: int):
    """Runs pose detection on one BGR frame.

    Returns ``(image_landmarks, world_landmarks)`` — each a dict of
    ``{index: {"x", "y", "z", "visibility"}}`` — or ``(None, None)`` if no
    person was detected in this frame.
    """
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result = landmarker.detect_for_video(mp_image, timestamp_ms)

    if not result.pose_landmarks or not result.pose_world_landmarks:
        return None, None

    h, w, _ = frame_bgr.shape
    image_person = result.pose_landmarks[0]
    world_person = result.pose_world_landmarks[0]

    image_landmarks = {
        idx: {"x": lm.x * w, "y": lm.y * h, "z": lm.z * w, "visibility": lm.visibility}
        for idx, lm in enumerate(image_person)
    }
    world_landmarks = {
        idx: {"x": lm.x, "y": lm.y, "z": lm.z, "visibility": lm.visibility}
        for idx, lm in enumerate(world_person)
    }
    return image_landmarks, world_landmarks
