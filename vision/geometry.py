"""
Part 3 — Geometry Engine

Converts pixel coordinates into real angles in degrees, using plain vector
math (no ML). Covers knee/elbow flexion and spine tilt.
"""

from __future__ import annotations

import numpy as np

from . import config
from .skeleton import midpoint


def vector(p1, p2):
    return np.array([p2["x"] - p1["x"], p2["y"] - p1["y"]])


def angle_between_vectors(v1, v2) -> float:
    dot = np.dot(v1, v2)
    norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norm_product == 0:
        return 0.0
    cos_angle = np.clip(dot / norm_product, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def joint_angle(landmarks, a_idx: int, vertex_idx: int, b_idx: int) -> float:
    vertex = landmarks[vertex_idx]
    v1 = vector(vertex, landmarks[a_idx])
    v2 = vector(vertex, landmarks[b_idx])
    return angle_between_vectors(v1, v2)


def knee_flexion(landmarks, hip_idx: int, knee_idx: int, ankle_idx: int) -> float:
    return 180 - joint_angle(landmarks, hip_idx, knee_idx, ankle_idx)


def elbow_flexion(landmarks, shoulder_idx: int, elbow_idx: int, wrist_idx: int) -> float:
    """180 minus the raw joint angle = how bent the elbow is (a straight arm = 0)."""
    return 180 - joint_angle(landmarks, shoulder_idx, elbow_idx, wrist_idx)


def spine_tilt_from_vertical(landmarks) -> float:
    mid_shoulder = midpoint(landmarks[11], landmarks[12])
    mid_hip = midpoint(landmarks[23], landmarks[24])
    spine_vector = vector(mid_shoulder, mid_hip)
    return angle_between_vectors(spine_vector, np.array([0, 1]))


def _visible_enough(landmarks, *indices) -> bool:
    return all(landmarks[i]["visibility"] >= config.VISIBILITY_THRESHOLD for i in indices)


def compute_geometry_metrics(image_landmarks: dict) -> dict:
    """Every metric returns ``None`` if any joint it depends on isn't visible
    with enough confidence — instead of computing an angle from a guessed
    coordinate (which was the original bug this guards against)."""

    def safe(fn, *idx):
        return round(fn(), 1) if _visible_enough(image_landmarks, *idx) else None

    return {
        "spine_tilt": safe(lambda: spine_tilt_from_vertical(image_landmarks), 11, 12, 23, 24),
        "lead_knee_flexion": safe(lambda: knee_flexion(image_landmarks, 23, 25, 27), 23, 25, 27),
        "trail_knee_flexion": safe(lambda: knee_flexion(image_landmarks, 24, 26, 28), 24, 26, 28),
        "lead_elbow_flexion": safe(lambda: elbow_flexion(image_landmarks, 11, 13, 15), 11, 13, 15),
        "trail_elbow_flexion": safe(lambda: elbow_flexion(image_landmarks, 12, 14, 16), 12, 14, 16),
    }
