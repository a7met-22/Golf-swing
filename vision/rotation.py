"""
Part 4 — Rotation Engine

Uses the real-world (meter) landmarks instead of pixels, so rotation around
the vertical axis can be measured even though the camera only sees 2D.
"""

from __future__ import annotations

import math

from .geometry import _visible_enough


def hip_shoulder_angle_top_view(world_landmarks, left_idx: int, right_idx: int) -> float:
    left = world_landmarks[left_idx]
    right = world_landmarks[right_idx]
    dx = right["x"] - left["x"]
    dz = right["z"] - left["z"]
    return math.degrees(math.atan2(dz, dx))


def compute_rotation(world_landmarks, left_idx: int, right_idx: int, reference_angle: float) -> float:
    current_angle = hip_shoulder_angle_top_view(world_landmarks, left_idx, right_idx)
    rotation = current_angle - reference_angle
    if rotation > 180:
        rotation -= 360
    elif rotation < -180:
        rotation += 360
    return rotation


def compute_rotation_metrics(world_landmarks, address_hip_angle: float, address_shoulder_angle: float) -> dict:
    """Same visibility-gating idea as geometry.py, applied to hip/shoulder meters."""
    hip_ok = _visible_enough(world_landmarks, 23, 24)
    sh_ok = _visible_enough(world_landmarks, 11, 12)

    hip_rotation = round(compute_rotation(world_landmarks, 23, 24, address_hip_angle), 1) if hip_ok else None
    shoulder_rotation = (
        round(compute_rotation(world_landmarks, 11, 12, address_shoulder_angle), 1) if sh_ok else None
    )
    x_factor = round(shoulder_rotation - hip_rotation, 1) if (hip_ok and sh_ok) else None

    return {
        "hip_rotation": hip_rotation,
        "shoulder_rotation": shoulder_rotation,
        "x_factor": x_factor,
    }
