"""
Part 2 — Skeleton Drawing

A visual sanity check: if the overlay looks wrong or flipped, you know
something is off in the data before trusting any computed number.
"""

from __future__ import annotations

import cv2

from . import config


def _pt(landmarks, idx):
    return (int(landmarks[idx]["x"]), int(landmarks[idx]["y"]))


def _visible(landmarks, idx):
    return landmarks[idx]["visibility"] >= config.VISIBILITY_THRESHOLD


def midpoint(p1, p2):
    return {"x": (p1["x"] + p2["x"]) / 2, "y": (p1["y"] + p2["y"]) / 2}


def draw_skeleton(frame, image_landmarks):
    """Draws the lead/trail limb chains and spine line onto ``frame`` in place."""
    if image_landmarks is None:
        return frame

    chains = [
        (config.LEAD_CHAIN, config.COLOR_LEAD),
        (config.TRAIL_CHAIN, config.COLOR_TRAIL),
        (config.LEAD_ARM, config.COLOR_LEAD),
        (config.TRAIL_ARM, config.COLOR_TRAIL),
    ]
    for chain, color in chains:
        for a, b in chain:
            if _visible(image_landmarks, a) and _visible(image_landmarks, b):
                cv2.line(frame, _pt(image_landmarks, a), _pt(image_landmarks, b), color, 4)
                cv2.circle(frame, _pt(image_landmarks, a), 5, color, -1)
                cv2.circle(frame, _pt(image_landmarks, b), 5, color, -1)

    for a, b in config.SPINE_LINKS:
        if _visible(image_landmarks, a) and _visible(image_landmarks, b):
            cv2.line(frame, _pt(image_landmarks, a), _pt(image_landmarks, b), config.COLOR_SPINE, 4)

    mid_shoulder = midpoint(image_landmarks[11], image_landmarks[12])
    mid_hip = midpoint(image_landmarks[23], image_landmarks[24])
    cv2.line(
        frame,
        (int(mid_shoulder["x"]), int(mid_shoulder["y"])),
        (int(mid_hip["x"]), int(mid_hip["y"])),
        config.COLOR_SPINE,
        4,
    )
    return frame
