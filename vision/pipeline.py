"""
Part 7 — Pipeline

The main loop: reads the video frame by frame, extracts landmarks,
computes every metric, draws the overlay, and writes the annotated output
video. Swing numbering continues across a detector rebuild (12+ consecutive
frames of lost tracking), so the (player, swing_id) key stays unique in the
exported CSV even across a body-tracking dropout mid-video.
"""

from __future__ import annotations

from collections import Counter, deque

import cv2

from . import config
from .dashboard import compose_frame
from .geometry import compute_geometry_metrics
from .phase_detector import AngleBasedSwingDetector, get_body_unit, get_hand_y, is_hands_low
from .pose import extract_landmarks
from .quality import SwingQualityAnalyzer
from .rotation import compute_rotation_metrics, hip_shoulder_angle_top_view
from .skeleton import draw_skeleton

EVENT_LABELS = {
    "swing_start": "start",
    "top": "top",
    "downswing": "downswing",
    "impact": "impact",
    "swing_end": "end",
    "swing_cancelled": "cancelled waggle",
    "swing_abort": "aborted",
    "scene_reset": "scene cut",
    "address_set": "calibrated",
}


def run_pipeline(video_path: str, output_path: str, landmarker) -> dict:
    """Processes ``video_path`` end to end and writes the annotated video to
    ``output_path``. Returns ``{"swings": [...], "fps": float, "counts": {...}}``
    for the caller to pass straight into :class:`~vision.export.SwingDatasetExporter`.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w, h = int(cap.get(3)), int(cap.get(4))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w + config.DASHBOARD_WIDTH, h))

    frame_count = 0
    lost_frames = 0
    hand_y_history = deque(maxlen=config.HAND_SMOOTHING_WINDOW)
    detector: AngleBasedSwingDetector | None = None
    quality = SwingQualityAnalyzer()
    address_hip_angle = address_shoulder_angle = None
    last_good = None
    all_events = []
    max_swing_num = 0  # highest swing number ever started — numbering survives a detector rebuild

    def process_events(events, hip_raw, shoulder_raw):
        nonlocal address_hip_angle, address_shoulder_angle, max_swing_num
        for kind, ev_frame, num in events:
            if kind == "address_set":
                address_hip_angle = hip_raw
                address_shoulder_angle = shoulder_raw
            elif kind == "scene_reset":
                hand_y_history.clear()
                address_hip_angle = address_shoulder_angle = None
            elif kind == "swing_start":
                max_swing_num = max(max_swing_num, num)
                quality.start(num, ev_frame)
            elif kind in ("swing_cancelled", "swing_abort"):
                for i in range(len(quality.swings) - 1, -1, -1):
                    if quality.swings[i]["n"] == num:
                        quality.swings.pop(i)
                        break
            elif kind == "top":
                if quality.swings and quality.swings[-1]["n"] == num:
                    quality.swings[-1]["top"] = ev_frame
            elif kind == "downswing":
                if quality.swings and quality.swings[-1]["n"] == num:
                    quality.swings[-1]["down"] = ev_frame
            elif kind == "impact":
                if quality.swings and quality.swings[-1]["n"] == num:
                    quality.swings[-1]["impact"] = ev_frame
            elif kind == "swing_end":
                if quality.swings and quality.swings[-1]["n"] == num:
                    quality.swings[-1]["end"] = ev_frame

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        timestamp_ms = int((frame_count / fps) * 1000)
        image_landmarks, world_landmarks = extract_landmarks(landmarker, frame, timestamp_ms)

        if image_landmarks is None:
            lost_frames += 1
            if lost_frames >= config.RESET_LOST_FRAMES and detector is not None:
                process_events(detector.hard_reset(frame_count, "body lost"), None, None)
                detector = None
                hand_y_history.clear()
                address_hip_angle = address_shoulder_angle = None
            if last_good is not None:
                out.write(compose_frame(frame, last_good["phase"], last_good["metrics"], last_good["swing"]))
            frame_count += 1
            continue

        lost_frames = 0

        if detector is None:
            detector = AngleBasedSwingDetector(body_unit=get_body_unit(image_landmarks), fps_scale=fps / 30.0)
            detector.swing_count = max_swing_num  # numbering resumes where it left off
            hand_y_history.clear()

        hand_y_history.append(get_hand_y(image_landmarks))
        smoothed_hand_y = sum(hand_y_history) / len(hand_y_history)

        shoulder_raw = hip_shoulder_angle_top_view(world_landmarks, 11, 12)
        hip_raw = hip_shoulder_angle_top_view(world_landmarks, 23, 24)
        hands_low = is_hands_low(image_landmarks)

        phase, events = detector.update(shoulder_raw, hip_raw, smoothed_hand_y, hands_low, frame_count)
        all_events.extend(events)
        process_events(events, hip_raw, shoulder_raw)

        hip_ref = address_hip_angle if address_hip_angle is not None else 0.0
        sh_ref = address_shoulder_angle if address_shoulder_angle is not None else 0.0

        metrics = compute_geometry_metrics(image_landmarks)
        metrics.update(compute_rotation_metrics(world_landmarks, hip_ref, sh_ref))

        if phase in ("TOP", "IMPACT", "FOLLOW_THROUGH"):
            quality.snapshot(phase, metrics)

        frame = draw_skeleton(frame, image_landmarks)
        swing_num = None if phase in ("READY", "ADDRESS") else detector.swing_count
        out.write(compose_frame(frame, phase, metrics, swing_num))

        last_good = {"phase": phase, "metrics": metrics, "swing": swing_num}
        frame_count += 1

    cap.release()
    out.release()

    print(f"Done! {frame_count} frames processed. Output video: {output_path}")
    print(quality.report_text(fps))

    counts = Counter(kind for kind, _, _ in all_events)
    print("\n=== Event counts (diagnostics) ===")
    for k, v in counts.most_common():
        print(f"  {k}: {v}")

    print("\n=== First 25 events ===")
    for kind, f, num in all_events[:25]:
        print(f"  [{f:5d} | {f/fps:6.2f}s] {EVENT_LABELS.get(kind, kind)}" + (f" #{num}" if num else ""))

    if detector is not None and detector.timeline:
        print("\n=== Last 30 state-machine log entries ===")
        for f, msg in detector.timeline[-30:]:
            print(f"  [{f:5d} | {f/fps:6.2f}s] {msg}")

    return {"swings": quality.swings, "fps": fps, "counts": dict(counts)}
