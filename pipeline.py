"""
Part 7 — تجميع كل الأجزاء في لوب واحد كامل: يقرأ الفيديو فريم فريم ويصدّر النتيجة النهائية
"""
from collections import deque

import cv2

import config
from pose_extraction import extract_landmarks
from skeleton_drawing import draw_skeleton
from geometry_engine import compute_geometry_metrics
from rotation_engine import hip_shoulder_angle_top_view, compute_rotation_metrics
from phase_detector import SwingPhaseDetector, get_hand_y
from dashboard import compose_frame


def run_pipeline(video_path, output_path, landmarker):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"مقدرش أفتح الفيديو: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w, h = int(cap.get(3)), int(cap.get(4))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w + config.DASHBOARD_WIDTH, h))

    frame_count = 0
    hand_y_history = deque(maxlen=config.HAND_SMOOTHING_WINDOW)
    detector = None
    address_hip_angle = address_shoulder_angle = None
    last_good = None  # آخر نتيجة صحيحة، لو فريم معين مفيهوش جسم ظاهر

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        timestamp_ms = int((frame_count / fps) * 1000)
        image_landmarks, world_landmarks = extract_landmarks(landmarker, frame, timestamp_ms)

        if image_landmarks is None:
            if last_good is not None:
                out.write(compose_frame(frame, last_good["phase"], last_good["metrics"]))
            frame_count += 1
            continue

        if detector is None:  # أول فريم صالح = نلتقط قيم الـ Address المرجعية
            hand_y_history.append(get_hand_y(image_landmarks))
            detector = SwingPhaseDetector(address_hand_y=hand_y_history[-1])
            address_hip_angle = hip_shoulder_angle_top_view(world_landmarks, 23, 24)
            address_shoulder_angle = hip_shoulder_angle_top_view(world_landmarks, 11, 12)

        hand_y_history.append(get_hand_y(image_landmarks))
        smoothed_hand_y = sum(hand_y_history) / len(hand_y_history)
        phase = detector.update(smoothed_hand_y)

        metrics = compute_geometry_metrics(image_landmarks)
        metrics.update(compute_rotation_metrics(world_landmarks, address_hip_angle, address_shoulder_angle))

        frame = draw_skeleton(frame, image_landmarks)
        out.write(compose_frame(frame, phase, metrics))

        last_good = {"phase": phase, "metrics": metrics}
        frame_count += 1

    cap.release()
    out.release()
    print(f"تم! {frame_count} فريم اتعالجوا. الفيديو النهائي: {output_path}")
