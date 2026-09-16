"""
Part 6 — رسم لوحة الداشبورد الحي (النصوص + الأشرطة الملونة) ولزقها جنب الفيديو
"""
import cv2
import numpy as np

import config


def _value_to_bar_width(value, min_val, max_val, max_bar_width):
    fraction = (value - min_val) / (max_val - min_val)
    fraction = max(0.0, min(1.0, fraction))  # يمنع الشريط يطلع برا حدود اللوحة
    return int(fraction * max_bar_width)


def _draw_metric_row(panel, y, label, value, min_val, max_val, color, bar_max_width):
    cv2.putText(panel, label, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(panel, f"{value:.1f} deg", (bar_max_width - 50, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    bar_y = y + 12
    bar_width = _value_to_bar_width(value, min_val, max_val, bar_max_width)
    cv2.rectangle(panel, (20, bar_y), (20 + bar_max_width, bar_y + 8), (60, 60, 60), -1)  # الخلفية
    if bar_width > 0:
        cv2.rectangle(panel, (20, bar_y), (20 + bar_width, bar_y + 8), color, -1)          # الجزء المليان


def _draw_phase_box(panel, phase):
    color = config.PHASE_COLORS.get(phase, (255, 255, 255))
    cv2.rectangle(panel, (20, 45), (300, 85), color, 2)
    cv2.putText(panel, phase, (32, 73), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)


def draw_dashboard(height, phase, metrics, width=config.DASHBOARD_WIDTH):
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(panel, "LIVE JOINT KINEMATICS", (20, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    _draw_phase_box(panel, phase)

    rows = [
        ("Spine tilt (from vertical)", metrics["spine_tilt"], 0, 60, (200, 180, 255)),
        ("Lead (L) knee flexion", metrics["lead_knee_flexion"], 0, 60, (255, 255, 0)),
        ("Trail (R) knee flexion", metrics["trail_knee_flexion"], 0, 60, (0, 165, 255)),
        ("Hip rotation", metrics["hip_rotation"], -90, 90, (0, 255, 0)),
        ("Shoulder rotation", metrics["shoulder_rotation"], -90, 90, (0, 255, 0)),
        ("X-Factor (sep.)", metrics["x_factor"], -60, 60, (0, 255, 255)),
    ]

    y = 130
    for label, value, min_v, max_v, color in rows:
        _draw_metric_row(panel, y, label, value, min_v, max_v, color, bar_max_width=width - 120)
        y += 55

    return panel


def compose_frame(video_frame, phase, metrics):
    """يلزق الفيديو (بعد رسم الـ Skeleton عليه) جنب لوحة الداشبورد"""
    panel = draw_dashboard(video_frame.shape[0], phase, metrics)
    return np.hstack([video_frame, panel])
