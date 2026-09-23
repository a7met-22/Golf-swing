"""
Unit tests for the pure-math parts of the vision package. These need only
numpy — no OpenCV window, no MediaPipe model, no video file.
"""

import math

from vision.geometry import angle_between_vectors, compute_geometry_metrics, joint_angle, knee_flexion
from vision.rotation import compute_rotation, hip_shoulder_angle_top_view


def _lm(x, y, z=0.0, visibility=1.0):
    return {"x": x, "y": y, "z": z, "visibility": visibility}


def test_angle_between_vectors_right_angle():
    import numpy as np

    v1 = np.array([1, 0])
    v2 = np.array([0, 1])
    assert math.isclose(angle_between_vectors(v1, v2), 90.0, abs_tol=1e-6)


def test_angle_between_vectors_zero_vector_is_safe():
    import numpy as np

    v1 = np.array([0, 0])
    v2 = np.array([1, 0])
    assert angle_between_vectors(v1, v2) == 0.0


def test_knee_flexion_straight_leg_is_zero():
    # hip, knee, ankle in a straight vertical line -> 180 degree joint angle -> 0 flexion
    landmarks = {23: _lm(0, 0), 25: _lm(0, 1), 27: _lm(0, 2)}
    assert math.isclose(knee_flexion(landmarks, 23, 25, 27), 0.0, abs_tol=1e-6)


def test_knee_flexion_right_angle_bend():
    # hip above knee, ankle straight out to the side -> 90 degree joint angle -> 90 flexion
    landmarks = {23: _lm(0, 0), 25: _lm(0, 1), 27: _lm(1, 1)}
    assert math.isclose(knee_flexion(landmarks, 23, 25, 27), 90.0, abs_tol=1e-6)


def test_compute_geometry_metrics_returns_none_for_low_visibility():
    landmarks = {
        11: _lm(0, 0),
        12: _lm(1, 0),
        23: _lm(0, 1, visibility=0.1),  # below threshold
        24: _lm(1, 1),
        25: _lm(0, 2),
        27: _lm(0, 3),
        26: _lm(1, 2),
        28: _lm(1, 3),
        13: _lm(-1, 0),
        15: _lm(-2, 0),
        14: _lm(2, 0),
        16: _lm(3, 0),
    }
    metrics = compute_geometry_metrics(landmarks)
    # lead_knee_flexion depends on landmark 23, which is below the visibility threshold
    assert metrics["lead_knee_flexion"] is None


def test_hip_shoulder_angle_top_view_and_rotation_reference():
    world = {11: _lm(-0.2, 0, 0.0), 12: _lm(0.2, 0, 0.0)}
    address_angle = hip_shoulder_angle_top_view(world, 11, 12)
    # no rotation yet -> rotation relative to itself is 0
    assert math.isclose(compute_rotation(world, 11, 12, address_angle), 0.0, abs_tol=1e-6)

    rotated = {11: _lm(-0.2, 0, 0.2), 12: _lm(0.2, 0, -0.2)}
    rot = compute_rotation(rotated, 11, 12, address_angle)
    assert rot != 0.0


def test_joint_angle_straight_line_is_180():
    landmarks = {1: _lm(0, 0), 2: _lm(0, 1), 3: _lm(0, 2)}
    assert math.isclose(joint_angle(landmarks, 1, 2, 3), 180.0, abs_tol=1e-6)
