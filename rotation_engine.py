"""
Part 4 — حساب لفة الحوض والكتف والـ X-Factor من الإحداثيات الحقيقية بالمتر (world landmarks)
"""
import math


def hip_shoulder_angle_top_view(world_landmarks, left_idx, right_idx):
    """اتجاه الخط الواصل بين نقطتين وهو متبوّس من فوق (بيرجع زاوية بالدرجات)"""
    left = world_landmarks[left_idx]
    right = world_landmarks[right_idx]
    dx = right["x"] - left["x"]
    dz = right["z"] - left["z"]
    return math.degrees(math.atan2(dz, dx))


def compute_rotation(world_landmarks, left_idx, right_idx, reference_angle):
    """الفرق بين الاتجاه الحالي والاتجاه المرجعي وقت الـ Address"""
    current_angle = hip_shoulder_angle_top_view(world_landmarks, left_idx, right_idx)
    rotation = current_angle - reference_angle

    # تصحيح دوران الزوايا (359 قريبة من صفر، مش بعيدة عنه)
    if rotation > 180:
        rotation -= 360
    elif rotation < -180:
        rotation += 360
    return rotation


def compute_rotation_metrics(world_landmarks, address_hip_angle, address_shoulder_angle):
    hip_rotation = compute_rotation(world_landmarks, 23, 24, address_hip_angle)
    shoulder_rotation = compute_rotation(world_landmarks, 11, 12, address_shoulder_angle)
    return {
        "hip_rotation": round(hip_rotation, 1),
        "shoulder_rotation": round(shoulder_rotation, 1),
        "x_factor": round(shoulder_rotation - hip_rotation, 1),
    }
