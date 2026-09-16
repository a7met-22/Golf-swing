"""
Part 3 — Geometry Engine: تحويل إحداثيات البكسل لزوايا حقيقية (spine tilt, knee flexion)
كله رياضيات بحتة (vector math) - مفيش أي Machine Learning هنا.
"""
import numpy as np


def vector(p1, p2):
    """المتجه من نقطة p1 لنقطة p2"""
    return np.array([p2["x"] - p1["x"], p2["y"] - p1["y"]])


def angle_between_vectors(v1, v2):
    """الزاوية بين متجهين بالدرجات: cos(theta) = (A.B) / (|A| * |B|)"""
    dot = np.dot(v1, v2)
    norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norm_product == 0:
        return 0.0  # حماية: مفصلين فوق بعض بالظبط
    cos_angle = np.clip(dot / norm_product, -1.0, 1.0)  # يمنع خطأ تقريب يطلع برا [-1,1]
    return float(np.degrees(np.arccos(cos_angle)))


def midpoint(p1, p2):
    return {"x": (p1["x"] + p2["x"]) / 2, "y": (p1["y"] + p2["y"]) / 2}


def joint_angle(landmarks, a_idx, vertex_idx, b_idx):
    """الزاوية الفعلية عند مفصل (vertex) بين طرفين (a) و (b)"""
    vertex = landmarks[vertex_idx]
    v1 = vector(vertex, landmarks[a_idx])
    v2 = vector(vertex, landmarks[b_idx])
    return angle_between_vectors(v1, v2)


def knee_flexion(landmarks, hip_idx, knee_idx, ankle_idx):
    """180 - الزاوية الفعلية = مقدار الانثناء (رجل مفرودة = صفر)"""
    return 180 - joint_angle(landmarks, hip_idx, knee_idx, ankle_idx)


def spine_tilt_from_vertical(landmarks):
    """ميل خط (منتصف الكتف -> منتصف الحوض) عن الاتجاه الرأسي"""
    mid_shoulder = midpoint(landmarks[11], landmarks[12])
    mid_hip = midpoint(landmarks[23], landmarks[24])
    spine_vector = vector(mid_shoulder, mid_hip)
    return angle_between_vectors(spine_vector, np.array([0, 1]))


def compute_geometry_metrics(image_landmarks):
    """يجمع كل مقاييس الهندسة في dictionary واحد"""
    return {
        "spine_tilt": round(spine_tilt_from_vertical(image_landmarks), 1),
        "lead_knee_flexion": round(knee_flexion(image_landmarks, 23, 25, 27), 1),
        "trail_knee_flexion": round(knee_flexion(image_landmarks, 24, 26, 28), 1),
    }
