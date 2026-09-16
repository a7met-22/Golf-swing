"""
Part 1 — استخراج نقاط الجسم (Landmarks) من كل فريم باستخدام MediaPipe Pose (Tasks API)

ملاحظة مهمة: الإصدارات الحديثة من مكتبة mediapipe شالت الـ Legacy API القديم
(mp.solutions.pose) خالص، فالملف ده مبني على الـ Tasks API الجديد اللي هو
الطريقة الرسمية الوحيدة الشغالة حاليًا.
"""
import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

import config


def create_landmarker(model_path=config.DEFAULT_MODEL_PATH):
    """
    بينشئ كاشف الجسم (PoseLandmarker) مرة واحدة بس في بداية البرنامج.
    استخدمه جوه "with" أو اقفله بـ landmarker.close() لما تخلص.
    """
    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO,  # VIDEO مش IMAGE عشان يستخدم الفريم اللي قبله في التتبع
        min_pose_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
    )
    return vision.PoseLandmarker.create_from_options(options)


def extract_landmarks(landmarker, frame_bgr, timestamp_ms):
    """
    فريم واحد (BGR) + الوقت بالمللي ثانية -> (image_landmarks, world_landmarks)

    - image_landmarks: إحداثيات بالبكسل (نستخدمها في الرسم وحساب زوايا الركبة/العمود الفقري)
    - world_landmarks: إحداثيات حقيقية بالمتر (نستخدمها في حساب اللفة/الـ rotation)

    بيرجع (None, None) لو الجسم مش ظاهر في الفريم ده.
    """
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    result = landmarker.detect_for_video(mp_image, timestamp_ms)  # استدعاء واحد بس للفريم

    if not result.pose_landmarks or not result.pose_world_landmarks:
        return None, None

    h, w, _ = frame_bgr.shape
    image_person = result.pose_landmarks[0]        # أول شخص مكتشف في الفريم
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
