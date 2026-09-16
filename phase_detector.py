"""
Part 5 — كاشف مرحلة السوينج (State Machine بسيط، بدون أي Machine Learning)
بيراقب سرعة ارتفاع الإيدين (متوسط الرسغين) عشان يعرف احنا في أنهي مرحلة.
"""
import config


class SwingPhaseDetector:
    def __init__(self, address_hand_y, speed_threshold=config.HAND_SPEED_THRESHOLD):
        self.phase = "ADDRESS"
        self.address_hand_y = address_hand_y
        self.speed_threshold = speed_threshold
        self.prev_hand_y = address_hand_y

    def update(self, hand_y):
        velocity = hand_y - self.prev_hand_y  # سالب = طالع لفوق، موجب = نازل لتحت
        self.prev_hand_y = hand_y

        if self.phase == "ADDRESS" and velocity < -self.speed_threshold:
            self.phase = "BACKSWING"
        elif self.phase == "BACKSWING" and abs(velocity) < self.speed_threshold:
            self.phase = "TOP"
        elif self.phase == "TOP" and velocity > self.speed_threshold:
            self.phase = "DOWNSWING"
        elif self.phase == "DOWNSWING" and hand_y >= self.address_hand_y:
            self.phase = "IMPACT"
        elif self.phase == "IMPACT":
            self.phase = "FOLLOW_THROUGH"

        return self.phase


def get_hand_y(image_landmarks):
    """متوسط ارتفاع الرسغين - تقريب بسيط لمكان الإيدين (بديل عن تتبع العصا)"""
    return (image_landmarks[15]["y"] + image_landmarks[16]["y"]) / 2
