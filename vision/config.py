"""
All tunable numbers for the computer-vision pipeline live here, and only
here. Detection sensitivity, colors, and export paths are grouped so you
never have to hunt through the pipeline code to tweak behavior.
"""

# ---------------------------------------------------------------------------
# MediaPipe pose detection confidence
# ---------------------------------------------------------------------------
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5
VISIBILITY_THRESHOLD = 0.5

# Pose Landmarker model (auto-downloaded by vision.model_loader if missing)
MODEL_PATH = "pose_landmarker_full.task"
MODEL_URLS = {
    "lite": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
            "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    "full": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
            "pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    "heavy": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
             "pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
}

# ---------------------------------------------------------------------------
# Skeleton overlay — colors (BGR, OpenCV convention) and joint chains
# ---------------------------------------------------------------------------
COLOR_LEAD = (255, 255, 0)     # cyan   - lead (front) leg/arm
COLOR_TRAIL = (0, 165, 255)    # orange - trail (back) leg/arm
COLOR_SPINE = (0, 255, 0)      # green  - spine line

# Index pairs from MediaPipe Pose's 33 landmarks
LEAD_CHAIN = [(11, 23), (23, 25), (25, 27)]
TRAIL_CHAIN = [(12, 24), (24, 26), (26, 28)]
SPINE_LINKS = [(11, 12), (23, 24)]
LEAD_ARM = [(11, 13), (13, 15)]     # shoulder -> elbow -> wrist (left)
TRAIL_ARM = [(12, 14), (14, 16)]    # shoulder -> elbow -> wrist (right)

# ---------------------------------------------------------------------------
# Swing-phase state machine
# ---------------------------------------------------------------------------
ANGLE_EMA = 0.35
ANGULAR_STILL_DEG = 3.0
BACKSWING_START_DEG = 18
TOP_MIN_DEG = 45
TOP_MAX_DEG = 125
TOP_PEAK_DROP_DEG = 2.5
DOWNSWING_UNWIND_DEG = 2.5
IMPACT_ROT_MAX_DEG = 25
WAGGLE_CANCEL_DEG = 8
CONFIRM_FRAMES = 2
RESET_RETURN_DEG = 15
RESET_HOLD_FRAMES = 8

# Timeout guards (in frames, approx. @30fps) so a stuck phase auto-resets
STATE_TIMEOUT_FRAMES = {
    "BACKSWING": 75,
    "TOP": 25,
    "DOWNSWING": 45,
    "FOLLOW_THROUGH": 120,
}

# Cut/scene-change detection (both conditions must fire together)
CUT_SHOULDER_JUMP_DEG = 80
CUT_HAND_JUMP_RATIO = 0.7

# Address-position calibration
CALIB_STILL_FRAMES = 10
CALIB_MAX_FRAMES = 60
HANDS_LOW_RATIO = 0.30
RESET_LOST_FRAMES = 12
HAND_SMOOTHING_WINDOW = 5

# ---------------------------------------------------------------------------
# Dataset export (CSV + comparison plots)
# ---------------------------------------------------------------------------
PLAYER_NAME = ""              # leave blank to derive it from the video filename
DATASET_DIR = "swing_data"    # output folder
MASTER_CSV = "all_swings_dataset.csv"   # unified dataset the agent reads
TEMPO_PRO_RANGE = (2.5, 3.5)  # pro-like tempo ratio (the well-known 3:1 rule)

# ---------------------------------------------------------------------------
# Dashboard overlay
# ---------------------------------------------------------------------------
DASHBOARD_WIDTH = 420

PHASE_COLORS = {
    "READY": (120, 120, 120),
    "ADDRESS": (180, 180, 180),
    "BACKSWING": (255, 200, 0),
    "TOP": (0, 200, 255),
    "DOWNSWING": (0, 140, 255),
    "IMPACT": (0, 0, 255),
    "FOLLOW_THROUGH": (0, 255, 0),
}
