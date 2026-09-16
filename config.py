"""
إعدادات وثوابت المشروع كله في مكان واحد.
لو حبيت تغيّر أي رقم (لون، حساسية، مدى...) غيّره من هنا بدل ما تدور عليه جوه الملفات.
"""

# مسار ملف موديل MediaPipe الافتراضي (لازم تحمّله بنفسك - التعليمات في README.md)
DEFAULT_MODEL_PATH = "models/pose_landmarker_full.task"

# إعدادات دقة كشف الجسم
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

# ألوان الـ Skeleton بصيغة BGR (اللي بتستخدمها OpenCV، مش RGB العادية)
COLOR_LEAD = (255, 255, 0)     # cyan   - الرجل الأمامية (Lead)
COLOR_TRAIL = (0, 165, 255)    # orange - الرجل الخلفية (Trail)
COLOR_SPINE = (0, 255, 0)      # green  - العمود الفقري

# سلاسل الاتصال بين المفاصل (الأرقام دي indices من الـ 33 نقطة بتاعة MediaPipe Pose)
LEAD_CHAIN = [(11, 23), (23, 25), (25, 27)]     # كتف -> حوض -> ركبة -> كاحل (شمال)
TRAIL_CHAIN = [(12, 24), (24, 26), (26, 28)]    # نفس الحاجة يمين
SPINE_LINKS = [(11, 12), (23, 24)]              # خط الكتفين + خط الحوض

VISIBILITY_THRESHOLD = 0.5   # تحت الرقم ده منرسمش النقطة أصلًا

# إعدادات كاشف مرحلة السوينج (Part 5)
HAND_SPEED_THRESHOLD = 8     # بكسل/فريم - لازم تظبطه حسب دقة الفيديو بتاعك (شوف README)
HAND_SMOOTHING_WINDOW = 5    # عدد الفريمات في المتوسط المتحرك لتنعيم الإشارة

# إعدادات لوحة الداشبورد
DASHBOARD_WIDTH = 420

PHASE_COLORS = {
    "ADDRESS": (180, 180, 180),
    "BACKSWING": (255, 200, 0),
    "TOP": (0, 200, 255),
    "DOWNSWING": (0, 140, 255),
    "IMPACT": (0, 0, 255),
    "FOLLOW_THROUGH": (0, 255, 0),
}
