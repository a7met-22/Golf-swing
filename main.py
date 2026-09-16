"""
نقطة تشغيل المشروع

الاستخدام:
    python main.py --video swing.mp4
    python main.py --video swing.mp4 --output result.mp4 --model models/pose_landmarker_full.task
"""
import argparse

import config
from pose_extraction import create_landmarker
from pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser(description="Golf Swing Biomechanics Analyzer")
    parser.add_argument("--video", required=True, help="مسار فيديو السوينج المدخل")
    parser.add_argument("--output", default="output_with_dashboard.mp4", help="مسار الفيديو الناتج")
    parser.add_argument("--model", default=config.DEFAULT_MODEL_PATH, help="مسار ملف موديل MediaPipe (.task)")
    args = parser.parse_args()

    landmarker = create_landmarker(args.model)
    try:
        run_pipeline(args.video, args.output, landmarker)
    finally:
        landmarker.close()


if __name__ == "__main__":
    main()
