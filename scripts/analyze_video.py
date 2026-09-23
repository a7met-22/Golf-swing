#!/usr/bin/env python3
"""
Runs the full computer-vision pipeline on one swing video:
  1. downloads the MediaPipe pose model if it isn't already on disk
  2. processes the video, writing an annotated preview with the live dashboard
  3. exports the per-swing CSV (+ appends to the master dataset the agent reads)
  4. saves a comparison plot across this video's swings

Usage:
    python scripts/analyze_video.py path/to/swing.mp4
    python scripts/analyze_video.py path/to/swing.mp4 --player ahmed --actual-swings 11
    python scripts/analyze_video.py path/to/swing.mp4 --model-variant lite --no-plot
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vision import config as vconfig
from vision.export import SwingDatasetExporter
from vision.model_loader import ensure_model
from vision.pipeline import run_pipeline
from vision.pose import create_landmarker


def parse_args():
    p = argparse.ArgumentParser(description="Analyze a golf swing video.")
    p.add_argument("video", help="Path to the input video file.")
    p.add_argument("-o", "--output", default="output_with_dashboard.mp4", help="Annotated output video path.")
    p.add_argument("--player", default=vconfig.PLAYER_NAME, help="Player name (defaults to the video filename).")
    p.add_argument(
        "--model-variant",
        choices=["lite", "full", "heavy"],
        default="full",
        help="Pose model accuracy/speed tradeoff.",
    )
    p.add_argument("--model-path", default=vconfig.MODEL_PATH, help="Where to cache the pose model file.")
    p.add_argument("--actual-swings", type=int, default=None, help="Ground-truth swing count, to print detection accuracy.")
    p.add_argument("--no-plot", action="store_true", help="Skip saving the comparison plot.")
    return p.parse_args()


def main():
    args = parse_args()

    model_path = ensure_model(args.model_path, variant=args.model_variant)
    landmarker = create_landmarker(model_path)
    try:
        result = run_pipeline(args.video, args.output, landmarker)
    finally:
        landmarker.close()

    if args.actual_swings:
        detected = len(result["swings"])
        accuracy = detected / args.actual_swings
        print(f"\nDetection: {detected} swings out of {args.actual_swings} actual -> {accuracy:.0%}")

    exporter = SwingDatasetExporter(result, args.video, args.player)
    df_swings = exporter.save()
    if not args.no_plot and not df_swings.empty:
        exporter.plot(df_swings)

    print(f"\nAnnotated video: {args.output}")
    print(f"Dataset folder:  {vconfig.DATASET_DIR}/")


if __name__ == "__main__":
    main()
