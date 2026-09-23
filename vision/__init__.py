"""
vision
======
Computer-vision half of the project: everything needed to turn a raw golf
swing video into pose landmarks, biomechanics metrics, a live dashboard
overlay, and a per-swing CSV dataset.

This package has no knowledge of the LLM agent in ``agent/`` — it only
produces data (CSV files) that the agent later reads. That separation is
intentional: you can run swing analysis on a machine with no GPU/LLM setup
at all, and you can develop the agent against a dataset without ever
touching OpenCV or MediaPipe.
"""

from .pipeline import run_pipeline

__all__ = ["run_pipeline"]
