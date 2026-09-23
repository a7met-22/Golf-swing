"""
agent
=====
The conversational "Golf Swing Analyst" — a locally-hosted LLM (Qwen3-4B by
default) that answers questions about the swing dataset produced by the
``vision`` package, by calling a small whitelisted set of read-only data
tools. It never touches OpenCV, MediaPipe, or the video itself — it only
ever sees the exported CSV.

This package is intentionally independent of ``vision``: it only needs a
CSV file that follows the schema ``vision.export.SwingDatasetExporter``
produces. You can develop or test the agent against any CSV with that
shape without ever running the video pipeline.

``GolfSwingAgent`` is exposed lazily (see ``__getattr__`` below) so that
importing e.g. ``agent.security`` or ``agent.tools`` for unit testing never
forces torch/transformers to be installed — only building or loading an
actual agent does.
"""

__all__ = ["GolfSwingAgent"]


def __getattr__(name):
    if name == "GolfSwingAgent":
        from .engine import GolfSwingAgent

        return GolfSwingAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
