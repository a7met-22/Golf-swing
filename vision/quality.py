"""
Part 5A — Quality Standards (PRO / AMATEUR / OUT)

Approximate reference ranges from golf-biomechanics literature (pro vs.
solid-amateur averages). We snapshot metrics at TOP / IMPACT /
FOLLOW_THROUGH and classify each one:
  PRO  |  AMATEUR (acceptable beginner-plus)  |  OUT (outside both ranges)

Rotation angles are judged by their absolute value, since the sign is just
an axis convention (left vs. right), not a quality signal.
"""

from __future__ import annotations

QUALITY_STANDARDS = {
    "TOP": {
        "shoulder_rotation": {"pro": (85, 110), "beginner": (55, 85)},
        "hip_rotation": {"pro": (40, 60), "beginner": (25, 45)},
        "x_factor": {"pro": (35, 55), "beginner": (15, 35)},
        "lead_knee_flexion": {"pro": (20, 40), "beginner": (15, 45)},
        "trail_knee_flexion": {"pro": (25, 45), "beginner": (20, 50)},
        "spine_tilt": {"pro": (25, 45), "beginner": (20, 50)},
    },
    "IMPACT": {
        "hip_rotation": {"pro": (15, 40), "beginner": (5, 40)},
        "spine_tilt": {"pro": (25, 40), "beginner": (20, 45)},
        "lead_knee_flexion": {"pro": (10, 30), "beginner": (10, 35)},
        "lead_elbow_flexion": {"pro": (0, 20), "beginner": (0, 30)},
    },
    "FOLLOW_THROUGH": {
        "shoulder_rotation": {"pro": (85, 125), "beginner": (55, 125)},
        "hip_rotation": {"pro": (50, 100), "beginner": (30, 100)},
    },
}

ROTATION_KEYS = {"shoulder_rotation", "hip_rotation", "x_factor"}
TEMPO_PRO_LABEL = "PRO ~ 3:1 (acceptable range 2.5-3.5)"


def classify_metric(key: str, value: float, std: dict) -> tuple[str, float]:
    v = abs(value) if key in ROTATION_KEYS else value
    if std["pro"][0] <= v <= std["pro"][1]:
        return "PRO", v
    if std["beginner"][0] <= v <= std["beginner"][1]:
        return "AMATEUR", v
    return "OUT", v


class SwingQualityAnalyzer:
    """Tracks per-swing timing and takes metric snapshots at TOP / IMPACT /
    FOLLOW_THROUGH — kept separate from :class:`~vision.phase_detector.AngleBasedSwingDetector`
    on purpose: phase *detection* and quality *evaluation* are different concerns."""

    def __init__(self):
        self.swings: list[dict] = []

    def start(self, n, frame):
        # top/down are pre-initialized, not appended dynamically later
        self.swings.append(
            {"n": n, "start": frame, "top": None, "down": None, "impact": None, "end": None, "snapshots": {}}
        )

    def cancel(self):
        if self.swings:
            self.swings.pop()

    def snapshot(self, phase: str, metrics: dict):
        """Only the first frame of each phase is captured, to measure the value at entry."""
        if not self.swings or phase in self.swings[-1]["snapshots"]:
            return
        self.swings[-1]["snapshots"][phase] = dict(metrics)

    def report_text(self, fps: float) -> str:
        lines = ["", "=== Swing report & quality (PRO / acceptable amateur) ==="]
        if not self.swings:
            lines.append("No completed swings were detected.")
            return "\n".join(lines)
        marks = {"PRO": "PRO", "AMATEUR": "acceptable (beginner-plus)", "OUT": "outside range"}
        for s in self.swings:
            st = f"{s['start']/fps:.2f}s" if s["start"] is not None else "-"
            im = f"{s['impact']/fps:.2f}s" if s["impact"] is not None else "-"
            en = f"{s['end']/fps:.2f}s" if s["end"] is not None else "-"
            lines.append(f"\nSwing #{s['n']} — starts {st} | impact {im} | ends {en}")

            if None not in (s["start"], s["top"], s["down"], s["impact"]):
                bs = (s["top"] - s["start"]) / fps
                ds = (s["impact"] - s["down"]) / fps
                tempo = bs / ds if ds > 0 else float("nan")
                lines.append(f"  tempo: backswing {bs:.2f}s | downswing {ds:.2f}s | ratio {tempo:.2f}   ({TEMPO_PRO_LABEL})")

            for phase, snap in s["snapshots"].items():
                lines.append(f"  {phase}:")
                for key, std in QUALITY_STANDARDS.get(phase, {}).items():
                    if key not in snap or snap[key] is None:
                        continue
                    label, v = classify_metric(key, snap[key], std)
                    lines.append(
                        f"      {key:22s} = {v:6.1f}  -> {marks[label]}"
                        f"  [PRO {std['pro'][0]}-{std['pro'][1]}"
                        f" | amateur {std['beginner'][0]}-{std['beginner'][1]}]"
                    )
        return "\n".join(lines)
