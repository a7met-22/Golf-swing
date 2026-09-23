"""
Part 5B — Dataset Export

Writes one row per swing, with English snake_case column names so the CSV
loads directly with pandas — this is the exact hand-off point to the agent
package, which never sees the video, only this data.

- Each metric gets a companion ``*_class`` column (PRO / AMATEUR / OUT)
- ``tempo_ratio`` = backswing time / downswing time, computed from raw
  frame counts (not rounded seconds) for accuracy
- One CSV per video, plus a master dataset that accumulates across videos
  (de-duplicated per player)
- Unified key for the agent: (player, swing_id) — numbering continues
  across scene cuts within one video, so it never repeats
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import pandas as pd

from . import config
from .quality import QUALITY_STANDARDS, classify_metric

SNAPSHOT_TO_CSV = {
    "TOP": {
        "shoulder_rotation": "top_shoulder_rotation_deg",
        "hip_rotation": "top_hip_rotation_deg",
        "x_factor": "top_x_factor_deg",
        "spine_tilt": "top_spine_tilt_deg",
        "lead_knee_flexion": "top_lead_knee_flexion_deg",
        "trail_knee_flexion": "top_trail_knee_flexion_deg",
    },
    "IMPACT": {
        "hip_rotation": "impact_hip_rotation_deg",
        "spine_tilt": "impact_spine_tilt_deg",
        "lead_knee_flexion": "impact_lead_knee_flexion_deg",
        "lead_elbow_flexion": "impact_lead_elbow_flexion_deg",
    },
    "FOLLOW_THROUGH": {
        "shoulder_rotation": "follow_shoulder_rotation_deg",
        "hip_rotation": "follow_hip_rotation_deg",
    },
}

# Max number of metrics that could ever be measured (if every joint were 100% visible)
TOTAL_POSSIBLE_METRICS = sum(len(v) for v in SNAPSHOT_TO_CSV.values())


class SwingDatasetExporter:
    """Builds and saves the per-swing CSV (and a comparison PNG) from a
    :class:`~vision.pipeline.run_pipeline` result."""

    def __init__(self, result: dict, video_path: str, player_name: str = ""):
        self.swings = result["swings"]
        self.fps = result["fps"]
        self.video_path = video_path
        self.player = player_name or os.path.splitext(os.path.basename(video_path))[0]

    def _sec(self, frame):
        return round(frame / self.fps, 2) if frame is not None else None

    def build_dataframe(self) -> pd.DataFrame:
        rows = []
        for s in self.swings:
            start, top = s.get("start"), s.get("top")
            down, impact = s.get("down"), s.get("impact")
            end = s.get("end")

            # Timing from raw frame counts — far more accurate than subtracting rounded seconds
            bs_f = (top - start) if (top is not None and start is not None) else None
            ds_f = (impact - down) if (impact is not None and down is not None) else None

            row = {
                "player": self.player,
                "video": os.path.basename(self.video_path),
                "swing_id": s["n"],
                "start_s": self._sec(start),
                "top_s": self._sec(top),
                "down_s": self._sec(down),
                "impact_s": self._sec(impact),
                "end_s": self._sec(end),
                "backswing_time_s": round(bs_f / self.fps, 2) if bs_f else None,
                "downswing_time_s": round(ds_f / self.fps, 2) if ds_f else None,
                "tempo_ratio": round(bs_f / ds_f, 2) if (bs_f and ds_f) else None,
                "duration_total_s": (self._sec(end - start) if (end is not None and start is not None) else None),
            }

            pro_n, total_n = 0, 0
            for phase, mapping in SNAPSHOT_TO_CSV.items():
                snap = s["snapshots"].get(phase, {})
                for metric_key, col in mapping.items():
                    val = snap.get(metric_key)
                    row[col] = val
                    class_col = col.replace("_deg", "_class")
                    std = QUALITY_STANDARDS.get(phase, {}).get(metric_key)
                    if val is not None and std:
                        label, _ = classify_metric(metric_key, val, std)
                        row[class_col] = label
                        total_n += 1
                        pro_n += label == "PRO"
                    else:
                        row[class_col] = None
            row["metrics_evaluated"] = total_n
            row["metrics_pro"] = pro_n
            row["quality_score_pct"] = round(100 * pro_n / total_n, 1) if total_n else None
            # Not every quality_score_pct is equally reliable (100% of 2 metrics vs. of 12
            # metrics aren't the same) — this confidence column disambiguates the two
            row["quality_confidence_pct"] = round(100 * total_n / TOTAL_POSSIBLE_METRICS, 1)

            top_rot = row.get("top_shoulder_rotation_deg")
            row["shallow_top_flag"] = bool(top_rot is not None and abs(top_rot) < (config.TOP_MIN_DEG + 10))

            rows.append(row)
        return pd.DataFrame(rows)

    def save(self) -> pd.DataFrame:
        os.makedirs(config.DATASET_DIR, exist_ok=True)
        df = self.build_dataframe()
        per_video = os.path.join(config.DATASET_DIR, f"swings_{self.player}.csv")
        df.to_csv(per_video, index=False, encoding="utf-8-sig")

        if df.empty:
            print(f"No swings recorded — wrote an empty file: {per_video}")
            print("   (existing master rows for this player are left untouched)")
            return df

        master_path = os.path.join(config.DATASET_DIR, config.MASTER_CSV)
        if os.path.exists(master_path):
            try:
                master = pd.read_csv(master_path)
            except pd.errors.EmptyDataError:
                master = pd.DataFrame(columns=df.columns)  # empty file from a previous run
            master = master[master["player"] != self.player]  # de-dup: latest analysis wins
            master = pd.concat([master, df], ignore_index=True)
        else:
            master = df
        master.to_csv(master_path, index=False, encoding="utf-8-sig")

        print(f"Per-swing CSV: {per_video} ({len(df)} swings)")
        print(
            f"Master dataset: {master_path} -> {len(master)} swings | "
            f"{master['player'].nunique()} player(s): {sorted(master['player'].unique())}"
        )
        return df

    def plot(self, df: pd.DataFrame | None = None):
        df = self.build_dataframe() if df is None else df
        panels = [
            ("top_shoulder_rotation_deg", "Shoulder rotation @ TOP (deg)", QUALITY_STANDARDS["TOP"]["shoulder_rotation"]["pro"]),
            ("top_hip_rotation_deg", "Hip rotation @ TOP (deg)", QUALITY_STANDARDS["TOP"]["hip_rotation"]["pro"]),
            ("top_x_factor_deg", "X-Factor @ TOP (deg)", QUALITY_STANDARDS["TOP"]["x_factor"]["pro"]),
            ("tempo_ratio", "Tempo (backswing / downswing)", config.TEMPO_PRO_RANGE),
            ("follow_shoulder_rotation_deg", "Shoulder rotation @ FOLLOW (deg)", QUALITY_STANDARDS["FOLLOW_THROUGH"]["shoulder_rotation"]["pro"]),
        ]

        fig, axes = plt.subplots(2, 3, figsize=(19, 9))
        axes = axes.ravel()
        for ax, (col, title, band) in zip(axes, panels):
            sub = df[["swing_id", col]].dropna()
            ax.axhspan(band[0], band[1], color="green", alpha=0.15, label="PRO range")
            if len(sub):
                bars = ax.bar(sub["swing_id"].astype(str), sub[col], color="#4C9BE8")
                ax.bar_label(bars, fmt="%.1f", fontsize=9)
                ax.axhline(float(sub[col].mean()), color="red", ls="--", lw=1, label="avg")
            else:
                ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title, fontsize=11)
            ax.set_xlabel("swing #")
            ax.grid(axis="y", alpha=0.3)
            ax.legend(fontsize=8)

        ax = axes[5]
        sub = df[["swing_id", "quality_score_pct"]].dropna()
        if len(sub):
            colors = ["green" if v >= 70 else "orange" if v >= 40 else "red" for v in sub["quality_score_pct"]]
            bars = ax.bar(sub["swing_id"].astype(str), sub["quality_score_pct"], color=colors)
            ax.bar_label(bars, fmt="%.0f%%", fontsize=9)
            ax.set_ylim(0, 100)
        else:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Quality score (% metrics in PRO range)", fontsize=11)
        ax.set_xlabel("swing #")
        ax.grid(axis="y", alpha=0.3)

        fig.suptitle(f"Swing comparison — {self.player}", fontsize=14)
        plt.tight_layout()
        out = os.path.join(config.DATASET_DIR, f"comparison_{self.player}.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Comparison plot: {out}")
        return out
