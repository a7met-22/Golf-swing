"""
Loads the dataset produced by ``vision.export.SwingDatasetExporter`` (B1 +
B2 in the original notebook) and renders the system prompt template with
the real column names, player names, and video list.
"""

from __future__ import annotations

import os

import pandas as pd

from . import config


class SwingDataStore:
    """Read-only view of the master swing dataset, plus the logic to turn
    :data:`agent.config.AGENT_SYSTEM_PROMPT` into a fully-injected prompt
    for a given run."""

    def __init__(self, data_dir: str = config.DATA_DIR, master_csv: str = config.MASTER_CSV):
        self.data_dir = data_dir
        self.master_path = os.path.join(data_dir, master_csv)
        self.df: pd.DataFrame = pd.DataFrame()
        self.players: list[str] = []
        self.numeric_columns: list[str] = []

    def load(self) -> "SwingDataStore":
        if not os.path.exists(self.master_path):
            raise FileNotFoundError(
                f"Dataset not found: {self.master_path}\n"
                "Run the vision pipeline first (see scripts/analyze_video.py) to produce it."
            )

        df = pd.read_csv(self.master_path)
        self.players = sorted(df["player"].dropna().unique().tolist())
        self.numeric_columns = [
            c for c in df.columns if c not in ("player", "video") and pd.api.types.is_numeric_dtype(df[c])
        ]

        # swing_uid: a single human-friendly key the model can refer to
        # (e.g. "amir#3") — kept as text, never used inside expr calculations.
        if "swing_id" in df.columns:
            sid = pd.to_numeric(df["swing_id"], errors="coerce")
            df["swing_uid"] = df["player"].astype(str) + "#" + sid.map(
                lambda x: str(int(x)) if pd.notna(x) else "?"
            )
        else:
            df["swing_uid"] = df["player"].astype(str) + "#0"

        self.df = df
        print(
            f"Loaded {len(df)} swings | {len(self.players)} player(s) | "
            f"{len(self.numeric_columns)} numeric columns"
        )
        print(f"  players: {self.players}")
        return self

    def build_system_prompt(self, memory_hint: str = "") -> str:
        df = self.df
        id_cols = ", ".join(c for c in ("player", "video", "swing_id") if c in df.columns)
        lines = [
            f"- identity: {id_cols}",
            "- numeric columns (use these names EXACTLY, no prefixes): " + ", ".join(self.numeric_columns),
        ]
        notes = []
        if "quality_confidence_pct" in self.numeric_columns:
            notes.append(
                "quality_confidence_pct: % of all possible metrics actually measured "
                "(occlusion lowers it; flag low reliability if <50 or very different confidences)"
            )
        if "shallow_top_flag" in self.numeric_columns:
            notes.append("shallow_top_flag: true if backswing barely passed TOP threshold")
        if notes:
            lines.append("- notes: " + "; ".join(notes))

        videos = ", ".join(sorted(df["video"].dropna().unique())) if "video" in df else "(unknown)"
        players = ", ".join(self.players) if self.players else "(none yet)"
        hints = memory_hint or "(none)"
        return (
            config.AGENT_SYSTEM_PROMPT.replace("<<COLUMNS>>", "\n".join(lines))
            .replace("<<VIDEOS>>", videos)
            .replace("<<PLAYERS>>", players)
            .replace("<<MEMORY>>", hints)
        )
