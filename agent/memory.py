"""
Two kinds of memory, kept deliberately separate (this split fixed a bug in
an earlier version where a restart could resurface a poisoned session):

- ``player_summaries`` — durable, written to ``agent_memory.json`` on disk.
  Short, model-authored notes about a player, carried into future prompts
  as a hint (the model is still required to verify every number via a tool
  call — memory is context, never a source of truth).
- session state (last question/answer, "am I awaiting a clarification?") —
  in-memory only, intentionally reset on every process restart.
"""

from __future__ import annotations

import json
import os
import re

from . import config


class AgentMemory:
    def __init__(self, data_dir: str = config.DATA_DIR, filename: str = config.MEMORY_FILENAME):
        self.path = os.path.join(data_dir, filename)
        self.player_summaries: dict[str, str] = {}
        self.last_player: str | None = None
        # session-only state
        self.last_question: str | None = None
        self.last_answer: str | None = None
        self.awaiting_clarify: bool = False

        self._load()

    def _load(self):
        try:
            with open(self.path, encoding="utf-8") as f:
                disk = json.load(f)
            if isinstance(disk.get("player_summaries"), dict):
                self.player_summaries = disk["player_summaries"]
        except Exception:
            pass

    def save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump({"player_summaries": self.player_summaries}, f, ensure_ascii=False, indent=1)

    def hint(self) -> str:
        lines = [f"- {p}: {s}" for p, s in list(self.player_summaries.items())[-6:]]
        if self.last_player:
            lines.append(f"- last discussed player: {self.last_player}")
        return "\n".join(lines)

    def with_context(self, question: str, players: list[str]) -> str:
        """Wraps a short follow-up message with the previous turn's context
        so the model can resolve pronouns / bare answers like a player name
        typed on its own in reply to a clarifying question."""
        q = question.strip()
        prev_q, prev_a = self.last_question, self.last_answer
        if not prev_q:
            return question
        if self.awaiting_clarify:
            return (
                f'[Context: your previous answer asked the user for more information. '
                f'The user\'s original request: "{prev_q}". '
                f'Your request for information was: "{prev_a}". '
                f'The user\'s new message is the reply to it: "{q}". '
                f"Combine them and fulfill the original request now. If the reply does "
                f"not answer your question, treat it as a new request and fulfill it.]"
            )
        if q in players or len(q.split()) <= config.CONTEXT_MAX_WORDS:
            return (
                f'[Context: the user is continuing the previous exchange. '
                f'Previous user request: "{prev_q}". '
                f'The user\'s new message is the answer/continuation: "{q}"]'
            )
        return question

    @staticmethod
    def extract_trailer(resp: str) -> tuple[str, tuple[str, str] | None]:
        """Splits the model's trailing ``@@MEM player=<name> | <note>`` line
        (if present) off the visible answer, returning ``(answer, mem_pair)``."""
        final = resp.split("@@MEM")[0].strip()
        m = re.search(r"@@MEM\s+player=(.+?)\s*\|\s*(.+)", resp)
        return final, ((m.group(1).strip(), m.group(2).strip()) if m else None)

    def remember_turn(self, question: str, answer: str, clarify_rx: re.Pattern, mem_pair):
        if mem_pair:
            self.player_summaries[mem_pair[0]] = mem_pair[1]
            self.last_player = mem_pair[0]
        self.last_question = question
        self.last_answer = answer
        self.awaiting_clarify = bool(clarify_rx.search(answer))
        self.save()
