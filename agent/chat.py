"""
Interactive terminal chat loop (B5 in the original notebook). Kept as a
thin wrapper around :meth:`agent.engine.GolfSwingAgent.ask` — all behavior
lives in the engine, this module is only the input/output shell.
"""

from __future__ import annotations

import time

from .engine import GolfSwingAgent


def chat(agent: GolfSwingAgent) -> None:
    print("Golf swing analyst ready — type your question (exit/quit to leave)")
    while True:
        try:
            q = input("\nyou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nSession ended")
            break
        if not q:
            continue
        if q.lower() in ("exit", "quit"):
            print("Session ended")
            break
        t0 = time.time()
        resp, meta = agent.ask(q, echo=False)
        print(f"agent ({time.time() - t0:.1f}s | attempts: {meta['attempts']}):\n{resp}")
