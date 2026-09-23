#!/usr/bin/env python3
"""
Starts an interactive chat session with the Golf Swing Analyst agent.
Requires a dataset already produced by scripts/analyze_video.py.

Usage:
    python scripts/run_agent_chat.py
    python scripts/run_agent_chat.py --data-dir swing_data --debug
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import config as aconfig
from agent.chat import chat
from agent.engine import GolfSwingAgent


def parse_args():
    p = argparse.ArgumentParser(description="Chat with the Golf Swing Analyst agent.")
    p.add_argument("--data-dir", default=aconfig.DATA_DIR, help="Folder containing the master dataset CSV.")
    p.add_argument("--master-csv", default=aconfig.MASTER_CSV, help="Master dataset filename.")
    p.add_argument("--debug", action="store_true", help="Print every raw model reply.")
    return p.parse_args()


def main():
    args = parse_args()
    if args.debug:
        aconfig.DEBUG = True

    agent = GolfSwingAgent.build(data_dir=args.data_dir, master_csv=args.master_csv)
    chat(agent)


if __name__ == "__main__":
    main()
