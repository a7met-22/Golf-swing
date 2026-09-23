#!/usr/bin/env python3
"""
Runs the 13-scenario compliance suite against the live model: scope
detection, tool use, refusals, and clarification requests (Arabic +
English). This is a behavioral smoke test that needs the model loaded —
for fast, GPU-free checks of the deterministic logic, see tests/ instead.

Usage:
    python scripts/run_agent_tests.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.engine import GolfSwingAgent
from agent.tests import run_compliance_suite


def main():
    agent = GolfSwingAgent.build()
    run_compliance_suite(agent)


if __name__ == "__main__":
    main()
