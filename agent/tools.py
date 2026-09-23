"""
Tool execution (the second half of B3 in the original notebook). Once
:mod:`agent.security` has approved a tool call, this module is what
actually touches the dataframe.

Calculations (``query_stats`` / ``compare``) always run on the raw,
untouched column values. ``get_table`` is the one place a standardized
display sign (:data:`agent.config.INVERTED_SIGN_COLS`) is applied, because
a table's column headers give the reader the context to interpret it
correctly — a bare computed number would not.
"""

from __future__ import annotations

import ast
import json
import re

import numpy as np
import pandas as pd

from . import config

SAFE_FUNCS = {
    "mean": np.mean,
    "std": np.std,
    "median": np.median,
    "min": np.min,
    "max": np.max,
    "sum": np.sum,
    "count": lambda a: float(len(a)),
    "first": lambda a: float(a[0]),
    "last": lambda a: float(a[-1]),
    "range": lambda a: float(np.max(a) - np.min(a)),
    "slope": lambda a: (float(np.polyfit(np.arange(len(a)), a, 1)[0]) if len(a) > 1 else 0.0),
    "maxjump": lambda a: (float(np.max(np.diff(a))) if len(a) > 1 else 0.0),
    "maxdrop": lambda a: (float(np.min(np.diff(a))) if len(a) > 1 else 0.0),
    "values": lambda a: float(np.mean(a)),  # placeholder — the "values(col)" case is special-cased below
}


def _signed(col: str, v: float) -> float:
    """Display-only sign flip — never used by query_stats/compare."""
    return -float(v) if col in config.INVERTED_SIGN_COLS else float(v)


def _eval_expr(expr: str, group_df: pd.DataFrame) -> float:
    """Isolated AST evaluator — Names resolve to raw CSV column values only."""

    def col_values(name):
        return group_df[name].dropna().to_numpy(dtype=float)  # raw, no _signed

    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.BinOp):
            l, r = _eval(node.left), _eval(node.right)
            if isinstance(node.op, ast.Add):
                return l + r
            if isinstance(node.op, ast.Sub):
                return l - r
            if isinstance(node.op, ast.Mult):
                return l * r
            if isinstance(node.op, ast.Div):
                if abs(r) < 1e-12:
                    raise ZeroDivisionError("division by zero in expr")
                return l / r
        if isinstance(node, ast.UnaryOp):
            v = _eval(node.operand)
            return -v if isinstance(node.op, ast.USub) else +v
        if isinstance(node, ast.Call):
            return SAFE_FUNCS[node.func.id](col_values(node.args[0].id))
        if isinstance(node, ast.Name):
            return col_values(node.id)
        if isinstance(node, ast.Constant):
            return float(node.value)
        raise ValueError("disallowed node")

    val = float(_eval(ast.parse(expr, mode="eval")))
    if not np.isfinite(val):
        raise ValueError("expression produced non-finite value")
    return round(val, 2)


class ToolExecutor:
    """Executes an already-validated tool call against a
    :class:`~agent.data_loader.SwingDataStore`."""

    def __init__(self, data_store):
        self.data_store = data_store

    def execute(self, tool: str, args: dict) -> tuple[dict, set[float]]:
        df = self.data_store.df
        players = self.data_store.players
        numeric_cols = self.data_store.numeric_columns

        if tool == "get_catalog":
            res = {
                "players": {p: int((df["player"] == p).sum()) for p in players},
                "numeric_columns": numeric_cols,
            }

        elif tool == "get_table":
            p = args.get("player")
            cols = [c for c in args.get("columns", []) if c in numeric_cols][:5] or config.DEFAULT_COLS
            base = ["swing_uid", "swing_id"] if "swing_uid" in df.columns else ["swing_id"]
            if p:
                sub = df[df["player"] == p].sort_values("swing_id")
                rows = sub[base + cols].head(20).to_dict("records")
                for r in rows:
                    for c in cols:
                        if isinstance(r.get(c), (int, float)):
                            r[c] = round(_signed(c, r[c]), 2)
                res = {"player": p, "rows": rows}
            else:
                sub = df.sort_values(["player", "swing_id"])
                rows = sub[["player"] + base + cols].head(30).to_dict("records")
                for r in rows:
                    for c in cols:
                        if isinstance(r.get(c), (int, float)):
                            r[c] = round(_signed(c, r[c]), 2)
                res = {"rows": rows}

        elif tool == "compare":
            pls = [x for x in args["players"] if x in players]
            cols = [c for c in args.get("columns", []) if c in numeric_cols][:4] or config.DEFAULT_COLS
            res = {pl: {c: round(float(df[df["player"] == pl][c].mean()), 2) for c in cols} for pl in pls}

        elif tool == "query_stats":
            expr = args["expr"]
            p = args.get("player")
            targets = (
                [(p, df[df["player"] == p].sort_values("swing_id"))]
                if p
                else [(pl, df[df["player"] == pl].sort_values("swing_id")) for pl in players]
            )
            mvals = re.fullmatch(r"\s*values\(\s*(\w+)\s*\)\s*", expr)
            res = {}
            for pl, sub in targets:
                if sub.empty:
                    if p:
                        return {"error": f"no data for player '{p}'"}, set()
                    continue
                try:
                    if mvals:
                        col = mvals.group(1)
                        if col not in numeric_cols:
                            raise ValueError(f"unknown column '{col}'")
                        pairs = sub[["swing_id", col]].dropna()
                        res[pl] = {
                            "swing_ids": [int(x) for x in pairs["swing_id"]],
                            col: [round(float(v), 2) for v in pairs[col]],  # raw
                        }
                    else:
                        res[pl] = _eval_expr(expr, sub)
                except Exception as e:
                    if p:
                        return {"error": str(e)}, set()
            if not res:
                return {"error": "expression failed for all players"}, set()
        else:
            return {"error": "unhandled tool"}, set()

        nums = {float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", json.dumps(res, ensure_ascii=False))}
        return res, nums
