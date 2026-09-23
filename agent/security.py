"""
Security layer (B3 in the original notebook), split from tool execution on
purpose: this module only ever answers "is this allowed?" — it never reads
the dataset or computes anything. Three independent layers:

  L1 security_check()   — banned-phrase check on the raw question, before
                           any model call at all.
  L2 parse_tool_call()   — strict whitelist + AST validation of whatever
                           tool call the model tries to make.
  L3 numbers_guard()     — every number in the model's final answer must be
                           traceable to an actual tool result.
"""

from __future__ import annotations

import ast
import json
import re

from . import config


# ---------------------------------------------------------------------------
# L1 — banned request patterns, checked before any model call
# ---------------------------------------------------------------------------
def security_check(question: str) -> tuple[bool, str | None]:
    for pat in config.BANNED_PATTERNS:
        if re.search(pat, question, flags=re.IGNORECASE):
            return False, "REFUSE: This violates security rules."
    return True, None


# ---------------------------------------------------------------------------
# L2 — expression sandbox (deny-by-default AST walk) + tool-call parsing
# ---------------------------------------------------------------------------
_ALLOWED_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div)
_ALLOWED_UNARY = (ast.USub, ast.UAdd)

# Function names allowed inside a query_stats expression. Kept here (rather
# than in tools.py) because validation must agree exactly with execution —
# see agent.tools.SAFE_FUNCS for the matching implementations.
SAFE_FUNC_NAMES = {
    "mean", "std", "median", "min", "max", "sum", "count",
    "first", "last", "range", "slope", "maxjump", "maxdrop", "values",
}


def validate_expr(expr: str, numeric_columns: list[str]) -> str | None:
    """Walks the expression's AST node by node — anything not explicitly
    whitelisted is rejected (deny-by-default). Returns an error string, or
    ``None`` if the expression is safe."""
    if not isinstance(expr, str) or not (1 <= len(expr) <= config.EXPR_MAX_CHARS):
        return f"expr must be 1-{config.EXPR_MAX_CHARS} chars"
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return "invalid expression syntax"
    if sum(1 for _ in ast.walk(tree)) > config.EXPR_MAX_NODES:
        return "expression too complex"

    used_cols: list[str] = []

    def _walk(node):
        if isinstance(node, ast.Expression):
            return _walk(node.body)
        if isinstance(node, ast.BinOp) and isinstance(node.op, _ALLOWED_BINOPS):
            return _walk(node.left), _walk(node.right)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, _ALLOWED_UNARY):
            return _walk(node.operand)
        if isinstance(node, ast.Call):
            if (
                node.keywords
                or len(node.args) != 1
                or not isinstance(node.func, ast.Name)
                or node.func.id not in SAFE_FUNC_NAMES
            ):
                return "only whitelisted function(col) calls allowed"
            return _walk(node.args[0])
        if isinstance(node, ast.Name):
            if node.id in SAFE_FUNC_NAMES:
                return "function name used outside a call"
            if node.id not in numeric_columns:
                return f"unknown column '{node.id}'"
            used_cols.append(node.id)
            return None
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return None
        return f"expression element not allowed: {type(node).__name__}"

    def _flatten(r):
        if isinstance(r, tuple):
            for x in r:
                _flatten(x)
        elif isinstance(r, str) and r:
            raise ValueError(r)

    try:
        _flatten(_walk(tree))
    except ValueError as e:
        return str(e)
    if not used_cols:
        return "expr must reference at least one column"
    return None


def _extract_json_objects(text: str) -> list[str]:
    """Extracts balanced-brace JSON object substrings from free text."""
    objs, depth, start, in_str, esc = [], 0, None, False, False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    objs.append(text[start : i + 1])
                    start = None
    return objs


def _try_parse_obj(cand: str):
    fixed = cand.replace("\u201c", '"').replace("\u201d", '"').replace("\u2018", "'").replace("\u2019", "'")
    for c in (cand, fixed):
        try:
            return json.loads(c)
        except Exception:
            pass
        try:
            return ast.literal_eval(c)
        except Exception:
            pass
    return None


def parse_tool_call(text: str, numeric_columns: list[str], players: list[str]):
    """Returns ``(tool, args, None)`` for the first valid tool call found in
    ``text``, or ``(None, None, reason)`` explaining why nothing valid was
    found. Never trusts the model's JSON blindly — every argument is
    checked against the real dataset (known columns, known players)."""
    first_err = "no valid tool JSON"
    for cand in _extract_json_objects(text):
        obj = _try_parse_obj(cand)
        if not (isinstance(obj, dict) and "tool" in obj):
            first_err = "JSON object without a 'tool' key"
            continue
        tool, args = obj["tool"], obj.get("args", {})
        if not isinstance(args, dict):
            return None, None, "args must be a JSON object"
        cols = args.get("columns") or []
        bad = [c for c in cols if c not in numeric_columns]
        p = args.get("player")

        if tool == "query_stats":
            err = validate_expr(args.get("expr"), numeric_columns)
            if err:
                first_err = f"bad expr: {err}. Example: mean(top_shoulder_rotation_deg)"
                continue
            if p and p not in players:
                first_err = f"unknown player '{p}'. Available: {players}"
                continue
            return tool, args, None

        if tool == "compare":
            pls = args.get("players", [])
            bad_p = [x for x in pls if x not in players]
            if bad_p:
                first_err = f"unknown players {bad_p}. Available: {players}"
                continue
            if not (1 < len(pls) <= 4):
                first_err = "compare needs 2-4 valid player names"
                continue
            if bad:
                first_err = f"unknown columns {bad}. Valid: {numeric_columns[:6]}..."
                continue
            return tool, args, None

        if tool == "get_table":
            if bad:
                first_err = f"unknown columns {bad}. Valid: {numeric_columns[:6]}..."
                continue
            if p and p not in players:
                first_err = f"unknown player '{p}'. Available: {players}"
                continue
            return tool, args, None

        if tool == "get_catalog":
            return tool, args, None

        first_err = f"unknown tool '{tool}'. Allowed: {sorted(config.ALLOWED_TOOLS)}"

    if first_err.startswith("no valid"):
        first_err = f"{first_err} in: {text[:120]!r}"
    return None, None, first_err


# ---------------------------------------------------------------------------
# L3 — numbers guard on the final answer
# ---------------------------------------------------------------------------
def numbers_guard(text: str, allowed_numbers: set[float], players: list[str]) -> tuple[bool, float | None]:
    """Every number in the reply must trace back to a real tool result —
    player names and swing_uid tokens (``name#3``) are stripped first so
    they're never mistaken for invented numbers."""
    cleaned = re.sub(r"\w+#\d+", "", text)
    for pl in players:
        cleaned = cleaned.replace(pl, "")
    cleaned = re.sub(r"^\s*\d+[.)]\s", "", cleaned, flags=re.M)
    for x in re.findall(r"-?\d+(?:\.\d+)?", cleaned):
        n = float(x)
        if not any(abs(n - a) <= 0.02 * max(1.0, abs(a)) for a in allowed_numbers):
            return False, n
    return True, None
