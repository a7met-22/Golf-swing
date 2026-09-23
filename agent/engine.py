"""
The agent loop itself (B4 in the original notebook): security check ->
model call -> either a tool call, an invalid tool call, a refusal, or a
final answer -> repeat until a final answer is produced or the attempt
budget runs out. On timeout, a fallback summary is built from whatever
real tool results were already collected, so the user never gets stuck
with nothing.
"""

from __future__ import annotations

import json

from . import config
from .data_loader import SwingDataStore
from .llm import LocalLLM
from .memory import AgentMemory
from .security import numbers_guard, parse_tool_call, security_check
from .tools import ToolExecutor


class GolfSwingAgent:
    """Ties the model, dataset, tool executor, memory, and safety layers
    together behind one method: :meth:`ask`."""

    def __init__(self, llm: LocalLLM, data_store: SwingDataStore, memory: AgentMemory | None = None):
        self.llm = llm
        self.data_store = data_store
        self.memory = memory or AgentMemory()
        self.tools = ToolExecutor(data_store)

    # -- convenience constructor ------------------------------------------------
    @classmethod
    def build(cls, data_dir: str = config.DATA_DIR, master_csv: str = config.MASTER_CSV) -> "GolfSwingAgent":
        data_store = SwingDataStore(data_dir, master_csv).load()
        llm = LocalLLM().load()
        memory = AgentMemory(data_dir)
        return cls(llm, data_store, memory)

    # -- main entry point --------------------------------------------------------
    def ask(self, question: str, echo: bool = True) -> tuple[str, dict]:
        clean, reason = security_check(question)
        if not clean:
            if echo:
                print(f"[security] {reason}")
            return reason, {"used_tool": False, "attempts": 0, "refused": True}

        sys_prompt = self.data_store.build_system_prompt(self.memory.hint())
        contextualized = self.memory.with_context(question, self.data_store.players)
        messages = [{"role": "user", "content": contextualized}]

        allowed_numbers: set[float] = set()
        used_tool = False
        no_progress, last_err = 0, "unknown"
        refusal_bounces, tool_calls = 0, 0
        repeat_offenses, force_failures = 0, 0
        force_final = False
        done_calls: set[str] = set()
        collected_results: list[dict] = []

        def fallback_summary():
            if not collected_results:
                return None
            lines = ["Automatic summary from the system — real results from the file:"]
            for r in collected_results[-4:]:
                lines.append(f"- {json.dumps(r, ensure_ascii=False)}")
            return "\n".join(lines)

        for attempt in range(1, config.MAX_ATTEMPTS + 1):
            try:
                resp = self.llm.generate(messages, system=sys_prompt)
            except Exception as e:
                last_err = f"model error: {e}"
                no_progress += 1
                if no_progress >= config.NO_PROGRESS_LIMIT:
                    break
                continue

            if config.DEBUG:
                print(f"   [{attempt}] model: {resp[:160]}")

            # 1) model refused -> guard against over-refusal on in-scope questions
            if resp.strip().startswith("REFUSE"):
                if config.IN_SCOPE_RX.search(question) and refusal_bounces < 2:
                    refusal_bounces += 1
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "CORRECTION: this request is about the analyzed swing data — it IS in "
                                "scope. Do NOT refuse. Follow the DECISION LADDER: send a tool call now, "
                                "or get_catalog then one clarifying question."
                            ),
                        }
                    )
                    last_err = "over-refusal corrected by policy audit"
                    no_progress += 1
                    if no_progress >= config.NO_PROGRESS_LIMIT:
                        break
                    continue
                if echo:
                    print(f"[agent] {resp.strip()}")
                return resp.strip(), {"used_tool": used_tool, "attempts": attempt, "refused": True}

            # 2) a tool call
            tool, args, err = parse_tool_call(resp, self.data_store.numeric_columns, self.data_store.players)
            if tool:
                sig = json.dumps({"t": tool, "a": args}, sort_keys=True, ensure_ascii=False)
                if sig in done_calls:
                    repeat_offenses += 1
                    if repeat_offenses == 1:
                        messages.append(
                            {
                                "role": "user",
                                "content": (
                                    "ERROR: you already ran this exact call — its [TOOL_RESULT] is "
                                    "above. Summarize the results, or call a DIFFERENT tool."
                                ),
                            }
                        )
                        last_err = "repeated identical tool call"
                        no_progress += 1
                    else:
                        force_final = True
                        messages.append(
                            {
                                "role": "user",
                                "content": (
                                    "STOP: repeated call again. Write your FINAL ANSWER now from the "
                                    "[TOOL_RESULT]s above. No more tool calls."
                                ),
                            }
                        )
                        last_err = "forced final after repeats"
                        no_progress += 1
                elif tool_calls >= config.MAX_TOOL_CALLS or force_final:
                    messages.append(
                        {
                            "role": "user",
                            "content": "STOP: no more tool calls. Write your FINAL ANSWER now summarizing the [TOOL_RESULT]s above.",
                        }
                    )
                    last_err = "forced final answer"
                    no_progress += 1
                else:
                    result, nums = self.tools.execute(tool, args)
                    if isinstance(result, dict) and "error" in result:
                        messages.append(
                            {
                                "role": "user",
                                "content": f"ERROR: {result['error']}. Fix the call and resend, or get_catalog then clarify.",
                            }
                        )
                        last_err = f"tool error: {result['error']}"
                        no_progress += 1
                    else:
                        tool_calls += 1
                        done_calls.add(sig)
                        collected_results.append(result)
                        if len(collected_results) > 6:
                            collected_results.pop(0)
                        used_tool = True
                        grew = len(allowed_numbers | nums) > len(allowed_numbers)
                        allowed_numbers |= nums
                        messages.append({"role": "user", "content": f"[TOOL_RESULT] {json.dumps(result, ensure_ascii=False)}"})
                        last_err = "tool executed, waiting for final answer"
                        no_progress = 0 if grew else no_progress + 1

            # 3) malformed tool-call shape -> feed the error back to the model
            elif resp.lstrip().startswith("{") or '"tool"' in resp:
                if force_final:
                    force_failures += 1
                    if force_failures >= 2:
                        last_err = "model kept calling tools after forced summary"
                        no_progress = config.NO_PROGRESS_LIMIT
                        continue
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"ERROR: invalid tool call ({err}). Schema: "
                            f'{{"tool":"query_stats","args":{{"expr":"mean(COL)","player":"NAME"}}}} '
                            f"— or write a final answer without numbers."
                        ),
                    }
                )
                last_err = f"invalid tool call: {err}"
                no_progress += 1

            # 4) final text answer
            else:
                final, mem_pair = self.memory.extract_trailer(resp)
                ok, bad = numbers_guard(final, allowed_numbers, self.data_store.players)
                if not ok:
                    messages.append(
                        {
                            "role": "user",
                            "content": f"ERROR: number {bad} is not from any [TOOL_RESULT]. State only real tool values, or call a tool first.",
                        }
                    )
                    last_err = f"invented number {bad}"
                    no_progress += 1
                else:
                    self.memory.remember_turn(question, final, config.CLARIFY_RX, mem_pair)
                    if echo:
                        print(f"[agent] {final}")
                    return final, {"used_tool": used_tool, "attempts": attempt, "refused": False}

            if no_progress >= config.NO_PROGRESS_LIMIT:
                break

        # Safe exit: if we collected real results, summarize them instead of just failing
        fb = fallback_summary()
        if fb:
            self.memory.remember_turn(question, fb, config.CLARIFY_RX, None)
            if echo:
                print(f"[agent] {fb}")
            return fb, {"used_tool": used_tool, "attempts": attempt, "refused": False, "fallback": True}

        fail = f"Sorry, I couldn't complete the request: {last_err} (after {attempt} attempts)"
        if echo:
            print(fail)
        return fail, {"used_tool": used_tool, "attempts": attempt, "refused": True}
