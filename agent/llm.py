"""
Local LLM backend (A1 in the original notebook). Loads a 4-bit-quantized
Qwen3-4B-Instruct with transformers/bitsandbytes and exposes a single
``generate(messages, system)`` call with a hard wall-clock timeout, so a
runaway generation can never hang the agent loop.
"""

from __future__ import annotations

import time

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    StoppingCriteria,
    StoppingCriteriaList,
)

from . import config


class TimeoutStoppingCriteria(StoppingCriteria):
    """Stops generation immediately once the time budget is spent, even if
    the model still wants to keep going. Checked after every new token —
    transformers' official mechanism for a custom stop condition."""

    def __init__(self, max_seconds: float):
        self.max_seconds = max_seconds
        self.start_time = time.time()
        self.timed_out = False

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        if time.time() - self.start_time > self.max_seconds:
            self.timed_out = True
            return True
        return False


class LocalLLM:
    """Wraps tokenizer + model loading and a single-turn ``generate`` call.

    The chat "contract" is intentionally narrow: one system message
    (always first) plus a run of user messages — never assistant turns —
    because the agent loop in :mod:`agent.engine` re-sends the full
    context as a sequence of user messages (tool results included) rather
    than a real multi-role conversation.
    """

    def __init__(self, model_id: str = config.MODEL_ID, max_new_tokens: int = config.MAX_NEW_TOKENS):
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.tokenizer = None
        self.model = None

    def load(self) -> "LocalLLM":
        gpu_available = torch.cuda.is_available()
        if gpu_available:
            props = torch.cuda.get_device_properties(0)
            print(f"GPU available: {props.name} | {props.total_memory / 1e9:.1f} GB VRAM")
        else:
            print("=" * 60)
            print("WARNING: no GPU detected — the model will run on CPU and every")
            print("reply will take minutes instead of seconds.")
            print("=" * 60)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

        print("Loading model...")
        t0 = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        self.model.eval()
        print(f"Model ready in {time.time() - t0:.0f}s")

        devices = {p.device.type for p in self.model.parameters()}
        print(f"Model is actually loaded on: {devices}")
        if gpu_available and "cuda" not in devices:
            print("ERROR: GPU is available but the model is on CPU — reload the cell/process.")
        return self

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    @staticmethod
    def _normalize(messages: list[dict], fallback_system: str) -> list[dict]:
        if messages and messages[0]["role"] == "system":
            sys_msg, rest = messages[0], messages[1:]
        else:
            sys_msg, rest = {"role": "system", "content": fallback_system}, messages
        for m in rest:
            if m["role"] != "user":
                raise ValueError(f"Disallowed role: '{m['role']}' — only system (first) + user are allowed.")
        return [{"role": "system", "content": sys_msg["content"]}] + [
            {"role": "user", "content": m["content"]} for m in rest
        ]

    @torch.no_grad()
    def generate(self, messages: list[dict], system: str | None = None) -> str:
        if system is None:
            system = "Answer with the final answer only, no extra text."
        chat = self._normalize(messages, system)
        text = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        stopper = TimeoutStoppingCriteria(config.MAX_RESPONSE_SECONDS)
        out = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            stopping_criteria=StoppingCriteriaList([stopper]),
        )
        return self.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
