"""LLM access. `GeminiLLM` uses the free Gemini API; `FakeLLM` is offline, for tests."""

from __future__ import annotations

import json
import re
import time
from typing import Callable, Protocol


class LLM(Protocol):
    def generate_json(self, system: str, prompt: str) -> dict: ...


def parse_json(text: str) -> dict:
    """Parse a JSON object, tolerating ```json fences around it."""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f"expected a JSON object, got {type(data).__name__}")
    return data


DEFAULT_MODEL = "gemini-3.8-flash"
DEFAULT_FALLBACKS = ("gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-3.1-flash-lite")

TRANSIENT_CODES = ("429", "500", "503", "504", "RESOURCE_EXHAUSTED", "UNAVAILABLE", "DEADLINE_EXCEEDED")
TRANSIENT_NETWORK = ("RemoteProtocolError", "ConnectError", "ReadError", "Timeout")  # matched in class names


def is_transient(exc: Exception) -> bool:
    msg, kind = str(exc), type(exc).__name__
    return any(c in msg for c in TRANSIENT_CODES) or any(n in kind for n in TRANSIENT_NETWORK)


def call_with_fallback(call: Callable[[str], object], models: list[str], attempts: int | None = None, sleep=time.sleep):
    """Call `call(model)`; on a transient error, retry with the next model in the list.

    Rotating models beats waiting on one: when a model is overloaded (503) the
    others usually still answer. A full pass through the list backs off before the next.
    """
    attempts = attempts or 3 * len(models)  # three full passes through the list
    for i in range(attempts):
        model = models[i % len(models)]
        try:
            return model, call(model)
        except Exception as exc:
            if not is_transient(exc) or i == attempts - 1:
                raise
            if (i + 1) % len(models) == 0:
                sleep(min(60, 10 * 2 ** (i // len(models))))  # 10 s, then 20 s between passes


class GeminiLLM:
    def __init__(self, api_key: str, model: str = DEFAULT_MODEL, fallbacks: list[str] | tuple[str, ...] = DEFAULT_FALLBACKS):
        if not api_key:
            raise ValueError("GEMINI_API_KEY is missing: create one at https://aistudio.google.com/apikey")
        from google import genai
        from google.genai import types

        # 60 s per request: a dropped connection fails fast and is retried instead of hanging.
        self._client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=60_000))
        self._types = types
        self.models = [model, *[m for m in fallbacks if m != model]]
        self.used: list[str] = []  # which model answered each call (shown in eval results)

    def _generate(self, prompt: str, system: str | None, json_mode: bool) -> str:
        cfg = self._types.GenerateContentConfig(
            system_instruction=system,
            response_mime_type="application/json" if json_mode else None,
            temperature=0,
        )
        model, res = call_with_fallback(
            lambda m: self._client.models.generate_content(model=m, contents=prompt, config=cfg), self.models
        )
        self.used.append(model)
        return res.text or ""

    def generate_json(self, system: str, prompt: str) -> dict:
        return parse_json(self._generate(prompt, system, json_mode=True) or "{}")

    def generate_text(self, prompt: str, json_mode: bool = False) -> str:
        return self._generate(prompt, None, json_mode)


class FakeLLM:
    """Returns canned answers chosen by the task name on the system prompt's first line."""

    def __init__(self, responses: dict[str, dict] | Callable[[str, str], dict]):
        self.responses = responses
        self.calls: list[str] = []

    def generate_json(self, system: str, prompt: str) -> dict:
        task = system.splitlines()[0].removeprefix("Task:").strip()
        self.calls.append(task)
        if callable(self.responses):
            return self.responses(system, prompt)
        return self.responses[task]
