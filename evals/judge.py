"""DeepEval judge backed by GeminiLLM, so the judge gets the same model fallback as the app.

DeepEval's built-in GeminiModel is pinned to one model; on the free tier that model
hits its daily quota (20 requests) quickly, which fails the whole eval run.
"""

from __future__ import annotations

from deepeval.models import DeepEvalBaseLLM

from medtriage.llm import GeminiLLM, parse_json


class FallbackGeminiJudge(DeepEvalBaseLLM):
    def __init__(self, llm: GeminiLLM):
        self.llm = llm
        super().__init__(model=llm.models[0])

    def load_model(self):
        return self.llm

    def generate(self, prompt: str, schema=None):
        if schema is None:
            return self.llm.generate_text(prompt)
        # DeepEval passes a pydantic model when it wants structured output (scores, steps).
        return schema(**parse_json(self.llm.generate_text(prompt, json_mode=True)))

    async def a_generate(self, prompt: str, schema=None):
        return self.generate(prompt, schema=schema)

    def get_model_name(self) -> str:
        return "Gemini with fallback: " + " > ".join(self.llm.models)
