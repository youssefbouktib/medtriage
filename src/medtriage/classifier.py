"""Specialty classification with a Hugging Face zero-shot model (no training needed)."""

from __future__ import annotations

from typing import Protocol

DEFAULT_MODEL = "MoritzLaurer/deberta-v3-base-zeroshot-v2.0"


class Classifier(Protocol):
    def classify(self, text: str, labels: list[str]) -> tuple[str, float]: ...


class HFZeroShotClassifier:
    """Natural-language-inference model that scores "This medical note is about <label>."."""

    def __init__(self, model: str = DEFAULT_MODEL, descriptions: dict[str, str] | None = None):
        from transformers import pipeline  # heavy import, only when actually used

        from .state import SPECIALTY_DESCRIPTIONS

        import torch

        # float32 explicitly: the checkpoint is stored in float16, and recent transformers load
        # it as is. float16 runs ~50x slower on a CPU (18 s instead of ~0.3 s per comparison).
        self._pipe = pipeline("zero-shot-classification", model=model, dtype=torch.float32)
        self.descriptions = SPECIALTY_DESCRIPTIONS if descriptions is None else descriptions

    def classify(self, text: str, labels: list[str]) -> tuple[str, float]:
        # The model scores a plain-language description of each label; map the winner back.
        hypotheses = {self.descriptions.get(label, label): label for label in labels}
        res = self._pipe(
            text[:2000],  # the model reads ~512 tokens; the start of a note is the most informative
            candidate_labels=list(hypotheses),
            hypothesis_template="This medical note is about {}.",
            batch_size=len(hypotheses),  # score all labels in one forward pass
        )
        return hypotheses[res["labels"][0]], float(res["scores"][0])


class FakeClassifier:
    def __init__(self, label: str = "general medicine", score: float = 0.9):
        self.label, self.score = label, score

    def classify(self, text: str, labels: list[str]) -> tuple[str, float]:
        return self.label, self.score
