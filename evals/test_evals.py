"""Evals with the real models on the golden set (synthetic notes with expert labels).

Run: pytest evals -m eval -s      (needs GEMINI_API_KEY; downloads the HF model once)
     EVAL_REUSE_OUTPUTS=1 pytest evals -m eval -s   re-scores evals/results/outputs.json
                                                    without re-running the agent (saves quota)

Metrics:
- urgency accuracy: share of notes triaged at the labelled level
- under-triage of emergencies: emergent notes triaged lower; this must be 0
- specialty accuracy: zero-shot classifier vs label (reported, softer threshold)
- summary faithfulness: DeepEval G-Eval; the judge is a different Gemini model than the app's
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True))

pytestmark = [
    pytest.mark.eval,
    pytest.mark.skipif(not os.getenv("GEMINI_API_KEY"), reason="GEMINI_API_KEY not set"),
]

HERE = Path(__file__).parent
RESULTS = HERE / "results"
CASES = [json.loads(line) for line in (HERE / "golden.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]


@pytest.fixture(scope="module")
def outputs() -> list[dict]:
    saved = RESULTS / "outputs.json"
    if os.getenv("EVAL_REUSE_OUTPUTS") == "1" and saved.exists():
        rows = {r["id"]: r for r in json.loads(saved.read_text(encoding="utf-8"))}
        return [rows[c["id"]] for c in CASES]

    from medtriage.graph import build_default_graph

    graph = build_default_graph()
    outs = [graph.invoke({"note": c["note"]}) for c in CASES]
    RESULTS.mkdir(exist_ok=True)
    rows = [{"id": c["id"], "expected": c["urgency"], **{k: v for k, v in o.items() if k != "note"}} for c, o in zip(CASES, outs)]
    (RESULTS / "outputs.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return outs


def test_urgency_accuracy(outputs):
    correct = sum(o["urgency"] == c["urgency"] for c, o in zip(CASES, outputs))
    accuracy = correct / len(CASES)
    print(f"\nurgency accuracy: {correct}/{len(CASES)} = {accuracy:.0%}")
    assert accuracy >= 0.8


def test_no_emergency_is_under_triaged(outputs):
    missed = [c["id"] for c, o in zip(CASES, outputs) if c["urgency"] == "emergent" and o["urgency"] != "emergent"]
    assert missed == [], f"emergent notes triaged lower: {missed}"


def test_specialty_accuracy(outputs):
    correct = sum(o["specialty"] == c["specialty"] for c, o in zip(CASES, outputs))
    print(f"\nspecialty accuracy: {correct}/{len(CASES)} = {correct / len(CASES):.0%}")
    assert correct / len(CASES) >= 0.6


def test_summary_faithfulness(outputs):
    from deepeval.metrics import GEval
    from deepeval.test_case import LLMTestCase, LLMTestCaseParams
    from judge import FallbackGeminiJudge

    from medtriage.llm import GeminiLLM

    models = os.getenv("JUDGE_MODELS", "gemini-3.7-flash,gemini-3.5-flash,gemini-3.5-flash-lite").split(",")
    judge = FallbackGeminiJudge(GeminiLLM(os.environ["GEMINI_API_KEY"], models[0].strip(), [m.strip() for m in models[1:]]))
    metric = GEval(
        name="Faithfulness",
        criteria=(
            "The actual output summarizes the clinical note given as input, and states the triage level "
            "given with it. Penalize any fact (symptom, vital sign, medication, history) that is not in the "
            "note, and omission of the chief complaint."
        ),
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        model=judge,
        threshold=0.7,
        async_mode=False,
    )
    scores = {}
    for c, o in zip(CASES, outputs):
        # The judge sees exactly what the summarizer saw: the note plus the assigned triage level.
        source = f"{c['note']}\n\nTriage level assigned: {o['urgency']}"
        metric.measure(LLMTestCase(input=source, actual_output=o["summary"]))
        scores[c["id"]] = round(metric.score, 2)
    mean = sum(scores.values()) / len(scores)
    print(f"\nsummary faithfulness (G-Eval): mean {mean:.2f} {scores}")
    (RESULTS / "faithfulness.json").write_text(json.dumps(scores, indent=2), encoding="utf-8")
    assert mean >= 0.7
