"""The triage workflow as a LangGraph state graph.

    START ─┬─► classify (Hugging Face) ─┐
           └─► extract  (LLM) ──────────┴─► red_flags (rules) ─┬─► escalate (rules) ─────┐
                                                               └─► assess_urgency (LLM) ─┴─► summarize (LLM) ─► END

classify and extract run in parallel. If the deterministic rules find a red flag,
the note is escalated to "emergent" and the LLM is never asked to judge urgency:
the model can't talk a red flag down.
"""

from __future__ import annotations

import json

from langgraph.graph import END, START, StateGraph

from . import prompts
from .classifier import Classifier
from .llm import LLM
from .red_flags import find_red_flags
from .state import SPECIALTIES, URGENCY_LEVELS, Extraction, TriageState


def normalize_extraction(data: dict) -> Extraction:
    def as_list(v) -> list[str]:
        return [str(x) for x in v] if isinstance(v, list) else ([str(v)] if v else [])

    vitals = data.get("vitals")
    return {
        "chief_complaint": str(data.get("chief_complaint") or ""),
        "symptoms": as_list(data.get("symptoms")),
        "vitals": {str(k): str(v) for k, v in vitals.items()} if isinstance(vitals, dict) else {},
        "medications": as_list(data.get("medications")),
        "allergies": as_list(data.get("allergies")),
    }


def build_graph(llm: LLM, classifier: Classifier, specialties: list[str] = SPECIALTIES):
    def classify(state: TriageState) -> TriageState:
        label, score = classifier.classify(state["note"], specialties)
        return {"specialty": label, "specialty_score": round(score, 3)}

    def extract(state: TriageState) -> TriageState:
        data = llm.generate_json(prompts.EXTRACT_SYSTEM, prompts.NOTE_PROMPT.format(note=state["note"]))
        return {"extraction": normalize_extraction(data)}

    def red_flags(state: TriageState) -> TriageState:
        return {"red_flags": find_red_flags(state["note"])}

    def route(state: TriageState) -> str:
        return "escalate" if state["red_flags"] else "assess_urgency"

    def escalate(state: TriageState) -> TriageState:
        return {
            "urgency": "emergent",
            "urgency_reason": "Red flags: " + ", ".join(state["red_flags"]),
            "urgency_source": "rules",
        }

    def assess_urgency(state: TriageState) -> TriageState:
        prompt = prompts.URGENCY_PROMPT.format(
            note=state["note"], extraction=json.dumps(state.get("extraction", {}), indent=1)
        )
        data = llm.generate_json(prompts.URGENCY_SYSTEM, prompt)
        urgency = str(data.get("urgency", "")).lower().strip()
        reason = str(data.get("reason", ""))
        if urgency not in URGENCY_LEVELS:  # fail safe: an unreadable answer is never "routine"
            urgency, reason = "urgent", f"Model returned an invalid level ({urgency!r}); defaulted to urgent."
        return {"urgency": urgency, "urgency_reason": reason, "urgency_source": "llm"}

    def summarize(state: TriageState) -> TriageState:
        prompt = prompts.SUMMARY_PROMPT.format(
            note=state["note"], urgency=state["urgency"], reason=state["urgency_reason"]
        )
        data = llm.generate_json(prompts.SUMMARY_SYSTEM, prompt)
        return {"summary": str(data.get("summary", ""))}

    g = StateGraph(TriageState)
    for name, fn in [
        ("classify", classify),
        ("extract", extract),
        ("red_flags", red_flags),
        ("escalate", escalate),
        ("assess_urgency", assess_urgency),
        ("summarize", summarize),
    ]:
        g.add_node(name, fn)

    g.add_edge(START, "classify")
    g.add_edge(START, "extract")
    g.add_edge(["classify", "extract"], "red_flags")  # wait for both branches
    g.add_conditional_edges("red_flags", route, ["escalate", "assess_urgency"])
    g.add_edge("escalate", "summarize")
    g.add_edge("assess_urgency", "summarize")
    g.add_edge("summarize", END)
    return g.compile()


def gemini_from_env():
    """GeminiLLM configured from .env (GEMINI_API_KEY, CHAT_MODEL, FALLBACK_MODELS)."""
    import os

    from dotenv import find_dotenv, load_dotenv

    from .llm import DEFAULT_FALLBACKS, DEFAULT_MODEL, GeminiLLM

    load_dotenv(find_dotenv(usecwd=True))
    fallbacks = os.getenv("FALLBACK_MODELS")
    return GeminiLLM(
        os.getenv("GEMINI_API_KEY", ""),
        os.getenv("CHAT_MODEL", DEFAULT_MODEL),
        [m.strip() for m in fallbacks.split(",") if m.strip()] if fallbacks else DEFAULT_FALLBACKS,
    )


def build_default_graph():
    """Real models: Gemini (with fallbacks) + the Hugging Face zero-shot classifier."""
    import os

    from .classifier import DEFAULT_MODEL, HFZeroShotClassifier

    llm = gemini_from_env()
    clf = HFZeroShotClassifier(os.getenv("HF_ZERO_SHOT_MODEL", DEFAULT_MODEL))
    return build_graph(llm, clf)
