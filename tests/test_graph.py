"""Graph tests with a fake LLM and classifier: offline, fast, run on every push."""

from medtriage.classifier import FakeClassifier
from medtriage.graph import build_graph
from medtriage.llm import FakeLLM, parse_json

EXTRACTION = {
    "chief_complaint": "knee pain",
    "symptoms": ["knee pain"],
    "vitals": {"BP": "128/80"},
    "medications": "ibuprofen",  # a string instead of a list: must be normalized
    "allergies": [],
}


def make_llm(urgency: dict) -> FakeLLM:
    return FakeLLM({"extract": EXTRACTION, "urgency": urgency, "summary": {"summary": "Short summary."}})


def test_routine_note_goes_through_llm_urgency():
    llm = make_llm({"urgency": "routine", "reason": "Chronic, stable."})
    graph = build_graph(llm, FakeClassifier("orthopedics", 0.81))

    out = graph.invoke({"note": "Chronic knee pain for 2 years. BP 128/80."})

    assert out["specialty"] == "orthopedics"
    assert out["red_flags"] == []
    assert (out["urgency"], out["urgency_source"]) == ("routine", "llm")
    assert out["extraction"]["medications"] == ["ibuprofen"]
    assert out["summary"] == "Short summary."
    assert sorted(llm.calls) == ["extract", "summary", "urgency"]


def test_red_flag_escalates_without_asking_the_llm():
    llm = make_llm({"urgency": "routine", "reason": "should never be used"})
    graph = build_graph(llm, FakeClassifier("cardiology"))

    out = graph.invoke({"note": "Crushing chest pain for 40 minutes."})

    assert (out["urgency"], out["urgency_source"]) == ("emergent", "rules")
    assert "chest pain" in out["urgency_reason"]
    assert "urgency" not in llm.calls


def test_invalid_llm_urgency_fails_safe_to_urgent():
    llm = make_llm({"urgency": "maybe later", "reason": "?"})
    out = build_graph(llm, FakeClassifier()).invoke({"note": "Mild rash on the arm."})
    assert out["urgency"] == "urgent"


def test_parse_json_strips_code_fences():
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}
