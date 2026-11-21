"""The state that flows through the triage graph. Each node fills in some keys."""

from __future__ import annotations

from typing import Literal, TypedDict

Urgency = Literal["emergent", "urgent", "routine"]
URGENCY_LEVELS: tuple[str, ...] = ("emergent", "urgent", "routine")

# Labels for the zero-shot specialty classifier (subset of MTSamples specialties).
SPECIALTIES: list[str] = [
    "cardiology",
    "neurology",
    "pulmonology",
    "gastroenterology",
    "orthopedics",
    "psychiatry",
    "dermatology",
    "obstetrics and gynecology",
    "urology",
    "general medicine",
]

# What the zero-shot model actually reads: bare names like "urology" are too terse for it.
SPECIALTY_DESCRIPTIONS: dict[str, str] = {
    "cardiology": "the heart, chest pain or heart attack",
    "neurology": "the brain or nerves, such as stroke, seizures or headaches",
    "pulmonology": "the lungs or breathing, such as asthma, COPD or pneumonia",
    "gastroenterology": "the stomach, bowels, liver or digestive bleeding",
    "orthopedics": "bones, joints, muscles or sports injuries",
    "psychiatry": "mental health, such as depression, anxiety or suicidal thoughts",
    "dermatology": "the skin, moles or rashes",
    "obstetrics and gynecology": "pregnancy or women's reproductive health",
    "urology": "the urinary tract, bladder or kidneys",
    "general medicine": "a routine check-up, physical exam or medication refill",
}


class Extraction(TypedDict):
    chief_complaint: str
    symptoms: list[str]
    vitals: dict[str, str]
    medications: list[str]
    allergies: list[str]


class TriageState(TypedDict, total=False):
    note: str                 # input: the clinical note
    specialty: str            # classify node (Hugging Face)
    specialty_score: float
    extraction: Extraction    # extract node (LLM)
    red_flags: list[str]      # red_flags node (deterministic rules)
    urgency: Urgency          # escalate (rules) or assess_urgency (LLM)
    urgency_reason: str
    urgency_source: Literal["rules", "llm"]
    summary: str              # summarize node (LLM)
