"""Deterministic red-flag detection: the safety net that does not depend on the LLM.

If any red flag is found, the graph escalates the note to "emergent" without asking
the LLM. Rules are simple regexes with basic negation handling ("denies chest pain"
is not a red flag), in the spirit of the NegEx algorithm.
"""

from __future__ import annotations

import re

KEYWORD_FLAGS: dict[str, str] = {
    "chest pain": r"\bchest (?:pain|pressure|tightness)\b",
    "stroke signs": r"\b(?:facial droop|slurred speech|hemiparesis|sudden (?:weakness|numbness))\b",
    "difficulty breathing": r"\b(?:shortness of breath|respiratory distress|unable to breathe|can'?t breathe)\b",
    "suicidal ideation": r"\b(?:suicidal|wants? to die|plan to (?:kill|hurt) (?:him|her|them)sel(?:f|ves))\b",
    "anaphylaxis": r"\b(?:anaphyla\w*|throat (?:swelling|closing))\b",
    "loss of consciousness": r"\b(?:syncope|loss of consciousness|unresponsive|passed out)\b",
    "severe bleeding": r"\b(?:hematemesis|vomiting blood|hemorrhag\w*|massive bleeding)\b",
}

NEGATION = re.compile(r"\b(?:denies|denied|no|negative for|without|not|resolved)\b", re.I)
SENTENCE_BREAK = re.compile(r"[.;\n]")

SPO2 = re.compile(r"(?:spo2|o2 sat(?:uration)?|oxygen saturation)\D{0,12}(\d{2,3})\s*%", re.I)
BP = re.compile(r"\b(?:bp|blood pressure)\D{0,12}(\d{2,3})\s*/\s*(\d{2,3})", re.I)
HR = re.compile(r"\b(?:hr|heart rate|pulse)\D{0,12}(\d{2,3})\b", re.I)


def _negated(text: str, start: int) -> bool:
    """True if a negation word appears earlier in the same sentence."""
    breaks = [m.end() for m in SENTENCE_BREAK.finditer(text, 0, start)]
    prefix = text[breaks[-1] if breaks else 0 : start]
    return bool(NEGATION.search(prefix))


def find_red_flags(note: str) -> list[str]:
    flags: list[str] = []
    for name, pattern in KEYWORD_FLAGS.items():
        for m in re.finditer(pattern, note, re.I):
            if not _negated(note, m.start()):
                flags.append(name)
                break

    if any(int(v) < 90 for v in SPO2.findall(note)):
        flags.append("hypoxia (SpO2 < 90%)")
    if any(int(sys) < 90 for sys, _ in BP.findall(note)):
        flags.append("hypotension (systolic < 90)")
    if any(int(v) > 130 or int(v) < 40 for v in HR.findall(note)):
        flags.append("abnormal heart rate")
    return flags
