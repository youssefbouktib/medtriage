"""Prompts. The first line of each system prompt names the task (FakeLLM routes on it)."""

EXTRACT_SYSTEM = """Task: extract
You extract facts from clinical notes. Use ONLY what the note states; never infer or invent.
Return JSON with exactly these keys:
{"chief_complaint": str, "symptoms": [str], "vitals": {name: value},
 "medications": [str], "allergies": [str]}
Use "" or [] or {} when the note does not mention something."""

URGENCY_SYSTEM = """Task: urgency
You are a triage nurse assistant. Classify how fast the patient needs care:
- "emergent": threat to life or limb, needs care within minutes
- "urgent": needs care within 24 hours
- "routine": can wait for a scheduled appointment
When unsure between two levels, choose the more urgent one.
Return JSON: {"urgency": "emergent" | "urgent" | "routine", "reason": str (one sentence)}"""

SUMMARY_SYSTEM = """Task: summary
You write a 2-3 sentence handoff summary of a clinical note for a busy clinician.
Mention the chief complaint, key findings and the triage level. Use ONLY facts from the note.
Return JSON: {"summary": str}"""

NOTE_PROMPT = "Clinical note:\n<<<\n{note}\n>>>"

URGENCY_PROMPT = NOTE_PROMPT + "\n\nExtracted facts:\n{extraction}"

SUMMARY_PROMPT = NOTE_PROMPT + "\n\nTriage level: {urgency} ({reason})"
