"""Streamlit demo: paste a clinical note, get a triage tag.  Run: streamlit run app.py"""

from __future__ import annotations

import json
from html import escape
from pathlib import Path

import streamlit as st

from medtriage.graph import build_default_graph

st.set_page_config(page_title="MedTriage", page_icon="🏷️", layout="wide")

ROOT = Path(__file__).parent
NOTES = {
    row["id"]: row["note"]
    for row in map(json.loads, (ROOT / "evals" / "golden.jsonl").read_text(encoding="utf-8").splitlines())
}
EXAMPLES = {  # short names for the example notes, in the order they are offered
    "Chest pressure": "g01",
    "Slurred speech": "g02",
    "Asthma attack": "g03",
    "Crisis call": "g04",
    "Kidney infection": "g05",
    "Twisted ankle": "g06",
    "Black stools": "g07",
    "Mole check": "g08",
    "Annual physical": "g09",
}
LEVELS = {
    "emergent": ("Emergent", "Care within minutes"),
    "urgent": ("Urgent", "Care within 24 hours"),
    "routine": ("Routine", "Book an appointment"),
}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Atkinson+Hyperlegible:wght@400;700&family=Barlow+Condensed:wght@600;700&display=swap');

:root {
  --paper: #F3F6F7; --ink: #1B2A41; --muted: #5B6B7A; --rule: #C9D3D8;
  --teal: #0E6E6E; --card: #FFFFFF;
  --emergent: #C8102E; --urgent: #F2B300; --routine: #2F8F5B;
}
html, body, [class*="st-"], .stMarkdown, textarea, button, input {
  font-family: 'Atkinson Hyperlegible', system-ui, sans-serif;
}
/* background: ECG paper (5 mm major / 1 mm minor grid) with a heartbeat trace along the top */
.stApp {
  color: var(--ink);
  background-color: var(--paper);
  background-image:
    url("data:image/svg+xml;utf8,%3Csvg xmlns='http://www.w3.org/2000/svg' width='260' height='90' viewBox='0 0 260 90'%3E%3Cpath d='M0 60 H70 q7 -9 14 0 H104 l5 7 l8 -52 l8 62 l6 -17 H160 q12 -16 24 0 H260' fill='none' stroke='%230E6E6E' stroke-opacity='.32' stroke-width='2' stroke-linejoin='round' stroke-linecap='round'/%3E%3C/svg%3E"),
    linear-gradient(rgba(200,16,46,.085) 1px, transparent 1px),
    linear-gradient(90deg, rgba(200,16,46,.085) 1px, transparent 1px),
    linear-gradient(rgba(200,16,46,.04) 1px, transparent 1px),
    linear-gradient(90deg, rgba(200,16,46,.04) 1px, transparent 1px);
  background-size: 260px 90px, 40px 40px, 40px 40px, 8px 8px, 8px 8px;
  background-repeat: repeat-x, repeat, repeat, repeat, repeat;
  background-position: 0 -6px, 0 0, 0 0, 0 0, 0 0;
}
.block-container { max-width: 1120px; padding-top: 5.6rem; }  /* room for the ECG trace */
header[data-testid="stHeader"] { background: transparent; }

/* masthead */
.mt-mast { display: flex; align-items: center; gap: .9rem; margin-bottom: 1.6rem; }
.mt-mast svg { flex: none; }
.mt-mast h1 { font: 700 1.55rem/1.1 'Atkinson Hyperlegible', sans-serif; margin: 0; padding: 0; color: var(--ink); }
.mt-mast p { margin: .15rem 0 0; color: var(--muted); font-size: .98rem; }

/* inputs */
.stTextArea textarea {
  background: var(--card); border: 1px solid var(--rule); border-radius: 6px;
  font-size: 1rem; line-height: 1.55; color: var(--ink);
}
.stTextArea textarea:focus { border-color: var(--teal); box-shadow: 0 0 0 3px rgba(14,110,110,.18); }
.stButton > button[kind="primary"] {
  background: var(--teal); border: none; border-radius: 6px; font-weight: 700;
  padding: .6rem 1.4rem; font-size: 1rem;
}
.stButton > button[kind="primary"]:hover { background: #0B5A5A; }
.stButton > button[kind="primary"]:disabled { background: #DCE3E6; color: var(--muted); }
[role="radiogroup"] { flex-wrap: wrap !important; overflow: visible !important; }

/* mark: a small triage tag drawn in CSS */
.mt-mark {
  flex: none; width: 30px; height: 40px; border-radius: 4px 4px 3px 3px;
  background:
    radial-gradient(circle at 50% 9px, var(--paper) 0 3.5px, transparent 4px),
    linear-gradient(var(--emergent) 0 55%, #fff 55% 60%, var(--urgent) 60% 78%, var(--routine) 78%);
  box-shadow: 0 6px 12px -8px rgba(27,42,65,.6);
  transform: rotate(-8deg);
}
/* Streamlit indents definition lists by default */
.tag__meta, .facts dl { padding: 0 !important; }
.tag__meta dt, .tag__meta dd, .facts dt, .facts dd { margin-left: 0 !important; }
.saved-note { color: var(--muted); font-size: .9rem; margin: .6rem 0 0; text-align: center; }
.stButton > button:focus-visible { outline: 3px solid var(--ink); outline-offset: 2px; }

/* the triage tag */
.tag-wrap { display: flex; justify-content: center; padding-top: .4rem; }
.tag {
  position: relative; width: min(100%, 380px); background: var(--card);
  border-radius: 10px 10px 6px 6px; box-shadow: 0 1px 0 var(--rule), 0 14px 28px -18px rgba(27,42,65,.45);
  transform-origin: 50% 14px;
}
.tag--live { animation: swing 900ms cubic-bezier(.2,.9,.3,1.2) both; }
@keyframes swing {
  0% { transform: rotate(-7deg) translateY(-10px); opacity: 0; }
  35% { opacity: 1; }
  60% { transform: rotate(2.5deg); }
  100% { transform: rotate(0); }
}
@media (prefers-reduced-motion: reduce) { .tag--live { animation: none; } }
.tag__hole {
  position: absolute; top: 9px; left: 50%; width: 14px; height: 14px; margin-left: -7px;
  border-radius: 50%; background: var(--paper); box-shadow: inset 0 0 0 3px #DCE3E6; z-index: 2;
}
.tag__band { padding: 2.1rem 1.4rem 1rem; border-radius: 10px 10px 0 0; color: #fff; }
.tag--emergent .tag__band { background: var(--emergent); }
.tag--urgent .tag__band { background: var(--urgent); color: var(--ink); }
.tag--routine .tag__band { background: var(--routine); }
.tag__level {
  display: block; font: 700 3.6rem/.9 'Barlow Condensed', sans-serif; letter-spacing: .01em;
}
.tag__when { display: block; margin-top: .35rem; font-size: 1rem; opacity: .92; }
.tag__body { padding: 1.1rem 1.4rem 1.2rem; }
.tag__reason { margin: 0 0 .9rem; line-height: 1.5; }
.tag__flags { display: flex; flex-wrap: wrap; gap: .4rem; margin-bottom: .9rem; }
.tag__flag {
  border: 1.5px solid var(--emergent); color: var(--emergent); border-radius: 999px;
  padding: .12rem .65rem; font-size: .88rem; font-weight: 700;
}
.tag__meta { display: grid; grid-template-columns: auto 1fr; gap: .3rem 1rem; margin: 0; font-size: .95rem; }
.tag__meta dt { color: var(--muted); }
.tag__meta dd { margin: 0; font-weight: 700; }
.tag__meta small { font-weight: 400; color: var(--muted); }
.tag__stub {
  border-top: 2px dashed var(--rule); padding: 1rem 1.4rem 1.2rem; font-size: .95rem; line-height: 1.55;
  color: var(--ink); background: #FAFBFB; border-radius: 0 0 6px 6px;
}
.tag__stub b { display: block; color: var(--muted); font-weight: 400; margin-bottom: .25rem; }

.tag--empty { box-shadow: none; border: 2px dashed var(--rule); background: transparent; }
.tag--empty .tag__band { color: var(--muted); padding-bottom: 2.4rem; }
.tag--empty .tag__level { color: #B5C1C7; }

/* path the agent took */
.path { margin: 2.4rem 0 1rem; }
.path h3, .facts h3 { font-size: 1.05rem; font-weight: 700; margin: 0 0 .8rem; color: var(--ink); }
.path ol { list-style: none; display: flex; flex-wrap: wrap; gap: .5rem; padding: 0; margin: 0; counter-reset: step; }
.path li {
  counter-increment: step; display: flex; align-items: center; gap: .5rem;
  background: var(--card); border: 1px solid var(--rule); border-radius: 6px; padding: .5rem .8rem; font-size: .93rem;
}
.path li::before {
  content: counter(step); display: inline-grid; place-items: center; width: 1.4rem; height: 1.4rem;
  border-radius: 50%; background: var(--ink); color: #fff; font-size: .78rem; font-weight: 700;
}
.path li.decided { border-color: var(--teal); box-shadow: inset 3px 0 0 var(--teal); }
.path li small { color: var(--muted); }

/* extracted facts */
.facts { margin-top: 1.6rem; }
.facts dl { display: grid; grid-template-columns: 11rem 1fr; gap: 0; margin: 0; background: var(--card);
  border: 1px solid var(--rule); border-radius: 6px; }
.facts dt, .facts dd { padding: .65rem .9rem; border-top: 1px solid #E4EAEC; margin: 0; }
.facts dt:first-of-type, .facts dd:first-of-type { border-top: none; }
.facts dt { color: var(--muted); }
.facts .none { color: #9AA8B1; }

.disclaimer { margin-top: 2.4rem; color: var(--muted); font-size: .85rem; }

@media (max-width: 640px) {
  .tag__level { font-size: 2.8rem; }
  .facts dl { grid-template-columns: 1fr; }
  .facts dd { border-top: none; padding-top: 0; }
}
</style>
"""

MAST = """
<div class="mt-mast">
  <div class="mt-mark" aria-hidden="true"></div>
  <div>
    <h1>MedTriage</h1>
    <p>Paste a clinical note. The agent reads it and writes the triage tag.</p>
  </div>
</div>
"""


@st.cache_resource(show_spinner="Loading the models. The first run downloads the classifier, about a minute.")
def graph():
    return build_default_graph()


def use_example() -> None:
    choice = st.session_state.get("example")
    if choice:
        st.session_state.note = NOTES[EXAMPLES[choice]]
        st.session_state.pop("result", None)


def tag_html(out: dict | None) -> str:
    if out is None:
        return (
            '<div class="tag-wrap"><div class="tag tag--empty"><div class="tag__hole"></div>'
            '<div class="tag__band"><span class="tag__level">No tag yet</span>'
            '<span class="tag__when">Choose an example or paste a note, then select Triage note.</span>'
            "</div></div></div>"
        )
    level, when = LEVELS[out["urgency"]]
    flags = "".join(f'<span class="tag__flag">{escape(f)}</span>' for f in out["red_flags"])
    decided = "Safety rules" if out["urgency_source"] == "rules" else "Gemini"
    # When the rules decide, the red-flag chips already say why; repeating it as a sentence adds nothing.
    reason = (
        '<p class="tag__reason">A safety rule matched, so the note was escalated without asking Gemini.</p>'
        if out["urgency_source"] == "rules"
        else f'<p class="tag__reason">{escape(out["urgency_reason"])}</p>'
    )
    return f"""
<div class="tag-wrap"><div class="tag tag--live tag--{out['urgency']}" role="status" aria-label="Triage level {level}">
  <div class="tag__hole"></div>
  <div class="tag__band"><span class="tag__level">{level}</span><span class="tag__when">{when}</span></div>
  <div class="tag__body">
    {reason}
    {f'<div class="tag__flags">{flags}</div>' if flags else ''}
    <dl class="tag__meta">
      <dt>Specialty</dt><dd>{escape(out['specialty'].capitalize())} <small>{out['specialty_score']:.0%} sure</small></dd>
      <dt>Decided by</dt><dd>{decided}</dd>
    </dl>
  </div>
  <div class="tag__stub"><b>Handoff summary</b>{escape(out['summary'])}</div>
</div></div>"""


def path_html(out: dict) -> str:
    rules = out["urgency_source"] == "rules"
    flags = ", ".join(out["red_flags"])
    steps = [
        ("Read the specialty", f"<small>{escape(out['specialty'])}</small>", False),
        ("Pulled out the facts", "", False),
        ("Checked safety rules", f"<small>{escape(flags) if rules else 'nothing found'}</small>", False),
        (
            ("Escalated by the rules", "<small>Gemini not asked</small>", True)
            if rules
            else ("Gemini judged urgency", f"<small>{escape(out['urgency'])}</small>", True)
        ),
        ("Wrote the summary", "", False),
    ]
    items = "".join(
        f'<li class="{"decided" if decided else ""}">{escape(name)} {detail}</li>' for name, detail, decided in steps
    )
    return f'<section class="path"><h3>How the agent got there</h3><ol>{items}</ol></section>'


def facts_html(ex: dict) -> str:
    def show(v) -> str:
        if isinstance(v, dict):
            v = ", ".join(f"{k} {val}" for k, val in v.items())
        elif isinstance(v, list):
            v = ", ".join(v)
        return escape(v) if v else '<span class="none">None noted</span>'

    rows = [
        ("Chief complaint", ex["chief_complaint"]),
        ("Symptoms", ex["symptoms"]),
        ("Vital signs", ex["vitals"]),
        ("Medications", ex["medications"]),
        ("Allergies", ex["allergies"]),
    ]
    body = "".join(f"<dt>{k}</dt><dd>{show(v)}</dd>" for k, v in rows)
    return f'<section class="facts"><h3>What the note says</h3><dl>{body}</dl></section>'


st.html(CSS)
st.html(MAST)

# ?saved=g01 shows a result saved by the eval run: a demo that works when Gemini is down.
saved_id = st.query_params.get("saved")
saved_file = ROOT / "evals" / "results" / "outputs.json"
if saved_id and saved_id in NOTES and saved_file.exists() and st.session_state.get("saved_id") != saved_id:
    saved = {r["id"]: r for r in json.loads(saved_file.read_text(encoding="utf-8"))}
    if saved_id in saved:
        st.session_state.saved_id = saved_id
        st.session_state.note = NOTES[saved_id]
        st.session_state.result = {**saved[saved_id], "saved": True}

left, right = st.columns([1.15, 1], gap="large")
with left:
    st.pills("Try an example", list(EXAMPLES), key="example", on_change=use_example)
    note = st.text_area(
        "Clinical note",
        key="note",
        height=230,
        placeholder="58-year-old with crushing chest pressure for 40 minutes, BP 150/95...",
    )
    if st.button("Triage note", type="primary", disabled=not (note or "").strip()):
        with st.spinner("Reading the note..."):
            try:
                st.session_state.result = graph().invoke({"note": note})
            except Exception as exc:  # e.g. every Gemini model overloaded
                st.session_state.pop("result", None)
                st.error(
                    f"Triage didn't finish: {type(exc).__name__}. Gemini may be overloaded; "
                    "wait a minute and select Triage note again."
                )

result = st.session_state.get("result")
with right:
    st.html(tag_html(result))
    if result and result.get("saved"):
        st.html('<p class="saved-note">Saved result from the evaluation run, not a live call.</p>')

if result:
    st.html(path_html(result))
    st.html(facts_html(result["extraction"]))

st.html(
    '<p class="disclaimer">A portfolio demo, not a medical device. Use synthetic notes only. '
    "Never paste real patient information.</p>"
)
