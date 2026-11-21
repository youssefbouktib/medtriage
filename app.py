"""Streamlit demo: paste a clinical note, get the triage.  Run: streamlit run app.py"""

import json
from pathlib import Path

import streamlit as st

from medtriage.graph import build_default_graph

st.set_page_config(page_title="MedTriage Agent", page_icon="🩺", layout="wide")

BADGE = {"emergent": ("🔴", "red"), "urgent": ("🟠", "orange"), "routine": ("🟢", "green")}
EXAMPLES = {
    json.loads(line)["id"]: json.loads(line)["note"]
    for line in (Path(__file__).parent / "evals" / "golden.jsonl").read_text(encoding="utf-8").splitlines()
    if line.strip()
}


@st.cache_resource(show_spinner="Loading models (first run downloads the classifier)...")
def graph():
    return build_default_graph()


st.title("🩺 MedTriage Agent")
st.caption("Portfolio demo, not a medical device. Use synthetic notes only; never paste real patient data.")

example = st.selectbox("Start from an example note", ["(write my own)", *EXAMPLES])
note = st.text_area(
    "Clinical note", value="" if example == "(write my own)" else EXAMPLES[example], height=160
)

if st.button("Triage", type="primary", disabled=not note.strip()):
    with st.spinner("Running the agent..."):
        out = graph().invoke({"note": note})

    icon, color = BADGE[out["urgency"]]
    st.subheader(f"{icon} :{color}[{out['urgency'].upper()}]")
    source = "safety rules (LLM not consulted)" if out["urgency_source"] == "rules" else "LLM assessment"
    st.write(f"**Why:** {out['urgency_reason']}  \n*Decided by: {source}*")

    if out["red_flags"]:
        st.error("Red flags: " + ", ".join(out["red_flags"]))

    st.info(f"**Handoff summary:** {out['summary']}")

    left, right = st.columns(2)
    with left:
        st.metric("Specialty", out["specialty"], f"confidence {out['specialty_score']:.0%}", delta_color="off")
    with right:
        st.write("**Extracted facts**")
        st.json(out["extraction"])
