# MedTriage Agent

A **LangGraph** agent that reads a clinical note and returns, in seconds:
the **specialty** it belongs to, the **key facts** (complaint, symptoms, vitals, medications,
allergies), **red flags**, an **urgency level** (emergent / urgent / routine) with its reason,
and a **2-3 sentence handoff summary**.

Built with free tools: **Gemini API** (extraction, urgency, summary), a **Hugging Face**
zero-shot model (specialty), **DeepEval** (LLM-as-judge evals), **Streamlit** (demo UI)
and **GitHub Actions** (CI).

> ⚠️ Portfolio project, not a medical device. Notes in this repo are synthetic.
> Never send real patient data to a cloud API.

## Results

On `evals/golden.jsonl`, 12 synthetic notes labelled with urgency and specialty:

| Metric | Target | Result |
|---|---|---|
| Urgency accuracy | ≥ 80 % | **92 %** (11/12) |
| Emergencies under-triaged | 0 | **0** (4/4 caught by the safety rules) |
| Specialty accuracy (zero-shot, no training) | ≥ 60 % | **83 %** (10/12) |
| Summary faithfulness (DeepEval G-Eval) | ≥ 0.7 | **0.97** |

The one urgency miss is an *over*-triage: melena in a patient on daily NSAIDs, labelled
*urgent*, triaged *emergent*. Over- and under-triage are tracked separately because
their costs differ: sending a patient to the ER too early is cheap, missing one is not.

12 notes is a small test set: treat these numbers as a smoke test, not a benchmark.

## How it works

```
START ─┬─► classify  (Hugging Face zero-shot) ─┐
       └─► extract   (Gemini, JSON) ───────────┴─► red_flags (rules) ─┬─► escalate (rules) ─────┐
                                                                      └─► assess_urgency (LLM) ─┴─► summarize (LLM) ─► END
```

Design choices:
- **Rules before the LLM on safety.** Red flags (chest pain, stroke signs, SpO2 < 90 %,
  suicidal ideation...) are found by deterministic rules with negation handling
  ("denies chest pain" is not a flag). A red flag escalates straight to *emergent*:
  the LLM is never asked, so it can't talk an emergency down.
- **Fail safe.** If the LLM returns an invalid urgency, the note becomes *urgent*, never *routine*.
- **Model fallback.** On an overloaded model (503), a rate limit (429) or a dropped
  connection, the next call goes to the next model in the list
  (`gemini-3.8-flash` → `gemini-3.5-flash` → `gemini-3.5-flash-lite`). During
  development the primary model was overloaded for hours; without fallback every run failed.
- **Zero-shot labels as descriptions.** The classifier scores "This note is about *the
  urinary tract, bladder or kidneys*" instead of the bare word "urology":
  specialty accuracy went from 67 % to 83 %, with no training.
- **Parallel branches.** Classification and extraction are independent and run in parallel.
- **Testable core.** The LLM and the classifier are interfaces; unit tests run offline with fakes.

Eval design:
- **The judge is a different model than the app** (`gemini-3.7-flash` first), to avoid a
  model grading its own writing.
- **The judge sees what the summarizer saw** (the note *and* the assigned triage level).
  A first version showed it only the note, and it penalized every summary for "inventing"
  the triage level: faithfulness 0.73 → 0.97 once the eval was fixed, with no change to the app.
- `EVAL_REUSE_OUTPUTS=1` re-scores saved outputs without re-running the agent, to save
  the free-tier quota (20 requests per model per day).

## Setup

```bash
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install --extra-index-url https://download.pytorch.org/whl/cpu -e ".[dev,hf,eval,ui]"
copy .env.example .env      # then paste your free Gemini key
```

```bash
streamlit run app.py                     # web demo
medtriage "Sudden slurred speech and facial droop for 1 hour."
medtriage --file evals/golden.jsonl
pytest                                   # offline unit tests (17)
pytest evals -m eval -s                  # real-model evals
```

The classifier model (~700 MB) downloads on first use and needs ~2 GB of free RAM.

## CI

Two GitHub Actions workflows:
- **ci**: the offline unit tests, on every push (free, ~30 s).
- **evals**: the real-model evals, only when `src/`, `evals/` or `pyproject.toml` change on
  `main`, or on demand (*Actions → evals → Run workflow*). One run at a time: the free Gemini
  tier allows about 20 requests per model per day. Needs the `GEMINI_API_KEY` repository
  secret; eval outputs are uploaded as a build artifact.

Measured on CI, where the primary model was overloaded and the fallbacks answered:
urgency 83 % (10/12), 0 emergencies under-triaged, specialty 83 %, faithfulness 0.98.
Accuracy dips slightly on the smaller fallback models; the safety rules don't depend on the model.

## Project layout

```
src/medtriage/
  graph.py       LangGraph workflow
  red_flags.py   deterministic safety rules
  llm.py         Gemini client with model fallback, fake LLM for tests
  classifier.py  Hugging Face zero-shot specialty classifier
  prompts.py     prompts
  state.py       graph state, labels
app.py           Streamlit demo
evals/           golden set, eval suite, DeepEval judge
tests/           offline unit tests
```

## Roadmap
- [ ] Run on the public **MTSamples** dataset (~5,000 de-identified transcriptions, Kaggle)
      and grow the golden set beyond 12 notes
- [ ] Compare the zero-shot classifier with a fine-tuned clinical model (Bio_ClinicalBERT)
- [ ] Langfuse tracing
