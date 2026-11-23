"""Benchmark the zero-shot specialty classifier on real (de-identified) MTSamples notes.

Download first:  kaggle datasets download tboyle10/medicaltranscriptions -p data/raw --unzip
Run:             python evals/mtsamples_specialty.py [--per-class 30] [--seed 0]

Only MTSamples categories that are clinical specialties are used ("Surgery", "Consult",
"Discharge Summary"... are document types, not specialties). "Cardiovascular / Pulmonary"
is one MTSamples category, so either cardiology or pulmonology counts as correct for it.
No LLM calls: free and unlimited.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

from medtriage.classifier import HFZeroShotClassifier
from medtriage.state import SPECIALTIES

ROOT = Path(__file__).resolve().parent.parent

# MTSamples category -> accepted labels
MAPPING: dict[str, set[str]] = {
    "Cardiovascular / Pulmonary": {"cardiology", "pulmonology"},
    "Neurology": {"neurology"},
    "Gastroenterology": {"gastroenterology"},
    "Orthopedic": {"orthopedics"},
    "Psychiatry / Psychology": {"psychiatry"},
    "Dermatology": {"dermatology"},
    "Obstetrics / Gynecology": {"obstetrics and gynecology"},
    "Urology": {"urology"},
    "General Medicine": {"general medicine"},
}


def load(per_class: int, seed: int) -> list[dict]:
    by_cat: dict[str, list[dict]] = defaultdict(list)
    with open(ROOT / "data/raw/mtsamples.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cat = row["medical_specialty"].strip()
            if cat in MAPPING and row["transcription"].strip():
                by_cat[cat].append(row)
    rng = random.Random(seed)
    sample = []
    for cat, rows in sorted(by_cat.items()):
        sample += rng.sample(rows, min(per_class, len(rows)))
    return sample


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--per-class", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rows = load(args.per_class, args.seed)
    clf = HFZeroShotClassifier()
    start = time.time()
    per_cat: dict[str, Counter] = defaultdict(Counter)
    confusions: Counter = Counter()
    for i, row in enumerate(rows, 1):
        cat = row["medical_specialty"].strip()
        label, _ = clf.classify(row["transcription"], SPECIALTIES)
        ok = label in MAPPING[cat]
        per_cat[cat]["correct" if ok else "wrong"] += 1
        if not ok:
            confusions[f"{cat} -> {label}"] += 1
        if i % 25 == 0:
            print(f"{i}/{len(rows)} ({time.time() - start:.0f}s)", flush=True)

    correct = sum(c["correct"] for c in per_cat.values())
    report = {
        "notes": len(rows),
        "accuracy": round(correct / len(rows), 3),
        "per_category": {
            cat: f"{c['correct']}/{c['correct'] + c['wrong']}" for cat, c in sorted(per_cat.items())
        },
        "top_confusions": confusions.most_common(8),
        "seconds": round(time.time() - start),
    }
    out = ROOT / "evals/results/mtsamples_specialty.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
