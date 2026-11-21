"""Command line: `medtriage "note text"` or `medtriage --file notes.jsonl`."""

from __future__ import annotations

import argparse
import json
import sys


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="medtriage", description="Triage clinical notes.")
    p.add_argument("note", nargs="?", help="note text (or use --file)")
    p.add_argument("--file", help="JSONL file with one {'note': ...} object per line")
    args = p.parse_args(argv)

    if args.file:
        with open(args.file, encoding="utf-8") as f:
            notes = [json.loads(line)["note"] for line in f if line.strip()]
    elif args.note:
        notes = [args.note]
    else:
        p.error("give a note or --file")

    from .graph import build_default_graph

    graph = build_default_graph()
    for note in notes:
        result = graph.invoke({"note": note})
        result.pop("note")
        json.dump(result, sys.stdout, indent=2, ensure_ascii=False)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
