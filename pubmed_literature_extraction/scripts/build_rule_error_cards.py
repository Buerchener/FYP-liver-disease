#!/usr/bin/env python3
"""Export Agent v3 ErrorCards from a completed, frozen development replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cognitive_agent.rule_memory import error_cards_from_records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="induction")
    args = parser.parse_args()
    payload = json.loads(args.results.read_text(encoding="utf-8"))
    cards = error_cards_from_records(
        payload.get("records", payload.get("articles", [])), split=args.split,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(card.to_dict(), ensure_ascii=False) + "\n" for card in cards),
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), "error_cards": len(cards)}))


if __name__ == "__main__":
    main()
