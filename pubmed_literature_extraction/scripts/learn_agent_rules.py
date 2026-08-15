#!/usr/bin/env python3
"""Offline entry point for Agent v3 rule induction and promotion."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cognitive_agent.aux_model_registry import AuxModelRegistry
from cognitive_agent.rule_learning import RuleLearner, write_rule_artifacts
from cognitive_agent.rule_memory import ErrorCard, RuleBundle


def load_json(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


def load_error_cards(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    try:
        payload = json.loads(text)
        return payload.get("error_cards", []) if isinstance(payload, dict) else payload
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--error-cards", type=Path, required=True)
    parser.add_argument("--replay-metrics", type=Path, required=True)
    parser.add_argument("--existing-bundle", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shadow-completed", action="store_true")
    parser.add_argument("--aux-primary-model", default="deepseek-v4-flash")
    parser.add_argument("--aux-critic-model", default="qwen3.6-flash")
    parser.add_argument("--timeout", type=float, default=90.0)
    args = parser.parse_args()

    raw_cards = load_error_cards(args.error_cards)
    cards = [ErrorCard(**item) for item in raw_cards]
    replay_metrics = load_json(args.replay_metrics)
    existing = (
        RuleBundle.from_dict(load_json(args.existing_bundle))
        if args.existing_bundle and args.existing_bundle.exists() else RuleBundle()
    )
    registry = AuxModelRegistry.from_environment(
        primary_model=args.aux_primary_model,
        critic_model=args.aux_critic_model,
        timeout_s=args.timeout,
    )
    if not registry.configured("primary") or not registry.configured("critic"):
        print(json.dumps({
            "status": "BLOCKED",
            "reason": "both DeepSeek primary and Qwen critic must be configured for automatic promotion",
            "models": registry.public_config(),
        }, ensure_ascii=False, indent=2))
        return 2
    bundle, audit = RuleLearner(registry).run(
        cards, existing=existing, replay_metrics=replay_metrics,
        shadow_completed=args.shadow_completed,
    )
    write_rule_artifacts(args.output_dir, bundle, audit)
    print(json.dumps({
        "status": audit.status, "candidate_count": len(audit.candidates),
        "promoted_count": len(audit.promoted), "bundle_hash": bundle.bundle_hash,
        "output_dir": str(args.output_dir), "model_audit": registry.audit(),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
