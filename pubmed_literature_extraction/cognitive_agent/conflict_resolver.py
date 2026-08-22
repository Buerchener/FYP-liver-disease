#!/usr/bin/env python3
"""
cognitive_agent/conflict_resolver.py — Phase 4: Conflict Detection & Resolution

Implements the 8-row decision table from the architecture document:

| Old evidence | New evidence | Old conf | New conf | Decision      |
| exists       | same_dir     | any      | higher   | UPDATE        |
| exists       | same_dir     | any      | lower    | KEEP_OLD      |
| exists       | opposite     | <0.6     | >0.8     | UPDATE        |
| exists       | opposite     | >0.8     | >0.8     | DISPUTE       |
| exists       | opposite     | >0.8     | <0.6     | KEEP_OLD      |
| no_exists    | —            | —        | >0.7     | CREATE        |
| no_exists    | —            | —        | 0.4–0.7  | CREATE_WITH_FLAG |
| no_exists    | —            | —        | <0.4     | DISCARD       |

4 conflict types:
  - DIRECT_CONTRADICTION: same subject/object, opposite direction
  - EVIDENCE_STRENGTH: same direction, differing confidence
  - METHODOLOGICAL_DIFF: different experimental context
  - TEMPORAL_DRIFT: older evidence vs newer evidence
"""

from __future__ import annotations

from dataclasses import dataclass, field
from cognitive_agent.verifier import VerifiedRelation, VerifiedExtraction


@dataclass
class ResolutionItem:
    """Resolution decision for a single relation."""
    subject: str = ""
    predicate: str = ""
    object: str = ""
    conflict_type: str = "NONE"     # NONE | DIRECT_CONTRADICTION | EVIDENCE_STRENGTH | METHODOLOGICAL_DIFF | TEMPORAL_DRIFT
    decision: str = "NO_ACTION"     # CREATE | UPDATE | KEEP_OLD | DISPUTE | DISCARD | CREATE_WITH_FLAG | NO_ACTION
    reasoning_trace: str = ""
    adjusted_confidence: float = 0.0

    def to_dict(self) -> dict:
        return {
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "conflict_type": self.conflict_type,
            "decision": self.decision,
            "adjusted_confidence": self.adjusted_confidence,
            "reasoning": self.reasoning_trace,
        }


@dataclass
class ResolutionResult:
    """Aggregated conflict resolution result."""
    items: list[ResolutionItem] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "total": len(self.items),
            "decisions": self._count_decisions(),
            "items": [i.to_dict() for i in self.items],
        }

    def _count_decisions(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for item in self.items:
            counts[item.decision] = counts.get(item.decision, 0) + 1
        return counts


class ConflictResolver:
    """Conflict detection and resolution engine.

    Pure logic: takes VerifiedExtraction, returns ResolutionResult.
    No side effects. DecisionEngine calls this before making final decisions.
    """

    def resolve(self, verified: VerifiedExtraction) -> ResolutionResult:
        """Resolve conflicts for all relations in a verified extraction.

        Args:
            verified: Verified extraction with neo4j_status and existing_confidence

        Returns:
            ResolutionResult with one ResolutionItem per relation
        """
        result = ResolutionResult()

        for rel in verified.relations:
            item = self._resolve_one(rel)
            result.items.append(item)

        return result

    def _resolve_one(self, rel: VerifiedRelation) -> ResolutionItem:
        """Apply the 8-row decision table to a single relation."""
        item = ResolutionItem(
            subject=rel.subject,
            predicate=rel.predicate,
            object=rel.object,
        )

        # ── Determine new evidence confidence ──
        if rel.uncertain:
            new_conf = 0.5
        elif rel.negated:
            new_conf = 0.4
        else:
            new_conf = 0.8  # default for well-formed extraction

        item.adjusted_confidence = new_conf

        # ── No existing evidence → create or discard ──
        if rel.neo4j_status == "NOVEL":
            if not getattr(rel, "candidate_schema_valid", rel.schema_valid):
                item.decision = "DISCARD"
                item.conflict_type = "NONE"
                item.reasoning_trace = "Candidate schema invalid — discard"
            elif not getattr(rel, "write_contract_valid", False):
                item.decision = "NO_ACTION"
                item.conflict_type = "NONE"
                reasons = getattr(rel, "schema_gap_reasons", []) or []
                item.reasoning_trace = (
                    "Main-KG schema gap — preserve candidate"
                    + (f" ({', '.join(reasons[:4])})" if reasons else "")
                )
            elif new_conf > 0.7:
                item.decision = "CREATE"
                item.conflict_type = "NONE"
                item.reasoning_trace = f"Novel relation, high confidence ({new_conf:.1f})"
            elif new_conf >= 0.4:
                item.decision = "CREATE_WITH_FLAG"
                item.conflict_type = "NONE"
                item.reasoning_trace = f"Novel relation, moderate confidence ({new_conf:.1f}) — flag for review"
            else:
                item.decision = "DISCARD"
                item.conflict_type = "NONE"
                item.reasoning_trace = f"Novel relation, low confidence ({new_conf:.1f})"
            return item

        # ── Known relation: check for contradiction ──
        old_conf = rel.existing_confidence if rel.existing_confidence > 0 else 0.7

        if rel.neo4j_status == "INVERTED":
            item.decision = "NO_ACTION"
            item.conflict_type = "NONE"
            item.reasoning_trace = "Inverse relation exists — reverse edge requires explicit schema review"
            return item

        if rel.neo4j_status == "CONTRADICTING":
            # Determine if direction is truly opposite
            item.conflict_type = "DIRECT_CONTRADICTION"

            if new_conf > 0.8 and old_conf < 0.6:
                item.decision = "UPDATE"
                item.reasoning_trace = (
                    f"New high-confidence evidence (conf={new_conf:.1f}) "
                    f"overrides old low-confidence evidence (conf={old_conf:.1f})"
                )
            elif new_conf > 0.8 and old_conf > 0.8:
                item.decision = "DISPUTE"
                item.conflict_type = "DIRECT_CONTRADICTION"
                item.reasoning_trace = (
                    f"Both old (conf={old_conf:.1f}) and new (conf={new_conf:.1f}) "
                    f"are high-confidence but contradictory — marking as disputed"
                )
            elif new_conf < 0.6 and old_conf > 0.8:
                item.decision = "KEEP_OLD"
                item.reasoning_trace = (
                    f"New evidence (conf={new_conf:.1f}) insufficient "
                    f"to override old (conf={old_conf:.1f})"
                )
            else:
                item.decision = "KEEP_OLD"
                item.reasoning_trace = (
                    f"Contradicting evidence, insufficient to override "
                    f"(new={new_conf:.1f}, old={old_conf:.1f})"
                )
            return item

        # ── Known, same direction: update or keep ──
        if rel.neo4j_status == "KNOWN":
            if new_conf > old_conf:
                item.decision = "UPDATE"
                item.conflict_type = "EVIDENCE_STRENGTH"
                item.reasoning_trace = (
                    f"New evidence (conf={new_conf:.1f}) strengthens "
                    f"existing evidence (conf={old_conf:.1f})"
                )
                item.adjusted_confidence = max(new_conf, old_conf)
            else:
                item.decision = "KEEP_OLD"
                item.conflict_type = "EVIDENCE_STRENGTH"
                item.reasoning_trace = (
                    f"New evidence (conf={new_conf:.1f}) does not improve "
                    f"on existing evidence (conf={old_conf:.1f})"
                )
                item.adjusted_confidence = old_conf
            return item

        # ── Fallback ──
        item.decision = "NO_ACTION"
        item.reasoning_trace = f"Unhandled status: {rel.neo4j_status}"
        return item
