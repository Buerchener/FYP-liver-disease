#!/usr/bin/env python3
"""
cognitive_agent/causal_reasoner.py — Phase 3+: Causal Chain Inference

Transitivity-based reasoning on extracted and known relations:
  A → B (extracted) + B → C (known in KG or extracted) ⇒ A → C (inferred)

Predicate composition rules determine the inferred predicate type.
Inferred confidence = min(step1.confidence, step2.confidence) × 0.8 discount.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
from cognitive_agent.memory.kg_memory import KGMemory
from cognitive_agent.verifier import VerifiedRelation


# ── 谓词组合规则 ──
# (predicate1, predicate2) → composed_predicate
PREDICATE_COMPOSITION: dict[tuple[str, str], str] = {
    ("ASSOCIATED_WITH", "PROGRESSES_TO"): "ASSOCIATED_WITH",
    ("PROGNOSTIC_IN", "PROGRESSES_TO"): "PROGNOSTIC_IN",
    ("ASSOCIATED_WITH", "ASSOCIATED_WITH"): "ASSOCIATED_WITH",
    ("PARTICIPATES_IN", "ASSOCIATED_WITH"): "ASSOCIATED_WITH",
    ("ENCODES", "INTERACTS_WITH"): "ASSOCIATED_WITH",
    ("INTERACTS_WITH", "PARTICIPATES_IN"): "PARTICIPATES_IN",
    ("ASSOCIATED_WITH_METABOLITE", "ASSOCIATED_WITH"): "ASSOCIATED_WITH",
    ("EXPRESSED_IN", "ASSOCIATED_WITH"): "ASSOCIATED_WITH",
}


@dataclass
class CausalStep:
    """One step in a causal chain."""
    subject: str
    predicate: str
    object: str
    subject_type: str = ""
    object_type: str = ""
    source: str = "extracted"  # "extracted" or "known"

    def to_dict(self) -> dict:
        return {
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "subject_type": self.subject_type,
            "object_type": self.object_type,
            "source": self.source,
        }


@dataclass
class CausalChain:
    """A derived causal chain with inferred relation."""
    steps: list[CausalStep] = field(default_factory=list)
    inferred_subject: str = ""
    inferred_predicate: str = ""
    inferred_object: str = ""
    inferred_subject_type: str = ""
    inferred_object_type: str = ""
    confidence: float = 0.0
    derivation: str = "causal_transitivity"

    def to_dict(self) -> dict:
        return {
            "steps": [s.to_dict() for s in self.steps],
            "inferred_relation": f"{self.inferred_subject}"
            f" -[{self.inferred_predicate}]-> {self.inferred_object}",
            "confidence": round(self.confidence, 3),
            "derivation": self.derivation,
        }


class CausalReasoner:
    """Transitivity-based causal chain reasoner.

    Operates on extracted relations first, then optionally extends
    with known KG relations if a KGMemory instance is available.
    """

    def __init__(self, kg_memory: Optional[KGMemory] = None):
        self.kg_memory = kg_memory

    def infer_causal_chains(
        self, relations: list[VerifiedRelation]
    ) -> list[CausalChain]:
        """Infer causal chains from a set of verified relations.

        For each pair of relations where rel_A.object ≈ rel_B.subject,
        attempts to compose them into a new inferred relation.

        Args:
            relations: Verified relations from the current extraction

        Returns:
            List of CausalChain objects
        """
        chains: list[CausalChain] = []

        # 1. Build intra-article lookup: object_text → list of relations
        #    (what points TO this entity)
        incoming: dict[str, list[VerifiedRelation]] = {}
        for rel in relations:
            obj_lower = rel.object.lower().strip()
            if obj_lower not in incoming:
                incoming[obj_lower] = []
            incoming[obj_lower].append(rel)

        # 2. For each relation, check if its subject is someone else's object
        for rel in relations:
            subj_lower = rel.subject.lower().strip()

            # a) Intra-article: check other extracted relations
            downstream_rels = incoming.get(subj_lower, [])
            for down_rel in downstream_rels:
                # Skip self-chaining
                if rel.subject == down_rel.subject and rel.object == down_rel.object:
                    continue
                chain = self._build_chain(rel, down_rel, "extracted")
                if chain:
                    chains.append(chain)

        # 3. KG-backed extension (if Neo4j available)
        if self.kg_memory and self.kg_memory.is_connected:
            for rel in relations:
                kg_chains = self._extend_from_kg(rel)
                chains.extend(kg_chains)

        # 4. Deduplicate by inferred relation triple
        seen: set[tuple[str, str, str]] = set()
        unique_chains: list[CausalChain] = []
        for chain in chains:
            key = (chain.inferred_subject, chain.inferred_predicate, chain.inferred_object)
            if key not in seen:
                seen.add(key)
                unique_chains.append(chain)

        return unique_chains

    def _build_chain(
        self,
        upstream: VerifiedRelation,
        downstream: VerifiedRelation,
        downstream_source: str,
    ) -> Optional[CausalChain]:
        """Attempt to compose two adjacent relations into a causal chain."""
        composed = self._compose_predicate(upstream.predicate, downstream.predicate)
        if not composed:
            return None

        # Confidence: min of the two, discounted by 0.8 for inference
        up_conf = upstream.existing_confidence if upstream.existing_confidence > 0 else 0.7
        down_conf = downstream.existing_confidence if downstream.existing_confidence > 0 else 0.7
        chain_conf = round(min(up_conf, down_conf) * 0.8, 3)

        # Don't generate very low-confidence chains
        if chain_conf < 0.3:
            return None

        return CausalChain(
            steps=[
                CausalStep(
                    upstream.subject, upstream.predicate, upstream.object,
                    upstream.subject_type, upstream.object_type, source="extracted",
                ),
                CausalStep(
                    downstream.subject, downstream.predicate, downstream.object,
                    downstream.subject_type, downstream.object_type,
                    source=downstream_source,
                ),
            ],
            inferred_subject=upstream.subject,
            inferred_predicate=composed,
            inferred_object=downstream.object,
            inferred_subject_type=upstream.subject_type,
            inferred_object_type=downstream.object_type,
            confidence=chain_conf,
        )

    def _compose_predicate(self, p1: str, p2: str) -> Optional[str]:
        """Compose two predicates through transitivity."""
        return PREDICATE_COMPOSITION.get((p1, p2))

    def _extend_from_kg(self, relation: VerifiedRelation) -> list[CausalChain]:
        """Extend a chain by querying Neo4j for downstream relations."""
        chains: list[CausalChain] = []
        if not self.kg_memory:
            return chains

        # Find the object entity in KG, get its outgoing relations
        obj_entity = self.kg_memory.find_entity(relation.object)
        if not obj_entity:
            return chains

        known_rels = self.kg_memory.get_relations(obj_entity["element_id"])
        for kr in known_rels:
            # Build a synthetic downstream step from KG
            vr = VerifiedRelation(
                subject=relation.object,
                predicate=kr.get("predicate", ""),
                object=kr.get("target_name", ""),
                subject_type=relation.object_type,
                object_type="",  # unknown from KG query
                existing_confidence=0.9,  # KG data is higher confidence
            )
            chain = self._build_chain(relation, vr, downstream_source="known")
            if chain:
                chains.append(chain)

        return chains
