"""Dataset-scoped candidate schema profiles.

The verification algorithm is shared, while predicates, endpoint signatures
and lexical support rules are supplied by the active dataset.  This module is
deliberately independent from ``write_contract``: accepting a benchmark
relation type never grants permission to write it to the LiverKG Neo4j graph.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping


SupportMatcher = Callable[..., dict[str, Any]]
SUPPORTED_SEMANTIC_TARGETS = frozenset({"CURRENT_FINDING", "RELATION_TRUTH"})
SUPPORTED_ENTITY_VALIDATION = frozenset({"liverkg_quality", "source_grounded", "given_entity"})


def _surface_pattern(value: str) -> re.Pattern[str] | None:
    tokens = re.findall(r"\w+", str(value or ""), flags=re.UNICODE)
    if not tokens:
        return None
    return re.compile(
        r"(?<!\w)" + r"(?:[\W_]+)".join(re.escape(token) for token in tokens) + r"(?!\w)",
        flags=re.IGNORECASE,
    )


def _alias_spans(text: str, aliases: Iterable[str]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for alias in aliases:
        pattern = _surface_pattern(alias)
        if pattern is not None:
            spans.extend((match.start(), match.end()) for match in pattern.finditer(text))
    return sorted(set(spans))


def _trigger_spans(patterns: Iterable[str], text: str) -> list[tuple[int, int]]:
    return [
        (match.start(), match.end())
        for pattern in patterns
        for match in re.finditer(pattern, text, flags=re.IGNORECASE)
    ]


def _links_endpoints(
    triggers: Iterable[tuple[int, int]],
    subjects: Iterable[tuple[int, int]],
    objects: Iterable[tuple[int, int]],
) -> bool:
    for subject in subjects:
        for object_ in objects:
            low, high = sorted(((subject[0] + subject[1]) / 2, (object_[0] + object_[1]) / 2))
            if any(end >= low and start <= high for start, end in triggers):
                return True
    return False


@dataclass(frozen=True)
class PredicateRule:
    """One dataset predicate and the type/evidence rules used to validate it."""

    name: str
    allowed_signatures: frozenset[tuple[str, str]]
    symmetric: bool = False
    description: str = ""
    explicit_patterns: tuple[str, ...] = ()
    weak_patterns: tuple[str, ...] = ()
    exclusion_patterns: tuple[str, ...] = ()
    relation_direction: str = "UNKNOWN"
    association_sign: str = "UNKNOWN"

    def normalized_signature(self, left_type: str, right_type: str) -> tuple[str, str]:
        pair = (str(left_type or ""), str(right_type or ""))
        return tuple(sorted(pair)) if self.symmetric else pair

    def valid(self, left_type: str, right_type: str) -> bool:
        return self.normalized_signature(left_type, right_type) in self.allowed_signatures

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed_signatures": [list(item) for item in sorted(self.allowed_signatures)],
            "symmetric": self.symmetric,
            "description": self.description,
            "explicit_patterns": list(self.explicit_patterns),
            "weak_patterns": list(self.weak_patterns),
            "exclusion_patterns": list(self.exclusion_patterns),
            "relation_direction": self.relation_direction,
            "association_sign": self.association_sign,
        }


@dataclass(frozen=True)
class SchemaProfile:
    """Read-only schema contract selected for one dataset or experiment."""

    name: str
    contract_version: str
    semantic_target: str
    predicates: Mapping[str, PredicateRule]
    entity_types: frozenset[str] = frozenset()
    entity_validation: str = "source_grounded"
    article_quality_mode: str = "none"
    source: str = "builtin"
    support_matcher_override: SupportMatcher | None = field(
        default=None, repr=False, compare=False,
    )

    def __post_init__(self) -> None:
        if not str(self.name).strip():
            raise ValueError("schema profile name is required")
        if not str(self.contract_version).strip():
            raise ValueError("schema profile contract_version is required")
        if self.semantic_target not in SUPPORTED_SEMANTIC_TARGETS:
            raise ValueError(
                f"unsupported semantic_target {self.semantic_target!r}; "
                f"expected one of {sorted(SUPPORTED_SEMANTIC_TARGETS)}"
            )
        if not self.predicates:
            raise ValueError("schema profile must define at least one predicate")
        if self.entity_validation not in SUPPORTED_ENTITY_VALIDATION:
            raise ValueError(
                f"unsupported entity_validation {self.entity_validation!r}; "
                f"expected one of {sorted(SUPPORTED_ENTITY_VALIDATION)}"
            )
        if self.article_quality_mode not in {"none", "liverkg"}:
            raise ValueError("article_quality_mode must be 'none' or 'liverkg'")

    def rule(self, predicate: str) -> PredicateRule | None:
        value = str(predicate or "")
        if value in self.predicates:
            return self.predicates[value]
        folded = value.casefold()
        return next(
            (rule for name, rule in self.predicates.items() if name.casefold() == folded),
            None,
        )

    def valid(self, predicate: str, left_type: str, right_type: str) -> bool:
        rule = self.rule(predicate)
        return bool(rule and rule.valid(left_type, right_type))

    def support_match(
        self,
        predicate: str,
        subject_type: str,
        object_type: str,
        evidence: str,
        *,
        subject_aliases: list[str] | None = None,
        object_aliases: list[str] | None = None,
    ) -> dict[str, Any]:
        if self.support_matcher_override is not None:
            return self.support_matcher_override(
                predicate,
                subject_type,
                object_type,
                evidence,
                subject_aliases=subject_aliases,
                object_aliases=object_aliases,
            )
        rule = self.rule(predicate)
        if rule is None:
            return {"match": "NONE", "reason_codes": ["unknown_dataset_predicate"]}
        if not rule.valid(subject_type, object_type):
            return {"match": "CONFLICT", "reason_codes": ["dataset_type_signature_mismatch"]}

        text = str(evidence or "")
        if any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in rule.exclusion_patterns):
            return {"match": "CONFLICT", "reason_codes": ["dataset_predicate_exclusion_cue"]}
        subject_spans = _alias_spans(text, subject_aliases or [])
        object_spans = _alias_spans(text, object_aliases or [])
        endpoints_closed = bool(subject_spans and object_spans)
        explicit_spans = _trigger_spans(rule.explicit_patterns, text)
        weak_spans = _trigger_spans(rule.weak_patterns, text)
        if endpoints_closed and _links_endpoints(explicit_spans, subject_spans, object_spans):
            return {"match": "EXPLICIT", "reason_codes": ["dataset_explicit_relation_support"]}
        if endpoints_closed and _links_endpoints(weak_spans, subject_spans, object_spans):
            return {"match": "WEAK", "reason_codes": ["dataset_weak_relation_support"]}
        if (explicit_spans or weak_spans) and (subject_spans or object_spans) and not endpoints_closed:
            return {
                "match": "WEAK",
                "reason_codes": ["dataset_trigger_endpoint_closure_external"],
            }
        reasons = ["dataset_predicate_support_absent"]
        if not endpoints_closed:
            reasons.append("support_endpoints_not_closed")
        elif explicit_spans or weak_spans:
            reasons.append("trigger_not_linking_endpoints")
        return {"match": "NONE", "reason_codes": reasons}

    def manifest(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "artifact_type": "dataset_schema_profile",
            "name": self.name,
            "contract_version": self.contract_version,
            "semantic_target": self.semantic_target,
            "entity_types": sorted(self.entity_types),
            "entity_validation": self.entity_validation,
            "article_quality_mode": self.article_quality_mode,
            "source": self.source,
            "predicates": {
                name: rule.to_dict() for name, rule in sorted(self.predicates.items())
            },
            # Candidate validation is intentionally not a database write grant.
            "authorizes_neo4j_write": False,
        }
        # File location is provenance, not contract semantics.  The same
        # checked-in profile copied to another machine must keep one hash.
        hash_payload = {key: value for key, value in payload.items() if key != "source"}
        encoded = json.dumps(hash_payload, sort_keys=True, separators=(",", ":"))
        payload["manifest_sha256"] = hashlib.sha256(encoded.encode()).hexdigest()
        return payload

    @property
    def manifest_sha256(self) -> str:
        return str(self.manifest()["manifest_sha256"])


def _validated_patterns(value: Any, *, field_name: str, predicate: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{predicate}.{field_name} must be a list of regex strings")
    for pattern in value:
        try:
            re.compile(pattern)
        except re.error as exc:
            raise ValueError(f"invalid regex in {predicate}.{field_name}: {pattern!r}") from exc
    return tuple(value)


def schema_profile_from_dict(payload: Mapping[str, Any], *, source: str = "memory") -> SchemaProfile:
    """Create a safe declarative profile; no executable Python is loaded."""

    if str(payload.get("artifact_type", "dataset_schema_profile")) != "dataset_schema_profile":
        raise ValueError("invalid schema profile artifact_type")
    raw_predicates = payload.get("predicates")
    if not isinstance(raw_predicates, Mapping) or not raw_predicates:
        raise ValueError("schema profile predicates must be a non-empty object")
    declared_types = frozenset(str(item) for item in (payload.get("entity_types") or []) if str(item))
    rules: dict[str, PredicateRule] = {}
    for raw_name, raw_rule in raw_predicates.items():
        name = str(raw_name or "").strip()
        if not name or not isinstance(raw_rule, Mapping):
            raise ValueError("each schema predicate must have a name and object definition")
        symmetric = bool(raw_rule.get("symmetric", False))
        raw_signatures = raw_rule.get("allowed_signatures")
        if not isinstance(raw_signatures, list) or not raw_signatures:
            raise ValueError(f"{name}.allowed_signatures must be a non-empty list")
        signatures: set[tuple[str, str]] = set()
        for signature in raw_signatures:
            if not isinstance(signature, (list, tuple)) or len(signature) != 2:
                raise ValueError(f"{name} contains an invalid endpoint signature")
            left, right = str(signature[0]), str(signature[1])
            if not left or not right:
                raise ValueError(f"{name} contains an empty endpoint type")
            if declared_types and (left not in declared_types or right not in declared_types):
                raise ValueError(f"{name} signature uses an undeclared entity type")
            pair = tuple(sorted((left, right))) if symmetric else (left, right)
            signatures.add(pair)
        rules[name] = PredicateRule(
            name=name,
            allowed_signatures=frozenset(signatures),
            symmetric=symmetric,
            description=str(raw_rule.get("description", "") or ""),
            explicit_patterns=_validated_patterns(
                raw_rule.get("explicit_patterns", []), field_name="explicit_patterns", predicate=name,
            ),
            weak_patterns=_validated_patterns(
                raw_rule.get("weak_patterns", []), field_name="weak_patterns", predicate=name,
            ),
            exclusion_patterns=_validated_patterns(
                raw_rule.get("exclusion_patterns", []), field_name="exclusion_patterns", predicate=name,
            ),
            relation_direction=str(raw_rule.get("relation_direction", "UNKNOWN") or "UNKNOWN"),
            association_sign=str(raw_rule.get("association_sign", "UNKNOWN") or "UNKNOWN"),
        )
    return SchemaProfile(
        name=str(payload.get("name", "") or "").strip(),
        contract_version=str(payload.get("contract_version", "") or "").strip(),
        semantic_target=str(payload.get("semantic_target", "CURRENT_FINDING") or "CURRENT_FINDING").upper(),
        predicates=rules,
        entity_types=declared_types,
        entity_validation=str(payload.get("entity_validation", "source_grounded") or "source_grounded"),
        article_quality_mode=str(payload.get("article_quality_mode", "none") or "none"),
        source=source,
    )


def load_schema_profile(path: str | Path) -> SchemaProfile:
    resolved = Path(path).expanduser().resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("schema profile must contain a JSON object")
    return schema_profile_from_dict(payload, source=str(resolved))


def liverkg_schema_profile() -> SchemaProfile:
    """Return the backwards-compatible built-in LiverKG candidate profile."""

    from cognitive_agent.schema.predicate_cards import RELATION_CARDS, predicate_support_match
    from cognitive_agent.schema.relation_signatures import LITERATURE_CANDIDATE_SIGNATURES

    rules = {
        predicate: PredicateRule(
            name=predicate,
            # Preserve the existing ordered candidate signatures exactly.
            allowed_signatures=frozenset(signatures),
            symmetric=False,
            description=RELATION_CARDS[predicate].description if predicate in RELATION_CARDS else "",
            explicit_patterns=(
                RELATION_CARDS[predicate].high_precision_patterns
                if predicate in RELATION_CARDS else ()
            ),
            weak_patterns=(
                RELATION_CARDS[predicate].trigger_patterns
                if predicate in RELATION_CARDS else ()
            ),
            exclusion_patterns=(
                RELATION_CARDS[predicate].exclusion_patterns
                if predicate in RELATION_CARDS else ()
            ),
            relation_direction=(
                RELATION_CARDS[predicate].relation_direction
                if predicate in RELATION_CARDS else "UNKNOWN"
            ),
        )
        for predicate, signatures in LITERATURE_CANDIDATE_SIGNATURES.items()
    }
    return SchemaProfile(
        name="LiverKG",
        contract_version="liverkg-candidate-v1",
        semantic_target="CURRENT_FINDING",
        predicates=rules,
        entity_types=frozenset(
            endpoint_type
            for signatures in LITERATURE_CANDIDATE_SIGNATURES.values()
            for signature in signatures
            for endpoint_type in signature
        ),
        entity_validation="liverkg_quality",
        article_quality_mode="liverkg",
        source="builtin:liverkg",
        support_matcher_override=predicate_support_match,
    )


def resolve_schema_profile(value: SchemaProfile | str | Path | None) -> SchemaProfile:
    if isinstance(value, SchemaProfile):
        return value
    if value:
        return load_schema_profile(value)
    return liverkg_schema_profile()
