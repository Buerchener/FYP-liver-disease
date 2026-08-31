"""BioRED native given-entity adapter.

BioRED stays in its official ontology.  This module provides strict PubTator
parsing, native predicate cards/signatures, stable Dev partitioning and the
official-Test release guard; it has no Neo4j dependency.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from cognitive_agent.candidate_lineage import edited_version, lineage_key, normalize_lineage
from cognitive_agent.evidence_pack import EvidencePackBuilder
from cognitive_agent.evidence_units import ArticleEvidenceReader
from cognitive_agent.extraction_quality import normalize_surface


BIORED_RELATION_TYPES = (
    "Association", "Bind", "Comparison", "Conversion", "Cotreatment",
    "Drug_Interaction", "Negative_Correlation", "Positive_Correlation",
)
BIORED_NOVELTY = ("Novel", "No", "Unknown")
BIORED_ASSOCIATION_LABELS = (
    "Association", "Positive_Correlation", "Negative_Correlation",
)


@dataclass(frozen=True)
class BioREDMention:
    start: int
    end: int
    text: str
    entity_type: str


@dataclass
class BioREDConcept:
    concept_id: str
    entity_type: str
    mentions: list[BioREDMention] = field(default_factory=list)


@dataclass(frozen=True)
class BioREDRelation:
    arg1_id: str
    arg2_id: str
    relation_type: str
    novelty: str = "Unknown"

    @property
    def label_key(self) -> tuple[str, str, str]:
        left, right = sorted((self.arg1_id, self.arg2_id))
        return left, right, self.relation_type

    @property
    def pair(self) -> tuple[str, str]:
        return tuple(sorted((self.arg1_id, self.arg2_id)))

    @property
    def novelty_key(self) -> tuple[str, str, str, str]:
        return (*self.pair, self.relation_type, self.novelty)


@dataclass
class BioREDDocument:
    pmid: str
    title: str
    abstract: str
    text: str
    concepts: dict[str, BioREDConcept]
    relations: list[BioREDRelation]


@dataclass(frozen=True)
class BioREDPredicateCard:
    relation_type: str
    allowed_signatures: frozenset[tuple[str, str]]
    description: str = ""
    boundary_note: str = ""
    relation_direction: str = "NON_DIRECTIONAL"
    association_sign: str = "UNKNOWN"
    explicit_patterns: tuple[str, ...] = ()
    weak_patterns: tuple[str, ...] = ()
    exclusion_patterns: tuple[str, ...] = ()


BIORED_CARD_PATTERNS: dict[str, dict[str, tuple[str, ...]]] = {
    "Association": {
        "explicit": (
            r"\bassociat\w+\b", r"\blinked?\s+(?:to|with)\b",
            r"\brelated\s+to\b", r"\brisk\s+(?:of|for)\b",
            r"\b(?:mutation|variant)s?\b.{0,100}\b(?:cause|caused|identified|found|observed|reported)\b",
        ),
        "weak": (
            r"\b(?:identified|found|observed|detected|carrying|harbou?r(?:ing|ed)?)\b",
            r"\b(?:mutation|variant|genotype|phenotype|expression)\b",
        ),
    },
    "Positive_Correlation": {
        "explicit": (
            r"\bpositive(?:ly)?\s+correlat\w*\b", r"\bincreas\w*\b",
            r"\belevat\w*\b", r"\bhigher\b", r"\bupregulat\w*\b",
            r"\bpromot\w*\b", r"\benhanc\w*\b", r"\bgreater\b",
            r"\binduc\w*\b", r"\bcaus\w*\b", r"\bresponsible\s+for\b",
            r"\badverse events?\b.{0,180}\b(?:following|after)\s+treatment\s+with\b",
            r"\breported\s+for\s+patients?\s+treated\s+with\b",
        ),
        "weak": (r"\bcorrelat\w*\b", r"\bassociat\w*\b", r"\brisk\b"),
    },
    "Negative_Correlation": {
        "explicit": (
            r"\bnegative(?:ly)?\s+correlat\w*\b", r"\binverse(?:ly)?\b",
            r"\bdecreas\w*\b", r"\breduc\w*\b", r"\blower\b",
            r"\bdownregulat\w*\b", r"\binhibit\w*\b", r"\bblock\w*\b",
            r"\bdiminish\w*\b", r"\bprotect\w*\b", r"\battenuat\w*\b",
            r"\balleviat\w*\b", r"\bameliorat\w*\b", r"\bimprov\w*\b",
            r"\bcorrect\w*\b", r"\brevers\w*\b", r"\brescu\w*\b",
            r"\bfail\w*\s+to\b", r"\bloss\s+of\b", r"\bdeficien\w*\b",
        ),
        "weak": (r"\bcorrelat\w*\b", r"\bassociat\w*\b", r"\baffect\w*\b"),
    },
    "Bind": {
        "explicit": (
            r"\bbind\w*\b", r"\bbound\s+to\b", r"\binteract\w*\b",
            r"\bcomplex(?:es)?\s+with\b", r"\bcoimmunoprecipitat\w*\b",
        ),
        "weak": (r"\bphysical(?:ly)?\b", r"\baffinity\b", r"\breceptor\b"),
    },
    "Comparison": {
        "explicit": (
            r"\bcompar\w*\b", r"\bversus\b", r"\bcompared\s+with\b",
            r"\bthan\b", r"\bno\s+(?:significant\s+)?difference\b",
        ),
        "weak": (r"\bdifference\b", r"\bsimilar\b"),
    },
    "Conversion": {
        "explicit": (
            r"\bconvert\w*\s+(?:to|into)\b", r"\bmetaboli[sz]\w*\s+(?:to|into)\b",
            r"\btransform\w*\s+(?:to|into)\b", r"\bproduct\s+of\b",
        ),
        "weak": (r"\bconversion\b", r"\bmetaboli[st]\w*\b"),
    },
    "Cotreatment": {
        "explicit": (
            r"\bco[- ]?(?:treat|administ)\w*\b", r"\bcombined?\s+(?:with|treatment|therapy)\b",
            r"\bcombination\s+(?:of|with|therapy|treatment)\b", r"\btogether\s+with\b",
        ),
        "weak": (r"\bconcomitant\w*\b", r"\badditional\b.{0,80}\b(?:mg|treatment|therapy)\b"),
    },
    "Drug_Interaction": {
        "explicit": (
            r"\bdrug[- ]drug\s+interaction\b", r"\bpharmacokinetic\s+interaction\b",
            r"\binteract\w*\s+with\b", r"\bcoadministr\w*\b",
        ),
        "weak": (
            r"\b(?:clearance|exposure|bioavailability|concentration)s?\b.{0,100}\baffect\w*\b",
            r"\b(?:inhibitor|inducer)\b",
        ),
    },
}

BIORED_CARD_DEFINITIONS: dict[str, tuple[str, str]] = {
    "Association": (
        "The article asserts a direct biomedical association but does not state a signed positive or negative effect.",
        "Do not use when the text clearly states increase, decrease, induction, inhibition, protection, or another signed effect.",
    ),
    "Positive_Correlation": (
        "One endpoint increases, induces, promotes, causes, elevates, exacerbates, or is positively correlated with the other.",
        "The sign must be supported by the article; generic co-occurrence or an unsigned association is Association.",
    ),
    "Negative_Correlation": (
        "One endpoint decreases, inhibits, blocks, protects against, attenuates, or is inversely correlated with the other.",
        "Treatment of a condition is not automatically negative unless an improving or suppressive effect is asserted.",
    ),
    "Bind": (
        "The endpoints physically bind, form a molecular complex, or directly interact at the molecular level.",
        "Functional regulation or statistical association without physical interaction is not Bind.",
    ),
    "Comparison": (
        "The article directly compares two biomedical entities, interventions, or effects.",
        "A sentence mentioning two entities without a comparative assertion is not Comparison.",
    ),
    "Conversion": (
        "One chemical or biomedical entity is converted, metabolized, or transformed into the other.",
        "Pathway participation or co-occurrence is not Conversion.",
    ),
    "Cotreatment": (
        "Two treatments or chemicals are jointly administered or explicitly used as a combination therapy.",
        "Sequential mention or a drug-drug interaction without joint treatment is not Cotreatment.",
    ),
    "Drug_Interaction": (
        "Two drugs have an asserted pharmacokinetic or pharmacodynamic interaction.",
        "Simple coadministration is Cotreatment unless an interaction or altered exposure/effect is stated.",
    ),
}

# These boundaries are transcribed from the public BioRED annotation guideline,
# not inferred from Dev labels.  The same surface cue has different meanings for
# different endpoint signatures, so the generic RelationCard alone is not a
# sufficient native-label contract.
BIORED_SIGNATURE_BOUNDARIES: dict[tuple[str, str], tuple[str, ...]] = {
    ("ChemicalEntity", "DiseaseOrPhenotypicFeature"): (
        "Positive_Correlation: the chemical induces/causes the disease, raises its risk, or higher exposure is positively correlated with it.",
        "Negative_Correlation: the chemical treats/prevents the disease or reduces disease susceptibility.",
        "Association: the direct chemical-disease relation is explicit but its sign is not clear.",
    ),
    ("DiseaseOrPhenotypicFeature", "GeneOrGeneProduct"): (
        "Association is the default for an associated gene and for a corresponding gene inherited from a confirmed variant-disease relation.",
        "Do not assign a sign from an increase/decrease word unless it describes the target gene's effect on the target disease.",
        "Per the guideline, loss/knockout of a gene preventing disease supports Positive_Correlation for the intact gene, while loss/knockout causing disease supports Negative_Correlation; do not map the surface word reduce directly to Negative.",
        "A perturbed/knocked-down gene is not the same endpoint state as the intact gene; ambiguous perturbation scope stays Association or REVIEW.",
    ),
    ("DiseaseOrPhenotypicFeature", "SequenceVariant"): (
        "Positive_Correlation requires that the variant causes, predisposes to, significantly contributes to, or raises susceptibility/risk of the disease.",
        "Negative_Correlation requires that the variant decreases disease risk.",
        "Association covers a variant observed/carried in affected patients when causation or signed risk is not established.",
    ),
    ("GeneOrGeneProduct", "GeneOrGeneProduct"): (
        "Bind requires physical interaction, receptor binding, or two or more proteins explicitly belonging to a complex.",
        "Positive/Negative_Correlation requires target-pair up/down regulation or signed expression correlation.",
        "Modification and an otherwise unsigned functional relation are Association.",
    ),
    ("ChemicalEntity", "GeneOrGeneProduct"): (
        "Bind covers a chemical receptor or direct binding; signed expression, response, sensitivity, or resistance uses Positive/Negative_Correlation.",
        "An unsigned direct chemical-gene relation is Association.",
    ),
    ("ChemicalEntity", "ChemicalEntity"): (
        "Cotreatment is joint/combination treatment; Drug_Interaction is an asserted pharmacokinetic or pharmacodynamic interaction.",
        "Conversion is actual conversion of one chemical to the other; signed changes in sensitivity/effect use Positive/Negative_Correlation.",
        "Do not infer a pair relation merely because two chemicals affect a third endpoint in the same sentence.",
    ),
    ("ChemicalEntity", "SequenceVariant"): (
        "Positive/Negative_Correlation requires variant-specific sensitivity, response, resistance, or adverse-effect evidence.",
        "Association is used when the confirmed chemical-variant relation is unsigned.",
    ),
    ("GeneOrGeneProduct", "SequenceVariant"): (
        "Use only the labels allowed by BioRED Train for this rare signature; gene membership alone is not a free relation edge.",
    ),
    ("SequenceVariant", "SequenceVariant"): (
        "Use Association only for an explicitly asserted relation between the two variants; shared gene membership is insufficient.",
    ),
}


class BioREDPredicateRegistry:
    semantic_target = "RELATION_TRUTH"

    def __init__(
        self, signatures: dict[str, Iterable[tuple[str, str]]],
        *, signature_label_counts: dict[tuple[str, str], dict[str, int]] | None = None,
    ):
        self.signature_label_counts = {
            tuple(sorted((str(signature[0]), str(signature[1])))): {
                str(label): int(count) for label, count in counts.items()
                if str(label) in BIORED_RELATION_TYPES and int(count) > 0
            }
            for signature, counts in (signature_label_counts or {}).items()
        }
        self.cards = {
            label: BioREDPredicateCard(
                relation_type=label,
                allowed_signatures=frozenset(
                    tuple(sorted((str(left), str(right))))
                    for left, right in signatures.get(label, [])
                ),
                description=BIORED_CARD_DEFINITIONS.get(label, ("", ""))[0],
                boundary_note=BIORED_CARD_DEFINITIONS.get(label, ("", ""))[1],
                association_sign=(
                    "POSITIVE" if label == "Positive_Correlation"
                    else "NEGATIVE" if label == "Negative_Correlation" else "UNKNOWN"
                ),
                explicit_patterns=BIORED_CARD_PATTERNS.get(label, {}).get("explicit", ()),
                weak_patterns=BIORED_CARD_PATTERNS.get(label, {}).get("weak", ()),
                exclusion_patterns=BIORED_CARD_PATTERNS.get(label, {}).get("exclusion", ()),
            )
            for label in BIORED_RELATION_TYPES
        }

    def valid(self, label: str, left_type: str, right_type: str) -> bool:
        card = self.cards.get(label)
        return bool(card and tuple(sorted((left_type, right_type))) in card.allowed_signatures)

    def label_priors(self, left_type: str, right_type: str) -> dict[str, float]:
        counts = self.signature_label_counts.get(tuple(sorted((left_type, right_type))), {})
        total = sum(counts.values())
        if not total:
            return {}
        return {
            label: count / total for label, count in sorted(counts.items())
        }

    def dominant_label(
        self, left_type: str, right_type: str, *, min_support: int = 100,
        min_fraction: float = 0.90,
    ) -> str:
        counts = self.signature_label_counts.get(tuple(sorted((left_type, right_type))), {})
        total = sum(counts.values())
        if total < min_support or not counts:
            return ""
        label, count = max(counts.items(), key=lambda item: (item[1], item[0]))
        return label if count / total >= min_fraction else ""

    def manifest(self, *, training_sha256: str) -> dict[str, Any]:
        payload = {
            "registry_contract_version": "biored-native-v2",
            "semantic_target": self.semantic_target,
            "training_sha256": training_sha256,
            "cards": {
                label: {
                    "allowed_signatures": [list(item) for item in sorted(card.allowed_signatures)],
                    "relation_direction": card.relation_direction,
                    "association_sign": card.association_sign,
                    "description": card.description,
                    "boundary_note": card.boundary_note,
                    "explicit_patterns": list(card.explicit_patterns),
                    "weak_patterns": list(card.weak_patterns),
                    "exclusion_patterns": list(card.exclusion_patterns),
                }
                for label, card in sorted(self.cards.items())
            },
            "signature_label_counts": {
                "|".join(signature): dict(sorted(counts.items()))
                for signature, counts in sorted(self.signature_label_counts.items())
            },
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        payload["manifest_sha256"] = hashlib.sha256(encoded.encode()).hexdigest()
        return payload

    def prompt_cards(self) -> list[dict[str, Any]]:
        return [
            {
                "relation_type": label,
                "definition": self.cards[label].description,
                "boundary": self.cards[label].boundary_note,
            }
            for label in BIORED_RELATION_TYPES
        ]

    def prompt_contract(self, left_type: str, right_type: str) -> dict[str, Any]:
        """Return the Train/guideline contract for one endpoint signature."""
        signature = tuple(sorted((str(left_type), str(right_type))))
        allowed = [
            label for label in BIORED_RELATION_TYPES
            if self.valid(label, *signature)
        ]
        return {
            "endpoint_signature": list(signature),
            "allowed_relation_types": allowed,
            "train_label_priors": self.label_priors(*signature),
            "signature_boundaries": list(
                BIORED_SIGNATURE_BOUNDARIES.get(signature, ())
            ),
            "relation_cards": [
                {
                    "relation_type": label,
                    "definition": self.cards[label].description,
                    "boundary": self.cards[label].boundary_note,
                }
                for label in allowed
            ],
        }

    def to_schema_profile(self, *, source: str = "builtin:biored-train"):
        """Expose the native registry through the shared verifier contract.

        BioRED signatures remain native and symmetric; they are never mapped
        into LiverKG predicates or its database write contract.
        """
        from cognitive_agent.schema.schema_profile import PredicateRule, SchemaProfile

        rules = {
            label: PredicateRule(
                name=label,
                allowed_signatures=card.allowed_signatures,
                symmetric=True,
                description=card.description,
                explicit_patterns=card.explicit_patterns,
                weak_patterns=card.weak_patterns,
                exclusion_patterns=card.exclusion_patterns,
                relation_direction=card.relation_direction,
                association_sign=card.association_sign,
            )
            for label, card in self.cards.items()
        }
        return SchemaProfile(
            name="BioRED",
            contract_version="biored-native-v2",
            semantic_target=self.semantic_target,
            predicates=rules,
            entity_types=frozenset(
                endpoint_type
                for card in self.cards.values()
                for signature in card.allowed_signatures
                for endpoint_type in signature
            ),
            entity_validation="given_entity",
            article_quality_mode="none",
            source=source,
            support_matcher_override=make_biored_support_matcher(self),
        )


def parse_biored_pubtator(path: str | Path) -> list[BioREDDocument]:
    path = Path(path)
    documents: list[BioREDDocument] = []
    raw = path.read_text(encoding="utf-8").strip()
    for block in re.split(r"\n\s*\n", raw):
        lines = [line for line in block.splitlines() if line.strip()]
        if len(lines) < 2 or "|t|" not in lines[0] or "|a|" not in lines[1]:
            continue
        pmid, title = lines[0].split("|t|", 1)
        abstract_pmid, abstract = lines[1].split("|a|", 1)
        if abstract_pmid != pmid:
            raise ValueError(f"PubTator PMID mismatch: {pmid} != {abstract_pmid}")
        text = title + " " + abstract
        concepts: dict[str, BioREDConcept] = {}
        relations: list[BioREDRelation] = []
        for line in lines[2:]:
            fields = line.split("\t")
            if len(fields) == 6 and fields[1].isdigit() and fields[2].isdigit():
                row_pmid, start, end, mention, entity_type, raw_ids = fields
                if row_pmid != pmid:
                    raise ValueError(f"annotation PMID mismatch in {pmid}")
                start_i, end_i = int(start), int(end)
                if start_i < 0 or end_i > len(text) or start_i >= end_i:
                    raise ValueError(f"invalid mention offset in {pmid}: {start_i}:{end_i}")
                # Official offsets are authoritative; retain the supplied
                # mention but audit offset disagreement instead of rewriting.
                for concept_id in raw_ids.split(","):
                    concept_id = concept_id.strip()
                    if not concept_id or concept_id == "-":
                        continue
                    concept = concepts.setdefault(
                        concept_id, BioREDConcept(concept_id, entity_type)
                    )
                    if concept.entity_type != entity_type:
                        raise ValueError(f"concept type conflict {pmid}:{concept_id}")
                    item = BioREDMention(start_i, end_i, mention, entity_type)
                    if item not in concept.mentions:
                        concept.mentions.append(item)
            elif len(fields) == 5 and fields[1] in BIORED_RELATION_TYPES:
                row_pmid, label, left, right, novelty = fields
                if row_pmid != pmid:
                    raise ValueError(f"relation PMID mismatch in {pmid}")
                if left not in concepts or right not in concepts:
                    raise ValueError(f"invalid BioRED relation endpoint in {pmid}")
                relations.append(BioREDRelation(
                    left, right, label, novelty if novelty in {"Novel", "No"} else "Unknown"
                ))
        documents.append(BioREDDocument(
            pmid=pmid, title=title, abstract=abstract, text=text,
            concepts=concepts, relations=relations,
        ))
    return documents


def registry_from_training(path: str | Path) -> tuple[BioREDPredicateRegistry, dict[str, Any]]:
    path = Path(path)
    documents = parse_biored_pubtator(path)
    signatures: dict[str, set[tuple[str, str]]] = {
        label: set() for label in BIORED_RELATION_TYPES
    }
    signature_label_counts: dict[tuple[str, str], dict[str, int]] = {}
    for document in documents:
        for relation in document.relations:
            signature = tuple(sorted((
                document.concepts[relation.arg1_id].entity_type,
                document.concepts[relation.arg2_id].entity_type,
            )))
            signatures[relation.relation_type].add(signature)
            counts = signature_label_counts.setdefault(signature, {})
            counts[relation.relation_type] = counts.get(relation.relation_type, 0) + 1
    registry = BioREDPredicateRegistry(
        signatures, signature_label_counts=signature_label_counts,
    )
    training_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    return registry, registry.manifest(training_sha256=training_sha256)


def _stable_id(prefix: str, *parts: str) -> str:
    payload = "\x1f".join(str(part) for part in parts)
    return f"{prefix}-{hashlib.sha256(payload.encode()).hexdigest()[:20]}"


def concept_aliases(concept: BioREDConcept) -> list[str]:
    """Return source aliases plus safe notation variants.

    Variant annotations frequently use ``c.835del18`` while model evidence
    quotes use ``835del18``.  Dropping only the formal c./g./p. prefix is a
    notation normalization, not a biomedical synonym expansion.
    """
    aliases: list[str] = []
    for mention in concept.mentions:
        text = str(mention.text or "").strip()
        if not text:
            continue
        aliases.append(text)
        stripped = re.sub(r"^(?:c|g|p)\.\s*", "", text, flags=re.IGNORECASE)
        if stripped != text:
            aliases.append(stripped)
        aliases.append(text.replace(" ", ""))
    return list(dict.fromkeys(item for item in aliases if item))


def _alias_present(text: str, aliases: Iterable[str]) -> bool:
    normalized = normalize_surface(text)
    compact = re.sub(r"\W+", "", str(text or "").casefold())
    for alias in aliases:
        alias_normalized = normalize_surface(alias)
        alias_compact = re.sub(r"\W+", "", str(alias or "").casefold())
        if alias_normalized and alias_normalized in normalized:
            return True
        if len(alias_compact) >= 4 and alias_compact in compact:
            return True
    return False


def make_biored_support_matcher(registry: BioREDPredicateRegistry):
    canonical = {label.casefold(): label for label in BIORED_RELATION_TYPES}

    def match(
        relation_type: str,
        left_type: str,
        right_type: str,
        evidence: str,
        *,
        subject_aliases: list[str] | None = None,
        object_aliases: list[str] | None = None,
    ) -> dict[str, Any]:
        label = canonical.get(str(relation_type or "").casefold(), "")
        card = registry.cards.get(label)
        if card is None:
            return {"match": "NONE", "reason_codes": ["unknown_native_relation_type"]}
        if not registry.valid(label, left_type, right_type):
            return {"match": "CONFLICT", "reason_codes": ["native_type_signature_mismatch"]}
        text = str(evidence or "")
        left_present = _alias_present(text, subject_aliases or [])
        right_present = _alias_present(text, object_aliases or [])
        if any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in card.exclusion_patterns):
            return {"match": "CONFLICT", "reason_codes": ["native_relation_exclusion_cue"]}
        explicit = any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in card.explicit_patterns)
        weak = any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in card.weak_patterns)
        if not (left_present and right_present):
            if (left_present or right_present) and (explicit or weak):
                return {
                    "match": "WEAK",
                    "reason_codes": ["native_trigger_endpoint_closure_external"],
                }
            return {"match": "NONE", "reason_codes": ["support_endpoints_not_closed"]}
        if explicit:
            return {"match": "EXPLICIT", "reason_codes": ["native_explicit_relation_support"]}
        if weak:
            return {"match": "WEAK", "reason_codes": ["native_weak_relation_support"]}
        return {"match": "NONE", "reason_codes": ["native_type_only_support"]}

    return match


def native_label_support_profile(
    value: dict[str, Any], document: BioREDDocument,
    registry: BioREDPredicateRegistry,
) -> dict[str, dict[str, Any]]:
    """Compare the three easily-confused native labels on identical spans.

    The profile is diagnostic only.  Train calibration shows that words such
    as ``increase`` and ``decrease`` are not precise enough to relabel a target
    concept pair on their own, so this function never edits a candidate.
    """
    left_id = str(value.get("arg1_id", "") or "")
    right_id = str(value.get("arg2_id", "") or "")
    if left_id not in document.concepts or right_id not in document.concepts:
        return {}
    left, right = document.concepts[left_id], document.concepts[right_id]
    pack = dict(value.get("evidence_pack", {}) or {})
    minimal_ids = set(pack.get("minimal_support_span_ids", []) or [])
    spans = [
        item for item in list(pack.get("spans", []) or [])
        if not minimal_ids or str(item.get("span_id", "")) in minimal_ids
    ]
    if not spans:
        evidence = str(
            value.get("evidence_quote", "")
            or value.get("evidence_window", "") or ""
        )
        spans = [{"span_id": "candidate-evidence", "text": evidence, "role": "OWNER"}]
    matcher = make_biored_support_matcher(registry)
    rank = {"CONFLICT": 0, "NONE": 1, "WEAK": 2, "EXPLICIT": 3}
    output: dict[str, dict[str, Any]] = {}
    for label in BIORED_RELATION_TYPES:
        if not registry.valid(label, left.entity_type, right.entity_type):
            continue
        best: dict[str, Any] | None = None
        for span in spans:
            result = matcher(
                label, left.entity_type, right.entity_type,
                str(span.get("text", "") or ""),
                subject_aliases=concept_aliases(left),
                object_aliases=concept_aliases(right),
            )
            candidate = {
                "match": str(result.get("match", "NONE") or "NONE"),
                "reason_codes": list(result.get("reason_codes", []) or []),
                "span_ids": [str(span.get("span_id", "") or "")],
            }
            if best is None or rank.get(candidate["match"], 0) > rank.get(best["match"], 0):
                best = candidate
        output[label] = best or {
            "match": "NONE", "reason_codes": ["no_candidate_evidence"], "span_ids": [],
        }
    return output


def build_native_adjudication_spans(
    value: dict[str, Any], document: BioREDDocument,
    registry: BioREDPredicateRegistry, *, max_spans: int = 8,
) -> list[dict[str, Any]]:
    """Retrieve pair-local, exact-offset evidence without consulting Gold.

    EvidencePack's minimal support set answers a structural closure question.
    Native BioRED label decisions additionally need pair-local context because
    the guideline permits cross-sentence and derived gene/variant relations.
    This retriever follows official mention offsets and keeps a small set of
    endpoint sentences plus their immediate bridges; it never invents text.
    """
    left_id = str(value.get("arg1_id", "") or "")
    right_id = str(value.get("arg2_id", "") or "")
    if left_id not in document.concepts or right_id not in document.concepts:
        return []
    sentences, indexes = _sentence_lattice(document)
    left_indexes = set(indexes.get(left_id, set()))
    right_indexes = set(indexes.get(right_id, set()))
    if not left_indexes and not right_indexes:
        return []
    candidate_indexes = set(left_indexes) | set(right_indexes)
    for index in list(candidate_indexes):
        if index > 0:
            candidate_indexes.add(index - 1)
        if index + 1 < len(sentences):
            candidate_indexes.add(index + 1)
    left = document.concepts[left_id]
    right = document.concepts[right_id]
    left_aliases, right_aliases = concept_aliases(left), concept_aliases(right)
    allowed = [
        label for label in BIORED_RELATION_TYPES
        if registry.valid(label, left.entity_type, right.entity_type)
    ]
    matcher = make_biored_support_matcher(registry)
    match_rank = {"CONFLICT": -1, "NONE": 0, "WEAK": 1, "EXPLICIT": 2}
    primary = normalize_surface(str(
        value.get("evidence_quote", "") or value.get("evidence_window", "") or ""
    ))
    rows: list[tuple[tuple[int, ...], dict[str, Any]]] = []
    for index in sorted(candidate_indexes):
        sentence = sentences[index]
        text = str(sentence.text or "")
        subject_covered = index in left_indexes
        object_covered = index in right_indexes
        label_matches: dict[str, dict[str, Any]] = {}
        for label in allowed:
            result = matcher(
                label, left.entity_type, right.entity_type, text,
                subject_aliases=left_aliases, object_aliases=right_aliases,
            )
            label_matches[label] = {
                "match": str(result.get("match", "NONE") or "NONE"),
                "reason_codes": list(result.get("reason_codes", []) or []),
            }
        current = label_matches.get(str(value.get("relation_type", "") or ""), {})
        best_match = max(
            (match_rank.get(item.get("match", "NONE"), 0) for item in label_matches.values()),
            default=0,
        )
        normalized = normalize_surface(text)
        primary_overlap = bool(
            primary and (normalized in primary or primary in normalized)
        )
        endpoint_count = int(subject_covered) + int(object_covered)
        role = (
            "BOTH_ENDPOINTS" if endpoint_count == 2
            else "SUBJECT_ONLY" if subject_covered
            else "OBJECT_ONLY" if object_covered else "BRIDGE_CONTEXT"
        )
        row = {
            "span_id": f"ep-{sentence.parent_sentence_id}-{sentence.char_start}-{sentence.char_end}",
            "text": text,
            "char_start": int(sentence.char_start),
            "char_end": int(sentence.char_end),
            "sentence_id": str(sentence.parent_sentence_id),
            "role": "OWNER",
            "alignment_status": "MATCH_EXACT",
            "subject_covered": subject_covered,
            "object_covered": object_covered,
            "trigger_match": str(current.get("match", "NONE") or "NONE"),
            "trigger_reason_codes": list(current.get("reason_codes", []) or []),
            "pair_context_role": role,
            "primary_evidence_overlap": primary_overlap,
            "label_support_profile": label_matches,
        }
        score = (
            -int(endpoint_count == 2), -int(primary_overlap), -best_match,
            -endpoint_count, len(text), index,
        )
        rows.append((score, row))

    selected: list[dict[str, Any]] = []
    # First preserve the shortest closest subject/object bridge when no single
    # sentence contains both endpoints.
    if not any(item[1]["subject_covered"] and item[1]["object_covered"] for item in rows):
        bridge_pairs = [
            (abs(left_index - right_index), left_index, right_index)
            for left_index in left_indexes for right_index in right_indexes
        ]
        if bridge_pairs:
            _, left_index, right_index = min(bridge_pairs)
            wanted = {left_index, right_index}
            selected.extend(
                row for _, row in rows
                if any(
                    sentence.parent_sentence_id == row["sentence_id"]
                    for i, sentence in enumerate(sentences) if i in wanted
                )
            )
    for _, row in sorted(rows, key=lambda item: item[0]):
        if row not in selected:
            selected.append(row)
        if len(selected) >= max(1, int(max_spans)):
            break
    return selected[:max(1, int(max_spans))]


def build_native_label_review_items(
    values: Iterable[dict[str, Any]], document: BioREDDocument,
    registry: BioREDPredicateRegistry,
) -> list[dict[str, Any]]:
    """Build ID-bound evidence packets for native-label adjudication."""
    reviews: list[dict[str, Any]] = []
    for value in values:
        left_id = str(value.get("arg1_id", "") or "")
        right_id = str(value.get("arg2_id", "") or "")
        if left_id not in document.concepts or right_id not in document.concepts:
            continue
        pack = dict(value.get("evidence_pack", {}) or {})
        minimal_ids = set(pack.get("minimal_support_span_ids", []) or [])
        spans = [
            {
                "span_id": str(item.get("span_id", "") or ""),
                "text": str(item.get("text", "") or ""),
                "role": str(item.get("role", "") or ""),
                "subject_covered": bool(item.get("subject_covered")),
                "object_covered": bool(item.get("object_covered")),
            }
            for item in list(pack.get("spans", []) or [])
            if str(item.get("span_id", "") or "") in minimal_ids
        ]
        if not spans:
            # A lexical RelationCard trigger is not required to give the
            # structured adjudicator source-grounded evidence.  Pick the
            # smallest owner-only endpoint closure, while leaving the formal
            # EvidencePack support_mode unchanged and fully auditable.
            owner_spans = [
                item for item in list(pack.get("spans", []) or [])
                if str(item.get("role", "") or "") == "OWNER"
            ]
            primary_text = normalize_surface(str(
                value.get("evidence_quote", "")
                or value.get("evidence_window", "")
                or value.get("evidence", "") or ""
            ))
            trigger_rank = {"EXPLICIT": 2, "WEAK": 1, "NONE": 0, "CONFLICT": -1}
            ordered = sorted(owner_spans, key=lambda item: (
                -int(bool(primary_text) and normalize_surface(str(item.get("text", "") or "")) == primary_text),
                -int(bool(item.get("subject_covered")) and bool(item.get("object_covered"))),
                -trigger_rank.get(str(item.get("trigger_match", "NONE") or "NONE"), 0),
                max(0, int(item.get("char_end", 0)) - int(item.get("char_start", 0))),
                str(item.get("span_id", "")),
            ))
            closure: tuple[dict[str, Any], ...] | None = None
            for size in range(1, min(3, len(owner_spans)) + 1):
                candidates = [
                    combo for combo in itertools.combinations(owner_spans, size)
                    if any(bool(item.get("subject_covered")) for item in combo)
                    and any(bool(item.get("object_covered")) for item in combo)
                ]
                if candidates:
                    closure = min(candidates, key=lambda combo: tuple(
                        ordered.index(item) for item in combo
                    ))
                    break
            selected = list(closure or ())
            for item in ordered:
                if item not in selected and len(selected) < 3:
                    selected.append(item)
            spans = [
                {
                    "span_id": str(item.get("span_id", "") or ""),
                    "text": str(item.get("text", "") or ""),
                    "role": str(item.get("role", "") or ""),
                    "subject_covered": bool(item.get("subject_covered")),
                    "object_covered": bool(item.get("object_covered")),
                }
                for item in selected
            ]
        adjudication_spans = list(pack.get("adjudication_spans", []) or [])
        by_span_id = {
            str(item.get("span_id", "") or ""): {
                "span_id": str(item.get("span_id", "") or ""),
                "text": str(item.get("text", "") or ""),
                "role": str(item.get("role", "") or ""),
                "subject_covered": bool(item.get("subject_covered")),
                "object_covered": bool(item.get("object_covered")),
                "pair_context_role": str(item.get("pair_context_role", "") or ""),
                "label_support_profile": dict(item.get("label_support_profile", {}) or {}),
            }
            for item in [*spans, *adjudication_spans]
            if str(item.get("span_id", "") or "")
        }
        spans = list(by_span_id.values())
        left, right = document.concepts[left_id], document.concepts[right_id]
        allowed = [
            label for label in BIORED_RELATION_TYPES
            if registry.valid(label, left.entity_type, right.entity_type)
        ]
        reviews.append({
            "candidate_id": str(value.get("candidate_id", "") or ""),
            "candidate_version": int(value.get("candidate_version", 1) or 1),
            "pair_candidate_id": str(value.get("pair_candidate_id", "") or ""),
            "arg1_id": left_id,
            "arg2_id": right_id,
            "arg1_type": left.entity_type,
            "arg2_type": right.entity_type,
            "arg1_mentions": concept_aliases(left),
            "arg2_mentions": concept_aliases(right),
            "current_relation_type": str(value.get("relation_type", "") or ""),
            "untrusted_current_relation_type": str(value.get("relation_type", "") or ""),
            "allowed_relation_types": allowed,
            "minimal_support_spans": spans,
            "support_profile": native_label_support_profile(value, document, registry),
            "signature_contract": registry.prompt_contract(
                left.entity_type, right.entity_type,
            ),
            "candidate_confidence": float(value.get("confidence", 0.0) or 0.0),
            "candidate_lane": str(value.get("candidate_lane", "") or ""),
        })
    return reviews


def apply_train_signature_prior_guard(
    values: Iterable[dict[str, Any]], document: BioREDDocument,
    registry: BioREDPredicateRegistry,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Version non-explicit labels when Train has a >=90% type prior.

    The guard is intentionally narrow: only Association/sign labels are
    eligible, explicit predicate evidence always wins, and the prior is
    derived solely from BioRED Train.  Dev labels never enter the decision.
    """
    output: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    for raw in values:
        value = dict(raw)
        left_id, right_id = str(value.get("arg1_id", "")), str(value.get("arg2_id", ""))
        left, right = document.concepts[left_id], document.concepts[right_id]
        current = str(value.get("relation_type", "") or "")
        dominant = registry.dominant_label(left.entity_type, right.entity_type)
        should_edit = bool(
            dominant and dominant != current
            and current in BIORED_ASSOCIATION_LABELS
            and dominant in BIORED_ASSOCIATION_LABELS
            and str(value.get("relation_card_match", "NONE")) != "EXPLICIT"
        )
        if not should_edit:
            output.append(value)
            continue
        edited = edited_version(value, reason_code="train_signature_dominant_label")
        edited["relation_type"] = dominant
        edited["association_sign"] = registry.cards[dominant].association_sign
        edited["train_signature_prior"] = {
            "contract_version": "biored-train-signature-prior-v1",
            "original_relation_type": current,
            "final_relation_type": dominant,
            "label_priors": registry.label_priors(left.entity_type, right.entity_type),
            "explicit_evidence_override": False,
        }
        lane = str(edited.get("candidate_lane", "extracted_hint") or "extracted_hint")
        edited = enrich_native_relation(edited, document, registry, lane=lane)
        edited["quality_flags"] = sorted(
            set(edited.get("quality_flags", []) or []) | {"train_signature_prior_applied"}
        )
        output.append(edited)
        audit.append({
            "candidate_id": edited.get("candidate_id", ""),
            "candidate_version_before": value.get("candidate_version", 1),
            "candidate_version_after": edited.get("candidate_version", 2),
            "original_relation_type": current,
            "final_relation_type": dominant,
            "label_priors": registry.label_priors(left.entity_type, right.entity_type),
            "reason_code": "train_signature_dominant_label",
        })
    return output, audit


def apply_native_label_adjudication(
    values: Iterable[dict[str, Any]], document: BioREDDocument,
    registry: BioREDPredicateRegistry,
    judge_decisions: Iterable[dict[str, Any]],
    critic_decisions: Iterable[dict[str, Any]] = (),
    *, require_critic: bool = True, confidence_threshold: float | None = None,
    relation_confidence_threshold: float = 0.90,
    label_confidence_threshold: float = 0.80,
    edit_confidence_threshold: float = 0.90,
    allow_model_label_edits: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Apply only lineage-bound, span-grounded, dual-confirmed label edits.

    KEEP never creates a version.  A material label edit creates version+1 and
    is re-enriched from the unchanged source evidence.  Missing, malformed or
    disagreeing decisions leave the original label in semantic REVIEW.
    """
    judge_by_key = {
        key: dict(item) for item in judge_decisions
        if isinstance(item, dict) and (key := lineage_key(item)) is not None
    }
    critic_by_key = {
        key: dict(item) for item in critic_decisions
        if isinstance(item, dict) and (key := lineage_key(item)) is not None
    }
    output: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []

    if confidence_threshold is not None:
        # Legacy callers can still request the original single threshold.
        relation_confidence_threshold = float(confidence_threshold)
        label_confidence_threshold = float(confidence_threshold)

    def validated_decision(
        decision: dict[str, Any] | None, value: dict[str, Any],
    ) -> dict[str, Any]:
        if not decision:
            return {
                "valid": False, "relation_supported": False,
                "label_supported": False, "target": "",
                "error": "missing_decision", "relation_confidence": 0.0,
                "label_confidence": 0.0, "span_ids": [],
            }
        verdict = str(decision.get("verdict", "ABSTAIN") or "ABSTAIN").upper()
        target = str(decision.get("recommended_relation_type", "") or "")
        try:
            legacy_confidence = float(decision.get("confidence", 0.0) or 0.0)
            relation_confidence = float(
                decision.get("relation_confidence", legacy_confidence) or 0.0
            )
            label_confidence = float(
                decision.get("label_confidence", legacy_confidence) or 0.0
            )
        except (TypeError, ValueError):
            relation_confidence = label_confidence = 0.0
        span_ids = [str(item) for item in decision.get("supporting_span_ids", []) or []]
        pack = dict(value.get("evidence_pack", {}) or {})
        by_id = {
            str(item.get("span_id", "") or ""): item
            for item in [
                *list(pack.get("spans", []) or []),
                *list(pack.get("adjudication_spans", []) or []),
            ]
        }
        selected = [by_id[item] for item in span_ids if item in by_id]
        spans_valid = bool(
            span_ids and len(selected) == len(span_ids)
            and all(str(item.get("role", "")) == "OWNER" for item in selected)
            and any(bool(item.get("subject_covered")) for item in selected)
            and any(bool(item.get("object_covered")) for item in selected)
        )
        if verdict == "KEEP":
            target = str(value.get("relation_type", "") or "")
        asserted = str(
            decision.get("relation_asserted", "AMBIGUOUS") or "AMBIGUOUS"
        ).upper()
        error = ""
        if verdict not in {"KEEP", "EDIT"}:
            error = "adjudicator_abstained"
        elif target not in BIORED_RELATION_TYPES:
            error = "adjudicator_invalid_target_label"
        else:
            left_id = str(value.get("arg1_id", ""))
            right_id = str(value.get("arg2_id", ""))
            if not registry.valid(
                target, document.concepts[left_id].entity_type,
                document.concepts[right_id].entity_type,
            ):
                error = "adjudicator_type_signature_mismatch"
            elif not spans_valid:
                error = "adjudicator_span_support_insufficient"
        valid = not error
        relation_supported = bool(
            valid and asserted == "YES"
            and relation_confidence >= relation_confidence_threshold
        )
        label_supported = bool(
            relation_supported and label_confidence >= label_confidence_threshold
        )
        if valid and asserted == "YES" and not relation_supported:
            error = "adjudicator_relation_low_confidence"
        elif relation_supported and not label_supported:
            error = "adjudicator_label_low_confidence"
        return {
            "valid": valid, "relation_supported": relation_supported,
            "label_supported": label_supported, "target": target,
            "error": error, "relation_confidence": relation_confidence,
            "label_confidence": label_confidence, "span_ids": span_ids,
        }

    for raw in values:
        value = dict(raw)
        key = lineage_key(value)
        judge = judge_by_key.get(key) if key else None
        judge_validation = validated_decision(judge, value)
        judge_ok = bool(judge_validation["label_supported"])
        judge_relation_ok = bool(judge_validation["relation_supported"])
        judge_target = str(judge_validation["target"])
        judge_error = str(judge_validation["error"])
        judge_relation_conf = float(judge_validation["relation_confidence"])
        judge_label_conf = float(judge_validation["label_confidence"])
        judge_spans = list(judge_validation["span_ids"])
        judge_asserted = str((judge or {}).get("relation_asserted", "AMBIGUOUS") or "AMBIGUOUS").upper()
        critic = critic_by_key.get(key) if key else None
        critic_validation = (
            validated_decision(critic, value) if require_critic
            and judge_target != value.get("relation_type")
            else {
                "valid": True, "relation_supported": True,
                "label_supported": True, "target": judge_target, "error": "",
                "relation_confidence": 1.0, "label_confidence": 1.0,
                "span_ids": judge_spans,
            }
        )
        critic_ok = bool(critic_validation["label_supported"])
        critic_target = str(critic_validation["target"])
        critic_error = str(critic_validation["error"])
        critic_relation_conf = float(critic_validation["relation_confidence"])
        critic_label_conf = float(critic_validation["label_confidence"])
        critic_spans = list(critic_validation["span_ids"])
        critic_asserted = str((critic or {}).get("relation_asserted", "AMBIGUOUS") or "AMBIGUOUS").upper()
        current = str(value.get("relation_type", "") or "")
        association_family = set(BIORED_ASSOCIATION_LABELS)
        edit_transition_allowed = bool(
            allow_model_label_edits
            and current in association_family and judge_target in association_family
        )
        confirmed_edit = bool(
            judge_ok and judge_target != current
            and critic_ok and critic_target == judge_target
            and judge_asserted == "YES" and critic_asserted == "YES"
            and judge_relation_conf >= edit_confidence_threshold
            and judge_label_conf >= edit_confidence_threshold
            and critic_relation_conf >= edit_confidence_threshold
            and critic_label_conf >= edit_confidence_threshold
            and edit_transition_allowed
        )
        left_id, right_id = str(value.get("arg1_id", "")), str(value.get("arg2_id", ""))
        left_type = document.concepts[left_id].entity_type
        right_type = document.concepts[right_id].entity_type
        dominant = registry.dominant_label(left_type, right_type)
        train_dominant_keep = bool(
            dominant and current == dominant and judge_relation_ok
            and str(value.get("relation_card_match", "NONE") or "NONE") == "NONE"
            and float(value.get("confidence", 0.0) or 0.0) >= 0.90
            and bool(value.get("endpoint_support_closed"))
        )
        # A >=90% Train signature prior is a calibrated tie-breaker for
        # type-only evidence.  Two LLMs may share the same surface-polarity
        # bias, so consensus alone cannot override that prior without an
        # explicit target-pair cue.
        confirmed_edit = bool(confirmed_edit and not train_dominant_keep)
        adjudication = {
            "contract_version": "biored-label-adjudication-v2",
            "original_relation_type": current,
            "judge": dict(judge or {}),
            "critic": dict(critic or {}),
            "decision": "EDIT" if confirmed_edit else (
                "KEEP" if (
                    judge_ok and judge_target == current
                ) or train_dominant_keep else "REVIEW"
            ),
            "final_relation_type": judge_target if confirmed_edit else current,
            "relation_truth_supported": judge_relation_ok,
            "train_dominant_label": dominant,
            "train_dominant_keep": train_dominant_keep,
        }
        reason_codes = [item for item in (judge_error, critic_error) if item]
        if judge_ok and judge_target != current and not edit_transition_allowed:
            reason_codes.append("label_edit_transition_not_calibrated")
        if judge_ok and judge_target != current and require_critic and critic_target != judge_target:
            reason_codes.append("label_adjudicators_disagree")
        if confirmed_edit:
            edited = edited_version(value, reason_code="native_relation_label_edit")
            edited["relation_type"] = judge_target
            edited["association_sign"] = registry.cards[judge_target].association_sign
            edited["label_adjudication"] = adjudication
            lane = str(edited.get("candidate_lane", "extracted_hint") or "extracted_hint")
            edited = enrich_native_relation(edited, document, registry, lane=lane)
            edited["semantic_status"] = "ACCEPTED"
            edited["quality_flags"] = sorted(
                set(edited.get("quality_flags", []) or [])
                | {"native_relation_label_edited", "adjudicated_native_relation_support"}
            )
            output.append(edited)
        else:
            value["label_adjudication"] = adjudication
            keep_supported = bool(
                judge_ok and judge_target == current and judge_asserted == "YES"
                or train_dominant_keep
            )
            if keep_supported:
                value["semantic_status"] = "ACCEPTED"
                value["quality_flags"] = sorted(
                    set(value.get("quality_flags", []) or [])
                    | {"adjudicated_native_relation_support"}
                    | ({"train_dominant_adjudication_keep"} if train_dominant_keep else set())
                )
            material_conflict = bool(
                not keep_supported and judge_ok and judge_asserted == "NO"
                or (
                    not keep_supported and judge_ok and judge_target != current
                    and edit_transition_allowed
                )
            )
            if material_conflict:
                value["semantic_status"] = "REVIEW"
                value["quality_flags"] = sorted(
                    set(value.get("quality_flags", []) or [])
                    | {"native_relation_label_ambiguous", *reason_codes}
                )
            output.append(value)
        audit.append({
            "candidate_id": value.get("candidate_id", ""),
            "candidate_version_before": value.get("candidate_version", 1),
            "candidate_version_after": output[-1].get("candidate_version", 1),
            "original_relation_type": current,
            "final_relation_type": output[-1].get("relation_type", current),
            "decision": adjudication["decision"],
            "reason_codes": sorted(set(reason_codes)),
            "judge_confidence": judge_label_conf,
            "judge_relation_confidence": judge_relation_conf,
            "judge_label_confidence": judge_label_conf,
            "critic_confidence": critic_label_conf if critic else None,
            "critic_relation_confidence": critic_relation_conf if critic else None,
            "critic_label_confidence": critic_label_conf if critic else None,
            "judge_relation_asserted": judge_asserted,
            "critic_relation_asserted": critic_asserted if critic else None,
            "judge_supporting_span_ids": judge_spans,
            "critic_supporting_span_ids": critic_spans if critic else [],
        })
    return output, audit


def _sentence_lattice(document: BioREDDocument) -> tuple[list[Any], dict[str, set[int]]]:
    reader = ArticleEvidenceReader()
    sentences = reader.parent_units(document.text, reader.read(document.text))
    sentence_indexes: dict[str, set[int]] = {concept_id: set() for concept_id in document.concepts}
    for concept_id, concept in document.concepts.items():
        for mention in concept.mentions:
            for index, sentence in enumerate(sentences):
                if sentence.char_start <= mention.start and mention.end <= sentence.char_end:
                    sentence_indexes[concept_id].add(index)
                    break
    return sentences, sentence_indexes


def build_native_pair_candidates(
    document: BioREDDocument,
    registry: BioREDPredicateRegistry,
    *,
    existing_pairs: Iterable[tuple[str, str]] = (),
    explicit_only: bool = False,
    max_candidates: int = 96,
    max_sentence_distance: int = 1,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Build type-valid, owner-local official concept pairs without Gold use."""
    sentences, indexes = _sentence_lattice(document)
    existing = {tuple(sorted(pair)) for pair in existing_pairs}
    matcher = make_biored_support_matcher(registry)
    candidates: list[dict[str, Any]] = []
    filtered_nonlocal = filtered_no_support = 0
    for left_id, right_id in itertools.combinations(sorted(document.concepts), 2):
        pair = tuple(sorted((left_id, right_id)))
        if pair in existing:
            continue
        left, right = document.concepts[left_id], document.concepts[right_id]
        allowed = [
            label for label in BIORED_RELATION_TYPES
            if registry.valid(label, left.entity_type, right.entity_type)
        ]
        if not allowed:
            continue
        distances = [
            abs(first - second)
            for first in indexes.get(left_id, set())
            for second in indexes.get(right_id, set())
        ]
        if not distances or min(distances) > max(0, int(max_sentence_distance)):
            filtered_nonlocal += 1
            continue
        windows: list[tuple[int, int, str, list[str]]] = []
        for first in indexes.get(left_id, set()):
            for second in indexes.get(right_id, set()):
                if abs(first - second) > max(0, int(max_sentence_distance)):
                    continue
                low, high = sorted((first, second))
                start, end = sentences[low].char_start, sentences[high].char_end
                windows.append((
                    low, high, document.text[start:end],
                    list(dict.fromkeys(sentences[index].parent_sentence_id for index in range(low, high + 1))),
                ))
        left_aliases, right_aliases = concept_aliases(left), concept_aliases(right)
        label_priors = registry.label_priors(left.entity_type, right.entity_type)
        dominant_label = registry.dominant_label(left.entity_type, right.entity_type)
        scored: list[tuple[int, int, int, str, str, list[str], list[str]]] = []
        for low, high, evidence, sentence_ids in windows:
            for label in allowed:
                support = matcher(
                    label, left.entity_type, right.entity_type, evidence,
                    subject_aliases=left_aliases, object_aliases=right_aliases,
                )
                support_value = str(support.get("match", "NONE"))
                scored.append((
                    {"EXPLICIT": 3, "WEAK": 2, "NONE": 1, "CONFLICT": 0}.get(support_value, 0),
                    -(high - low), -len(evidence), label, support_value, sentence_ids,
                    list(support.get("reason_codes", []) or []),
                ))
        best = max(scored, default=None)
        if best is None:
            filtered_nonlocal += 1
            continue
        support_rank, _, _, _, support_value, sentence_ids, reason_codes = best
        if explicit_only and support_value not in {"EXPLICIT", "WEAK"}:
            filtered_no_support += 1
            continue
        best_windows = [item for item in windows if item[3] == sentence_ids]
        evidence = min((item[2] for item in best_windows), key=len, default="")
        pair_candidate_id = _stable_id("p-br", document.pmid, *pair)
        candidates.append({
            "pair_candidate_id": pair_candidate_id,
            "candidate_id": _stable_id("r-br", document.pmid, *pair),
            "candidate_version": 1,
            "candidate_lane": "recovery",
            "arg1_id": left_id,
            "arg2_id": right_id,
            "arg1_type": left.entity_type,
            "arg2_type": right.entity_type,
            "allowed_relation_types": allowed,
            "training_label_priors": label_priors,
            "training_dominant_label": dominant_label,
            "evidence_window": evidence,
            "owner_sentence_ids": sentence_ids,
            "support_match": support_value,
            "support_reason_codes": reason_codes,
            "sentence_distance": -best[1],
            "support_rank": support_rank,
        })
    candidates.sort(key=lambda item: (
        -int(item["support_rank"]), int(item["sentence_distance"]),
        len(str(item["evidence_window"])), item["pair_candidate_id"],
    ))
    kept = candidates[:max(0, int(max_candidates))]
    return kept, {
        "type_local_candidates": len(candidates),
        "filtered_nonlocal_pairs": filtered_nonlocal,
        "filtered_no_explicit_support": filtered_no_support,
        "budget_truncated_pairs": max(0, len(candidates) - len(kept)),
    }


def enrich_native_relation(
    value: dict[str, Any],
    document: BioREDDocument,
    registry: BioREDPredicateRegistry,
    *,
    lane: str,
) -> dict[str, Any]:
    """Attach lineage-v2, native RelationCard and a full EvidencePack."""
    item = dict(value)
    left_id, right_id = str(item.get("arg1_id", "")), str(item.get("arg2_id", ""))
    label = str(item.get("relation_type", ""))
    left, right = document.concepts[left_id], document.concepts[right_id]
    pair = tuple(sorted((left_id, right_id)))
    candidate_id = str(item.get("candidate_id", "") or "")
    if not candidate_id:
        prefix = "r-br" if lane == "recovery" else "h-br"
        candidate_id = _stable_id(prefix, document.pmid, *pair, label)
    aliases = {
        left_id: concept_aliases(left),
        right_id: concept_aliases(right),
    }
    relation = {
        "subject": aliases[left_id][0] if aliases[left_id] else left_id,
        "object": aliases[right_id][0] if aliases[right_id] else right_id,
        "subject_family": left_id,
        "object_family": right_id,
        "subject_type": left.entity_type,
        "object_type": right.entity_type,
        "predicate": label,
        "evidence": str(item.get("evidence_quote", "") or item.get("evidence_window", "") or ""),
        "evidence_candidates": [str(item.get("evidence_window", "") or "")],
        "owner_sentence_ids": list(item.get("owner_sentence_ids", []) or []),
        "context_sentence_ids": [],
        "quality_flags": [],
    }
    pack = EvidencePackBuilder(
        support_matcher=make_biored_support_matcher(registry), max_spans=3,
    ).build(relation, text=document.text, aliases_by_canonical=aliases)
    pack_payload = pack.to_dict()
    adjudication_spans = build_native_adjudication_spans(
        item, document, registry, max_spans=8,
    )
    pack_payload["adjudication_spans"] = adjudication_spans
    pack_payload["adjudication_span_ids"] = [
        span["span_id"] for span in adjudication_spans
    ]
    endpoint_single = [
        span["span_id"] for span in adjudication_spans
        if span.get("subject_covered") and span.get("object_covered")
    ]
    endpoint_subject = [
        span["span_id"] for span in adjudication_spans
        if span.get("subject_covered")
    ]
    endpoint_object = [
        span["span_id"] for span in adjudication_spans
        if span.get("object_covered")
    ]
    if endpoint_single:
        endpoint_support_mode = "SINGLE_OWNER_ENDPOINT_CLOSED"
        endpoint_support_span_ids = endpoint_single[:1]
    elif endpoint_subject and endpoint_object:
        endpoint_support_mode = "MULTI_OWNER_ENDPOINT_CLOSED"
        endpoint_support_span_ids = list(dict.fromkeys([
            endpoint_subject[0], endpoint_object[0],
        ]))
    else:
        endpoint_support_mode = "ENDPOINT_SUPPORT_UNRESOLVED"
        endpoint_support_span_ids = []
    endpoint_support_closed = endpoint_support_mode != "ENDPOINT_SUPPORT_UNRESOLVED"
    pack_payload["endpoint_support_mode"] = endpoint_support_mode
    pack_payload["endpoint_support_span_ids"] = endpoint_support_span_ids
    pack_payload["endpoint_support_closed"] = endpoint_support_closed
    support_match = str(pack.support_trigger_match or "NONE")
    source_traceable = bool(pack.source_traceable or adjudication_spans)
    endpoints_traceable = bool(pack.subject_covered and pack.object_covered)
    endpoints_closed = bool(
        pack.support_subject_covered and pack.support_object_covered
    )
    support_closed = bool(
        endpoint_support_closed and (
            pack.support_mode != "UNRESOLVED" and endpoints_closed
        )
    )
    # BioRED is relation-truth oriented.  An exact, endpoint-closed model quote
    # can be retained as weak support even when the broad Association label has
    # no lexical trigger; it remains explicitly audited as TYPE_ONLY.
    confidence = float(item.get("confidence", 0.0) or 0.0)
    model_grounded = bool(
        support_closed and source_traceable and confidence >= 0.90
        and support_match == "NONE"
        and lane == "extracted_hint"
    )
    semantic_accepted = bool(
        support_closed and source_traceable
        and (support_match in {"EXPLICIT", "WEAK"} or model_grounded)
    )
    flags: list[str] = []
    if not source_traceable:
        flags.append("evidence_unresolved_partial")
    if not endpoints_traceable:
        flags.append("evidence_endpoints_not_closed")
    if support_match == "NONE":
        flags.append("native_relation_card_type_only")
    if model_grounded:
        flags.append("model_grounded_type_only_support")
    previous_flags = set(item.get("quality_flags", []) or []) - {
        "native_relation_card_type_only", "model_grounded_type_only_support",
        "evidence_unresolved_partial", "evidence_endpoints_not_closed",
    }
    enriched = {
        **item,
        "candidate_id": candidate_id,
        "candidate_version": int(item.get("candidate_version", 1) or 1),
        "candidate_lane": lane,
        "source_candidate_ids": list(item.get("source_candidate_ids", []) or [candidate_id]),
        "source_lanes": list(item.get("source_lanes", []) or [lane]),
        "pair_candidate_id": str(item.get("pair_candidate_id", "") or ""),
        "evidence_pack": pack_payload,
        "endpoint_support_mode": endpoint_support_mode,
        "endpoint_support_span_ids": endpoint_support_span_ids,
        "endpoint_support_closed": endpoint_support_closed,
        "support_mode": pack.support_mode,
        "relation_card_match": support_match,
        "relation_card_reason_codes": list(pack.support_trigger_reason_codes),
        "factual_status": "VALID" if source_traceable else "REVIEW",
        "semantic_status": "ACCEPTED" if semantic_accepted else "REVIEW",
        "write_status": "BLOCKED",
        "write_reasons": ["benchmark_no_write"],
        "association_sign": registry.cards[label].association_sign,
        "quality_flags": sorted(previous_flags | set(flags)),
    }
    return normalize_lineage(enriched, lane=lane)


def seed_native_lineage(
    value: dict[str, Any], document: BioREDDocument, *, lane: str,
) -> dict[str, Any]:
    """Assign identity before validation so factual discards remain audited."""
    item = dict(value)
    left_id, right_id = str(item.get("arg1_id", "")), str(item.get("arg2_id", ""))
    pair = tuple(sorted((left_id, right_id)))
    label = str(item.get("relation_type", ""))
    candidate_id = str(item.get("candidate_id", "") or "")
    if not candidate_id:
        candidate_id = _stable_id(
            "r-br" if lane == "recovery" else "h-br",
            document.pmid, *pair, label,
        )
    item.update({
        "candidate_id": candidate_id,
        "candidate_version": int(item.get("candidate_version", 1) or 1),
        "candidate_lane": lane,
        "source_candidate_ids": list(item.get("source_candidate_ids", []) or [candidate_id]),
        "source_lanes": list(item.get("source_lanes", []) or [lane]),
    })
    return normalize_lineage(item, lane=lane)


def audit_false_negatives(
    records: Iterable[dict[str, Any]],
    documents: Iterable[BioREDDocument],
    *,
    relation_field: str = "raw_relations",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Separate exact-match FN causes without changing benchmark scoring."""
    by_pmid = {str(item.get("pmid", "")): item for item in records}
    ledger: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    by_label: dict[str, dict[str, int]] = {}
    label_confusion: dict[str, dict[str, int]] = {}
    for document in documents:
        predictions = list((by_pmid.get(document.pmid, {}) or {}).get(relation_field, []) or [])
        predicted_keys = {
            (*sorted((str(item.get("arg1_id", "")), str(item.get("arg2_id", "")))), str(item.get("relation_type", "")))
            for item in predictions
        }
        pair_labels: dict[tuple[str, str], set[str]] = {}
        active_ids: set[str] = set()
        for item in predictions:
            pair = tuple(sorted((str(item.get("arg1_id", "")), str(item.get("arg2_id", "")))))
            pair_labels.setdefault(pair, set()).add(str(item.get("relation_type", "")))
            active_ids.update(pair)
        for gold in document.relations:
            if gold.label_key in predicted_keys:
                continue
            if gold.pair in pair_labels:
                category = "RELATION_LABEL_WRONG"
            elif all(endpoint in active_ids for endpoint in gold.pair):
                category = "ENDPOINTS_FOUND_PAIR_NOT_GENERATED"
            else:
                category = "GEMINI_COMPLETE_MISS"
            counts[category] = counts.get(category, 0) + 1
            label_counts = by_label.setdefault(category, {})
            label_counts[gold.relation_type] = label_counts.get(gold.relation_type, 0) + 1
            if category == "RELATION_LABEL_WRONG":
                gold_confusion = label_confusion.setdefault(gold.relation_type, {})
                for predicted_label in sorted(pair_labels.get(gold.pair, set())):
                    gold_confusion[predicted_label] = gold_confusion.get(predicted_label, 0) + 1
            ledger.append({
                "pmid": document.pmid,
                "arg1_id": gold.pair[0],
                "arg2_id": gold.pair[1],
                "gold_relation_type": gold.relation_type,
                "gold_novelty": gold.novelty,
                "fn_category": category,
                "predicted_labels_for_pair": sorted(pair_labels.get(gold.pair, set())),
                "arg1_seen_in_any_prediction": gold.pair[0] in active_ids,
                "arg2_seen_in_any_prediction": gold.pair[1] in active_ids,
            })
    return ledger, {
        "total_false_negatives": len(ledger),
        "category_counts": counts,
        "category_relation_type_counts": by_label,
        "relation_label_confusion_gold_to_predicted": label_confusion,
        "audit_scope": "Gold exact mismatch only; factual support is not inferred",
    }


def stable_dev_partition(
    documents: Iterable[BioREDDocument], calibration_size: int = 20,
) -> tuple[list[BioREDDocument], list[BioREDDocument]]:
    ranked = sorted(
        documents,
        key=lambda item: (hashlib.sha256(item.pmid.encode()).hexdigest(), item.pmid),
    )
    calibration = ranked[:max(0, calibration_size)]
    eval_docs = ranked[max(0, calibration_size):]
    return calibration, eval_docs


def guard_biored_test(path: str | Path, release_manifest: str | Path | None = None) -> None:
    path = Path(path)
    if path.name.casefold() != "test.pubtator":
        return
    if not release_manifest:
        raise PermissionError("BioRED official Test requires an explicit release/freeze manifest")
    manifest = json.loads(Path(release_manifest).read_text(encoding="utf-8"))
    expected = str(manifest.get("test_sha256", "") or "")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if not bool(manifest.get("allow_official_test")) or expected != actual:
        raise PermissionError("BioRED Test release manifest is absent, disabled, or hash-mismatched")


def validate_native_predictions(
    values: Iterable[dict[str, Any]], document: BioREDDocument,
    registry: BioREDPredicateRegistry,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    clean: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for raw in values:
        item = dict(raw) if isinstance(raw, dict) else {}
        left = str(item.get("arg1_id", "") or "")
        right = str(item.get("arg2_id", "") or "")
        label = str(item.get("relation_type", "") or "")
        reasons = []
        if left not in document.concepts or right not in document.concepts:
            reasons.append("invented_or_invalid_concept_id")
        if label not in BIORED_RELATION_TYPES:
            reasons.append("invented_or_invalid_relation_label")
        if not reasons and not registry.valid(
            label, document.concepts[left].entity_type, document.concepts[right].entity_type,
        ):
            reasons.append("native_type_signature_mismatch")
        key = (*sorted((left, right)), label)
        if key in seen:
            reasons.append("duplicate_relation")
        novelty = str(item.get("novelty", "Unknown") or "Unknown")
        if novelty not in BIORED_NOVELTY:
            novelty = "Unknown"
            reasons.append("invalid_novelty_normalized_unknown")
        if reasons:
            audit.append({"candidate": item, "reason_codes": reasons})
            continue
        seen.add(key)
        try:
            confidence = max(0.0, min(1.0, float(item.get("confidence", 0.0))))
        except (TypeError, ValueError):
            confidence = 0.0
        clean.append({
            **item, "arg1_id": left, "arg2_id": right,
            "relation_type": label, "novelty": novelty, "confidence": confidence,
            "association_sign": registry.cards[label].association_sign,
            "write_status": "BLOCKED", "write_reasons": ["benchmark_no_write"],
            "candidate_lane": str(item.get("candidate_lane", "extracted_hint") or "extracted_hint"),
        })
    return clean, audit
