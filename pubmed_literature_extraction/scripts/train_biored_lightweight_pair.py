#!/usr/bin/env python3
"""Train a small BioRED-style entity-pair classifier.

This is an optional offline adapter, not a runtime dependency.  It maps only
BioRED labels that have an unambiguous project-ontology counterpart and keeps
``NO_RELATION`` as an explicit class.  The resulting joblib artifact is small
enough for the Agent's lightweight sklearn backend.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cognitive_agent.relation_pair_classifier import NO_RELATION, SklearnPairBackend
from cognitive_agent.schema.relation_signatures import RELATION_SIGNATURES


TYPE_MAP = {
    "GeneOrGeneProduct": "Gene",
    "DiseaseOrPhenotypicFeature": "Disease",
    "ChemicalEntity": "Metabolite",
}
LABEL_MAP = {
    "Association": "ASSOCIATED_WITH",
    "Positive_Correlation": "ASSOCIATED_WITH",
    "Negative_Correlation": "ASSOCIATED_WITH",
    "Bind": "INTERACTS_WITH",
}


@dataclass
class Concept:
    concept_id: str
    entity_type: str
    mentions: list[str] = field(default_factory=list)


@dataclass
class Document:
    pmid: str
    text: str
    concepts: dict[str, Concept]
    relations: dict[frozenset[str], str]


def parse_pubtator(raw: str) -> list[Document]:
    documents: list[Document] = []
    for block in re.split(r"\n\s*\n", raw.strip()):
        lines = [line for line in block.splitlines() if line.strip()]
        if len(lines) < 2 or "|t|" not in lines[0] or "|a|" not in lines[1]:
            continue
        pmid, title = lines[0].split("|t|", 1)
        _, abstract = lines[1].split("|a|", 1)
        concepts: dict[str, Concept] = {}
        relations: dict[frozenset[str], str] = {}
        for line in lines[2:]:
            fields = line.split("\t")
            if len(fields) == 6 and fields[1].isdigit():
                _, _, _, mention, raw_type, raw_ids = fields
                entity_type = TYPE_MAP.get(raw_type)
                if not entity_type:
                    continue
                for concept_id in raw_ids.split(","):
                    concept = concepts.setdefault(
                        concept_id, Concept(concept_id, entity_type)
                    )
                    if mention not in concept.mentions:
                        concept.mentions.append(mention)
            elif len(fields) == 5 and fields[1] in LABEL_MAP:
                _, raw_label, left, right, _novelty = fields
                relations[frozenset((left, right))] = LABEL_MAP[raw_label]
        documents.append(Document(pmid, f"{title} {abstract}", concepts, relations))
    return documents


def allowed_predicates(left: str, right: str) -> list[str]:
    return sorted(
        predicate for predicate, signatures in RELATION_SIGNATURES.items()
        if (left, right) in signatures
    )


def orient(left: Concept, right: Concept, predicate: str) -> tuple[Concept, Concept] | None:
    signatures = RELATION_SIGNATURES.get(predicate, set())
    if (left.entity_type, right.entity_type) in signatures:
        return left, right
    if (right.entity_type, left.entity_type) in signatures:
        return right, left
    return None


def evidence_window(text: str, left: Concept, right: Concept) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    for sentence in sentences:
        if any(re.search(re.escape(m), sentence, re.I) for m in left.mentions) and any(
            re.search(re.escape(m), sentence, re.I) for m in right.mentions
        ):
            return sentence[:1800]
    positions = [
        match.start()
        for mention in (*left.mentions, *right.mentions)
        for match in [re.search(re.escape(mention), text, re.I)] if match
    ]
    if not positions:
        return text[:1800]
    center = sum(positions) // len(positions)
    return text[max(0, center - 900):center + 900]


def feature(subject: Concept, obj: Concept, evidence: str) -> str:
    return SklearnPairBackend.feature_text(SimpleNamespace(
        subject_type=subject.entity_type,
        object_type=obj.entity_type,
        evidence_section="ABSTRACT",
        source_predicates=[],
        subject=subject.mentions[0],
        object=obj.mentions[0],
        evidence=evidence,
    ))


def instances(documents: list[Document], *, training: bool, seed: int) -> tuple[list[str], list[str]]:
    rng = random.Random(seed)
    rows: list[tuple[str, str]] = []
    for document in documents:
        concepts = list(document.concepts.values())
        positives: list[tuple[str, str]] = []
        negatives: list[tuple[str, str]] = []
        for index, left in enumerate(concepts):
            for right in concepts[index + 1:]:
                gold = document.relations.get(frozenset((left.concept_id, right.concept_id)))
                if gold:
                    oriented = orient(left, right, gold)
                    if not oriented:
                        continue
                    subject, obj = oriented
                    positives.append((feature(subject, obj, evidence_window(document.text, left, right)), gold))
                    continue
                orientations = []
                if allowed_predicates(left.entity_type, right.entity_type):
                    orientations.append((left, right))
                if allowed_predicates(right.entity_type, left.entity_type):
                    orientations.append((right, left))
                if orientations:
                    subject, obj = orientations[0]
                    negatives.append((feature(subject, obj, evidence_window(document.text, left, right)), NO_RELATION))
        if training:
            rng.shuffle(negatives)
            negatives = negatives[: max(8, 4 * len(positives))]
        rows.extend(positives)
        rows.extend(negatives)
    rng.shuffle(rows)
    return [row[0] for row in rows], [row[1] for row in rows]


def read_split(path: Path, split: str) -> list[Document]:
    with zipfile.ZipFile(path) as archive:
        return parse_pubtator(archive.read(f"BioRED/{split}.PubTator").decode("utf-8"))


def metrics(labels: list[str], predictions: list[str]) -> dict:
    from sklearn.metrics import classification_report

    return classification_report(labels, predictions, output_dict=True, zero_division=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--biored-zip", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    from joblib import dump
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.frozen import FrozenEstimator
    from sklearn.linear_model import SGDClassifier
    from sklearn.pipeline import Pipeline

    train_docs = read_split(args.biored_zip, "Train")
    dev_docs = read_split(args.biored_zip, "Dev")
    test_docs = read_split(args.biored_zip, "Test")
    train_x, train_y = instances(train_docs, training=True, seed=args.seed)
    dev_x, dev_y = instances(dev_docs, training=True, seed=args.seed + 1)
    test_x, test_y = instances(test_docs, training=False, seed=args.seed)
    base = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True, ngram_range=(1, 2), min_df=2, max_features=40_000,
            sublinear_tf=True,
        )),
        ("classifier", SGDClassifier(
            loss="log_loss", max_iter=300, tol=1e-4, class_weight="balanced",
            alpha=2e-5, random_state=args.seed,
        )),
    ])
    base.fit(train_x, train_y)
    model = CalibratedClassifierCV(FrozenEstimator(base), method="sigmoid")
    model.fit(dev_x, dev_y)
    report = metrics(test_y, model.predict(test_x).tolist())
    metadata = {
        "source": "NCBI BioRED train+dev",
        "mapped_labels": LABEL_MAP,
        "train_instances": len(train_y),
        "calibration_instances": len(dev_y),
        "test_instances": len(test_y),
        "test_report": report,
        "limitations": [
            "BioRED GeneOrGeneProduct maps to project Gene for transfer training.",
            "Only labels with an unambiguous project-ontology mapping are used.",
            "This lightweight TF-IDF model is not the official PubMedBERT model.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dump({"model": model, "metadata": metadata}, args.output)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
