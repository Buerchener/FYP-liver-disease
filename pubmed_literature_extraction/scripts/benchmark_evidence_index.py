"""Compare evidence-pack builds with and without an article-scoped index."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
from time import perf_counter

from cognitive_agent.evidence_pack import EvidencePackBuilder
from cognitive_agent.evidence_units import ArticleEvidenceReader


OUTPUT = Path(__file__).resolve().parents[1] / "extraction_output"


class CountingReader(ArticleEvidenceReader):
    def __init__(self):
        self.read_count = 0

    def read(self, text):
        self.read_count += 1
        return super().read(text)


def load_cases(sources_path: Path, results_path: Path) -> list[tuple[str, list[dict]]]:
    with sources_path.open(encoding="utf-8") as sources_file:
        sources = {}
        for line in sources_file:
            if line.strip():
                source = json.loads(line)
                sources[str(source["pmid"])] = source
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    cases = []
    for record in payload["records"]:
        candidates = record.get("phases", {}).get("extraction", {}).get("relations", [])
        if not candidates:
            continue
        source = sources.get(str(record["pmid"]))
        if source is None:
            raise ValueError(f"Missing source text for PMID {record['pmid']}")
        text = f"TITLE: {source['title']}\nABSTRACT: {source['abstract']}"
        cases.append((text, candidates))
    if not cases:
        raise ValueError("The input has no articles with extraction candidates")
    return cases


def measure(cases, *, passes: int, indexed: bool, capture: bool = False):
    builder = EvidencePackBuilder()
    reader = CountingReader()
    builder.reader = reader
    packs = []
    started = perf_counter()
    for text, candidates in cases:
        index = builder.index(text) if indexed else None
        for _ in range(passes):
            for candidate in candidates:
                pack = builder.build(candidate, text=text, source_index=index)
                if capture:
                    packs.append(pack.to_dict())
    return perf_counter() - started, reader.read_count, packs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, default=OUTPUT / "pubmed_converted_500.jsonl")
    parser.add_argument("--results", type=Path, default=OUTPUT / "agent_results_v3_final_quality_30.json")
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--passes", type=int, default=2)
    args = parser.parse_args()
    if min(args.repeat, args.passes) < 1:
        parser.error("--repeat and --passes must be positive")
    cases = load_cases(args.sources, args.results)
    _, uncached_reads, uncached = measure(cases, passes=args.passes, indexed=False, capture=True)
    _, indexed_reads, indexed = measure(cases, passes=args.passes, indexed=True, capture=True)
    if uncached != indexed:
        raise RuntimeError("The evidence packs differ when the article index is reused")

    durations = {False: [], True: []}
    for run in range(args.repeat):
        for use_index in (False, True) if run % 2 == 0 else (True, False):
            elapsed, _, _ = measure(cases, passes=args.passes, indexed=use_index)
            durations[use_index].append(elapsed)
    uncached_time = median(durations[False])
    indexed_time = median(durations[True])
    print(json.dumps({
        "articles_with_candidates": len(cases),
        "candidates": sum(len(candidates) for _, candidates in cases),
        "passes": args.passes,
        "evidence_packs_equal": True,
        "article_reads_without_reuse": uncached_reads,
        "article_reads_with_reuse": indexed_reads,
        "median_seconds_without_reuse": round(uncached_time, 4),
        "median_seconds_with_reuse": round(indexed_time, 4),
        "speedup": round(uncached_time / indexed_time, 2),
    }, indent=2))


if __name__ == "__main__":
    main()
