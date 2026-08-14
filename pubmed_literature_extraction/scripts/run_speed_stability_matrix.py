#!/usr/bin/env python3
"""Run a bounded four-article transport/prompt stability matrix.

The default is plan-only.  Pass ``--execute`` to make remote calls.  The
runner never opens Neo4j, never writes to it, disables environment proxies for
the measured client, and never serializes credentials.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import ssl
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.metadata import version as package_version
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import httpx
import langextract as lx
from langextract.core import exceptions as lx_exceptions
from langextract.core import types as core_types
from langextract.providers.openai import OpenAILanguageModel
from openai import OpenAI

from cognitive_agent.evidence_units import ArticleEvidenceReader
from cognitive_agent.extraction_kernel import ExtractionKernel, RawExtraction
from cognitive_agent.golden_examples import GoldenExampleSelector
from cognitive_agent.relation_pair_classifier import BioREDPairClassifier
from cognitive_agent.schema.examples import KG_EXTRACTION_PROMPT
from cognitive_agent.verifier import KGVerifier


DEFAULT_SOURCE = ROOT / "extraction_output/pubmed_converted_500.jsonl"
DEFAULT_GOLD = ROOT / "gold_annotations/pubmed_200_gold_v2_strict.jsonl"
DEFAULT_ENV = ROOT / "workstreams/literature_hmdb_kegg/.env"
DEFAULT_PMIDS = ("41475279", "41719003", "41686896", "41794448")
STRATA = {
    "41475279": ("review", "review_or_meta"),
    "41719003": ("clinical", "simple_clinical_observation"),
    "41686896": ("human_omics", "omics_or_computational"),
    "41794448": ("animal", "mixed_validation_complex"),
}
PROMPT_VARIANTS = ("full_4shot", "dynamic_2shot", "biored_pair")


@dataclass(frozen=True)
class MatrixCell:
    model_id: str
    prompt_variant: str
    outer_workers: int
    inner_workers: int = 1

    @property
    def cell_id(self) -> str:
        model = self.model_id.replace("/", "_").replace(".", "_")
        return f"{model}__{self.prompt_variant}__outer{self.outer_workers}_inner1"


class TransportRecorder:
    def __init__(self):
        self._local = threading.local()
        self._lock = threading.Lock()
        self.ttfb_s: list[float] = []
        self.request_s: list[float] = []
        self.prompt_tokens = 0
        self.output_tokens = 0
        self.reasoning_tokens = 0
        self.invalid_json = 0
        self.connection_errors = 0
        self.requests = 0

    def begin(self) -> float:
        started = time.perf_counter()
        self._local.started = started
        return started

    def response_headers(self, _response: httpx.Response) -> None:
        started = getattr(self._local, "started", None)
        if started is not None:
            with self._lock:
                self.ttfb_s.append(time.perf_counter() - started)

    def complete(self, started: float, response, output: str) -> None:
        usage = getattr(response, "usage", None)
        reasoning = 0
        completion_details = getattr(usage, "completion_tokens_details", None)
        if completion_details is not None:
            reasoning = int(getattr(completion_details, "reasoning_tokens", 0) or 0)
        invalid = 0
        try:
            json.loads(_strip_fence(output))
        except (TypeError, json.JSONDecodeError):
            invalid = 1
        with self._lock:
            self.requests += 1
            self.request_s.append(time.perf_counter() - started)
            self.prompt_tokens += int(getattr(usage, "prompt_tokens", 0) or 0)
            self.output_tokens += int(getattr(usage, "completion_tokens", 0) or 0)
            self.reasoning_tokens += reasoning
            self.invalid_json += invalid

    def connection_error(self) -> None:
        with self._lock:
            self.connection_errors += 1

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "requests": self.requests,
                "connect_error_count": self.connection_errors,
                "invalid_json_count": self.invalid_json,
                "ttfb_s": list(self.ttfb_s),
                "request_s": list(self.request_s),
                "prompt_tokens": self.prompt_tokens or None,
                "output_tokens": self.output_tokens or None,
                "reasoning_tokens": self.reasoning_tokens or None,
            }


class InstrumentedOpenAIModel(OpenAILanguageModel):
    """LangExtract OpenAI provider with explicit timeout/retry/pool controls."""

    def __init__(self, *, recorder: TransportRecorder, connect_timeout: float,
                 read_timeout: float, max_retries: int, **kwargs):
        super().__init__(**kwargs)
        self.recorder = recorder
        self._client.close()
        timeout = httpx.Timeout(
            connect=connect_timeout, read=read_timeout,
            write=read_timeout, pool=connect_timeout,
        )
        self._http_client = httpx.Client(
            timeout=timeout,
            trust_env=False,
            limits=httpx.Limits(max_connections=8, max_keepalive_connections=8),
            event_hooks={"response": [recorder.response_headers]},
        )
        self._client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            max_retries=max_retries,
            timeout=timeout,
            http_client=self._http_client,
        )

    def _process_single_prompt(self, prompt: str, config: dict) -> core_types.ScoredOutput:
        started = self.recorder.begin()
        try:
            response = self._client.chat.completions.create(
                **self._build_chat_completions_params(prompt, config)
            )
            output = response.choices[0].message.content or ""
            self.recorder.complete(started, response, output)
            return core_types.ScoredOutput(score=1.0, output=output)
        except Exception as exc:
            if "connection" in str(exc).casefold() or "timeout" in str(exc).casefold():
                self.recorder.connection_error()
            raise lx_exceptions.InferenceRuntimeError(
                f"OpenAI API error: {exc}", original=exc
            ) from exc

    def complete_json(self, prompt: str) -> dict:
        output = self._process_single_prompt(prompt, {"temperature": 0.0}).output
        return json.loads(_strip_fence(output))

    def close(self) -> None:
        self._client.close()


class OfflineKG:
    is_connected = False


def _strip_fence(value: str) -> str:
    value = str(value or "").strip()
    if value.startswith("```"):
        lines = value.splitlines()
        if lines and lines[-1].strip() == "```":
            lines = lines[1:-1]
        else:
            lines = lines[1:]
        value = "\n".join(lines)
    return value.strip()


def load_env(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        if name and name not in os.environ:
            os.environ[name] = value.strip().strip("'\"")


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def selected_rows(source: Path, gold: Path, pmids: tuple[str, ...]) -> tuple[list[dict], dict[str, dict]]:
    wanted = set(pmids)
    articles = {str(row.get("pmid")): row for row in load_jsonl(source) if str(row.get("pmid")) in wanted}
    gold_rows = {str(row.get("pmid")): row for row in load_jsonl(gold) if str(row.get("pmid")) in wanted}
    missing = [pmid for pmid in pmids if pmid not in articles or pmid not in gold_rows]
    if missing:
        raise ValueError(f"selected PMIDs missing from source or gold: {missing}")
    return [articles[pmid] for pmid in pmids], gold_rows


def build_plan(models: tuple[str, str], *, full_factorial: bool) -> list[MatrixCell]:
    if full_factorial:
        return [
            MatrixCell(model, prompt, workers)
            for model in models for prompt in PROMPT_VARIANTS for workers in (1, 2, 4)
        ]
    # Staged screening changes one dimension at a time and avoids 12 redundant,
    # expensive cells before a stable transport configuration is known.
    return [
        MatrixCell(models[0], "full_4shot", 1),
        MatrixCell(models[0], "full_4shot", 2),
        MatrixCell(models[0], "full_4shot", 4),
        MatrixCell(models[1], "full_4shot", 1),
        MatrixCell(models[0], "dynamic_2shot", 1),
        MatrixCell(models[0], "biored_pair", 1),
    ]


def tls_probe(api_base: str, timeout_s: float) -> dict:
    parsed = urlparse(api_base)
    host = parsed.hostname
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    if not host:
        return {"status": "error", "error": "invalid_api_base"}
    started = time.perf_counter()
    try:
        with socket.create_connection((host, port), timeout=timeout_s) as raw:
            tcp_s = time.perf_counter() - started
            if parsed.scheme == "https":
                context = ssl.create_default_context()
                with context.wrap_socket(raw, server_hostname=host):
                    tls_s = time.perf_counter() - started
            else:
                tls_s = tcp_s
        return {"status": "ok", "tcp_connect_s": round(tcp_s, 4), "tls_ready_s": round(tls_s, 4)}
    except Exception as exc:
        return {"status": "error", "error": type(exc).__name__}


def _normalise_lx_result(value) -> list:
    if isinstance(value, list):
        return value
    return [value] if hasattr(value, "extractions") else []


def run_langextract(article: dict, model: InstrumentedOpenAIModel, variant: str) -> RawExtraction:
    pmid = str(article["pmid"])
    text = f"TITLE: {article.get('title', '')}\nABSTRACT: {article.get('abstract', '')}"
    study_type = STRATA[pmid][0]
    selection = GoldenExampleSelector().select(text, study_type, max_examples=4, document_id=pmid)
    examples = selection.examples if variant == "full_4shot" else selection.examples[:2]
    result = lx.extract(
        text_or_documents=[lx.data.Document(document_id=pmid, text=text)],
        prompt_description=KG_EXTRACTION_PROMPT,
        examples=examples,
        model=model,
        temperature=0,
        max_workers=1,
        use_schema_constraints=False,
        show_progress=False,
        extraction_passes=1,
    )
    raw = RawExtraction(pmid=pmid)
    parser = ExtractionKernel(SimpleNamespace(model_id=model.model_id), inner_max_workers=1)
    for annotated in _normalise_lx_result(result):
        parser._parse(annotated, raw, source_text=text)
    return raw


def _pair_prompt(article: dict, gold: dict) -> tuple[str, list[dict]]:
    text = f"TITLE: {article.get('title', '')}\nABSTRACT: {article.get('abstract', '')}"
    entities = [
        {"mention": row.get("mention", ""), "type": row.get("type", "")}
        for row in gold.get("entities", [])
    ]
    units = ArticleEvidenceReader().read(text)
    candidates, _ = BioREDPairClassifier().build_candidates(entities, [], units)
    payload = [candidate.to_dict() for candidate in candidates[:16]]
    prompt = (
        "Classify each biomedical entity pair using only its exact evidence sentence. "
        "Choose one label from allowed_predicates or NO_RELATION. Never infer from external "
        "knowledge. Return JSON object {relations:[{candidate_id,subject,subject_type,predicate,"
        "object,object_type,direction,negated,uncertain,evidence,confidence}]}.\n"
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    )
    return prompt, entities


def run_pair(article: dict, gold: dict, model: InstrumentedOpenAIModel) -> RawExtraction:
    prompt, entities = _pair_prompt(article, gold)
    raw = RawExtraction(pmid=str(article["pmid"]), entities=entities)
    if not json.loads(prompt[prompt.index("\n") + 1:]):
        return raw
    response = model.complete_json(prompt)
    raw.relations = list(response.get("relations", []) or [])
    return raw


def classify_failure(exc: Exception) -> str:
    message = str(exc).casefold()
    if "401" in message or "invalid token" in message or "authentication" in message:
        return "provider_auth"
    if "model" in message and any(code in message for code in ("400", "404", "not found")):
        return "provider_config"
    if "connection" in message or "timeout" in message:
        return "transport"
    if isinstance(exc, (json.JSONDecodeError, ValueError)) and "json" in message:
        return "invalid_output"
    return "provider_runtime"


def article_metrics(article: dict, raw: RawExtraction, latency_s: float, error: str = "",
                    error_category: str = "") -> dict:
    text = f"TITLE: {article.get('title', '')}\nABSTRACT: {article.get('abstract', '')}"
    verified = KGVerifier(OfflineKG()).verify(raw.entities, raw.relations, pmid=raw.pmid, text=text)
    retained_entities = sum(item.filter_status == "retained" for item in verified.entities)
    contiguous = sum(item.evidence_contiguous for item in verified.relations)
    return {
        "pmid": raw.pmid,
        "stratum": STRATA[raw.pmid][1],
        "success": not bool(error),
        "error_type": error,
        "error_category": error_category,
        "latency_s": round(latency_s, 3),
        "entity_count": len(raw.entities),
        "relation_count": len(raw.relations),
        "verified_entity_count": retained_entities,
        "verified_relation_count": len(verified.relations),
        "import_ready_relation_count": sum(item.import_ready for item in verified.relations),
        "verifier_entity_retention": round(retained_entities / len(raw.entities), 4) if raw.entities else None,
        "verifier_relation_retention": round(len(verified.relations) / len(raw.relations), 4) if raw.relations else None,
        "evidence_contiguous_rate": round(contiguous / len(verified.relations), 4) if verified.relations else None,
    }


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int((len(ordered) - 1) * quantile + 0.999999)))
    return round(ordered[index], 3)


def summarize(rows: list[dict], transport: dict, probe: dict) -> dict:
    failures = [row for row in rows if not row["success"]]
    connection_failures = transport["connect_error_count"]
    failure_categories = sorted({row.get("error_category", "") for row in failures})
    provider_failure = any(value.startswith("provider_") for value in failure_categories)
    transport_failure = bool(connection_failures or "transport" in failure_categories)
    valid_rows = [] if failures or connection_failures else rows
    latencies = [row["latency_s"] for row in valid_rows]
    relation_den = sum(row["relation_count"] for row in valid_rows)
    verified_relations = sum(row["verified_relation_count"] for row in valid_rows)
    contiguous_den = sum(row["verified_relation_count"] for row in valid_rows)
    contiguous_num = sum(
        round((row["evidence_contiguous_rate"] or 0) * row["verified_relation_count"])
        for row in valid_rows
    )
    return {
        "quality_status": (
            "invalid_provider" if provider_failure
            else "invalid_transport" if transport_failure
            else "invalid_output" if failures
            else "valid"
        ),
        "failure_categories": failure_categories,
        "success_rate": round((len(rows) - len(failures)) / len(rows), 4) if rows else 0.0,
        "connection_error_rate": round(connection_failures / max(transport["requests"], 1), 4),
        "invalid_json_rate": round(transport["invalid_json_count"] / max(transport["requests"], 1), 4),
        "article_latency_p50_s": percentile(latencies, 0.50),
        "article_latency_p95_s": percentile(latencies, 0.95),
        "request_ttfb_p50_s": percentile(transport["ttfb_s"], 0.50),
        "request_ttfb_p95_s": percentile(transport["ttfb_s"], 0.95),
        "mean_entities_per_article": round(statistics.mean(row["entity_count"] for row in valid_rows), 3) if valid_rows else None,
        "mean_relations_per_article": round(statistics.mean(row["relation_count"] for row in valid_rows), 3) if valid_rows else None,
        "verifier_relation_retention": round(verified_relations / relation_den, 4) if relation_den else None,
        "evidence_contiguous_rate": round(contiguous_num / contiguous_den, 4) if contiguous_den else None,
        "token_usage": {
            "prompt_tokens": transport["prompt_tokens"],
            "output_tokens": transport["output_tokens"],
            "reasoning_tokens": transport["reasoning_tokens"],
            "availability": "available" if transport["prompt_tokens"] is not None else "unavailable",
        },
        "connect_probe": probe,
    }


def run_cell(cell: MatrixCell, articles: list[dict], gold: dict[str, dict], *, api_key: str,
             api_base: str, connect_timeout: float, read_timeout: float) -> dict:
    recorder = TransportRecorder()
    model = InstrumentedOpenAIModel(
        recorder=recorder,
        connect_timeout=connect_timeout,
        read_timeout=read_timeout,
        max_retries=1,
        model_id=cell.model_id,
        api_key=api_key,
        base_url=api_base,
        temperature=0.0,
        max_workers=1,
    )
    rows: list[dict] = []
    probe = tls_probe(api_base, connect_timeout)

    def one(article: dict) -> dict:
        started = time.perf_counter()
        try:
            raw = (
                run_pair(article, gold[str(article["pmid"])], model)
                if cell.prompt_variant == "biored_pair"
                else run_langextract(article, model, cell.prompt_variant)
            )
            return article_metrics(article, raw, time.perf_counter() - started)
        except Exception as exc:
            raw = RawExtraction(pmid=str(article["pmid"]))
            return article_metrics(
                article, raw, time.perf_counter() - started,
                error=type(exc).__name__,
                error_category=classify_failure(exc),
            )

    try:
        with ThreadPoolExecutor(max_workers=cell.outer_workers) as executor:
            futures = {executor.submit(one, article): article for article in articles}
            for future in as_completed(futures):
                rows.append(future.result())
    finally:
        model.close()
    rows.sort(key=lambda row: DEFAULT_PMIDS.index(row["pmid"]))
    transport = recorder.snapshot()
    return {
        "cell": asdict(cell),
        "cell_id": cell.cell_id,
        "articles": rows,
        "transport": transport,
        "summary": summarize(rows, transport, probe),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true", help="make remote model calls")
    parser.add_argument("--full-factorial", action="store_true", help="run all 18 cells")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "benchmark_output")
    parser.add_argument("--run-id", default=f"speed_matrix_{datetime.now():%Y%m%d_%H%M%S}")
    parser.add_argument("--api-base", default="")
    parser.add_argument("--model-a", default="count.gmcli-gemini-3-flash-preview")
    parser.add_argument("--model-b", default="count.gmcli-gemini-3.5-flash")
    parser.add_argument("--connect-timeout", type=float, default=10.0)
    parser.add_argument("--read-timeout", type=float, default=900.0)
    args = parser.parse_args()

    load_env(args.env_file)
    api_key = os.environ.get("GEMINI_API_KEY", "")
    api_base = args.api_base or os.environ.get("GEMINI_API_BASE", "")
    articles, gold = selected_rows(args.source, args.gold, DEFAULT_PMIDS)
    plan = build_plan((args.model_a, args.model_b), full_factorial=args.full_factorial)
    run_dir = args.output_dir / args.run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "run_id": args.run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "planned" if not args.execute else "running",
        "design": "full_factorial" if args.full_factorial else "staged_screening",
        "source": str(args.source.resolve()),
        "gold": str(args.gold.resolve()),
        "pmids": [
            {"pmid": pmid, "study_type": STRATA[pmid][0], "stratum": STRATA[pmid][1]}
            for pmid in DEFAULT_PMIDS
        ],
        "langextract_version": package_version("langextract"),
        "openai_version": package_version("openai"),
        "python": os.sys.version.split()[0],
        "api_endpoint_identity": api_base.rstrip("/"),
        "proxy_policy": "httpx trust_env=false; system proxy bypassed",
        "neo4j_write": False,
        "cache_mode": "off",
        "schema_constraints": False,
        "provider_max_retries": 1,
        "timeouts_s": {"connect": args.connect_timeout, "read_write": args.read_timeout},
        "cells": [asdict(cell) | {"cell_id": cell.cell_id} for cell in plan],
        "pair_prompt_note": "Uses strict-gold endpoint inventory only to isolate bounded pair-classification transport; do not compare its entity metrics as blind NER quality.",
        "failure_policy": "Any connection/timeout/article failure marks the cell invalid_transport; failed rows are excluded from quality aggregates.",
        "credential_fields_serialized": [],
    }
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    if not args.execute:
        print(json.dumps({"status": "planned", "run_dir": str(run_dir), "cells": len(plan)}, ensure_ascii=False))
        return 0
    if not api_key or not api_base:
        raise SystemExit("GEMINI_API_KEY and GEMINI_API_BASE must be configured; values are never printed")

    results = []
    for index, cell in enumerate(plan, 1):
        print(f"[{index}/{len(plan)}] {cell.cell_id}", flush=True)
        cell_result = run_cell(
            cell, articles, gold, api_key=api_key, api_base=api_base,
            connect_timeout=args.connect_timeout, read_timeout=args.read_timeout,
        )
        results.append(cell_result)
        (run_dir / f"{cell.cell_id}.json").write_text(
            json.dumps(cell_result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    cell_statuses = {item["summary"]["quality_status"] for item in results}
    manifest["status"] = (
        "invalid_provider" if "invalid_provider" in cell_statuses
        else "invalid_transport" if "invalid_transport" in cell_statuses
        else "invalid_output" if "invalid_output" in cell_statuses
        else "complete"
    )
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "summary.json").write_text(
        json.dumps({"manifest": manifest, "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"status": manifest["status"], "run_dir": str(run_dir)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
