"""Run the pinned LiverKG extractor in dry-run mode and expose its raw records."""

import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile
import tempfile

from .mentions import read_articles
from ..model import DEFAULT_CONFIG, live_provider, read_settings
from ..storage import atomic_json, read_json, run_lock


EXTRACTOR_COMMIT = "fc9ebd72971318a7738f15fc2b86369a552656c2"
EXTRACTOR_DIRECTORY = "pubmed_literature_extraction"
EXTRACTOR_REPOSITORY = "https://github.com/Buerchener/FYP-liver-disease.git"
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EXTRACTOR_CACHE = Path(".cache/entity_ontology/extractor")
PROFILE = "quality"
RUN_ID = "entity_ontology"
AUXILIARY_KEYS = {
    "SECOND_LLM_API_KEY", "SECOND_LLM_API_BASE", "SECOND_LLM_MODEL_ID",
    "DEEPSEEK_API_KEY", "DEEPSEEK_API_BASE",
}


def archive_source(runtime, repo):
    source = runtime / EXTRACTOR_COMMIT
    runtime.mkdir(parents=True, exist_ok=True)
    with (runtime / ".archive.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not source.exists():
            git = ["git", "-C", str(repo)]
            present = subprocess.run(
                git + ["cat-file", "-e", f"{EXTRACTOR_COMMIT}^{{commit}}"], capture_output=True,
            )
            if present.returncode:
                fetch = subprocess.run(
                    git + ["fetch", "--no-tags", "--no-write-fetch-head",
                           EXTRACTOR_REPOSITORY, EXTRACTOR_COMMIT], capture_output=True,
                )
                if fetch.returncode:
                    raise RuntimeError(
                        f"Could not fetch pinned LiverKG commit {EXTRACTOR_COMMIT}. "
                        "Check network access and Git credentials for the LiverKG repository."
                    )
            archive = subprocess.run(
                git + ["archive", "--format=tar", EXTRACTOR_COMMIT, EXTRACTOR_DIRECTORY],
                check=True, capture_output=True,
            ).stdout
            with tempfile.TemporaryDirectory(dir=runtime) as directory:
                unpacked = Path(directory) / "source"
                unpacked.mkdir()
                with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
                    tar.extractall(unpacked, filter="data")
                unpacked.rename(source)
    return source / EXTRACTOR_DIRECTORY


def auxiliary_settings(path):
    settings = {}
    if path is not None:
        for line in path.read_text(encoding="utf-8").splitlines():
            key, sep, value = line.strip().removeprefix("export ").partition("=")
            if sep and key.strip() in AUXILIARY_KEYS:
                parts = shlex.split(value.strip())
                if parts:
                    settings[key.strip()] = parts[0]
    for key in AUXILIARY_KEYS:
        if key in os.environ:
            settings[key] = os.environ[key]
    return settings


def extractor_env(runtime, source, config, output, auxiliary_env):
    connection, settings = read_settings(config, "audit_edges")
    live_provider(connection, settings)
    credential = os.environ[connection["api_key_env"]]
    output.mkdir(exist_ok=True)
    config_path = runtime / "extractor.toml"
    config_path.write_text("output_dir = " + json.dumps(str(output), ensure_ascii=False) + "\n")
    env = {k: v for k, v in os.environ.items() if not k.startswith("NEO4J_")}
    env.update(auxiliary_settings(auxiliary_env))
    env.update(
        GEMINI_MODEL=settings["model"], GEMINI_API_BASE=connection["base_url"],
        GEMINI_API_KEY=credential, AUX_CRITIC_MODEL=settings["model"],
        ALIYUN_MAAS_API_BASE=connection["base_url"], ALIYUN_MAAS_API_KEY=credential,
        NEO4J_PASSWORD="", NEO4J_URI="bolt://localhost:7687", NEO4J_RAG_ENABLED="false",
        LIVERKG_CONFIG=str(config_path), LIVERKG_RUNS_DIR=str(runtime / "runs"),
        LIVERKG_CACHE_DIR=str(runtime / "cache"),
        EXTRACTION_CACHE_PATH=str(runtime / "cache/extraction.sqlite"),
        PRIMARY_LLM_MAX_OUTPUT_TOKENS="8192", PRIMARY_LLM_REQUEST_TIMEOUT_S="100",
        PRIMARY_LLM_MAX_RETRIES="2", AUX_MODEL_MAX_RETRIES="1",
        SCHEMA_PROFILE="", RULE_BUNDLE="", CONFORMAL_CALIBRATION="",
        PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(source),
    )
    return env


def validate_record(record, pmid):
    if str(record.get("pmid")) != pmid or record.get("error"):
        raise ValueError(f"Extraction failed for PMID {pmid}. Check the native extractor log.")
    phases = record.get("phases", {})
    execution = phases.get("execution", {})
    if execution.get("dry_run") is not True or any(
        execution.get(key, 0) for key in ("entities_written", "relations_written", "relations_updated_in_neo4j")
    ):
        raise ValueError(f"PMID {pmid} lacks a safe dry-run result.")
    extraction = phases.get("extraction", {})
    if extraction.get("error"):
        raise ValueError(f"Extraction failed for PMID {pmid}. Check the native extractor log.")
    if not isinstance(extraction.get("entities"), list) or not isinstance(phases.get("reader", {}).get("units"), list):
        raise ValueError(f"PMID {pmid} lacks the expected raw extraction and evidence.")
    return record


def validate_output(document, pmids, model):
    records = document["records"]
    ids = [str(record.get("pmid")) for record in records]
    if len(ids) != len(set(ids)) or set(ids) != set(pmids):
        raise ValueError("Extraction output must contain exactly one record for every input PMID.")
    report = document.get("report")
    if report and (report.get("dry_run") is not True or report.get("neo4j_connected") is not False
                   or report.get("model") != model):
        raise ValueError("Unexpected extractor runtime, model, or Neo4j status.")
    for record in records:
        validate_record(record, str(record["pmid"]))


def extract_articles(articles, output_dir, *, config=None, repo=None, python=None, auxiliary_env=None):
    """Extract a JSONL batch and return the validated agent_results.json path."""
    articles = Path(articles).resolve()
    output_dir = Path(output_dir).resolve()
    config = Path(config or os.getenv("ENTITY_MODEL_CONFIG") or DEFAULT_CONFIG).expanduser().resolve()
    repo = Path(repo or REPOSITORY_ROOT).expanduser().resolve()
    python = Path(python or os.getenv("LIVERKG_EXTRACTOR_PYTHON") or sys.executable).expanduser().absolute()
    auxiliary_env = auxiliary_env or os.getenv("LIVERKG_EXTRACTOR_AUX_ENV")
    auxiliary_env = Path(auxiliary_env).expanduser() if auxiliary_env else None
    if articles.suffix != ".jsonl":
        raise ValueError("The extractor requires a PubMed JSONL file.")
    content = articles.read_bytes()
    pmids = [str(row["pmid"]) for row in read_articles(content)]
    runtime = output_dir / "extractor_runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    final = output_dir / "agent_results.json"
    _, settings = read_settings(config, "audit_edges")
    model = settings["model"]
    manifest = {
        "article_sha256": hashlib.sha256(content).hexdigest(),
        "model_config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
        "commit": EXTRACTOR_COMMIT, "model": model, "profile": PROFILE, "pmids": pmids,
    }

    with run_lock(runtime):
        snapshot = runtime / "input_manifest.json"
        if final.exists() and not snapshot.exists():
            raise ValueError("An extraction output exists without an input snapshot.")
        if snapshot.exists() and read_json(snapshot) != manifest:
            raise ValueError("Extraction input or model settings changed. Use a separate run directory.")
        atomic_json(snapshot, manifest)
        if final.exists():
            validate_output(read_json(final), pmids, model)
            return final

        native_output = runtime / "native_output"
        checkpoint_dir = runtime / "runs" / RUN_ID / "checkpoints"
        run_dir = checkpoint_dir.parent
        native_results = native_output / f"agent_results_{RUN_ID}.json"
        resume = (run_dir / "runspec.json").exists()
        valid = set()
        for pmid in pmids:
            checkpoint = checkpoint_dir / f"{pmid}.json"
            if checkpoint.exists():
                try:
                    validate_record(read_json(checkpoint), pmid)
                    valid.add(pmid)
                except (ValueError, KeyError):
                    rejected = runtime / "failed_checkpoints" / checkpoint.name
                    rejected.parent.mkdir(exist_ok=True)
                    checkpoint.replace(rejected)
        if len(valid) != len(pmids):
            source = archive_source(repo / EXTRACTOR_CACHE, repo)
            env = extractor_env(runtime, source, config, native_output, auxiliary_env)
            args = [str(python), "-m", "liverkg_cli.app", "extract", str(articles),
                    "--profile", PROFILE, "--limit", str(len(pmids)), "--run-id", RUN_ID]
            if resume:
                args.append("--resume")
            log_path = runtime / "liverkg.log"
            print(f"Extracting {len(pmids) - len(valid)} PubMed articles with {model}. Log: {log_path}", flush=True)
            descriptor = os.open(log_path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
            with os.fdopen(descriptor, "ab") as log:
                code = subprocess.run(args, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
            if code:
                raise RuntimeError(f"Extractor exited with status {code}. See {log_path}.")

        try:
            artifact = read_json(native_results)
        except (FileNotFoundError, json.JSONDecodeError):
            artifact = {}
        records = [validate_record(read_json(checkpoint_dir / f"{pmid}.json"), pmid) for pmid in pmids]
        if len(artifact.get("records", [])) == len(pmids):
            artifact = {**artifact, "records": records}
        else:
            artifact = {
                "records": records,
                "config": {"model_id": model, "skip_neo4j_write": True},
                "adapter": {
                    "record_source": "native_article_checkpoints",
                    "native_report": "extractor_runtime/native_output" if native_results.exists() else None,
                    "model": model, "dry_run": True,
                },
            }
        validate_output(artifact, pmids, model)
        atomic_json(final, artifact)
        return final
