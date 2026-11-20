"""Run the in-repository LiverKG extractor in dry-run mode and expose its raw records."""

import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from urllib.parse import urlsplit

from .mentions import read_articles
from ..model import DEFAULT_CONFIG, live_provider, read_settings
from ..storage import atomic_json, read_json, run_lock


# The extractor ships next to this package: ``pubmed_literature_extraction/``
# holds ``cognitive_agent/``, ``liverkg_cli/`` and ``entity_ontology/``.
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EXTRACTOR_PACKAGES = ("cognitive_agent", "liverkg_cli")
EXTRACTOR_MODULES = ("convert_pubmed_xml_to_jsonl.py",)
PROFILE = "quality"
RUN_ID = "entity_ontology"
AUXILIARY_KEYS = {
    "SECOND_LLM_API_KEY", "SECOND_LLM_API_BASE", "SECOND_LLM_MODEL_ID",
    "DEEPSEEK_API_KEY", "DEEPSEEK_API_BASE",
}


def extractor_source(root=None):
    """Return the extractor project root, which must contain its packages."""
    root = Path(root or REPOSITORY_ROOT).resolve()
    missing = [name for name in (*EXTRACTOR_PACKAGES, *EXTRACTOR_MODULES) if not (root / name).exists()]
    if missing:
        raise RuntimeError(
            f"The LiverKG extractor was not found under {root} (missing: {', '.join(missing)}). "
            "Run the ontology workflow from the FYP-liver-disease repository."
        )
    return root


def extractor_fingerprint(source):
    """Hash the extractor source so results are tied to the exact code that produced them."""
    digest = hashlib.sha256()
    files = sorted(path for package in EXTRACTOR_PACKAGES for path in (source / package).rglob("*.py"))
    files += [source / module for module in EXTRACTOR_MODULES]
    for path in files:
        digest.update(path.relative_to(source).as_posix().encode("utf-8") + b"\0")
        digest.update(path.read_bytes() + b"\0")
    return digest.hexdigest()


def git_revision(source):
    """Record the Git commit for provenance; it is informational, not part of run identity."""
    try:
        head = subprocess.run(["git", "-C", str(source), "rev-parse", "HEAD"],
                              capture_output=True, text=True, timeout=10)
        if head.returncode:
            return None
        status = subprocess.run(
            ["git", "-C", str(source), "status", "--porcelain", "--",
             *EXTRACTOR_PACKAGES, *EXTRACTOR_MODULES],
            capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    dirty = bool(status.stdout.strip()) if status.returncode == 0 else None
    return {"commit": head.stdout.strip(), "dirty": dirty}


def extractor_identity(root=None):
    source = extractor_source(root)
    return {"fingerprint": extractor_fingerprint(source), "git": git_revision(source)}


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


def auxiliary_mode(settings):
    """'separate' when a second-model key is configured, otherwise 'single_model'."""
    return "separate" if settings.get("SECOND_LLM_API_KEY") or settings.get("DEEPSEEK_API_KEY") else "single_model"


def single_model_env(base_url, credential, model):
    # The extractor always enables its second model and DeepSeek-style
    # auxiliary judges. Without separate keys they reuse the configured model,
    # so one OpenAI-compatible endpoint is enough to run the whole system.
    return {
        "SECOND_LLM_API_KEY": credential, "SECOND_LLM_API_BASE": base_url, "SECOND_LLM_MODEL_ID": model,
        "DEEPSEEK_API_KEY": credential, "DEEPSEEK_API_BASE": base_url, "AUX_PRIMARY_MODEL": model,
        "ENTITY_RECOVERY_MODEL": model, "PAIRWISE_JUDGE_MODEL": model,
    }


def require_openai_compatible(base_url):
    # cognitive_agent.agent selects its OpenAI-compatible LangExtract provider
    # only for endpoints ending in /v1; any other base silently uses Gemini.
    if not urlsplit(base_url).path.rstrip("/").casefold().endswith("/v1"):
        raise ValueError(
            "The extractor needs an OpenAI-compatible base URL ending in /v1 "
            f"(got {base_url!r}). Set LIVERKG_API_BASE to the provider's /v1 endpoint."
        )


def extractor_env(runtime, source, config, output, auxiliary_env):
    connection, settings = read_settings(config, "audit_edges")
    live_provider(connection, settings)
    require_openai_compatible(connection["base_url"])
    credential = os.environ[connection["api_key_env"]]
    output.mkdir(exist_ok=True)
    config_path = runtime / "extractor.toml"
    config_path.write_text("output_dir = " + json.dumps(str(output), ensure_ascii=False) + "\n")
    env = {k: v for k, v in os.environ.items() if not k.startswith("NEO4J_")}
    auxiliary = auxiliary_settings(auxiliary_env)
    env.update(auxiliary)
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
        # Roles whose defaults point at another provider stay off unless the
        # shell environment would otherwise switch them on.
        ENTITY_RECOVERY_MODE="off", PAIRWISE_JUDGE_MODE="off",
        PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(source),
    )
    if auxiliary_mode(auxiliary) == "single_model":
        env.update(single_model_env(connection["base_url"], credential, settings["model"]))
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
    source = extractor_source(repo)
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
        "extractor_fingerprint": extractor_fingerprint(source),
        "auxiliary_models": auxiliary_mode(auxiliary_settings(auxiliary_env)),
        "model": model, "profile": PROFILE, "pmids": pmids,
    }

    with run_lock(runtime):
        snapshot = runtime / "input_manifest.json"
        if final.exists() and not snapshot.exists():
            raise ValueError("An extraction output exists without an input snapshot.")
        if snapshot.exists() and read_json(snapshot) != manifest:
            raise ValueError("Extraction input, extractor code or model settings changed. Use a separate run directory.")
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
