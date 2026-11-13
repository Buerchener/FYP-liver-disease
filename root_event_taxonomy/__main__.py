"""Run the root-constrained induction experiment from a Task 1 snapshot."""
import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path

import numpy
import scipy

from event_taxonomy.induction.dataset import load_dataset
from event_taxonomy.induction.graph import digest
from event_taxonomy.induction.journal import LLMJournal
from event_taxonomy.induction.providers import OpenAICompatibleProvider
from event_taxonomy.induction.storage import atomic_json, read_json, run_lock

from .data import InputData
from .decisions import PROMPTS
from .embedding import EmbeddingAPI, Embeddings
from .engine import Inducer
from .local import LocalMLXProvider, LocalSentenceTransformer, local_path

PACKAGE = Path(__file__).resolve().parent


def number(value, low, high, name):
    if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{name} must be a number in [{low}, {high}].")


def integer(value, minimum, name):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer at least {minimum}.")


def read_inputs(path):
    config = read_json(path)
    prompts, _ = implementation(config)
    for name, minimum in {
        "max_rounds": 1, "max_group_size": 2, "max_new_nodes": 0, "maximum_llm_calls": 1,
        "parent_candidates": 0, "reuse_candidates": 1, "max_ancestor_paths": 1, "audit_batch_size": 1,
        "rejected_group_split_depth": 0, "response_retries": 0, "evidence_per_concept": 1,
    }.items():
        integer(config[name], minimum, name)
    number(config["clustering"]["distance_cutoff"], 0, 2, "distance_cutoff")
    number(config["clustering"]["root_weight"], 0, 1, "root_weight")
    integer(config["embedding"]["batch_size"], 1, "embedding.batch_size")
    integer(config["embedding"].get("evidence_char_limit", 900), 0, "embedding.evidence_char_limit")
    if config["embedding"].get("dimensions") is not None:
        integer(config["embedding"]["dimensions"], 1, "embedding.dimensions")
    integer(config["llm"]["provider"]["timeout_seconds"], 1, "llm.provider.timeout_seconds")
    if set(config["llm"]["stages"]) != set(prompts):
        raise ValueError("Provide LLM settings for each configured algorithm stage.")
    if config.get("algorithm") == "five_step":
        for name, minimum in {"k": 1, "block_size": 1, "max_candidates_per_round": 1}.items():
            integer(config["discovery"][name], minimum, "discovery." + name)
        integer(config.get("placement_reserved_calls", 0), 0, "placement_reserved_calls")
        if config.get("placement_reserved_calls", 0) >= config["maximum_llm_calls"]:
            raise ValueError("Placement reserve must be smaller than the total call budget")
    for stage, settings in config["llm"]["stages"].items():
        integer(settings["max_tokens"], 1, stage + ".max_tokens")
        number(settings["temperature"], 0, 2, stage + ".temperature")
        if not isinstance(settings["model"], str) or not settings["model"].strip():
            raise ValueError("Missing model name for " + stage)
    for settings, local_kind in ((config["embedding"], "local_sentence_transformer"),
                                 (config["llm"]["provider"], "local_mlx")):
        kind = settings.get("kind", "openai_compatible")
        if kind == local_kind:
            if kind == "local_mlx":
                integer(settings["timeout_seconds"], 1, "llm.provider.timeout_seconds")
                if not (settings.get("python") or settings.get("python_env")):
                    raise ValueError("Configure the Python executable with MLX-LM installed.")
            else:
                integer(settings.get("torch_threads", 4), 1, "embedding.torch_threads")
            if not (settings.get("model_path") or settings.get("model_path_env")):
                raise ValueError("Configure a local model directory.")
        elif kind == "openai_compatible":
            integer(settings["timeout_seconds"], 1, "provider.timeout_seconds")
            if not settings.get("api_key_env") or not (settings.get("base_url") or settings.get("base_url_env")):
                raise ValueError("Configure the API URL and credential environment variable.")
        else:
            raise ValueError("Unknown provider kind: " + kind)
    if not isinstance(config["embedding"]["model"], str) or not config["embedding"]["model"].strip():
        raise ValueError("Missing embedding model name.")
    for name in ("dataset", "definitions_file"):
        if config.get(name):
            config[name] = str((path.parent / config[name]).resolve())

    selected = config.get("input_concept_ids")
    if selected is not None and (not isinstance(selected, list) or not all(isinstance(n, str) for n in selected)):
        raise ValueError("input_concept_ids must be an array of IDs.")
    overlay = read_json(config["definitions_file"]) if config.get("definitions_file") else {}
    if not isinstance(overlay, dict):
        raise ValueError("Definitions file must be a JSON object keyed by concept ID.")
    dataset = load_dataset(config["dataset"])
    return config, InputData(dataset, selected, overlay, config["evidence_per_concept"])


def connection(settings):
    resolved = dict(settings)
    resolved["base_url"] = settings.get("base_url") or os.environ.get(settings["base_url_env"])
    if not resolved["base_url"]:
        raise ValueError("Set " + settings["base_url_env"] + " to the API base URL.")
    if not os.environ.get(settings["api_key_env"]):
        raise ValueError("Set " + settings["api_key_env"] + " to the existing API credential.")
    return resolved


def implementation(config):
    if config.get("algorithm", "legacy") == "legacy":
        return PROMPTS, Inducer
    if config["algorithm"] == "five_step":
        from .refinement import FiveStepInducer, REFINEMENT_PROMPTS
        return REFINEMENT_PROMPTS, FiveStepInducer
    raise ValueError("Unknown induction algorithm")


def run_live(directory, config, data):
    prompts, inducer = implementation(config)
    embedding_config = config["embedding"]
    llm_config = config["llm"]["provider"]
    if embedding_config.get("kind") == "local_sentence_transformer":
        embedding_config = {**embedding_config, "model_path": local_path(embedding_config, "model_path")}
        embedding_provider = LocalSentenceTransformer(embedding_config)
    else:
        embedding_config = connection(embedding_config)
        embedding_provider = EmbeddingAPI(embedding_config)
    if llm_config.get("kind") == "local_mlx":
        llm_config = {**llm_config, "model_path": local_path(llm_config, "model_path"),
                      "python": local_path(llm_config, "python")}
        llm_provider = LocalMLXProvider(llm_config, directory)
    else:
        llm_config = connection(llm_config)
        llm_provider = nullcontext(OpenAICompatibleProvider(llm_config))
    config = {**config, "embedding": embedding_config, "llm": {**config["llm"], "provider": llm_config}}
    directory.mkdir(parents=True, exist_ok=False)
    with run_lock(directory), llm_provider as provider:
        atomic_json(directory / "input_dataset.json", data.dataset)
        atomic_json(directory / "definitions_overlay.json", data.overlay)
        atomic_json(directory / "manifest.json", {
            "algorithm": "root-constrained-event-taxonomy/" + ("2-five-step" if inducer is not Inducer else "1"),
            "config": config,
            "scope": data.scope(),
            "input_sha256": digest(data.dataset),
            "definition_overlay_sha256": digest(data.overlay),
            "prompt_sha256": {stage: digest(prompt) for stage, prompt in prompts.items()},
            "code_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in sorted(PACKAGE.glob("*.py"))},
            "numpy_version": numpy.__version__,
            "scipy_version": scipy.__version__,
        })
        embeddings = Embeddings(embedding_provider, embedding_config, data, directory / "embeddings")
        atomic_json(directory / "prompts.json", prompts)
        journal = LLMJournal(directory / "llm", provider, prompts,
                             config["llm"]["stages"], config["maximum_llm_calls"])
        return inducer(directory, config, data, embeddings, journal).run()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-input", action="store_true", help="Validate data and selection without model calls")
    mode.add_argument("--live", action="store_true", help="Run with the configured embedding and LLM providers")
    parser.add_argument("--output-parent", type=Path, default=PACKAGE / "runs")
    parser.add_argument("--run-id", default=datetime.now(timezone.utc).strftime("root-%Y%m%dT%H%M%S%fZ"))
    args = parser.parse_args()
    try:
        config, data = read_inputs(args.config.resolve())
        if args.check_input:
            print(json.dumps(data.scope(), ensure_ascii=False, indent=2))
            return 0
        if Path(args.run_id).name != args.run_id or args.run_id in (".", ".."):
            raise ValueError("Run ID must be one directory name.")
        directory = (args.output_parent / args.run_id).resolve()
        print(str(directory), flush=True)
        report = run_live(directory, config, data)
        print(json.dumps({
            k: report[k] for k in ("status", "stop_reason", "selected_root_reachable",
                                  "new_edges_retained", "llm_calls", "issues")
        }, ensure_ascii=False, indent=2))
        return 2 if report["issues"] else 0
    except (ValueError, KeyError, OSError) as error:
        parser.exit(1, str(error) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
