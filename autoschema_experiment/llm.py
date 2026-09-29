"""Reuse the existing structured client, config, credentials and usage accounting."""
import json
import logging
import os
from pathlib import Path
from .common import digest
from cognitive_agent.aux_model_registry import AuxModelRegistry, AuxModelSpec

LOG = logging.getLogger(__name__)


class Model:
    def __init__(self, output, *, replay=None, env_file=None, model=None, model_role="auxiliary"):
        self.output = Path(output)
        self.replay = Path(replay) if replay else None
        self.calls = []
        self.registry = None
        if self.replay:
            self.model = "recorded-replay"
            return
        from liverkg_cli.env import load_project_env, parse_env_file
        from liverkg_cli.config import load_config
        from liverkg_cli.security import get_secret
        for k, v in load_project_env().items():
            os.environ.setdefault(k, v)
        if env_file:
            for k, v in parse_env_file(Path(env_file)).items():
                os.environ.setdefault(k, v)
        config = load_config()
        self.model = model or (config.model_id if model_role == "extraction" else config.second_llm_model_id)
        spec = AuxModelSpec(
            role="primary", provider="openai", model_id=self.model,
            api_base=config.api_base if model_role == "extraction" else config.second_llm_api_base,
            api_key=get_secret("gemini_api_key" if model_role == "extraction" else "deepseek_api_key"), timeout_s=90, max_retries=1,
        )
        if not spec.configured:
            raise ValueError("Existing model configuration is missing")
        self.registry = AuxModelRegistry([spec])

    def call(self, stage, key, system, payload):
        request = {"system": system, "payload": payload}
        name = f"{stage}_{key}.json"
        request_hash = digest(request)
        raw_dir = self.output / "raw"
        raw_dir.mkdir(exist_ok=True)
        if self.replay:
            record = json.loads((self.replay / name).read_text())
            if record["request_hash"] != request_hash:
                raise ValueError("Replay request hash mismatch")
            result = record["payload"]
            meta = {**record["metadata"], "execution_mode": "replay"}
        else:
            LOG.info("%s PMID=%s model=%s", stage, key, self.model)
            response = self.registry.call_json(
                "primary", system_prompt=system,
                user_prompt=json.dumps(payload, ensure_ascii=False),
            )
            meta = {"model": self.model, "status": response.status,
                    "prompt_tokens": response.prompt_tokens, "output_tokens": response.output_tokens,
                    "latency_s": response.latency_s, "attempts": response.attempts,
                    "execution_mode": "live"}
            self.calls.append({"stage": stage, "key": key, **meta})
            if response.status != "OK":
                raise RuntimeError(f"{stage}/{key}: model status {response.status}")
            result = response.payload
        if self.replay:
            self.calls.append({"stage": stage, "key": key, **meta})
        (raw_dir / name).write_text(json.dumps({"request_hash": request_hash,
            "request": request, "payload": result, "metadata": meta}, ensure_ascii=False, indent=2))
        return result, meta
