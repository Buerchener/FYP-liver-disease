"""Offline adapters for cached Sentence Transformers and MLX models."""
import json
import os
from pathlib import Path
import select
import subprocess

from event_taxonomy.induction.contracts import ProviderError


def local_path(config, field):
    value = config.get(field)
    if not value and config.get(field + "_env"):
        value = os.environ.get(config[field + "_env"])
    if not value:
        raise ValueError("Provide " + field + " or set " + str(config.get(field + "_env")) + ".")
    path = Path(value).expanduser().resolve()
    if not path.exists():
        raise ValueError("Local path does not exist: " + str(path))
    return str(path)


class LocalSentenceTransformer:
    def __init__(self, config):
        self.config = config
        self.model = None

    def embed(self, texts):
        if self.model is None:
            from sentence_transformers import SentenceTransformer
            import torch

            cfg = self.config
            torch.set_num_threads(cfg.get("torch_threads", 4))
            try:
                self.model = SentenceTransformer(
                    cfg["model_path"], device=cfg.get("device", "cpu"), local_files_only=True,
                    model_kwargs={"dtype": torch.bfloat16}, truncate_dim=cfg.get("dimensions"),
                )
            except (ValueError, OSError, RuntimeError) as error:
                raise ProviderError("Local embedding load failed: " + type(error).__name__) from error
        try:
            vectors = self.model.encode(
                texts, batch_size=len(texts), normalize_embeddings=True, show_progress_bar=False,
            )
        except (ValueError, OSError, RuntimeError) as error:
            raise ProviderError("Local embedding inference failed: " + type(error).__name__) from error
        counts = self.model.tokenizer(texts, truncation=True, max_length=self.model.max_seq_length)
        total = sum(len(ids) for ids in counts["input_ids"])
        return {"vectors": vectors.tolist(), "model": self.config["model"],
                "usage": {"prompt_tokens": total, "total_tokens": total}}


class LocalMLXProvider:
    def __init__(self, config, directory):
        self.config, self.directory = config, directory
        self.worker = None

    def __enter__(self):
        return self

    def __exit__(self, *_error):
        self.close()

    def close(self):
        if self.worker is None:
            return
        worker, self.worker = self.worker, None
        worker.stdin.close()
        try:
            worker.wait(timeout=2)
        except subprocess.TimeoutExpired:
            worker.kill()
            worker.wait()
        worker.stdout.close()

    def start(self):
        cfg = self.config
        offline = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME")}
        offline.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1",
                       TOKENIZERS_PARALLELISM="false")
        with (self.directory / "mlx_worker.log").open("ab") as stderr:
            self.worker = subprocess.Popen(
                [cfg["python"], "-u", str(Path(__file__).with_name("mlx_worker.py")), cfg["model_path"]],
                env=offline, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr,
                encoding="utf-8",
            )

    def generate(self, stage, prompt, payload, parameters):
        if self.worker is None:
            try:
                self.start()
            except OSError as error:
                raise ProviderError("Could not start local MLX worker") from error
        request = {"prompt": prompt, "payload": payload, "parameters": parameters}
        try:
            self.worker.stdin.write(json.dumps(request, ensure_ascii=False, allow_nan=False) + "\n")
            self.worker.stdin.flush()
            ready, _, _ = select.select([self.worker.stdout], [], [], self.config["timeout_seconds"])
            if not ready:
                self.worker.kill()
                raise ProviderError("Local MLX generation timed out")
            line = self.worker.stdout.readline()
            if not line:
                raise ProviderError("Local MLX worker exited; see mlx_worker.log")
            response = json.loads(line)
            if response.get("error"):
                raise ProviderError("Local MLX inference failed; see mlx_worker.log")
            return response
        except (OSError, ValueError) as error:
            raise ProviderError("Local MLX worker communication failed") from error
