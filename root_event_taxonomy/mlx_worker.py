"""Read requests over stdin and generate locally with one loaded MLX model."""
import json
from pathlib import Path
import sys
import traceback


def main():
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler

    path = Path(sys.argv[1]).resolve(strict=True)
    model, tokenizer = load(str(path))
    for line in sys.stdin:
        try:
            request = json.loads(line)
            parameters = request["parameters"]
            user_json = json.dumps(request["payload"], ensure_ascii=False, separators=(",", ":"))
            prompt = tokenizer.apply_chat_template(
                [{"role": "system", "content": request["prompt"]}, {"role": "user", "content": user_json}],
                add_generation_prompt=True, tokenize=False,
            )
            parts = []
            for part in stream_generate(
                model, tokenizer, prompt, max_tokens=parameters["max_tokens"],
                sampler=make_sampler(temp=parameters["temperature"]),
            ):
                parts.append(part.text)
            response = {
                "content": "".join(parts),
                "model": path.name,
                "finish_reason": part.finish_reason,
                "usage": {
                    "prompt_tokens": part.prompt_tokens,
                    "completion_tokens": part.generation_tokens,
                    "total_tokens": part.prompt_tokens + part.generation_tokens,
                },
            }
        except Exception:
            traceback.print_exc(file=sys.stderr)
            response = {"error": "generation_failed"}
        print(json.dumps(response, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
