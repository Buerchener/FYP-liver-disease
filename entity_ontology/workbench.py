"""Read-only local workbench for completed entity ontology runs."""

import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import re
import threading
from urllib.parse import parse_qs, urlsplit
import webbrowser

from .order_entities import apply_entity_order
from .storage import read_json

WEB = Path(__file__).resolve().parent / "web"
RUN_ID = re.compile(r"[a-z0-9][a-z0-9-]{0,79}")
STATIC = {
    "/" + name: WEB / name
    for name in (
        "index.html", "app.js", "ui.js", "app.css", "overview.js", "workbench-shell.css",
        "evidence.js", "evidence.css", "incremental.js", "incremental.css",
        "preparation.js", "preparation.css",
    )
}
STATIC["/"] = STATIC["/index.html"]


def batch_manifest(runs_dir, run_id):
    if not RUN_ID.fullmatch(run_id):
        raise FileNotFoundError(run_id)
    folder = (runs_dir / run_id).resolve()
    if folder.parent != runs_dir or folder.name != run_id:
        raise FileNotFoundError(run_id)
    manifest = read_json(folder / "manifest.json")
    if manifest.get("run_id") != run_id:
        raise ValueError("The run ID and its manifest disagree.")
    return folder, manifest


def complete_batches(runs_dir):
    records = []
    for entry in runs_dir.glob("*/manifest.json"):
        try:
            _, manifest = batch_manifest(runs_dir, entry.parent.name)
            if manifest.get("status") != "complete":
                continue
            records.append({
                "run_id": manifest["run_id"],
                "title": manifest.get("title") or manifest["run_id"],
                "created_at": str(manifest.get("created_at") or ""),
                "article_count": len(manifest["pmids"]),
            })
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            continue
    return sorted(records, key=lambda row: (row["created_at"], row["run_id"]), reverse=True)


def read_articles(folder, manifest, inventory):
    articles_path = folder / "articles.jsonl"
    raw = articles_path.read_bytes()
    source_hash = next(item["sha256"] for item in inventory["sources"]
                       if item["filename"] == articles_path.name)
    if hashlib.sha256(raw).hexdigest() != manifest["articles_sha256"] or source_hash != manifest["articles_sha256"]:
        raise ValueError("Published abstracts do not match the batch and its mentions.")
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    pmids = [str(row["pmid"]) for row in rows]
    if (len(set(pmids)) != len(pmids) or pmids != list(map(str, manifest["pmids"]))
            or set(pmids) != set(map(str, inventory["pmids"]))):
        raise ValueError("Published article IDs do not match the batch and its mentions.")
    return [{"pmid": pmid, "title": row["title"], "abstract": row["abstract"]}
            for pmid, row in zip(pmids, rows)]


def batch_dataset(runs_dir, run_id):
    folder, manifest = batch_manifest(runs_dir, run_id)
    if manifest["status"] != "complete":
        raise ValueError("Only completed runs can be loaded.")
    equivalence_raw = (folder / "equivalence.json").read_bytes()
    order_raw = (folder / "concept_order.json").read_bytes()
    placement_raw = (folder / "incremental-placement/placement.json").read_bytes()
    placement = json.loads(placement_raw)
    provenance = placement["input_sha256"]
    if (placement["status"] != "complete"
            or hashlib.sha256(placement_raw).hexdigest() != manifest["placement_sha256"]
            or hashlib.sha256(equivalence_raw).hexdigest() != provenance["equivalence.json"]
            or hashlib.sha256(order_raw).hexdigest() != provenance["concept_order.json"]):
        raise ValueError("Placement and its inputs do not match the batch manifest.")
    inventory = read_json(folder / "input_mentions.json")
    equivalence = json.loads(equivalence_raw)
    order = json.loads(order_raw)
    if order["method"] != "llm_broad_first":
        raise ValueError("Unsupported concept ordering method.")
    apply_entity_order(equivalence["concepts"], order)
    run = {key: manifest.get(key) for key in (
        "run_id", "title", "created_at", "extraction_commit", "extraction_model",
    )}
    return {
        "run": run,
        "articles": read_articles(folder, manifest, inventory),
        "mentions": inventory["mentions"],
        "verification_candidates": inventory["verification_candidates"],
        "normalization": read_json(folder / "normalization.json"),
        "equivalence": equivalence,
        "retrieval": read_json(folder / "retrieval/retrieved_pairs.json"),
        "conceptOrder": order,
        "placement": placement,
    }


class Handler(BaseHTTPRequestHandler):
    runs_dir: Path

    def do_GET(self):
        url = urlsplit(self.path)
        path = url.path
        if path == "/api/batch":
            requested = parse_qs(url.query, keep_blank_values=True).get("run", [])
            if len(requested) > 1:
                self.send_error(400, "Select one run")
                return
            try:
                batches = complete_batches(self.runs_dir)
                run_id = requested[0] if requested else batches[0]["run_id"] if batches else None
                data = batch_dataset(self.runs_dir, run_id) if run_id is not None else {"run": None}
                data["availableRuns"] = batches
                body = json.dumps(data, ensure_ascii=False).encode("utf-8")
            except FileNotFoundError:
                self.send_error(404, "Entity run or a required artifact was not found")
                return
            except (OSError, ValueError, KeyError, StopIteration, TypeError, AttributeError):
                self.send_error(503, "Entity run data unavailable or inconsistent")
                return
            mime = "application/json; charset=utf-8"
        elif path == "/health":
            body, mime = b"LiverKG entity ontology workbench", "text/plain; charset=utf-8"
        elif path in STATIC and STATIC[path].is_file():
            body = STATIC[path].read_bytes()
            mime = mimetypes.guess_type(STATIC[path].name)[0] or "application/octet-stream"
            if mime in {"text/html", "text/css", "text/javascript", "application/javascript"}:
                mime += "; charset=utf-8"
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass


def serve(runs_dir, host="127.0.0.1", port=4320, open_browser=False):
    directory = Path(runs_dir).resolve()

    class BatchHandler(Handler):
        runs_dir = directory

    with ThreadingHTTPServer((host, port), BatchHandler) as server:
        url = f"http://{host}:{server.server_port}/"
        if open_browser:
            opener = threading.Timer(0.15, webbrowser.open, args=(url,))
            opener.daemon = True
            opener.start()
        print(f"LiverKG 实体本体工作台  {url}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
