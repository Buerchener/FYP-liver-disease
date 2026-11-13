"""Reproducible offline preparation; no LLM, embedding, or extraction imports."""
from datetime import datetime, timezone
from pathlib import Path
import glob
import hashlib
import json
import re
import subprocess

from . import __version__
from .adapters import load_sources, PROFILES
from .concepts.registry import normalize_candidates
from .models import to_dict
from .preparation.sources import SourceFile
from .schema import record_schema
from .taxonomy.initialization import build_initial_taxonomy, prepared_input_snapshot
from .validation import validate_bundle, validate_preparation

MODULE = Path(__file__).resolve().parent


def dump(path, value):
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def dump_lines(path, rows):
    with path.open("x", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def _resolved(base, name):
    path = Path(name).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def prepare(config_path, source_root=None, output_parent=None, run_id=None):
    config_source = SourceFile.read(config_path)
    config = config_source.json()
    allowed = {"schema_version", "profile", "namespace", "source_root", "input_files", "source_run_id",
               "data_origin", "roots_file", "assignments_file", "output_root", "allow_model_calls",
               "run_induction", "semantic_synonym_merging", "split"}
    if set(config) - allowed:
        raise ValueError("Unknown configuration fields: " + str(sorted(set(config) - allowed)))
    if config.get("schema_version") != "event-taxonomy-preparation-config/1":
        raise ValueError("Use a preparation config, not the design-only plan")
    if config.get("profile") not in PROFILES:
        raise ValueError("Unsupported adapter profile")
    if any(config.get(k) is not False for k in ("allow_model_calls", "run_induction", "semantic_synonym_merging")):
        raise ValueError("Task 1 requires model calls, induction and semantic merging disabled")
    base = config_source.path.parent
    root_value = source_root or config.get("source_root")
    if not root_value:
        raise ValueError("Source root required; supply --source-root")
    root = _resolved(base, root_value)
    patterns = config.get("input_files")
    if not isinstance(patterns, list) or not patterns or any(not isinstance(p, str) for p in patterns):
        raise ValueError("input_files must be a nonempty list of paths/globs")
    paths = []
    for pattern in patterns:
        matches = sorted(Path(p).resolve() for p in glob.glob(str(root / pattern)))
        if not matches:
            raise ValueError("No files match input: " + pattern)
        paths.extend(matches)
    if len(set(paths)) != len(paths):
        raise ValueError("Input globs overlap")
    sources = [SourceFile.read(path) for path in paths]
    roots_source = SourceFile.read(_resolved(base, config["roots_file"]))
    assignments_source = (SourceFile.read(_resolved(base, config["assignments_file"]))
                          if config.get("assignments_file") else None)
    assignments = assignments_source.json() if assignments_source else []
    if not isinstance(assignments, list):
        raise ValueError("Assignments file must be an array of explicit decisions")
    out_root = _resolved(base, output_parent or config["output_root"])
    identifier = run_id or (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + "-" + config["profile"])
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,119}", identifier):
        raise ValueError("Unsafe run ID")
    out = out_root / identifier
    # Never write into the existing extraction code, baselines, or source folders.
    protected = [MODULE.parent / "atlas_rag", MODULE.parent / "liver_baseline"] + [p.parent for p in paths]
    if any(out == p or p in out.parents for p in protected):
        raise ValueError("Output directory overlaps a protected source directory")
    out.mkdir(parents=True, exist_ok=False)
    (out / "sources").mkdir()
    manifest = {"schema_version": "event-taxonomy-run/1", "status": "preparing", "run_id": identifier,
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "module_version": __version__,
        "task": "offline_data_preparation", "new_api_calls": 0, "induction_executed": False,
        "config": config, "source_root": str(root), "split": config.get("split", "unspecified"),
        "input_semantically_frozen": False, "sources": [], "code_sha256": {}}
    try:
        for index, source in enumerate(sources + [roots_source, config_source] + ([assignments_source] if assignments_source else [])):
            name = "sources/{:03d}_{}".format(index, source.path.name)
            with (out / name).open("xb") as f:
                f.write(source.content)
            manifest["sources"].append({"original_path": str(source.path), "snapshot_path": name,
                                        "sha256": source.sha256})
        for path in sorted(MODULE.rglob("*.py")):
            relative = path.relative_to(MODULE)
            if "__pycache__" not in relative.parts and "runs" not in relative.parts:
                content = path.read_bytes()
                manifest["code_sha256"][str(relative)] = hashlib.sha256(content).hexdigest()
                code_copy = out / "code_snapshot/event_taxonomy" / relative
                code_copy.parent.mkdir(parents=True, exist_ok=True)
                with code_copy.open("xb") as f:
                    f.write(content)
        try:
            manifest["upstream_commit"] = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=MODULE.parent, text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            manifest["upstream_commit"] = None
        replay = dict(config, source_root=".",
                      input_files=[row["snapshot_path"] for row in manifest["sources"][:len(sources)]],
                      roots_file=manifest["sources"][len(sources)]["snapshot_path"], output_root="..",
                      assignments_file=manifest["sources"][-1]["snapshot_path"] if assignments_source else None)
        dump(out / "replay_config.json", replay)
        bundle = load_sources(sources, config["profile"], config["namespace"], config["data_origin"], config.get("source_run_id"))
        import_validation = validate_bundle(bundle, sources)
        concepts, mappings, registry_issues = normalize_candidates(bundle.candidates)
        taxonomy = build_initial_taxonomy(concepts, bundle.candidates, roots_source.json(), assignments)
        snapshot = prepared_input_snapshot(taxonomy)
        graph_validation = validate_preparation(bundle, concepts, mappings, taxonomy, snapshot)
        for name, rows in [("documents", bundle.documents), ("events", bundle.events),
                           ("evidence", bundle.evidence), ("concept_candidates", bundle.candidates),
                           ("unique_concepts", concepts)]:
            dump_lines(out / (name + ".jsonl"), [to_dict(r) for r in rows])
        dump(out / "candidate_concept_mapping.json", mappings)
        dump(out / "concept_generation_records.json", bundle.concept_generation_records)
        dump(out / "import_issues.json", bundle.issues + registry_issues)
        dump(out / "initial_taxonomy.json", to_dict(taxonomy))
        dump(out / "prepared_input_snapshot.json", snapshot)
        review = [{"concept_id": c.concept_id, "concept_label": c.concept_label,
                   "concept_description": c.concept_description, "candidate_ids": c.candidate_ids,
                   "supporting_event_ids": c.supporting_event_ids, "evidence_ids": c.evidence,
                   "granularity_status": c.granularity_status, "suggested_parent_ids": [],
                   "review_status": "pending", "reason": "Definition, granularity and type validity require review"}
                  for c in concepts if c.concept_role == "event_type"]
        dump(out / "concept_review_queue.json", review)
        validation = {"schema_version": "event-taxonomy-validation/1", "import": import_validation,
                      "taxonomy": graph_validation, "semantic_evaluation": "not_performed"}
        dump(out / "validation.json", validation)
        dump(out / "record_schema.json", record_schema())
        report = ["# Event Taxonomy Task 1 数据准备结果", "",
                  "本次仅离线导入和结构验证，没有运行模型或递归归纳。所有来源保留快照与哈希。", "",
                  "| 项目 | 数量／状态 |", "|---|---:|",
                  "| 文档 | {} |".format(len(bundle.documents)),
                  "| 事件实例 | {} |".format(len(bundle.events)),
                  "| 概念候选（逐次关联） | {} |".format(len(bundle.candidates)),
                  "| Unique Concept（全部角色） | {} |".format(len(concepts)),
                  "| 事件类型概念 | {} |".format(graph_validation["input_concepts"]),
                  "| 主题概念 | {} |".format(graph_validation["unique_concept_roles"].get("topic", 0)),
                  "| 初始图节点（含根） | {} |".format(len(taxonomy.nodes)),
                  "| 父子边 | {} |".format(len(taxonomy.parent_child_edges)),
                  "| 尚未归属的输入概念 | {} |".format(len(taxonomy.unplaced_concept_ids)),
                  "| 环路／悬空引用 | 0／0 |", "",
                  "证据定位状态：`" + json.dumps(import_validation["evidence_status"], ensure_ascii=False) + "`。", "",
                  "原始候选、主题、证据异常与排除状态均保留。字符串匹配不代表语义正确。",
                  "初始图状态为 partial。输入快照用于复现，尚未经过概念定义及粒度审核，不能视为正式细粒度 U0 或 Gold。", "",
                  "`concept_review_queue.json` 提供待审查概念及事件／证据 ID；`initial_taxonomy.json` 保存三个预定义根和全部输入类型概念。",
                  "`prepared_input_snapshot.json` 固定本次导入内容；后续算法必须保留输入概念及证据映射。", "",
                  "本批次数据划分：`" + str(config.get("split", "unspecified")) + "`。未计算算法效果或人工语义准确率。", ""]
        (out / "report.md").write_text("\n".join(report), encoding="utf-8")
        for source in sources + [roots_source, config_source] + ([assignments_source] if assignments_source else []):
            source.assert_unchanged()
        for relative, expected in manifest["code_sha256"].items():
            if hashlib.sha256((MODULE / relative).read_bytes()).hexdigest() != expected:
                raise ValueError("Module source changed during preparation: " + relative)
        manifest.update(status="complete", completed_at_utc=datetime.now(timezone.utc).isoformat(),
                        validation_file="validation.json", source_files_unchanged=True)
    except Exception as error:
        manifest.update(status="failed", error=str(error))
        dump(out / "manifest.json", manifest)
        raise
    dump(out / "manifest.json", manifest)
    dump(out / "checksums.json", {str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in sorted(out.rglob("*")) if p.is_file()})
    return out
