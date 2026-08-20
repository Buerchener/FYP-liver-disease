from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import typer
import requests
from rich.console import Console
from rich.table import Table

from .config import PROFILES, LiverKGConfig, load_config, set_config_value, write_user_config
from .i18n import detect_language, tr
from .normalizer import normalize_input
from .paths import cache_dir, project_root, user_config_path
from .runs import RunSpec, latest_run_id, list_runs, load_run_spec, read_status, save_run_spec, stop_run, update_status
from .runs import run_dir
from .security import get_secret, mask_secret, redact_mapping, set_secret
from .service import validate_write_safety


console = Console()
app = typer.Typer(help=tr("app_help"))
runs_app = typer.Typer(help="Manage LiverKG runs.")
config_app = typer.Typer(help="Manage non-sensitive LiverKG configuration.")
cache_app = typer.Typer(help="Inspect or clear LiverKG caches.")
research_app = typer.Typer(help="Research-only evaluation wrappers.")
app.add_typer(runs_app, name="runs")
app.add_typer(config_app, name="config")
app.add_typer(cache_app, name="cache")
app.add_typer(research_app, name="research")


LANG = "en"


@app.callback()
def _callback(
    lang: Optional[str] = typer.Option(None, "--lang", help="Language: zh or en."),
) -> None:
    global LANG
    LANG = detect_language(lang)


def _now_run_id(limit: int) -> str:
    return f"liverkg_{limit}_{time.strftime('%Y%m%d_%H%M%S')}"


def _console_table(title: str, columns: list[str]) -> Table:
    table = Table(title=title)
    for column in columns:
        table.add_column(column)
    return table


def _chat_preflight(api_base: str, api_key: str, model: str) -> str:
    if not api_base or not api_key or not model:
        return "missing api_base, api_key or model"
    url = api_base.rstrip("/") + "/chat/completions"
    response = requests.post(
        url,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": model,
            "messages": [{"role": "user", "content": "Return only {\"ok\":true}."}],
            "temperature": 0,
            "max_tokens": 16,
        },
        timeout=20,
    )
    response.raise_for_status()
    return "connected"


@app.command()
def init(
    non_interactive: bool = typer.Option(False, "--non-interactive", help="Write defaults without prompts."),
) -> None:
    """Create or update the local LiverKG configuration."""
    cfg = load_config()
    if not non_interactive:
        cfg.model_id = typer.prompt("Primary model", default=cfg.model_id)
        cfg.api_base = typer.prompt("Primary API base", default=cfg.api_base)
        cfg.default_profile = typer.prompt("Default profile", default=cfg.default_profile)
        cfg.neo4j_uri = typer.prompt("Neo4j URI", default=cfg.neo4j_uri)
        cfg.neo4j_user = typer.prompt("Neo4j user", default=cfg.neo4j_user)
        cfg.neo4j_database = typer.prompt("Neo4j database", default=cfg.neo4j_database)
        cfg.ncbi_email = typer.prompt("NCBI email", default=cfg.ncbi_email or "")
        if typer.confirm("Store or update API keys in the system keychain?", default=False):
            for name, label in (
                ("gemini_api_key", "Gemini-compatible API key"),
                ("deepseek_api_key", "DeepSeek API key"),
                ("qwen_api_key", "Qwen/DashScope API key"),
                ("neo4j_password", "Neo4j password"),
            ):
                value = typer.prompt(label, default="", hide_input=True)
                if value:
                    set_secret(name, value)
    path = write_user_config(cfg)
    console.print(f"{tr('init_done', LANG)} {path}")


@app.command()
def doctor(
    remote: bool = typer.Option(False, "--remote", help="Also perform remote connectivity checks."),
) -> None:
    """Check installation, configuration, credentials and safety posture."""
    cfg = load_config()
    table = _console_table(tr("doctor_title", LANG), ["Check", "Status", "Detail"])

    def row(name: str, ok: bool, detail: str = "", warn: bool = False) -> None:
        status = tr("ok", LANG) if ok and not warn else tr("warn", LANG) if warn else tr("fail", LANG)
        style = "green" if ok and not warn else "yellow" if warn else "red"
        table.add_row(name, f"[{style}]{status}[/{style}]", detail)

    row("Python", sys.version_info >= (3, 11), sys.version.split()[0])
    for module in ("typer", "rich", "requests", "neo4j", "langextract", "google.genai", "openai"):
        try:
            __import__(module)
            row(module, True)
        except Exception as exc:
            row(module, False, str(exc))
    row("Config", user_config_path().exists(), str(user_config_path()), warn=not user_config_path().exists())
    row("Primary key", bool(get_secret("gemini_api_key")), mask_secret(get_secret("gemini_api_key")), warn=not bool(get_secret("gemini_api_key")))
    row("DeepSeek key", bool(get_secret("deepseek_api_key")), mask_secret(get_secret("deepseek_api_key")), warn=not bool(get_secret("deepseek_api_key")))
    row("Qwen key", bool(get_secret("qwen_api_key")), mask_secret(get_secret("qwen_api_key")), warn=not bool(get_secret("qwen_api_key")))
    row("Neo4j password", bool(get_secret("neo4j_password")), mask_secret(get_secret("neo4j_password")), warn=not bool(get_secret("neo4j_password")))
    row("Default safety", True, tr("dry_run_default", LANG))
    if cfg.rule_bundle:
        row("Rule bundle", Path(cfg.rule_bundle).exists(), cfg.rule_bundle)
    else:
        row("Rule bundle", True, "not configured", warn=True)
    if cfg.conformal_calibration:
        row("Calibration", Path(cfg.conformal_calibration).exists(), cfg.conformal_calibration)
    else:
        row("Calibration", True, "not configured", warn=True)

    if remote:
        for name, api_base, model, secret_name in (
            ("Primary model", cfg.api_base, cfg.model_id, "gemini_api_key"),
            ("DeepSeek model", cfg.second_llm_api_base, cfg.second_llm_model_id, "deepseek_api_key"),
            ("Qwen model", cfg.qwen_api_base, cfg.aux_critic_model, "qwen_api_key"),
        ):
            try:
                detail = _chat_preflight(api_base, get_secret(secret_name), model)
                row(name, detail == "connected", detail)
            except Exception as exc:
                row(name, False, str(exc))
        try:
            from neo4j import GraphDatabase

            password = get_secret("neo4j_password")
            if password:
                with GraphDatabase.driver(cfg.neo4j_uri, auth=(cfg.neo4j_user, password)) as driver:
                    driver.verify_connectivity()
                row("Neo4j remote", True, cfg.neo4j_uri)
            else:
                row("Neo4j remote", False, "missing password")
        except Exception as exc:
            row("Neo4j remote", False, str(exc))

    console.print(table)


@app.command()
def extract(
    input: Optional[Path] = typer.Argument(None, help="Input JSONL or PubMed XML."),
    pmid: list[str] = typer.Option([], "--pmid", help="Fetch one PMID. Can be repeated."),
    profile: str = typer.Option(None, "--profile", help="quality, balanced, speed or legacy."),
    limit: int = typer.Option(50, "--limit", "-n"),
    run_id: Optional[str] = typer.Option(None, "--run-id"),
    detach: bool = typer.Option(False, "--detach", help="Run in the background."),
    resume: bool = typer.Option(False, "--resume", help="Resume from article checkpoints."),
    write_neo4j: bool = typer.Option(False, "--write-neo4j", help="Enable Neo4j writes after confirmation."),
    yes: bool = typer.Option(False, "--yes", help="Non-interactive confirmation for safe operations."),
) -> None:
    """Run PubMed extraction with a product profile."""
    cfg = load_config({"default_profile": profile} if profile else None)
    selected_profile = profile or cfg.default_profile
    if selected_profile not in PROFILES:
        raise typer.BadParameter(f"unknown profile: {selected_profile}")
    run_id = run_id or _now_run_id(limit)
    current_run_dir = run_dir(run_id)
    if current_run_dir.exists() and not resume:
        raise typer.BadParameter(f"run id already exists: {run_id}")

    confirmed = False
    if write_neo4j:
        if yes:
            confirmed = True
        elif sys.stdin.isatty():
            typed = typer.prompt(f"{tr('confirm_write', LANG)} [{run_id}]")
            confirmed = typed == run_id
        validate_write_safety(
            RunSpec(run_id=run_id, write_neo4j=True, neo4j_uri=cfg.neo4j_uri, neo4j_user=cfg.neo4j_user, neo4j_database=cfg.neo4j_database),
            confirmed=confirmed,
        )

    input_dir = current_run_dir / "inputs"
    snapshot_dir = current_run_dir / "snapshots"
    normalized_path = input_dir / "input.jsonl"
    if not resume:
        manifest = normalize_input(
            input_path=input,
            pmids=list(pmid or []),
            output_path=normalized_path,
            ncbi_email=cfg.ncbi_email,
            ncbi_tool=cfg.ncbi_tool,
            snapshot_dir=snapshot_dir,
        )
    else:
        spec = load_run_spec(run_id)
        manifest = spec.input_manifest
        normalized_path = Path(spec.normalized_input_path)

    profile_args = dict(PROFILES[selected_profile])
    spec = RunSpec(
        run_id=run_id,
        profile=selected_profile,
        input_path=str(input) if input else "",
        normalized_input_path=str(normalized_path),
        output_dir=cfg.output_dir,
        limit=limit,
        dry_run=not write_neo4j,
        write_neo4j=write_neo4j,
        max_workers=int(profile_args.get("max_workers", 5)),
        extraction_inner_max_workers=int(profile_args.get("extraction_inner_max_workers", 1)),
        model_id=cfg.model_id,
        api_base=cfg.api_base,
        neo4j_uri=cfg.neo4j_uri,
        neo4j_user=cfg.neo4j_user,
        neo4j_database=cfg.neo4j_database,
        ncbi_email=cfg.ncbi_email,
        ncbi_tool=cfg.ncbi_tool,
        second_llm_enabled=True,
        second_llm_model_id=cfg.second_llm_model_id,
        second_llm_api_base=cfg.second_llm_api_base,
        aux_primary_model=cfg.second_llm_model_id,
        aux_critic_model=cfg.aux_critic_model,
        rule_bundle=cfg.rule_bundle,
        conformal_calibration=cfg.conformal_calibration,
        agent_args={
            "rule_bundle": cfg.rule_bundle,
            "conformal_calibration": cfg.conformal_calibration,
            **profile_args,
        },
        input_manifest=manifest,
        project_root=str(project_root()),
    )
    save_run_spec(spec)
    update_status(run_id, state="queued", pid=None, log_path=str(current_run_dir / "logs" / "liverkg.log"))

    if detach:
        log_path = current_run_dir / "logs" / "liverkg.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_fh = log_path.open("ab")
        cmd = [sys.executable, "-m", "liverkg_cli.worker", run_id]
        if resume:
            cmd.append("--resume")
        proc = subprocess.Popen(cmd, cwd=project_root(), stdout=log_fh, stderr=subprocess.STDOUT, start_new_session=True)
        update_status(run_id, state="running", pid=proc.pid, log_path=str(log_path), detached=True)
        console.print(f"{tr('detached_started', LANG)} run_id={run_id} pid={proc.pid}")
        console.print(f"status: liverkg status {run_id}")
        console.print(f"logs:   liverkg logs {run_id}")
        return

    console.print(f"{tr('extract_started', LANG)} run_id={run_id}")
    from .worker import main as worker_main

    code = worker_main([run_id] + (["--resume"] if resume else []))
    raise typer.Exit(code)


@app.command()
def status(run_id: Optional[str] = typer.Argument(None)) -> None:
    """Show run status. Defaults to the latest run."""
    _show_status(run_id)


@app.command()
def logs(
    run_id: Optional[str] = typer.Argument(None),
    tail: int = typer.Option(80, "--tail"),
) -> None:
    """Print recent log lines for a run."""
    rid = run_id or latest_run_id()
    if not rid:
        console.print(tr("no_runs", LANG))
        return
    status_data = read_status(rid)
    path = Path(status_data.get("log_path") or run_dir(rid) / "logs" / "liverkg.log")
    if not path.exists():
        console.print(f"log not found: {path}")
        return
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    console.print("\n".join(lines[-tail:]))


@app.command()
def report(run_id: str) -> None:
    """Show paths and a compact summary for a completed run."""
    status_data = read_status(run_id)
    report_path = Path(status_data.get("report_path") or project_root() / "extraction_output" / f"agent_report_{run_id}.json")
    result_path = Path(status_data.get("result_path") or project_root() / "extraction_output" / f"agent_results_{run_id}.json")
    if not report_path.exists():
        console.print(f"{tr('report_missing', LANG)} {report_path}")
        return
    data = json.loads(report_path.read_text(encoding="utf-8"))
    table = _console_table(f"LiverKG report {run_id}", ["Field", "Value"])
    for key in ("run_id", "total_articles", "success_rate", "total_entities", "total_relations", "import_ready_relations"):
        if key in data:
            table.add_row(key, str(data[key]))
    table.add_row("report_json", str(report_path))
    table.add_row("results_json", str(result_path))
    table.add_row("run_dir", str(run_dir(run_id)))
    console.print(table)


@app.command()
def resume(run_id: str) -> None:
    """Resume a stopped or failed run from article checkpoints."""
    spec = load_run_spec(run_id)
    save_run_spec(spec)
    console.print(f"resuming {run_id}")
    from .worker import main as worker_main

    raise typer.Exit(worker_main([run_id, "--resume"]))


def _show_status(run_id: Optional[str]) -> None:
    rid = run_id or latest_run_id()
    if not rid:
        console.print(tr("no_runs", LANG))
        return
    try:
        status_data = read_status(rid)
    except FileNotFoundError:
        console.print(tr("not_found", LANG))
        return
    pid = status_data.get("pid") or status_data.get("child_pid")
    alive = False
    if pid:
        try:
            os.kill(int(pid), 0)
            alive = True
        except OSError:
            alive = False
    table = _console_table(f"LiverKG status {rid}", ["Field", "Value"])
    for key in ("state", "pid", "child_pid", "exit_code", "log_path", "result_path", "report_path", "error"):
        if key in status_data:
            table.add_row(key, str(status_data[key]))
    table.add_row("alive", str(alive))
    table.add_row("run_dir", str(run_dir(rid)))
    console.print(table)


@runs_app.command("list")
def runs_list() -> None:
    items = list_runs()
    if not items:
        console.print(tr("no_runs", LANG))
        return
    table = _console_table("LiverKG runs", ["Run", "State", "Profile", "Updated", "Log"])
    for item in items:
        status_data = item["status"]
        spec = item["spec"]
        table.add_row(
            item["run_id"],
            str(status_data.get("state", "")),
            str(spec.get("profile", "")),
            time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(float(status_data.get("updated_at", 0) or 0))),
            str(status_data.get("log_path", "")),
        )
    console.print(table)


@runs_app.command("status")
def runs_status(run_id: Optional[str] = typer.Argument(None)) -> None:
    _show_status(run_id)


@runs_app.command("logs")
def runs_logs(run_id: Optional[str] = typer.Argument(None)) -> None:
    logs(run_id)


@runs_app.command("stop")
def runs_stop(run_id: str) -> None:
    stop_run(run_id)
    console.print(tr("stopped", LANG))


@runs_app.command("resume")
def runs_resume(run_id: str) -> None:
    resume(run_id)


@config_app.command("show")
def config_show(redacted: bool = typer.Option(True, "--redacted/--plain")) -> None:
    cfg = load_config()
    data = cfg.to_toml_dict()
    if redacted:
        data = redact_mapping(data)
    console.print_json(json.dumps(data, ensure_ascii=False, default=str))


@config_app.command("set")
def config_set(key: str, value: str) -> None:
    if any(part in key.lower() for part in ("key", "password", "secret", "token")):
        raise typer.BadParameter("secrets must be stored with `liverkg init` or environment variables")
    path = set_config_value(key, value)
    console.print(f"updated {key} -> {path}")


@config_app.command("profiles")
def config_profiles() -> None:
    table = _console_table("LiverKG profiles", ["Profile", "Core mode", "Cache", "Workers"])
    for name, profile in PROFILES.items():
        table.add_row(
            name,
            f"{profile.get('execution_mode')} / {profile.get('relation_authority')}",
            str(profile.get("extraction_cache_mode")),
            str(profile.get("max_workers")),
        )
    console.print(table)


@cache_app.command("stats")
def cache_stats() -> None:
    roots = [cache_dir(), project_root() / ".cache"]
    table = _console_table("LiverKG cache", ["Path", "Files", "Size MB"])
    for root in roots:
        files = [path for path in root.rglob("*") if path.is_file()] if root.exists() else []
        size = sum(path.stat().st_size for path in files) / (1024 * 1024)
        table.add_row(str(root), str(len(files)), f"{size:.2f}")
    console.print(table)


@cache_app.command("clear")
def cache_clear(
    target: str = typer.Option("user", "--target", help="user, project or all."),
    yes: bool = typer.Option(False, "--yes"),
) -> None:
    targets = []
    if target in {"user", "all"}:
        targets.append(cache_dir())
    if target in {"project", "all"}:
        targets.append(project_root() / ".cache")
    if not targets:
        raise typer.BadParameter("target must be user, project or all")
    for path in targets:
        console.print(f"target: {path}")
    if not yes and not typer.confirm("Clear these cache directories?", default=False):
        raise typer.Exit(1)
    for root in targets:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                try:
                    path.rmdir()
                except OSError:
                    pass
    console.print("cache cleared")


def _run_script(script: str, args: list[str]) -> None:
    cmd = [sys.executable, str(project_root() / script), *args]
    raise typer.Exit(subprocess.call(cmd, cwd=project_root()))


@research_app.command("evaluate")
def research_evaluate(args: list[str] = typer.Argument(None)) -> None:
    _run_script("scripts/evaluate_unified.py", list(args or []))


@research_app.command("ablate")
def research_ablate(args: list[str] = typer.Argument(None)) -> None:
    _run_script("scripts/run_pairwise_judge_experiments.py", list(args or []))


@research_app.command("leakage-check")
def research_leakage_check(args: list[str] = typer.Argument(None)) -> None:
    _run_script("scripts/check_agent_v3_leakage.py", list(args or []))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
