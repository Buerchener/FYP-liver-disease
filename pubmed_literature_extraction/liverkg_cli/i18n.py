from __future__ import annotations

import locale
import os


MESSAGES: dict[str, dict[str, str]] = {
    "en": {
        "app_help": "LiverKG research CLI for PubMed knowledge graph extraction.",
        "init_done": "Configuration written.",
        "doctor_title": "LiverKG Doctor",
        "ok": "OK",
        "warn": "WARN",
        "fail": "FAIL",
        "dry_run_default": "Default mode is dry-run. Neo4j writes require explicit confirmation.",
        "extract_started": "Extraction started.",
        "detached_started": "Background run started.",
        "no_runs": "No runs found.",
        "not_found": "Run not found.",
        "report_missing": "Report not found yet.",
        "confirm_write": "Type the run id to confirm Neo4j writes",
        "stopped": "Stop signal sent.",
    },
    "zh": {
        "app_help": "LiverKG PubMed 文献知识图谱抽取科研 CLI。",
        "init_done": "配置已写入。",
        "doctor_title": "LiverKG 环境检查",
        "ok": "正常",
        "warn": "警告",
        "fail": "失败",
        "dry_run_default": "默认 dry-run；Neo4j 写入必须显式确认。",
        "extract_started": "抽取已启动。",
        "detached_started": "后台任务已启动。",
        "no_runs": "没有找到运行记录。",
        "not_found": "没有找到该运行。",
        "report_missing": "报告尚未生成。",
        "confirm_write": "请输入 run id 以确认 Neo4j 写入",
        "stopped": "已发送停止信号。",
    },
}


def detect_language(explicit: str | None = None) -> str:
    if explicit in {"zh", "en"}:
        return explicit
    env_lang = os.environ.get("LIVERKG_LANG", "").strip().lower()
    if env_lang in {"zh", "en"}:
        return env_lang
    loc = (locale.getlocale()[0] or locale.getdefaultlocale()[0] or "").lower()
    return "zh" if loc.startswith("zh") else "en"


def tr(key: str, lang: str = "en") -> str:
    return MESSAGES.get(lang, MESSAGES["en"]).get(key, MESSAGES["en"].get(key, key))
