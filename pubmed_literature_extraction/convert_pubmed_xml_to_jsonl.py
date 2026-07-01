#!/usr/bin/env python3
"""
convert_pubmed_xml_to_jsonl.py — 将 PubMed XML 批次文件转换为 Pipeline 标准 JSONL 输入。

输入:
    data_organized_backup_20260616/datasets/pubmed/pubmed_batch_*.xml

输出:
    extraction_output/pubmed_converted_500.jsonl  (默认 500 条, 质量优先)
    extraction_output/pubmed_converted_full.jsonl (全量)

用法:
    python3 convert_pubmed_xml_to_jsonl.py                  # 默认 500 条
    python3 convert_pubmed_xml_to_jsonl.py --limit 1000     # 指定条数
    python3 convert_pubmed_xml_to_jsonl.py --all            # 全量
    python3 convert_pubmed_xml_to_jsonl.py --limit 500 --quality-report  # 附带质量报告
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
XML_DIR = SCRIPT_DIR / "data_organized_backup_20260616" / "datasets" / "pubmed"
OUTPUT_DIR = SCRIPT_DIR / "extraction_output"

# ── 质量阈值 ──────────────────────────────────────────────
MIN_ABSTRACT_LENGTH = 200      # 摘要最少字符数（过滤空摘要/极短摘要）
MIN_TITLE_LENGTH = 10          # 标题最少字符数
MAX_ABSTRACT_LENGTH = 8000     # 摘要最大字符数（过滤异常长文本）
REJECT_TITLE_KEYWORDS = [      # 标题中出现这些词 → 降级/拒绝
    "correction", "retraction", "withdrawal", "erratum",
]


def parse_pubmed_xml(xml_path: Path) -> list[dict[str, Any]]:
    """解析单个 PubMed XML 文件，返回 (pmid, title, abstract) 列表。"""
    tree = ET.parse(str(xml_path))
    root = tree.getroot()
    articles: list[dict[str, Any]] = []

    for art in root.findall(".//PubmedArticle"):
        # PMID
        pmid_el = art.find(".//PMID")
        pmid = pmid_el.text.strip() if pmid_el is not None and pmid_el.text else ""

        # Title
        title_el = art.find(".//ArticleTitle")
        title = title_el.text.strip() if title_el is not None and title_el.text else ""
        # 有时候 title 在多个子元素中
        if not title and title_el is not None:
            title = " ".join(title_el.itertext()).strip()

        # Abstract — 可能有多个 AbstractText 子元素
        abstract_parts: list[str] = []
        abs_el = art.find(".//Abstract")
        if abs_el is not None:
            for abs_text in abs_el.findall("AbstractText"):
                label = abs_text.get("Label", "")
                text = abs_text.text or ""
                # 合并子元素文本
                full_text = "".join(abs_text.itertext()).strip()
                if label:
                    full_text = f"{label}: {full_text}"
                if full_text:
                    abstract_parts.append(full_text)
        abstract = " ".join(abstract_parts).strip()

        # 部分旧的 PubMed XML 用 Article/Abstract/AbstractText
        if not abstract:
            abs_texts = art.findall(".//AbstractText")
            parts = []
            for at in abs_texts:
                t = "".join(at.itertext()).strip()
                if t:
                    parts.append(t)
            abstract = " ".join(parts).strip()

        if pmid:
            articles.append({
                "pmid": pmid,
                "title": title,
                "abstract": abstract,
            })

    return articles


def assess_quality(article: dict[str, Any]) -> dict[str, Any]:
    """评估单篇文章质量,返回 flags + score。"""
    title = article.get("title", "")
    abstract = article.get("abstract", "")
    flags: list[str] = []
    score = 100

    # 1. 有标题
    if not title or len(title) < MIN_TITLE_LENGTH:
        flags.append("short_title")
        score -= 40

    # 2. 有摘要
    if not abstract:
        flags.append("no_abstract")
        score -= 100
    elif len(abstract) < MIN_ABSTRACT_LENGTH:
        flags.append("short_abstract")
        score -= 30
    elif len(abstract) > MAX_ABSTRACT_LENGTH:
        flags.append("long_abstract")
        score -= 5

    # 3. 标题含拒收关键词
    title_lower = title.lower()
    for kw in REJECT_TITLE_KEYWORDS:
        if kw in title_lower:
            flags.append(f"reject_keyword:{kw}")
            score -= 80
            break

    # 4. 标题看起来像英文 (ASCII 占比)
    ascii_chars = sum(1 for c in title if ord(c) < 128)
    if len(title) > 0 and ascii_chars / len(title) < 0.7:
        flags.append("non_english_title")
        score -= 20

    # 5. 摘要像英文
    if abstract:
        ascii_abs = sum(1 for c in abstract[:200] if ord(c) < 128)
        if ascii_abs / min(len(abstract), 200) < 0.7:
            flags.append("non_english_abstract")
            score -= 20

    article["quality_flags"] = flags
    article["quality_score"] = max(0, score)
    return article


def extract_mesh_terms(article: dict[str, Any], xml_articles: dict[str, Any]) -> None:
    """从原始解析结果中提取 MeSH terms（如果有的话）。"""
    # 当前不保留 MeSH,仅占位
    pass


def build_output(
    xml_dir: Path,
    limit: int | None = None,
    quality_threshold: int = 60,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    主转换逻辑。

    Args:
        xml_dir: XML 文件目录
        limit: 输出条数上限, None = 全量
        quality_threshold: 最低质量分数 (0-100), 低于此分不输出

    Returns:
        (output_articles, stats)
    """
    # ── 1. 解析所有 XML ──────────────────────────────────
    all_articles: list[dict[str, Any]] = []
    batch_stats: dict[str, int] = {}

    for xml_path in sorted(xml_dir.glob("pubmed_batch_*.xml")):
        articles = parse_pubmed_xml(xml_path)
        batch_name = xml_path.name
        batch_stats[batch_name] = len(articles)
        all_articles.extend(articles)

    raw_total = len(all_articles)

    # ── 2. 按 PMID 去重 ───────────────────────────────────
    seen: dict[str, dict[str, Any]] = {}
    dup_count = 0
    for art in all_articles:
        pmid = art["pmid"]
        if pmid in seen:
            # 保留摘要更长的那条
            if len(art.get("abstract", "")) > len(seen[pmid].get("abstract", "")):
                seen[pmid] = art
            dup_count += 1
        else:
            seen[pmid] = art
    deduped = list(seen.values())

    # ── 3. 质量评估 ───────────────────────────────────────
    for art in deduped:
        assess_quality(art)

    # ── 4. 按质量分数排序 ──────────────────────────────────
    deduped.sort(key=lambda a: a["quality_score"], reverse=True)

    # ── 5. 统计 ────────────────────────────────────────────
    quality_dist = Counter()
    no_abs = 0
    short_abs = 0
    reject_kw = 0
    non_eng = 0

    for art in deduped:
        score = art["quality_score"]
        quality_dist[score // 10 * 10] += 1
        flags = art.get("quality_flags", [])
        if "no_abstract" in flags:
            no_abs += 1
        if "short_abstract" in flags:
            short_abs += 1
        if any(f.startswith("reject_keyword") for f in flags):
            reject_kw += 1
        if "non_english_title" in flags:
            non_eng += 1

    # ── 6. 筛选输出 ────────────────────────────────────────
    eligible = [a for a in deduped if a["quality_score"] >= quality_threshold]
    if limit and limit < len(eligible):
        output = eligible[:limit]
    else:
        output = eligible

    stats = {
        "raw_total": raw_total,
        "duplicates_removed": dup_count,
        "unique_articles": len(deduped),
        "no_abstract": no_abs,
        "short_abstract": short_abs,
        "reject_keywords": reject_kw,
        "non_english_title": non_eng,
        "quality_distribution": dict(sorted(quality_dist.items(), reverse=True)),
        "eligible_above_threshold": len(eligible),
        "output_count": len(output),
        "quality_threshold": quality_threshold,
        "batch_stats": batch_stats,
        "abstract_lengths": {
            "min": min(len(a.get("abstract", "")) for a in output) if output else 0,
            "max": max(len(a.get("abstract", "")) for a in output) if output else 0,
            "avg": round(sum(len(a.get("abstract", "")) for a in output) / len(output)) if output else 0,
            "median": sorted(len(a.get("abstract", "")) for a in output)[len(output)//2] if output else 0,
        },
    }

    return output, stats


def print_quality_report(stats: dict[str, Any], output: list[dict[str, Any]]) -> None:
    """打印人可读的质量报告。"""
    print()
    print("=" * 70)
    print("  PubMed XML → JSONL 转换质量报告")
    print("=" * 70)
    print()
    print(f"  原始文章数:         {stats['raw_total']:>6}")
    print(f"  去重 (同PMID):       {stats['duplicates_removed']:>6}")
    print(f"  唯一文章数:         {stats['unique_articles']:>6}")
    print(f"  质量阈值:           {stats['quality_threshold']:>6} 分")
    print(f"  合格文章数:         {stats['eligible_above_threshold']:>6}")
    print(f"  最终输出:           {stats['output_count']:>6}")
    print()
    print("  质量问题分布:")
    print(f"    无摘要 (no_abstract):           {stats['no_abstract']:>5}")
    print(f"    摘要过短 (short_abstract):       {stats['short_abstract']:>5}")
    print(f"    拒收关键词 (correction/erratum):  {stats['reject_keywords']:>5}")
    print(f"    疑似非英文:                     {stats['non_english_title']:>5}")
    print()
    print("  质量分数分布:")
    for score_bucket in sorted(stats["quality_distribution"], reverse=True):
        count = stats["quality_distribution"][score_bucket]
        bar = "█" * (count // 5)
        print(f"    {score_bucket:>3}-{score_bucket+9:>3}: {count:>5}  {bar}")
    print()
    al = stats["abstract_lengths"]
    print(f"  输出摘要长度 — min: {al['min']} | median: {al['median']} | avg: {al['avg']} | max: {al['max']}")
    print()
    print("  各批次文章数:")
    for batch_name, count in sorted(stats["batch_stats"].items()):
        print(f"    {batch_name}: {count}")
    print()

    # ── 抽样展示 ──────────────────────────────────────────
    print("  ── 抽样展示 (质量最高 3 + 最低 3) ──")
    print()
    for i, art in enumerate(output[:3]):
        print(f"  ★ #{i+1}  PMID:{art['pmid']}  score={art['quality_score']}  flags={art.get('quality_flags',[])}")
        print(f"     Title: {art['title'][:120]}")
        print(f"     Abstract: {art['abstract'][:200]}...")
        print()
    print("  ...")
    for i, art in enumerate(output[-3:]):
        idx = len(output) - 3 + i + 1
        print(f"  · #{idx}  PMID:{art['pmid']}  score={art['quality_score']}  flags={art.get('quality_flags',[])}")
        print(f"     Title: {art['title'][:120]}")
        print(f"     Abstract: {art['abstract'][:200]}...")
    print()
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="将 PubMed XML 批次转换为 Pipeline 标准 JSONL 输入"
    )
    parser.add_argument(
        "--limit", "-n", type=int, default=500,
        help="输出条数上限 (默认 500)"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="输出全量合格文章"
    )
    parser.add_argument(
        "--quality-threshold", "-q", type=int, default=60,
        help="最低质量分数 (0-100), 默认 60"
    )
    parser.add_argument(
        "--quality-report", "-r", action="store_true",
        help="打印详细质量报告"
    )
    parser.add_argument(
        "--output-dir", "-o", default=str(OUTPUT_DIR),
        help="输出目录"
    )
    parser.add_argument(
        "--run-id", default="",
        help="输出文件后缀标识"
    )
    args = parser.parse_args()

    limit = None if args.all else args.limit

    # ── 检查 XML 目录 ─────────────────────────────────────
    if not XML_DIR.exists():
        print(f"[ERROR] XML 目录不存在: {XML_DIR}")
        sys.exit(1)

    xml_files = sorted(XML_DIR.glob("pubmed_batch_*.xml"))
    if not xml_files:
        print(f"[ERROR] 未找到 pubmed_batch_*.xml 文件")
        sys.exit(1)
    print(f"[INFO] 找到 {len(xml_files)} 个 XML 批次文件")

    # ── 转换 ───────────────────────────────────────────────
    output, stats = build_output(
        xml_dir=XML_DIR,
        limit=limit,
        quality_threshold=args.quality_threshold,
    )

    # ── 质量报告 ───────────────────────────────────────────
    if args.quality_report:
        print_quality_report(stats, output)

    # ── 写入 JSONL ─────────────────────────────────────────
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    suffix = f"_{args.run_id}" if args.run_id else ""
    limit_suffix = f"_{len(output)}" if args.all else f"_{len(output)}"
    jsonl_path = out_dir / f"pubmed_converted{suffix}{limit_suffix}.jsonl"

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for art in output:
            # 写入 pipeline 需要的标准字段: pmid, title, abstract, source
            record = {
                "pmid": art["pmid"],
                "title": art["title"],
                "abstract": art["abstract"],
                "source": "PubMed",
                "quality_score": art["quality_score"],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    stats_path = out_dir / f"pubmed_converted{suffix}{limit_suffix}_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"[OK] 输出 {len(output)} 条 → {jsonl_path}")
    print(f"[OK] 统计 → {stats_path}")
    print()
    print(f"  下一步运行:")
    print(f"  python3 multi_stage_extraction_pipeline.py \\")
    print(f"    --input {jsonl_path} \\")
    print(f"    --limit {min(len(output), 10)} \\")
    print(f"    --run-id pubmed_converted_dryrun_001 \\")
    print(f"    --skip-neo4j")


if __name__ == "__main__":
    main()
