"""Render a self-contained, offline evidence review from normalized run data."""

from __future__ import annotations

import json


def render_review_page(data: dict, *, language: str = "en") -> str:
    # The browser reads the payload as inert JSON. A quote containing HTML
    # cannot terminate the script element or become markup in the page.
    payload = json.dumps(data, separators=(",", ":")).replace("<", "\\u003c")
    return (
        PAGE.replace("__PAGE_LANG__", "zh-CN" if language == "zh" else "en")
        .replace("__REVIEW_DATA__", payload)
    )


PAGE = r"""<!doctype html>
<html lang="__PAGE_LANG__">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="light">
<title>LiverKG · Evidence Review</title>
<style>
:root {
  --ink: #1d2b2b;
  --muted: #586968;
  --paper: #f1f3ed;
  --surface: #fffef9;
  --line: #dce3d9;
  --deep: #112e2c;
  --mint: #cdf3d6;
  --green: #145c43;
  --ochre: #7b4b13;
  --red: #8d3c32;
  --sans: "Aptos", "Helvetica Neue", "PingFang SC", "Microsoft YaHei", sans-serif;
  --serif: Georgia, "Songti SC", "Noto Serif CJK SC", serif;
  --mono: "SFMono-Regular", Consolas, monospace;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--paper); color: var(--ink); font: 15px/1.5 var(--sans); }
button, input, select { font: inherit; }
button { cursor: pointer; }
a { color: var(--green); text-underline-offset: .23em; }
:focus-visible { outline: 3px solid #15815c; outline-offset: 3px; }
[hidden] { display: none !important; }
.wrap { max-width: 1432px; width: calc(100% - 88px); margin: auto; }
.mast { background: var(--deep); color: #f5f8ee; }
.topbar { display: flex; justify-content: space-between; align-items: center; gap: 16px; min-height: 64px; border-bottom: 1px solid #345251; }
.brand { display: flex; align-items: center; gap: 11px; font-size: 16px; font-weight: 720; letter-spacing: .03em; }
.brand-mark { display: inline-block; width: 23px; height: 23px; border: 1px solid #79ac9e; border-radius: 6px; position: relative; }
.brand-mark::before, .brand-mark::after { content: ""; position: absolute; background: #a8e6c5; border-radius: 1px; }
.brand-mark::before { width: 3px; height: 11px; top: 5px; left: 6px; }
.brand-mark::after { width: 9px; height: 3px; bottom: 5px; left: 6px; }
.top-actions { display: flex; align-items: center; gap: 12px; }
.mode, .language { border: 1px solid #50716b; border-radius: 6px; padding: 5px 10px; color: #e0ece1; background: transparent; font-size: 12px; }
.language:hover { background: #2c4c47; }
.language:focus-visible { outline-color: #9fe3bf; }
.hero { display: grid; grid-template-columns: minmax(0, 1fr) minmax(340px, .9fr); gap: clamp(24px, 6vw, 90px); align-items: center; padding: 36px 0 37px; }
h1 { font: 400 clamp(37px, 4vw, 63px)/1.02 var(--serif); margin: 0 0 12px; letter-spacing: -.045em; }
.hero p { margin: 0; color: #bcd3c6; max-width: 42ch; font-size: 15px; }
.hero-stats { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); margin: 0; }
.hero-stats div { display: flex; flex-direction: column-reverse; justify-content: flex-end; border-left: 1px solid #45665f; padding: 3px clamp(10px, 1.4vw, 20px); }
.hero-stats dt { color: #bbd6ca; font-size: 12px; max-width: 13ch; overflow-wrap: anywhere; }
.hero-stats dd { font: 400 clamp(27px, 3.5vw, 43px)/1 var(--serif); color: var(--mint); margin: 0 0 9px; font-variant-numeric: tabular-nums; }
main.wrap { display: grid; grid-template-columns: minmax(268px, 325px) minmax(0, 1fr); align-items: start; gap: 25px; margin-top: 26px; margin-bottom: 25px; }
.rail, .stage { border: 1px solid var(--line); background: var(--surface); border-radius: 14px; min-width: 0; }
.rail-tools { padding: 18px 16px 13px; border-bottom: 1px solid var(--line); }
.rail-tools h2 { margin: 0 0 12px; font-size: 16px; font-weight: 700; }
.filter-box { display: flex; flex-direction: column; gap: 8px; }
.filter-box input, .filter-box select { width: 100%; color: var(--ink); background: #f8f9f4; border: 1px solid #d4ddd2; border-radius: 7px; padding: 8px 10px; }
.filter-box input::placeholder { color: var(--muted); }
.list-count { margin: 10px 0 0; font-size: 12px; color: var(--muted); }
.queue { list-style: none; padding: 0; margin: 0; max-height: 574px; overflow: auto; scrollbar-color: #a9b8aa transparent; }
.queue li + li { border-top: 1px solid #e9ede4; }
.queue button { text-align: left; display: block; width: 100%; border: 0; border-left: 3px solid transparent; background: transparent; padding: 13px 13px 13px 14px; color: var(--ink); }
.queue button:hover { background: #f2f6ec; }
.queue button[aria-current="true"] { border-left-color: #268061; background: #e9f3e8; }
.queue button:focus-visible { outline-offset: -4px; }
.queue-meta { display: flex; flex-wrap: wrap; align-items: center; gap: 5px 8px; margin-bottom: 6px; font-size: 11px; }
.queue-source { color: var(--muted); font-family: var(--mono); }
.queue-edge { display: block; font-weight: 650; line-height: 1.28; overflow-wrap: anywhere; }
.queue-predicate { display: block; margin-top: 4px; font: 11px/1.4 var(--mono); color: var(--muted); overflow-wrap: anywhere; }
.stage { padding: clamp(16px, 2vw, 29px); }
.stage h2 { font: 400 clamp(21px, 2vw, 31px)/1.22 var(--serif); margin: 6px 0 18px; max-width: 60ch; overflow-wrap: anywhere; }
.source-line { display: flex; gap: 6px 14px; flex-wrap: wrap; color: var(--muted); font-size: 12px; }
.source-id { font-family: var(--mono); overflow-wrap: anywhere; }
.graph { display: grid; grid-template-columns: minmax(0, 1fr) minmax(110px, auto) minmax(0, 1fr); align-items: center; margin-bottom: 20px; }
.endpoint { min-width: 0; border: 1px solid #c4d8ce; border-radius: 9px; background: #edf6ef; padding: 13px 15px; }
.endpoint.target { border-color: #d6d7c3; background: #f4f2e5; }
.endpoint small { display: block; color: var(--muted); margin-bottom: 3px; font: 11px/1.4 var(--mono); overflow-wrap: anywhere; }
.endpoint strong { display: block; overflow-wrap: anywhere; font-size: clamp(14px, 1.55vw, 20px); }
.connector { display: grid; place-items: center; position: relative; min-width: 0; padding: 0 13px; text-align: center; }
.connector::before { content: ""; height: 1px; inset: 50% 0 auto 0; background: #9aae9f; position: absolute; }
.connector::after { content: ""; position: absolute; right: 3px; width: 6px; height: 6px; border-top: 1px solid #547365; border-right: 1px solid #547365; transform: rotate(45deg); }
.predicate { z-index: 1; padding: 4px 8px; color: #244e3c; border: 1px solid #c7d8cd; background: var(--surface); border-radius: 4px; font: 11px/1.35 var(--mono); max-width: 23ch; overflow-wrap: anywhere; }
.section-label { font-size: 12px; font-weight: 730; letter-spacing: .025em; margin: 0 0 8px; color: var(--green); }
.quote { border-left: 3px solid #48836a; background: #f4f6ed; margin: 0 0 18px; border-radius: 0 8px 8px 0; padding: 15px 17px; font: 400 clamp(16px, 1.55vw, 20px)/1.52 var(--serif); overflow-wrap: anywhere; white-space: pre-wrap; }
mark { padding: .03em .13em; color: inherit; border-radius: 3px; }
mark.subject { background: #d4edda; }
mark.object { background: #f3e5b4; }
.decision-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
.verdict { border: 1px solid var(--line); border-radius: 9px; padding: 12px 13px; min-width: 0; }
.verdict h3 { margin: 0 0 8px; font-size: 13px; }
.fact-line { font-size: 12px; color: var(--muted); margin: 8px 0 0; overflow-wrap: anywhere; }
.badge { display: inline-block; border-radius: 4px; padding: 2px 7px; font-size: 11px; font-weight: 700; color: #455758; background: #e9efec; line-height: 1.5; }
.badge.positive { color: var(--green); background: #dcefe0; }
.badge.caution { color: var(--ochre); background: #f5ecce; }
.badge.negative { color: var(--red); background: #f5e4dd; }
.reasons { display: flex; flex-wrap: wrap; gap: 5px; list-style: none; padding: 0; margin: 8px 0 0; }
.reasons li { padding: 3px 6px; border: 1px solid #e2e8df; border-radius: 4px; color: #415850; font: 11px/1.4 var(--mono); overflow-wrap: anywhere; max-width: 100%; }
.reason-gloss { display: block; color: var(--ink); font: 13px/1.4 var(--sans); margin-bottom: 3px; }
details { border-top: 1px solid var(--line); margin-top: 15px; padding-top: 10px; }
summary { cursor: pointer; color: var(--muted); font-size: 13px; font-weight: 650; }
.span-list { margin: 10px 0 0; padding-left: 19px; }
.span-list li { padding: 4px 0; overflow-wrap: anywhere; white-space: pre-wrap; }
.span-label { color: var(--muted); margin-right: 6px; font: 11px var(--mono); }
.empty { margin: 0; padding: 18px 0; color: var(--muted); overflow-wrap: anywhere; }
footer.wrap { color: var(--muted); display: flex; gap: 6px 20px; flex-wrap: wrap; font: 11px/1.5 var(--mono); margin-bottom: 24px; overflow-wrap: anywhere; }
@media (max-width: 990px) {
  .wrap { width: calc(100% - 48px); }
  main.wrap { grid-template-columns: minmax(215px, 260px) minmax(0, 1fr); gap: 16px; }
  .hero { gap: 20px; }
}
@media (max-width: 720px) {
  .wrap { width: calc(100% - 30px); }
  .hero, main.wrap { grid-template-columns: minmax(0, 1fr); gap: 18px; }
  .hero { padding: 23px 0; }
  .hero-stats div { padding: 0 8px; }
  .queue { max-height: 218px; }
  .stage { scroll-margin-top: 10px; }
  .graph { grid-template-columns: minmax(0, 1fr) minmax(85px, .65fr) minmax(0, 1fr); }
  .endpoint { padding: 9px; }
  .predicate { padding: 3px 4px; font-size: 10px; }
}
@media (max-width: 390px) {
  .graph { grid-template-columns: 1fr; gap: 7px; }
  .connector::after { display: none; }
  .decision-grid { grid-template-columns: 1fr; }
}
@media print {
  .top-actions, .rail { display: none; }
  .wrap { width: 100%; }
  main.wrap { display: block; }
  .mast { background: white; color: var(--ink); }
  .hero-stats dt, .hero-stats dd, .hero p { color: var(--ink); }
  .stage { border: 0; }
}
</style>
</head>
<body>
<header class="mast">
  <div class="wrap">
    <div class="topbar">
      <div class="brand"><span class="brand-mark" aria-hidden="true"></span>LiverKG</div>
      <div class="top-actions"><span class="mode" id="mode"></span><button type="button" class="language" id="language"></button></div>
    </div>
    <div class="hero">
      <div><h1 data-label="title"></h1><p id="intro"></p></div>
      <dl class="hero-stats" id="stats"></dl>
    </div>
  </div>
</header>
<main class="wrap">
  <aside class="rail" aria-labelledby="queue-title">
    <div class="rail-tools">
      <h2 id="queue-title" data-label="queue"></h2>
      <div class="filter-box">
        <input type="search" id="search" autocomplete="off">
        <select id="status"></select>
      </div>
      <p class="list-count" id="list-count" role="status"></p>
    </div>
    <ul class="queue" id="queue"></ul>
  </aside>
  <section class="stage" id="stage" aria-label="Evidence detail"></section>
</main>
<footer class="wrap" id="footer"></footer>
<script type="application/json" id="review-data">__REVIEW_DATA__</script>
<script>
"use strict";
const strings = {
  en: {
    title: "Evidence review", demo: "Offline demo", run: "Saved run",
    demoIntro: "Fictional examples processed by LiverKG’s verifier. Select a relation to inspect its evidence and decision.",
    runIntro: "Inspect each candidate’s evidence, semantic decision, and graph eligibility from this saved run.",
    articles: "Articles", candidates: "Candidates", accepted: "Accepted", eligible: "Write eligible",
    queue: "Candidate queue", search: "Search entity, predicate, or PMID", filter: "Filter semantic status",
    all: "All semantic statuses", results: "matching entries", detail: "Evidence detail",
    evidence: "Evidence in source", source: "Source", pubmed: "Open in PubMed ↗",
    semantic: "Semantic decision", graph: "Graph eligibility", schema: "Schema",
    candidateSchema: "Candidate schema", writeContract: "Graph contract",
    yes: "Pass", no: "Fail", missing: "Unrecorded",
    noEvidence: "No evidence quote was recorded.", noRelations: "No verified relations were recorded.",
    empty: "No candidates match the current filters.", error: "Article processing error",
    spans: "Evidence spans", flags: "Audit flags", factual: "Factual validation",
    eligibility: "Eligible for graph import", eligibleYes: "Yes", eligibleNo: "No",
    runId: "Run", profileHash: "Schema SHA-256",
    ACCEPTED: "Accepted", REVIEW: "Needs review", REJECTED: "Rejected", UNKNOWN: "Unrecorded",
    IMPORT_READY: "Import ready", SEMANTIC_ONLY: "Candidate export only", HUMAN_REVIEW: "Human review",
    BLOCKED: "Blocked", UNASSESSED: "Unassessed", VALID: "Valid", UNVERIFIED: "Unverified",
    language: "中文", switchLanguage: "Switch to Chinese"
  },
  zh: {
    title: "关系证据审查", demo: "离线演示", run: "已保存的运行",
    demoIntro: "用 LiverKG 验证器处理的虚构样例。选择一条关系，查看它的证据和判定。",
    runIntro: "逐条查看候选关系的原文证据、语义判定与图谱写入资格。",
    articles: "文章", candidates: "候选关系", accepted: "语义接受", eligible: "符合写入条件",
    queue: "候选关系", search: "搜索实体、关系或 PMID", filter: "筛选语义状态",
    all: "全部语义状态", results: "条匹配记录", detail: "关系证据详情",
    evidence: "证据原句", source: "来源", pubmed: "打开 PubMed ↗",
    semantic: "语义判定", graph: "图谱写入资格", schema: "Schema",
    candidateSchema: "候选 Schema", writeContract: "图谱写入契约",
    yes: "通过", no: "未通过", missing: "未记录",
    noEvidence: "这条记录未提供原文证据。", noRelations: "这篇文章没有已验证的关系记录。",
    empty: "当前筛选条件下没有匹配记录。", error: "文章处理错误",
    spans: "证据片段", flags: "审计标记", factual: "事实校验",
    eligibility: "运行时写入资格", eligibleYes: "符合", eligibleNo: "不符合",
    runId: "运行", profileHash: "Schema SHA-256",
    ACCEPTED: "接受", REVIEW: "待复核", REJECTED: "拒绝", UNKNOWN: "未记录",
    IMPORT_READY: "符合导入条件", SEMANTIC_ONLY: "仅导出候选", HUMAN_REVIEW: "人工复核",
    BLOCKED: "禁止写入", UNASSESSED: "未评估", VALID: "通过", UNVERIFIED: "未验证",
    language: "EN", switchLanguage: "切换到英文"
  }
};
const reasonGlosses = {
  dataset_predicate_support_absent: [
    "The quote lacks a configured trigger for this relation.", "引文没有提供该关系配置所需的触发证据。"
  ],
  dataset_predicate_exclusion_cue: [
    "The quote matches a configured exclusion rule.", "引文命中了配置里的排除规则。"
  ],
  dataset_candidate_only: [
    "This schema exports candidates for review.", "该 schema 用于导出候选供审查。"
  ],
  evidence_untraceable: [
    "The evidence quote could not be located in the source.", "证据引文无法在原文中定位。"
  ],
  schema_mismatch: [
    "The predicate or endpoint types fall outside the selected schema.", "谓词或端点类型超出当前 schema 的定义。"
  ]
};
const data = JSON.parse(document.getElementById("review-data").textContent);
const byId = (id) => document.getElementById(id);
const semanticLabels = ["ACCEPTED", "REVIEW", "REJECTED", "UNKNOWN"];
const rows = data.articles.flatMap((article) => article.relations.length
  ? article.relations.map((relation) => ({article, relation}))
  : [{article, relation: null}]);
const relations = rows.filter((row) => row.relation).map((row) => row.relation);
let selected = rows.find((row) => row.relation?.semantic_status === "ACCEPTED") || rows[0];
let lang = document.documentElement.lang.startsWith("zh") ? "zh" : "en";
const t = (key) => strings[lang][key] ?? key;

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function semanticStatus(relation) {
  return semanticLabels.includes(relation?.semantic_status) ? relation.semantic_status : "UNKNOWN";
}

function badge(status) {
  const tones = {
    ACCEPTED: "positive", IMPORT_READY: "positive", VALID: "positive",
    REVIEW: "caution", HUMAN_REVIEW: "caution",
    REJECTED: "negative", BLOCKED: "negative"
  };
  return el("span", "badge " + (tones[status] || ""), t(status || "UNKNOWN"));
}

function renderHeader() {
  document.documentElement.lang = lang === "zh" ? "zh-CN" : "en";
  document.title = "LiverKG · " + t("title");
  document.querySelectorAll("[data-label]").forEach((node) => {
    node.textContent = t(node.dataset.label);
  });
  byId("mode").textContent = t(data.mode === "demo" ? "demo" : "run");
  byId("intro").textContent = t(data.mode === "demo" ? "demoIntro" : "runIntro");
  byId("language").textContent = t("language");
  byId("language").setAttribute("aria-label", t("switchLanguage"));
  byId("stage").setAttribute("aria-label", t("detail"));
  byId("search").placeholder = t("search");
  byId("search").setAttribute("aria-label", t("search"));
  byId("status").setAttribute("aria-label", t("filter"));
  const filter = byId("status").value;
  byId("status").replaceChildren(...["", ...semanticLabels].map((value) => {
    const option = el("option", "", t(value || "all"));
    option.value = value;
    return option;
  }));
  byId("status").value = filter;

  const stats = [
    ["articles", data.articles.length],
    ["candidates", relations.length],
    ["accepted", knownCount(relations, "semantic_status", "ACCEPTED")],
    ["eligible", knownCount(relations, "import_ready", true)]
  ];
  byId("stats").replaceChildren(...stats.map(([label, count]) => {
    const item = el("div");
    item.append(el("dt", "", t(label)), el("dd", "", String(count)));
    return item;
  }));
  const footer = byId("footer");
  footer.replaceChildren();
  if (data.run_id) footer.append(el("span", "", t("runId") + " · " + data.run_id));
  if (data.schema_hash) footer.append(el("span", "", t("profileHash") + " · " + data.schema_hash));
}

function knownCount(items, key, value) {
  const recorded = items.filter((row) => row[key] !== null && row[key] !== "UNKNOWN");
  return items.length && !recorded.length ? "—" : recorded.filter((row) => row[key] === value).length;
}

function renderQueue() {
  const term = byId("search").value.trim().toLocaleLowerCase();
  const status = byId("status").value;
  const visible = rows.filter(({article, relation}) => {
    if (status && semanticStatus(relation) !== status) return false;
    return !term || [article.title, article.pmid, relation?.predicate, relation?.subject,
      relation?.object, relation?.evidence].some((value) => value?.toLocaleLowerCase().includes(term));
  });
  if (!visible.includes(selected)) selected = visible[0];
  byId("list-count").textContent = visible.length + " " + t("results");
  const queue = byId("queue");
  const items = document.createDocumentFragment();
  visible.forEach((row) => {
    const item = el("li");
    const button = el("button");
    button.type = "button";
    button.setAttribute("aria-current", String(row === selected));
    const meta = el("span", "queue-meta");
    meta.append(badge(semanticStatus(row.relation)), el("span", "queue-source", row.article.pmid));
    const relation = row.relation;
    button.append(meta, el("span", "queue-edge", relation
      ? relation.subject + " → " + relation.object
      : row.article.title || row.article.pmid || t("noRelations")));
    if (relation?.predicate) button.append(el("span", "queue-predicate", relation.predicate));
    button.addEventListener("click", () => {
      queue.querySelector('[aria-current="true"]')?.setAttribute("aria-current", "false");
      selected = row;
      button.setAttribute("aria-current", "true");
      renderDetail();
    });
    item.append(button);
    items.append(item);
  });
  queue.replaceChildren(items);
  renderDetail();
}

function endpoint(mention, type, target = false) {
  const card = el("div", target ? "endpoint target" : "endpoint");
  card.append(el("small", "", type || t("missing")), el("strong", "", mention || t("missing")));
  return card;
}

function quoteWithHighlights(quote, subject, object) {
  const block = el("blockquote", "quote");
  if (!quote) {
    block.textContent = t("noEvidence");
    return block;
  }
  const mentions = [subject, object].filter(Boolean).sort((a, b) => b.length - a.length);
  if (!mentions.length) {
    block.textContent = quote;
    return block;
  }
  const pattern = new RegExp(mentions.map((part) => part.replace(/[.*+?^\${}()|[\]\\]/g, "\\$&")).join("|"), "giu");
  let cursor = 0;
  for (const match of quote.matchAll(pattern)) {
    block.append(document.createTextNode(quote.slice(cursor, match.index)));
    const kind = match[0].toLocaleLowerCase() === subject.toLocaleLowerCase() ? "subject" : "object";
    block.append(el("mark", kind, match[0]));
    cursor = match.index + match[0].length;
  }
  block.append(document.createTextNode(quote.slice(cursor)));
  return block;
}

function reasons(codes, explain = false) {
  const list = el("ul", "reasons");
  list.replaceChildren(...codes.map((code) => {
    const item = el("li");
    const gloss = Object.hasOwn(reasonGlosses, code) ? reasonGlosses[code] : null;
    if (explain && gloss) item.append(el("span", "reason-gloss", gloss[lang === "zh" ? 1 : 0]));
    item.append(document.createTextNode(code));
    return item;
  }));
  return list;
}

function checkValue(value, pass = "yes", fail = "no") {
  return t(value === true ? pass : value === false ? fail : "missing");
}

function renderDetail() {
  const stage = byId("stage");
  stage.replaceChildren();
  if (!selected) {
    stage.append(el("p", "empty", rows.length ? t("empty") : t("noRelations")));
    return;
  }
  const {article, relation} = selected;
  const source = el("div", "source-line");
  if (article.pmid) source.append(el("span", "source-id",
    (/^\d+$/.test(article.pmid) ? "PMID · " : t("source") + " · ") + article.pmid));
  const profile = relation?.schema_name || data.schema_name;
  if (profile) {
    const profileLabel = el("span", "", t("schema") + " · " + profile);
    const hash = relation?.schema_hash || data.schema_hash;
    if (hash) profileLabel.title = t("profileHash") + " · " + hash;
    source.append(profileLabel);
  }
  if (/^\d+$/.test(article.pmid)) {
    const link = el("a", "", t("pubmed"));
    link.href = "https://pubmed.ncbi.nlm.nih.gov/" + article.pmid + "/";
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    source.append(link);
  }
  stage.append(source, el("h2", "", article.title || article.pmid || t("detail")));
  if (article.error) stage.append(el("p", "empty", t("error") + " · " + article.error));
  if (!relation) {
    stage.append(el("p", "empty", t("noRelations")));
    return;
  }
  const graph = el("div", "graph");
  const connector = el("div", "connector");
  connector.append(el("span", "predicate", relation.predicate || t("missing")));
  graph.append(endpoint(relation.subject, relation.subject_type), connector,
    endpoint(relation.object, relation.object_type, true));
  stage.append(graph, el("h3", "section-label", t("evidence")),
    quoteWithHighlights(relation.evidence, relation.subject, relation.object));

  const semantic = el("div", "verdict");
  semantic.append(el("h3", "", t("semantic")), badge(relation.semantic_status));
  semantic.append(el("p", "fact-line", t("candidateSchema") + " · " + checkValue(relation.candidate_schema_valid)));
  if (relation.factual_status && relation.factual_status !== "UNKNOWN") {
    semantic.append(el("p", "fact-line", t("factual") + " · " + t(relation.factual_status)));
  }
  if (relation.semantic_reasons.length) semantic.append(reasons(relation.semantic_reasons, true));

  const write = el("div", "verdict");
  write.append(el("h3", "", t("graph")), badge(relation.write_status));
  write.append(el("p", "fact-line", t("eligibility") + " · " + checkValue(relation.import_ready, "eligibleYes", "eligibleNo")));
  write.append(el("p", "fact-line", t("writeContract") + " · " + checkValue(relation.write_contract_valid)));
  if (relation.write_reasons.length) write.append(reasons(relation.write_reasons, true));
  const decisions = el("div", "decision-grid");
  decisions.append(semantic, write);
  stage.append(decisions);

  if (relation.spans.length) {
    const details = el("details");
    const list = el("ul", "span-list");
    relation.spans.forEach((span) => {
      const item = el("li");
      const range = Number.isInteger(span.char_start) && Number.isInteger(span.char_end)
        && span.char_start >= 0 && span.char_end >= span.char_start
        ? span.char_start + "–" + span.char_end : "";
      const label = [span.role, span.alignment_status, range].filter(Boolean).join(" / ");
      if (label) item.append(el("span", "span-label", label));
      item.append(document.createTextNode(span.text));
      list.append(item);
    });
    details.append(el("summary", "", t("spans") + " (" + relation.spans.length + ")"), list);
    stage.append(details);
  }
  if (relation.quality_flags.length) {
    const details = el("details");
    details.append(el("summary", "", t("flags") + " (" + relation.quality_flags.length + ")"),
      reasons(relation.quality_flags));
    stage.append(details);
  }
}

byId("search").addEventListener("input", renderQueue);
byId("status").addEventListener("change", renderQueue);
byId("language").addEventListener("click", () => {
  lang = lang === "en" ? "zh" : "en";
  renderHeader();
  renderQueue();
});
renderHeader();
renderQueue();
</script>
</body>
</html>
"""
