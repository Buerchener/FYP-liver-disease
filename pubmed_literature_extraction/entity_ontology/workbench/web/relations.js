import { el, status, articleLink, typeLabel } from "./ui.js";

const GATES = { AUTO: ["可自动写入", "good"], REVIEW: ["待复核", "warn"], BLOCKED: ["已拦截", "bad"] };
const NOVELTY = { KNOWN_IN_V2: ["v2 已有", "neutral"], NEW: ["v2 中没有", "good"], UNLINKED: ["端点未对齐 v2", "neutral"] };
const LINKS = { LINKED: ["已对齐", "good"], AMBIGUOUS: ["多处命中", "warn"], NEW: ["v2 中没有", "neutral"] };
const WRITE = { IMPORT_READY: ["可写入", "good"], SEMANTIC_ONLY: ["仅语义", "neutral"], HUMAN_REVIEW: ["人工复核", "warn"], BLOCKED: ["拦截", "bad"] };
const REASONS = {
  no_import_ready_evidence: "没有可直接写入的证据",
  counter_evidence: "存在否定、反证或统计未证实",
  signature_mismatch: "端点类型不符合谓词签名",
  unknown_predicate: "谓词不在 LiverKG 范围内",
  subject_type_mismatch: "主语类型与概念类型不一致",
  object_type_mismatch: "宾语类型与概念类型不一致",
};
const BASIS = { name: "名称或别名", normalized_id: "抽取器给出的 ID", normalized_id_conflicts_with_name: "ID 与名称指向不同节点" };

const side = word => word === "subject" ? "主语" : "宾语";
const pill = (table, key) => {
  const [label, tone] = table[key] || [key || "—", "neutral"];
  return status(label, tone);
};

function reasonLabel(code) {
  if (REASONS[code]) return REASONS[code];
  const placement = code.match(/^(subject|object)_concept_(.+)$/);
  if (placement) return side(placement[1]) + "概念尚未放入层级（" + placement[2] + "）";
  const endpoint = code.match(/^(subject|object)_(unresolved|ambiguous)$/);
  if (endpoint) return side(endpoint[1]) + (endpoint[2] === "unresolved" ? "无法对应到概念" : "对应到多个概念");
  return code;
}

function highlight(text, terms) {
  const words = [...new Set(terms.filter(Boolean))]
    .sort((a, b) => b.length - a.length)
    .map(term => term.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
  if (!text || !words.length) return [text || ""];
  return text.split(new RegExp("(" + words.join("|") + ")", "gi"))
    .map((part, index) => index % 2 ? el("mark", {}, part) : part);
}

function select(label, choices, value) {
  const control = el("select", { "aria-label": label });
  for (const [key, text] of choices) control.append(el("option", { value: key }, text));
  control.value = value;
  return el("label", { class: "rel-field" }, el("span", {}, label), control);
}

function search(placeholder) {
  return el("label", { class: "rel-field" }, el("span", {}, "搜索"),
    el("input", { type: "search", placeholder, "aria-label": placeholder, autocomplete: "off" }));
}

function evidenceQuote(row, articles) {
  const article = articles.get(row.pmid);
  return el("figure", { class: "source-quote" },
    el("blockquote", {}, highlight(row.evidence, [row.subject, row.object])),
    el("figcaption", { class: "rel-caption" },
      articleLink(row.pmid), pill(WRITE, row.write_status),
      row.contrast_status && status("对照：" + row.contrast_status, "warn"),
      row.negated && status("否定", "bad"),
      el("span", { class: "muted" }, article?.title || "")));
}

function relationCard(relation, articles, go) {
  const concept = (id, name, kind) => el("a", { href: "#links/" + encodeURIComponent(id),
    onClick: event => { event.preventDefault(); go("links", id); } }, name);
  return el("article", { class: "rel-card" },
    el("header", { class: "rel-head" },
      el("strong", {}, concept(relation.subject_concept_id, relation.subject_name)),
      el("span", { class: "muted" }, typeLabel(relation.subject_kind)),
      el("code", { class: "rel-predicate" }, relation.predicate),
      el("strong", {}, concept(relation.object_concept_id, relation.object_name)),
      el("span", { class: "muted" }, typeLabel(relation.object_kind)),
      el("span", { class: "rel-pills" }, pill(GATES, relation.gate), pill(NOVELTY, relation.novelty),
        status(relation.evidence_count + " 条证据 · " + relation.pmids.length + " 篇"))),
    relation.gate_reasons.length > 0 && el("p", { class: "muted rel-reasons" },
      "原因：" + relation.gate_reasons.map(reasonLabel).join("；")),
    relation.v2_relations.length > 0 && el("p", { class: "muted" },
      "v2 中两端之间已有：" + relation.v2_relations.map(row => row.type + "（" + row.source + "）").join("、")),
    relation.evidence.map(row => evidenceQuote(row, articles)));
}

function renderRelationList({ mount, data, go, selected }) {
  const projection = data.conceptRelations;
  const articles = new Map(data.articles.map(article => [article.pmid, article]));
  const counts = projection.counts;
  const gate = select("门禁", [["", "全部"], ["AUTO", "可自动写入"], ["REVIEW", "待复核"], ["BLOCKED", "已拦截"]], "");
  const novelty = select("与 v2 对照", [["", "全部"], ["NEW", "v2 中没有"], ["KNOWN_IN_V2", "v2 已有"], ["UNLINKED", "端点未对齐"]], "");
  const query = search("概念名称或谓词");
  const list = el("div", { class: "rel-list" });
  const focus = selected && data.conceptLinks.links.find(row => row.concept_id === selected);

  function update() {
    const wanted = query.querySelector("input").value.trim().toLowerCase();
    const rows = projection.relations.filter(relation =>
      (!focus || relation.subject_concept_id === selected || relation.object_concept_id === selected) &&
      (!gate.querySelector("select").value || relation.gate === gate.querySelector("select").value) &&
      (!novelty.querySelector("select").value || relation.novelty === novelty.querySelector("select").value) &&
      (!wanted || [relation.subject_name, relation.object_name, relation.predicate]
        .some(text => text.toLowerCase().includes(wanted))));
    list.replaceChildren(...(rows.length ? rows.map(relation => relationCard(relation, articles, go))
      : [el("p", { class: "empty" }, "没有符合条件的关系。")]));
  }
  for (const control of [gate, novelty, query]) control.addEventListener("input", update);

  mount.append(...[
    el("p", { class: "rel-summary" },
      `文章级关系 ${counts.article_relations} 条，合并成 ${counts.concept_relations} 条概念级关系（合并了 ${counts.merged_evidence} 条重复证据）。` +
      `可自动写入 ${counts.gate.AUTO}，待复核 ${counts.gate.REVIEW}，已拦截 ${counts.gate.BLOCKED}。`),
    focus && el("p", { class: "rel-focus" }, "只看与「" + focus.name + "」有关的关系 · ",
      el("a", { href: "#relations" }, "查看全部")),
    el("div", { class: "controls rel-controls" }, gate, novelty, query),
    list,
  ].filter(Boolean));
  update();
  if (projection.unresolved_evidence.length) {
    mount.append(el("details", { class: "rel-unresolved" },
      el("summary", {}, `端点没能对应到概念的证据（${projection.unresolved_evidence.length} 条）`),
      projection.unresolved_evidence.map(row => el("div", { class: "rel-card" },
        el("p", { class: "rel-head" }, el("strong", {}, row.subject), el("code", { class: "rel-predicate" }, row.predicate),
          el("strong", {}, row.object)),
        el("p", { class: "muted" }, "原因：" + row.reasons.map(reasonLabel).join("；")),
        evidenceQuote(row, articles)))));
  }
}

function renderLinkTable({ mount, data, go, selected }) {
  const links = data.conceptLinks;
  const counts = links.counts;
  const filter = select("对齐状态", [["", "全部"], ["LINKED", "已对齐"], ["AMBIGUOUS", "多处命中"], ["NEW", "v2 中没有"]], "");
  const query = search("概念名称");
  const body = el("tbody");
  const relationCount = new Map();
  for (const relation of data.conceptRelations.relations) {
    for (const id of [relation.subject_concept_id, relation.object_concept_id])
      relationCount.set(id, (relationCount.get(id) || 0) + 1);
  }

  function target(row) {
    if (row.target) return el("span", {}, el("code", {}, row.target.id), " ", row.target.name);
    if (row.candidates.length) return el("span", { class: "muted" },
      row.candidates.map(item => item.id + " " + item.name).join("；"));
    return el("span", { class: "muted" }, "—");
  }

  function update() {
    const wanted = query.querySelector("input").value.trim().toLowerCase();
    const wantedStatus = filter.querySelector("select").value;
    const rows = links.links.filter(row => (!wantedStatus || row.status === wantedStatus) &&
      (!wanted || row.name.toLowerCase().includes(wanted)));
    body.replaceChildren(...rows.map(row => el("tr", { class: row.concept_id === selected ? "rel-selected" : null },
      el("td", {}, el("strong", {}, row.name), el("div", { class: "muted" }, typeLabel(row.kind))),
      el("td", {}, pill(LINKS, row.status)),
      el("td", {}, target(row)),
      el("td", { class: "muted" }, row.matched_by.map(item => BASIS[item] || item).join("、") || "—"),
      el("td", {}, row.external_ids?.GO ? el("code", {}, row.external_ids.GO) : el("span", { class: "muted" }, "—")),
      el("td", {}, relationCount.get(row.concept_id)
        ? el("a", { href: "#relations/" + encodeURIComponent(row.concept_id),
          onClick: event => { event.preventDefault(); go("relations", row.concept_id); } },
          relationCount.get(row.concept_id) + " 条")
        : el("span", { class: "muted" }, "0")))));
  }
  for (const control of [filter, query]) control.addEventListener("input", update);
  mount.append(
    el("p", { class: "rel-summary" },
      `已对齐 ${counts.LINKED}，多处命中 ${counts.AMBIGUOUS}，v2 中没有 ${counts.NEW}。` +
      "对齐只读取仓库内的 v2 数据包，不连接数据库。"),
    el("div", { class: "controls rel-controls" }, filter, query),
    el("div", { class: "document-table-wrap" }, el("table", { class: "data-table" },
      el("thead", {}, el("tr", {}, ["概念", "对齐状态", "v2 节点", "依据", "GO", "关系"].map(text => el("th", {}, text)))),
      body)));
  update();
}

export function renderRelations({ stage, mount, data, go, selected }) {
  if (!data.conceptRelations || !data.conceptLinks) {
    mount.append(el("section", { class: "empty empty-state" },
      el("h2", {}, "这个批次没有关系结果"),
      el("p", {}, "它生成于关系投影接入之前。用当前版本重新运行这个批次即可看到概念对齐和概念级关系。")));
    return;
  }
  if (stage === "links") renderLinkTable({ mount, data, go, selected });
  else renderRelationList({ mount, data, go, selected });
}
