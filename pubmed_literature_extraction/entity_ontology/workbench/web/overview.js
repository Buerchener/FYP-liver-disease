import { el, articleLink } from "./ui.js";

export const overviewSections = [
  ["summary", "批次结果"],
  ["evidence", "原文与层级证据"],
  ["workflow", "构建流程"],
  ["sources", "文献来源"],
];

const jump = (label, stage, selected, css) => el("a", {
  class: css,
  href: "#" + stage + (selected ? "/" + encodeURIComponent(selected) : ""),
}, label);

function excerpt(text, phrase) {
  if (!phrase) return [text];
  const at = text.toLowerCase().indexOf(phrase.toLowerCase());
  return at < 0 ? [text] : [text.slice(0, at), el("mark", {}, text.slice(at, at + phrase.length)), text.slice(at + phrase.length)];
}

function table(label, headings, rows, css) {
  return el("div", { class: "document-table-wrap" },
    el("table", { class: "data-table document-table " + css, "aria-label": label },
      el("thead", {}, el("tr", {}, headings.map(heading => el("th", { scope: "col" }, heading)))),
      el("tbody", {}, rows.map(([name, ...rest]) => el("tr", {}, el("th", { scope: "row" }, name),
        rest.map(value => el("td", {}, value)))))));
}

function section(id, heading, ...content) {
  return el("section", { class: "document-section", id, "aria-labelledby": id + "-heading" },
    el("h2", { id: id + "-heading" }, heading), content);
}

function renderTrace(concept, evidence, parent, nodes) {
  const [id, heading] = overviewSections[1];
  if (!concept || !evidence) return section(id, heading, el("p", { class: "muted" }, "本批次暂无可追溯的概念。"));
  const sentence = evidence.context || evidence.evidence_contexts?.[0]?.text;
  const position = parent?.name || (nodes.has(concept.id) ? "已放置的概念" : "待放置");
  const trace = [
    ["原文提及", jump(evidence.mention, "extraction", evidence.id)],
    ["规范概念", jump(concept.name, "deduplication", concept.id)],
    [parent ? "候选父类" : "层级位置", jump(position, "tree", concept.id)],
  ];
  return section(id, heading,
    sentence && el("figure", { class: "source-quote" },
      el("blockquote", { lang: "en" }, excerpt(sentence, evidence.source_span || evidence.mention)),
      el("figcaption", {}, articleLink(evidence.pmid))),
    el("dl", { class: "trace-links" }, trace.map(([label, value]) => el("div", {},
      el("dt", {}, label), el("dd", {}, value)))));
}

function renderPapers(articles, pmidMentions, byMention) {
  const [id, heading] = overviewSections[3];
  return section(id, heading, el("ol", { class: "research-sources" }, articles.map(article => {
    const rows = pmidMentions.get(article.pmid) || [];
    const kept = rows.filter(row => byMention.get(row.id)?.status === "keep").length;
    return el("li", {},
      el("span", { class: "paper-title", lang: "en" },
        rows.length ? jump(article.title, "extraction", rows[0].id) : article.title),
      el("span", { class: "paper-meta" }, articleLink(article.pmid),
        el("span", {}, rows.length + " 条原始提及，" + kept + " 条保留")));
  })));
}

export function renderOverview({ mount, data, stages }) {
  const { run, articles, mentions, normalization, equivalence, retrieval, placement, conceptOrder } = data;
  const byMention = new Map(normalization.map(row => [row.id, row]));
  const rawMentions = new Map(mentions.map(row => [row.id, row]));
  const nodes = new Map(placement.nodes.map(node => [node.id, node]));
  const specificEdges = placement.edges.filter(edge => edge.support !== "schema_type");
  const concepts = new Map(equivalence.concepts.map(concept => [concept.id, concept]));
  const exampleEdge = specificEdges.find(edge => concepts.has(edge.child_id));
  const concept = concepts.get(exampleEdge?.child_id) || equivalence.concepts.find(row => nodes.has(row.id)) || equivalence.concepts[0];
  const sourceMentions = concept?.source_mentions.map(id => rawMentions.get(id)).filter(Boolean) || [];
  const evidence = sourceMentions.find(row => row.offset_status === "exact" && row.context) || sourceMentions[0];
  const kept = normalization.filter(row => row.status === "keep").length;
  const points = retrieval.points || [];
  const clusters = new Set(points.filter(row => row.cluster_id >= 0).map(row => row.cluster_id));
  const totals = {
    extraction: mentions.length + " 条提及",
    normalization: kept + " 条保留",
    embedding: points.length + " 个向量",
    clustering: clusters.size + " 个分组",
    deduplication: concepts.size + " 个概念",
    ordering: conceptOrder.groups.length + " 个实体类型",
    tree: placement.edges.length + " 条候选边",
  };
  const metrics = [
    [jump("原始提及", "extraction"), mentions.length, mentions.filter(row => row.offset_status === "exact").length + " 条精确定位"],
    [jump("保留提及", "normalization"), kept, normalization.filter(row => row.status === "review").length + " 条待复核"],
    [jump("去重后概念", "deduplication"), concepts.size, retrieval.candidate_count + " 对同义候选"],
    [jump("候选层级边", "tree"), placement.edges.length, specificEdges.length + " 条具体父子关系"],
  ];

  const pmidMentions = new Map();
  for (const mention of mentions) {
    if (!pmidMentions.has(mention.pmid)) pmidMentions.set(mention.pmid, []);
    pmidMentions.get(mention.pmid).push(mention);
  }

  const created = new Date(run.created_at || "");
  const date = Number.isNaN(created.valueOf()) ? null :
    el("time", { datetime: run.created_at }, new Intl.DateTimeFormat("zh-CN", { dateStyle: "long" }).format(created));
  const extractorCommit = run.extractor_git?.commit || run.extraction_commit;
  const relationCounts = data.conceptRelations?.counts;
  const metadata = [
    ["批次", el("code", {}, run.run_id)],
    extractorCommit && ["抽取器", el("code", {}, extractorCommit.slice(0, 12) + (run.extractor_git?.dirty ? " · 含未提交改动" : ""))],
    run.extraction_model && ["抽取模型", el("code", {}, run.extraction_model)],
    placement.model && ["构树模型", el("code", {}, placement.model)],
    ["GO 查询", (placement.go_reference_count ?? 0) + " 次"],
    relationCounts && ["概念级关系", `${relationCounts.concept_relations} 条（可自动写入 ${relationCounts.gate.AUTO}，待复核 ${relationCounts.gate.REVIEW}）`],
  ].filter(Boolean);

  mount.append(
    section(...overviewSections[0],
      el("p", { class: "batch-intro" }, el("strong", {}, run.title || run.run_id)),
      el("p", { class: "batch-meta" }, date && [date, "｜"], articles.length + " 篇 PubMed 文献"),
      table("批次结果", ["指标", "数量", "说明"], metrics, "metric-table")),
    renderTrace(concept, evidence, nodes.get(exampleEdge?.parent_id), nodes),
    section(...overviewSections[2],
      table("构建流程", ["阶段", "检查内容", "本批次结果"], stages.map(stage =>
        [jump(stage.label, stage.id), stage.detail, totals[stage.id]]), "workflow-table")),
    renderPapers(articles, pmidMentions, byMention),
    el("footer", { class: "origin-line", "aria-label": "批次信息" },
      el("dl", {}, metadata.map(([label, value]) => el("div", {}, el("dt", {}, label), el("dd", {}, value))))),
  );
}
