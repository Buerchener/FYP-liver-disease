import {el,articleLink} from "./ui.js";

const jump=(label,stage,selected,cls="text-link")=>el("a",{class:cls,href:"#"+stage+(selected?"/"+encodeURIComponent(selected):"")},label);
function excerpt(text,phrase) {
  const at=text.toLowerCase().indexOf(phrase.toLowerCase());
  return at<0?[text]:[text.slice(0,at),el("mark",{},text.slice(at,at+phrase.length)),text.slice(at+phrase.length)];
}
function metric(label,qty,hint,stage){
  return el("a",{class:"metric",href:"#"+stage},el("span",{class:"metric-label"},label),
     el("span",{class:"metric-value"},String(qty)),el("span",{class:"metric-hint"},hint));
}
function card(label,value,info,stage,id){
  return el("div",{class:"flow-cell"},el("span",{class:"flow-label"},label),
    jump(el("span",{},el("strong",{},value),el("small",{},info)),stage,id,"flow-node"));
}
export function renderOverview({mount,data}) {
  const {run,articles,mentions,normalization,equivalence,retrieval,placement} = data;
  const byMention = new Map(normalization.map(row => [row.id,row]));
  const placed = new Set(placement.nodes.map(node => node.id));
  const specificEdges = placement.edges.filter(edge => edge.support !== "schema_type");
  const exampleEdge = specificEdges.find(edge => equivalence.concepts.some(row => row.id===edge.child_id));
  const concept = equivalence.concepts.find(row => row.id===exampleEdge?.child_id) ||
    equivalence.concepts.find(row => placed.has(row.id)) || equivalence.concepts[0];
  const parent = exampleEdge && placement.nodes.find(node => node.id===exampleEdge.parent_id);
  const evidence = concept && mentions.find(row => concept.source_mentions.includes(row.id));
  const kept = normalization.filter(row => row.status === "keep").length;
  const stats = el("section",{class:"metrics","aria-label":"本批次结果"},
    metric("原始实体提及",mentions.length,mentions.filter(row=>row.offset_status==="exact").length+" 条可逐字定位","extraction"),
    metric("保留提及",kept,normalization.filter(row=>row.status==="review").length+" 条待复核","normalization"),
    metric("去重后概念",equivalence.concepts.length,retrieval.candidate_count+" 对同义候选","deduplication"),
    metric("候选层级边",placement.edges.length,specificEdges.length+" 条具体父子关系","tree"));

  const trace = el("section",{class:"trace-section"},
    el("div",{class:"section-top"},el("h2",{},"概念溯源"),jump("查看逐个放置","tree")));
  if (evidence) {
    const sentence = evidence.context || evidence.evidence_contexts[0]?.text;
    trace.append(el("div",{class:"concept-flow","aria-label":"从原始提及到构树"},
      card("原始提及",evidence.mention,evidence.raw_type,"extraction",evidence.id),
      card("规范概念",concept.name,concept.kind,"deduplication",concept.id),
      card("放置结果",parent?.name || (placed.has(concept.id)?"候选层级":"待挂接"),"查看放置理由","tree",concept.id)));
    if (sentence) trace.append(el("div",{class:"case-source"},
      el("div",{class:"source-meta"},el("span",{},"文献原文"),articleLink(evidence.pmid)),
      el("blockquote",{lang:"en"},...excerpt(sentence,evidence.source_span || evidence.mention))));
  } else trace.append(el("p",{class:"empty"},"这批文献没有可展示的归并概念。"));

  const papers = el("aside",{class:"papers"},
    el("div",{class:"section-top"},el("h2",{},"文献来源"),el("span",{class:"counter"},articles.length+" 篇")));
  for (const article of articles) {
    const rows = mentions.filter(row => row.pmid===article.pmid);
    const title = rows.length ? jump(article.title,"extraction",rows[0].id,"paper-title") :
      el("p",{class:"paper-title"},article.title);
    papers.append(el("div",{class:"paper-item"},articleLink(article.pmid),title,
      el("p",{class:"paper-meta"},rows.length+" 条原始提及 · "+rows.filter(row=>byMention.get(row.id)?.status==="keep").length+" 条保留")));
  }
  const sources = el("div",{class:"origin-line"},el("span",{},"输入批次",el("code",{},run.run_id)),
    run.extraction_commit && el("span",{},"抽取器版本",el("code",{},run.extraction_commit.slice(0,12))),
    run.extraction_model && el("span",{},"抽取模型",el("code",{},run.extraction_model)),
    el("span",{},"构树模型",el("code",{},placement.model)),
    el("span",{},"GO 查询",el("b",{},placement.go_reference_count+" 次")));
  mount.append(stats,el("div",{class:"overview-grid"},trace,papers),sources);
}
