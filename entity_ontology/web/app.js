import { el } from "./ui.js";
import { renderOverview } from "./overview.js";

const stages = [
  ["extraction", "实体抽取", "检查实体提及、模型标签和对应的原文位置。", "evidence", "renderEvidence"],
  ["normalization", "概念化", "结合原句和缩写释义，核对每条提及的名称、类型与去向。", "evidence", "renderEvidence"],
  ["embedding", "向量化", "查看名称、类型、定义和别名生成的语义向量。", "preparation", "renderPreparation"],
  ["clustering", "聚类", "浏览 HDBSCAN 分组及组外实体的向量分布。", "preparation", "renderPreparation"],
  ["deduplication", "同义去重", "查看由全局近邻和共享名称或别名找到的候选，回查同义判断和原句。", "evidence", "renderEvidence"],
  ["ordering", "排序", "按实体类型查看构树前的概念清单，组内从宽到窄排列。", "preparation", "renderPreparation"],
  ["tree", "构树", "按排序清单逐个放置，回查每次判断和证据。", "incremental", "renderIncremental"],
];
const title = document.querySelector("#heading");
const content = document.querySelector("#content");
const nav = document.querySelector("#stages");
const crumb = document.querySelector("#crumb");
const run = new URLSearchParams(location.search).get("run");
for (const [i, s] of stages.entries()) {
  nav.append(el("li",{},el("a",{href:"#"+s[0]},el("span",{class:"workbench-step","aria-hidden":"true"},String(i+1)),s[1])));
}
let data;
let renderVersion = 0;
function go(stage, selected) {
  const hash = "#" + stage + (selected ? "/" + encodeURIComponent(selected) : "");
  if (location.hash !== hash) location.hash = hash;
  else render();
}
const loadedStyles = new Map();
function stageStyle(module) {
  if (!loadedStyles.has(module)) {
    loadedStyles.set(module,new Promise((resolve,reject)=>{
      const link = el("link",{rel:"stylesheet",href:"/"+module+".css"});
      link.onload=resolve;
      link.onerror=()=>{loadedStyles.delete(module); reject(new Error("style"));};
      document.head.append(link);
    }));
  }
  return loadedStyles.get(module);
}
async function render() {
  const turn=++renderVersion;
  const [token, encoded] = location.hash.slice(1).split("/");
  const stage = data?.run && stages.find(s=>s[0]===token);
  let selected = null;
  if (encoded) {
    try { selected = decodeURIComponent(encoded); } catch { selected = null; }
  }
  for (const a of nav.querySelectorAll("a")) {
    if (a.hash==="#"+stage?.[0]) a.setAttribute("aria-current","step");
    else a.removeAttribute("aria-current");
  }
  for (const a of nav.querySelectorAll("[aria-current]")) {
    if (nav.scrollWidth>nav.clientWidth) nav.scrollLeft=a.offsetLeft-nav.offsetLeft-16;
  }
  crumb.replaceChildren(el("a",{href:"#overview"},"实体本体工作台"),el("span",{"aria-hidden":"true"},"/"),stage ? stage[1] : "批次总览");
  title.textContent=stage ? stage[1] : "实体本体构建工作台";
  document.title=(stage ? stage[1]+" · 实体本体" : "实体本体构建工作台")+" · LiverKG";
  const batchLabel = data?.run ? `${data.run.title || data.run.run_id} · ${data.articles.length} 篇文献` : "";
  document.querySelector("#subtitle").textContent=stage?.[2] || batchLabel;
  if (!data) return;
  window.scrollTo(0,0);
  const mount=el("div",{class:"stage-body",id:"stage-view"});
  if (!data.run) {
    mount.append(el("section",{class:"empty"},el("h2",{},"暂无已完成的批次"),
      el("p",{},"完成一次实体本体构建后，刷新这里就能查看结果。"),
      el("button",{class:"btn",type:"button",onClick:()=>location.reload()},"刷新批次")));
    content.replaceChildren(mount);
    return;
  }
  if (!stage) {
    content.replaceChildren(mount);
    renderOverview({mount,data,go});
    return;
  }
  try {
    const [mod] = await Promise.all([import("./"+stage[3]+".js"),stageStyle(stage[3])]);
    if(turn!==renderVersion) return;
    content.replaceChildren(mount);
    mod[stage[4]]({stage:stage[0],mount,data,placement:data.placement,go,selected});
  } catch (error) {
    if(turn===renderVersion) fail("这个阶段暂时无法显示。",error);
  }
}
function fail(message,error) {
  console.error(error);
  content.replaceChildren(el("div",{class:"empty",role:"alert"},el("p",{},message),
    el("div",{class:"controls"},el("button",{class:"btn",onClick:()=>location.reload()},"重新读取"),
      run && el("a",{href:"/",class:"btn"},"查看最新批次"))));
}
function loadChoices() {
  if (!data.availableRuns?.length) return;
  const select = document.querySelector("#batch-choice");
  for (const row of data.availableRuns) {
    select.append(el("option",{value:row.run_id},`${row.title || row.run_id} · ${row.article_count} 篇`));
  }
  select.value = data.run.run_id;
  select.addEventListener("change",()=> {
    const url = new URL(location.href);
    url.searchParams.set("run",select.value);
    url.hash = location.hash.split("/")[0];
    location.assign(url);
  });
  document.querySelector("#batch-switcher").hidden=false;
}
document.querySelector(".workbench-skip").addEventListener("click",event=>{
  event.preventDefault();
  document.querySelector("#main").focus();
});
window.addEventListener("hashchange",render);
render();
try {
  const query = run === null ? "" : "?" + new URLSearchParams({run});
  const response=await fetch("/api/batch" + query);
  if(!response.ok) throw new Error("HTTP "+response.status);
  data=await response.json();
  if (data.run) loadChoices();
  await render();
} catch(error) { fail(run ? "这个批次尚无法读取。确认运行完成后重新读取。" : "暂时无法读取批次结果。",error); }
