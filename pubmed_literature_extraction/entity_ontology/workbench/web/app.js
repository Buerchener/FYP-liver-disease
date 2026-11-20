import { el } from "./ui.js";
import { overviewSections, renderOverview } from "./overview.js";

const groups = [
  ["原文与实体", [
    { id: "extraction", label: "实体抽取", detail: "查看模型找到的实体、补漏结果，以及每条提及在原文中的位置。", module: "evidence", method: "renderEvidence" },
    { id: "normalization", label: "概念化", detail: "结合原句和缩写释义，核对每条提及的名称、类型与去向。", module: "evidence", method: "renderEvidence" },
  ]],
  ["概念准备", [
    { id: "embedding", label: "向量化", detail: "查看名称、类型、定义和别名生成的语义向量。", module: "preparation", method: "renderPreparation" },
    { id: "clustering", label: "聚类", detail: "浏览 HDBSCAN 分组及组外实体的向量分布。", module: "preparation", method: "renderPreparation" },
    { id: "deduplication", label: "同义去重", detail: "查看相似候选和同义判断，回查合并概念的原句。", module: "evidence", method: "renderEvidence" },
  ]],
  ["本体层级", [
    { id: "ordering", label: "排序", detail: "查看各实体类型组内从宽到窄的概念清单。", module: "preparation", method: "renderPreparation" },
    { id: "tree", label: "构树", detail: "查看逐个放置的结果，回查候选父类和每次判断。", module: "incremental", method: "renderIncremental" },
  ]],
  ["关系与图谱", [
    { id: "links", label: "图谱对齐", detail: "查看每个概念对应到 v2 图谱的哪个节点，以及对齐依据。", module: "relations", method: "renderRelations" },
    { id: "relations", label: "概念关系", detail: "查看抽取器验证过的关系合并到概念之后的结果、门禁状态和每条证据原句。", module: "relations", method: "renderRelations" },
  ]],
];
const stages = groups.flatMap(([, children]) => children);
const title = document.querySelector("#heading");
const content = document.querySelector("#content");
const nav = document.querySelector("#stages");
const crumb = document.querySelector("#crumb");
const subtitle = document.querySelector("#subtitle");
const layout = document.querySelector(".workbench-layout");
const toc = document.querySelector("#page-toc");
const tocLinks = document.querySelector("#toc-links");
const run = new URLSearchParams(location.search).get("run");
const overviewLink = el("a", { href: "#overview", class: "workbench-nav-link workbench-overview" }, "批次总览");
nav.append(overviewLink);
for (const [label, children] of groups) {
  nav.append(el("section", { class: "nav-group", "aria-label": label },
    el("p", { class: "nav-group-title", "aria-hidden": "true" }, label),
    el("ol", { class: "workbench-stage-list" }, children.map(stage =>
      el("li", {}, el("a", { class: "workbench-nav-link", href: "#" + stage.id,
        onClick: event => { if (!data?.run) event.preventDefault(); } },
        stage.label))))));
}
for (const [id, label] of overviewSections) {
  tocLinks.append(el("li", {}, el("a", { href: "#overview/" + id }, label)));
}

let data;
let renderVersion = 0;
let renderedStage;
const loadedStyles = new Map();

function go(stage, selected) {
  const hash = "#" + stage + (selected ? "/" + encodeURIComponent(selected) : "");
  if (location.hash !== hash) location.hash = hash;
  else render();
}

function stageStyle(module) {
  if (!loadedStyles.has(module)) {
    loadedStyles.set(module, new Promise((resolve, reject) => {
      const link = el("link", { rel: "stylesheet", href: "/" + module + ".css" });
      link.onload = resolve;
      link.onerror = () => {
        link.remove();
        loadedStyles.delete(module);
        reject(new Error("Stage stylesheet failed to load."));
      };
      document.head.append(link);
    }));
  }
  return loadedStyles.get(module);
}

function renderLocation(stage, section) {
  toc.hidden = Boolean(stage) || !data?.run;
  layout.dataset.toc = String(!toc.hidden);
  for (const link of tocLinks.querySelectorAll("a")) {
    if (link.hash === "#overview/" + section) link.setAttribute("aria-current", "location");
    else link.removeAttribute("aria-current");
  }
  for (const link of nav.querySelectorAll("a")) {
    if (link !== overviewLink && !data?.run) {
      link.setAttribute("aria-disabled", "true");
      link.tabIndex = -1;
    } else {
      link.removeAttribute("aria-disabled");
      link.removeAttribute("tabindex");
    }
    if (link.hash === "#" + (stage?.id || "overview")) {
      link.setAttribute("aria-current", stage ? "step" : "page");
      if (nav.scrollWidth > nav.clientWidth) {
        const bounds = nav.getBoundingClientRect();
        const item = link.getBoundingClientRect();
        if (item.left < bounds.left || item.right > bounds.right) {
          nav.scrollLeft += item.left - bounds.left - 16;
        }
      }
    } else {
      link.removeAttribute("aria-current");
    }
  }
  crumb.replaceChildren(stage ? el("a", { href: "#overview" }, "批次总览") : el("span", { "aria-current": "page" }, "批次总览"));
  if (stage) crumb.append(el("span", { "aria-hidden": "true" }, "/"), el("span", { "aria-current": "page" }, stage.label));
  title.textContent = stage?.label || "实体本体工作台";
  document.title = (stage?.label || "实体本体工作台") + " · LiverKG";
  subtitle.textContent = stage?.detail ||
    "LiverKG 从 PubMed 文献抽取实体，合并同义概念并生成候选层级。工作台保留原文出处与各阶段的判断记录。";
}

function adjacentStages(stage) {
  const index = stages.indexOf(stage);
  const previous = stages[index - 1];
  const next = stages[index + 1];
  return el("nav", { class: "adjacent-stages", "aria-label": "相邻步骤" },
    el("a", { href: previous ? "#" + previous.id : "#overview" },
      el("small", {}, previous ? "上一步" : "返回"), el("strong", {}, previous?.label || "批次总览")),
    next && el("a", { href: "#" + next.id },
      el("small", {}, "下一步"), el("strong", {}, next.label)));
}

async function render() {
  const turn = ++renderVersion;
  const [token, encoded] = location.hash.slice(1).split("/");
  const stage = data?.run && stages.find(item => item.id === token);
  let selected = null;
  if (encoded) {
    try { selected = decodeURIComponent(encoded); } catch { selected = null; }
  }
  const overviewSection = !stage && overviewSections.some(([id]) => id === selected) ? selected : null;
  renderLocation(stage, overviewSection);
  if (!data) return;
  const target = stage?.id || "overview";
  if (renderedStage !== target || (!overviewSection && !stage)) window.scrollTo(0, 0);
  renderedStage = target;

  const mount = el("div", { class: "stage-body", id: "stage-view" });
  if (!data.run) {
    mount.append(el("section", { class: "empty empty-state" },
      el("h2", {}, "暂无已完成的批次"),
      el("p", {}, "先运行一次实体本体构建。结果生成后，刷新这里就可以查看。"),
      el("button", { class: "btn", type: "button", onClick: () => location.reload() }, "刷新批次")));
    content.replaceChildren(mount);
    return;
  }
  if (!stage) {
    content.replaceChildren(mount);
    renderOverview({ mount, data, stages });
    if (overviewSection) document.getElementById(overviewSection)?.scrollIntoView();
    return;
  }
  try {
    const [module] = await Promise.all([import("./" + stage.module + ".js"), stageStyle(stage.module)]);
    if (turn !== renderVersion) return;
    content.replaceChildren(mount, adjacentStages(stage));
    module[stage.method]({ stage: stage.id, mount, data, placement: data.placement, go, selected });
  } catch (error) {
    if (turn === renderVersion) fail("这个阶段暂时无法显示。", error);
  }
}

function fail(message, error) {
  console.error(error);
  toc.hidden = true;
  layout.dataset.toc = "false";
  content.replaceChildren(el("section", { class: "empty empty-state", role: "alert" },
    el("h2", {}, "读取结果遇到问题"), el("p", {}, message),
    el("div", { class: "controls" },
      el("button", { class: "btn", type: "button", onClick: () => location.reload() }, "重新读取"),
      run && el("a", { href: "/", class: "text-link" }, "查看最新批次"))));
}

function loadChoices() {
  if (!data.availableRuns?.length) return;
  const select = document.querySelector("#batch-choice");
  for (const row of data.availableRuns) {
    select.append(el("option", { value: row.run_id }, row.title || row.run_id));
  }
  select.value = data.run.run_id;
  select.addEventListener("change", () => {
    const url = new URL(location.href);
    url.searchParams.set("run", select.value);
    url.hash = location.hash.split("/")[0];
    location.assign(url);
  });
  document.querySelector("#batch-switcher").hidden = false;
}

document.querySelector(".workbench-skip").addEventListener("click", event => {
  event.preventDefault();
  document.querySelector("#main").focus();
});
window.addEventListener("hashchange", render);
render();
try {
  const query = run === null ? "" : "?" + new URLSearchParams({ run });
  const response = await fetch("/api/batch" + query);
  if (!response.ok) throw new Error("HTTP " + response.status);
  data = await response.json();
  if (data.run) loadChoices();
  await render();
} catch (error) {
  fail(run ? "这个批次暂时无法读取，可以重新读取或查看最新的已完成批次。" : "暂时无法读取批次结果，请确认工作台服务仍在运行。", error);
}
