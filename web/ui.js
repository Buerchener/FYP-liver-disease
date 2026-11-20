export function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, val] of Object.entries(attrs || {})) {
    if (val == null || val === false) continue;
    if (key === "class" || key === "className") node.className = val;
    else if (key === "text" || key === "textContent") node.textContent = val;
    else if (/^on[A-Z]/.test(key)) node.addEventListener(key.slice(2).toLowerCase(), val);
    else if (key === "dataset") Object.assign(node.dataset, val);
    else if (key === "style" && typeof val === "object") Object.assign(node.style, val);
    else if (key === "htmlFor") node.htmlFor = val;
    else node.setAttribute(key, val === true ? "" : String(val));
  }
  for (const child of children.flat(Infinity))
    if (child !== null && child !== undefined && child !== false) node.append(child);
  return node;
}
export const status = (label, tone = "neutral") => el("span", {class: "pill tone-" + tone},label);
export const articleLink = (pmid) => el("a", {class: "article-link",href:"https://pubmed.ncbi.nlm.nih.gov/"+encodeURIComponent(pmid)+"/",target:"_blank",rel:"noreferrer"},"PMID "+pmid);
