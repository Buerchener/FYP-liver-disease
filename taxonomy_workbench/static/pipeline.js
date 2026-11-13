import {element, edgeTone, indexGraph, TaxonomyGraph} from './taxonomy-graph.js';

const byId = id => document.getElementById(id);
const stageNames = {events: '事件抽取', concepts: '概念化', leaves: 'Leaf 归一化', taxonomy: '层级归纳'};
const rootNames = {BIO: '生物事件', CARE: '临床照护', RESEARCH: '生物医学研究', unrooted: '未连顶层'};
const kindNames = {root: '预定义顶层', intermediate: '中间概念', leaf: 'Leaf Concept'};
let data, graph, nodes, scopes, eventById, eventBySource, documentByPmid;
let stage = 'taxonomy';
let scope = 'RESEARCH';
let selectedNode;
let selectedRecord;
const branchButtons = new Map();

function hashState() {
  return new URLSearchParams(location.hash.slice(1));
}

function storeView() {
  const params = new URLSearchParams({stage});
  if (stage === 'taxonomy') {
    params.set('scope', scope);
    params.set('node', selectedNode);
  } else if (selectedRecord) params.set('record', selectedRecord);
  history.replaceState(null, '', '#' + params.toString());
}

function leafCount(ids) {
  return [...ids].filter(id => nodes.get(id).kind === 'leaf').length;
}

function chooseNode(id, requestedScope, remember = true) {
  if (!nodes.has(id)) return;
  const nextScope = requestedScope || scope;
  scope = scopes.get(nextScope)?.has(id) ? nextScope : [...scopes].find(([, ids]) => ids.has(id))[0];
  selectedNode = id;
  for (const [key, button] of branchButtons) button.setAttribute('aria-pressed', String(key === scope));
  byId('unrooted-picker').hidden = scope !== 'unrooted';
  const options = [...byId('unrooted-select').options];
  byId('unrooted-select').value = options.some(option => option.value === id) ? id : '';
  graph.show(id);
  renderNode(nodes.get(id));
  if (remember) storeView();
}

function detailSection(heading) {
  const section = element('section', 'detail-section');
  section.append(element('h3', '', heading));
  return section;
}

function evidenceView(event) {
  const article = element('article', 'evidence');
  const source = element('a', 'source', 'PMID ' + event.pmid);
  source.href = 'https://pubmed.ncbi.nlm.nih.gov/' + encodeURIComponent(event.pmid) + '/';
  source.target = '_blank';
  source.rel = 'noopener noreferrer';
  source.title = documentByPmid.get(event.pmid)?.title || '';
  article.append(source, element('p', '', event.text));
  const quote = element('blockquote');
  const text = event.evidence || event.text;
  const at = event.trigger ? text.indexOf(event.trigger) : -1;
  if (at < 0) quote.textContent = text;
  else quote.append(document.createTextNode(text.slice(0, at)), element('mark', '', event.trigger),
    document.createTextNode(text.slice(at + event.trigger.length)));
  const evidence = element('details');
  evidence.append(element('summary', '', '原句与 Trigger'), quote);
  article.append(evidence);
  return article;
}

function appendEvidence(container, events) {
  if (!events.length) return;
  const section = detailSection('原始事件 · ' + events.length);
  section.append(...events.slice(0, 3).map(evidenceView));
  if (events.length > 3) {
    const more = element('details');
    more.append(element('summary', '', '其余 ' + (events.length - 3) + ' 条事件'));
    more.append(...events.slice(3).map(evidenceView));
    section.append(more);
  }
  container.append(section);
}

function relationView(edge, target) {
  const wrapper = element('div', 'relation');
  wrapper.dataset.edge = edge.id;
  const button = element('button', '', nodes.get(target).label);
  button.type = 'button';
  button.addEventListener('click', () => chooseNode(target));
  wrapper.append(button, element('span', 'verdict ' + edgeTone(edge), edge.review_label));
  if (edge.review_reason) wrapper.append(element('p', '', edge.review_reason));
  return wrapper;
}

function renderNode(node) {
  const pane = byId('inspector-content');
  pane.replaceChildren(
    element('p', 'node-type', kindNames[node.kind]),
    element('h2', '', node.label),
    element('p', 'definition', node.definition || '尚无定义'),
  );
  if (node.parents.length) {
    const parents = detailSection('直接父类');
    parents.append(...node.parents.map(edge => relationView(edge, edge.parent)));
    pane.append(parents);
  } else if (node.kind !== 'root') {
    pane.append(element('p', 'empty-note', '尚无直接父类'));
  }
  if (node.children.length) {
    const section = detailSection('子类关系 · ' + node.children.length);
    const details = element('details');
    details.append(element('summary', '', '查看逐边复查'));
    details.append(...node.children.map(edge => relationView(edge, edge.child)));
    section.append(details);
    pane.append(section);
  }
  appendEvidence(pane, node.event_ids.map(id => eventById.get(id)).filter(Boolean));
  if (node.kind === 'root') {
    const section = detailSection('本次运行');
    section.append(element('p', 'subtle',
      data.run.rounds.length + ' 轮归纳 · ' + data.run.llm_models.join(' / ')));
    const review = element('p', 'subtle', '连线颜色来自 Codex 定义复查，尚待人工审定。');
    section.append(review);
    pane.append(section);
  }
}

function revealEdge(edge) {
  const pane = byId('inspector-content');
  const row = [...pane.querySelectorAll('.relation')].find(item => item.dataset.edge === edge.id);
  if (!row) return;
  pane.querySelectorAll('.relation').forEach(item => item.classList.remove('selected'));
  if (row.closest('details')) row.closest('details').open = true;
  row.classList.add('selected');
  row.scrollIntoView({block: 'nearest', behavior: 'auto'});
}

function mountGraph() {
  graph = new TaxonomyGraph(byId('graph'), nodes, chooseNode, revealEdge);
  const unrooted = [...scopes.get('unrooted')].map(id => nodes.get(id));
  const sort = (a, b) => a.label.localeCompare(b.label);
  const branches = unrooted.filter(node => !node.parents.length && node.children.length).sort(sort);
  const isolated = unrooted.filter(node => !node.parents.length && !node.children.length).sort(sort);
  const defaultUnrooted = branches.find(node => node.name === 'Research Finding Statement') || branches[0] || isolated[0];
  for (const [key, ids] of scopes) {
    const button = element('button');
    button.type = 'button';
    button.append(element('span', '', rootNames[key] || nodes.get(key)?.name || key), element('span', 'count', String(leafCount(ids))));
    button.addEventListener('click', () => chooseNode(key === 'unrooted' ? defaultUnrooted.id : key, key));
    button.disabled = !ids.size;
    byId('branches').append(button);
    branchButtons.set(key, button);
  }
  const select = byId('unrooted-select');
  const placeholder = new Option('选择分支或独立概念', '');
  placeholder.disabled = true;
  select.append(placeholder);
  for (const [label, values] of [['未连顶层的分支', branches], ['没有父子边的 Leaf', isolated]]) {
    const group = element('optgroup');
    group.label = label;
    group.append(...values.map(node => new Option(node.label, node.id)));
    select.append(group);
  }
  select.addEventListener('change', () => chooseNode(select.value, 'unrooted'));
  byId('unrooted-count').textContent = data.stats.parentless + ' 个 Leaf 无直接父类';
  const middle = [...nodes.values()].filter(node => node.kind === 'intermediate').sort(sort);
  byId('middle-count').textContent = middle.length;
  byId('middle-list').append(...middle.map(node => {
    const button = element('button', '', node.label);
    button.type = 'button';
    button.addEventListener('click', () => chooseNode(node.id));
    return button;
  }));
  byId('show-intermediates').addEventListener('click', () => {
    const list = byId('middle-list');
    list.hidden = !list.hidden;
    byId('show-intermediates').setAttribute('aria-pressed', String(!list.hidden));
  });
}

function records() {
  if (stage === 'events') return data.events.map(event => ({
    id: event.id, name: event.text, definition: '', events: [event], pmids: [event.pmid],
  }));
  if (stage === 'concepts') return data.concepts.map(concept => {
    const event = eventBySource.get(concept.source_id);
    return {id: concept.source_id, name: concept.concept_name || '待概念化',
      definition: concept.description, events: event ? [event] : [], pmids: event ? [event.pmid] : []};
  });
  return data.leaves.map(leaf => {
    const events = leaf.source_events.map(id => eventBySource.get(id)).filter(Boolean);
    return {id: leaf.concept_id, name: leaf.concept_name, definition: leaf.description,
      events, pmids: [...new Set(events.map(event => event.pmid))], sources: leaf.source_concepts.length};
  });
}

function showRecord(row) {
  selectedRecord = row.id;
  for (const button of byId('record-list').children) button.setAttribute('aria-pressed', String(button.dataset.id === row.id));
  const detail = byId('record-detail');
  detail.replaceChildren(element('h2', '', stage === 'events' ? '事件与证据' : row.name));
  if (row.definition) detail.append(element('p', 'definition', row.definition));
  if (stage === 'leaves') {
    detail.append(element('p', 'empty-note', row.sources + ' 条原始概念记录'));
    const button = element('button', 'button', '在分类图中查看');
    button.type = 'button';
    button.addEventListener('click', () => {changeStage('taxonomy'); chooseNode(row.id);});
    detail.append(button);
  }
  appendEvidence(detail, row.events);
  storeView();
}

function renderRecords() {
  const needle = byId('record-search').value.toLocaleLowerCase().trim();
  const pmid = byId('article-select').value;
  const rows = records().filter(row => (!pmid || row.pmids.includes(pmid)) &&
    (!needle || [row.name, row.definition, ...row.pmids, ...row.events.map(event => event.text)].join(' ').toLocaleLowerCase().includes(needle)));
  byId('result-count').textContent = rows.length + ' 条';
  const list = byId('record-list');
  list.replaceChildren(...rows.map(row => {
    const button = element('button', 'record');
    button.type = 'button';
    button.dataset.id = row.id;
    button.append(element('strong', '', row.name), element('span', '', 'PMID ' + row.pmids.join(' · ')));
    button.addEventListener('click', () => showRecord(row));
    return button;
  }));
  const selected = rows.find(row => row.id === selectedRecord) || rows[0];
  if (selected) showRecord(selected);
  else {
    list.append(element('p', 'no-records', '没有匹配的记录'));
    byId('record-detail').replaceChildren();
    selectedRecord = null;
  }
}

function changeStage(nextStage, remember = true) {
  if (!stageNames[nextStage]) nextStage = 'taxonomy';
  stage = nextStage;
  for (const button of document.querySelectorAll('[data-stage]')) {
    if (button.dataset.stage === stage) button.setAttribute('aria-current', 'step');
    else button.removeAttribute('aria-current');
  }
  byId('page-title').textContent = stageNames[stage];
  const stats = data.stats;
  const meta = {
    events: stats.articles + ' 篇文献 · ' + stats.events + ' 条事件' +
      (stats.rejected_events == null ? '' : ' · ' + stats.rejected_events + ' 条未通过原文校验'),
    concepts: stats.events + ' 条事件 → ' + stats.concepts + ' 条概念记录 · ' + stats.abstained + ' 条待概念化',
    leaves: stats.concepts + ' 条概念记录 → ' + stats.leaves + ' 个 Leaf',
    taxonomy: stats.leaves + ' 个 Leaf · ' + stats.intermediates + ' 个中间概念 · ' +
      stats.rooted + ' 个 Leaf 已连到顶层 · ' + stats.edges + ' 条候选边',
  };
  byId('page-meta').textContent = meta[stage];
  byId('taxonomy-view').hidden = stage !== 'taxonomy';
  byId('records-view').hidden = stage === 'taxonomy';
  byId('export').hidden = stage !== 'taxonomy';
  if (stage === 'taxonomy') graph.schedule();
  else renderRecords();
  if (remember) storeView();
}

function restore() {
  const params = hashState();
  selectedRecord = params.get('record');
  chooseNode(nodes.has(params.get('node')) ? params.get('node') : data.roots[data.roots.length - 1],
    params.get('scope') || data.roots[data.roots.length - 1], false);
  changeStage(params.get('stage') || 'taxonomy', false);
}

async function start() {
  try {
    const response = await fetch('/api/pipeline');
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || '无法读取结果');
    data = result;
    if (!data.available) {
      byId('empty').hidden = false;
      byId('page-meta').textContent = '';
      document.querySelectorAll('[data-stage]').forEach(button => button.disabled = true);
      return;
    }
    ({nodes, scopes} = indexGraph(data));
    eventById = new Map(data.events.map(event => [event.id, event]));
    eventBySource = new Map(data.events.map(event => [event.source_id, event]));
    documentByPmid = new Map(data.documents.map(doc => [doc.pmid, doc]));
    mountGraph();
    byId('article-select').append(...data.documents.map(doc => new Option(doc.pmid + ' · ' + doc.title, doc.pmid)));
    document.querySelectorAll('[data-stage]').forEach(button => button.addEventListener('click', () => changeStage(button.dataset.stage)));
    byId('record-search').addEventListener('input', renderRecords);
    byId('article-select').addEventListener('change', renderRecords);
    window.addEventListener('hashchange', restore);
    restore();
  } catch (error) {
    byId('error').textContent = error.message;
    byId('error').hidden = false;
    byId('page-meta').textContent = '当前结果读取失败';
  }
}
start();
