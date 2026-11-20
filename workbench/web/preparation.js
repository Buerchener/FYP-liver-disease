import { el, status } from './ui.js';

const CLUSTER_COLORS = ['#166b70', '#7962a4', '#ab6325', '#467f4a', '#b74d6e', '#3275aa', '#8b7540', '#785b63'];
const VERDICTS = {
  equivalent: '同义',
  distinct: '不同',
  uncertain: '待定',
};

function indexBy(rows, key) {
  return new Map(rows.map(row => [row[key], row]));
}

function button(label, onClick, className = 'btn') {
  return el('button', {type: 'button', class: className, onClick}, label);
}

function sectionTitle(text) {
  return el('h2', {class: 'prep-section-title'}, text);
}

function labelledSelect(label, onChange) {
  const select = el('select', {'aria-label': label});
  select.addEventListener('change', () => onChange(select.value));
  return {select, wrapper: el('label', {class: 'prep-field'}, el('span', {}, label), select)};
}

function option(value, label) {
  return el('option', {value: String(value)}, label);
}

function keyValue(rows) {
  const dl = el('dl', {class: 'prep-fields'});
  for (const [key, value] of rows) {
    dl.append(el('dt', {}, key), el('dd', {}, value));
  }
  return dl;
}

function color(clusterId) {
  return clusterId < 0 ? '#566373' : CLUSTER_COLORS[clusterId % CLUSTER_COLORS.length];
}

function clusterText(id) {
  return Number(id) === -1 ? '噪声' : '簇 ' + id;
}

function statsLine(entries) {
  return el('dl', {class:'prep-strip'}, ...entries.map(([label, value]) =>
    el('div', {}, el('dt', {}, label), el('dd', {}, String(value)))));
}

function tickRange(values) {
  const lo = Math.min(...values);
  const hi = Math.max(...values);
  const padding = Math.max((hi - lo) * .06, .03);
  const delta = (hi - lo + padding * 2) / 5;
  const power = Math.pow(10, Math.floor(Math.log10(delta)));
  const step = [1, 2, 5, 10].map(x => x * power).find(x => x >= delta);
  return [Math.floor((lo - padding) / step) * step, Math.ceil((hi + padding) / step) * step, step];
}

function plot(points, ratios, colorByCluster, onPick) {
  const [xmin, xmax, xstep] = tickRange(points.map(p => p.x));
  const [ymin, ymax, ystep] = tickRange(points.map(p => p.y));
  const xPercent = x => 100 * (x - xmin) / (xmax - xmin);
  const yPercent = y => 100 * (ymax - y) / (ymax - ymin);
  const grid = el('div', {class: 'prep-grid', 'aria-label': '可选择的投影点'});
  const dotIndex = new Map();
  for (let x = xmin; x <= xmax + xstep * .01; x += xstep) {
    const line = el('span', {class: 'prep-gridline-x', 'aria-hidden':'true'},
      el('span', {}, String(Number(x.toFixed(2)))));
    line.style.left = xPercent(x) + '%';
    grid.append(line);
  }
  for (let y = ymin; y <= ymax + ystep * .01; y += ystep) {
    const line = el('span', {class: 'prep-gridline-y', 'aria-hidden':'true'},
      el('span', {}, String(Number(y.toFixed(2)))));
    line.style.top = yPercent(y) + '%';
    grid.append(line);
  }
  for (const point of points) {
    const label = point.concept_name + ' · ' + point.source_id;
    const dot = button(label, () => onPick(point.source_id), 'prep-dot');
    dot.replaceChildren(el('span', {class:'prep-sr-only'}, label));
    dot.title = label;
    dot.style.left = xPercent(point.x) + '%';
    dot.style.top = yPercent(point.y) + '%';
    dot.style.setProperty('--dot-color', colorByCluster ? color(point.cluster_id) : 'var(--accent, #076a70)');
    if (colorByCluster && point.cluster_id === -1) dot.classList.add('is-noise');
    grid.append(dot);
    dotIndex.set(point.source_id, dot);
  }
  const axis = (name, i) => name + (ratios?.[i] != null ? ' · ' + (ratios[i] * 100).toFixed(1) + '%' : '');
  const wrap = el('figure', {class: 'prep-figure'},
    el('div', {class:'prep-plot'},
      el('span', {class:'prep-axis-y'}, axis('PC2',1)),
      grid,
      el('span', {class:'prep-axis-x'}, axis('PC1',0))),
    el('figcaption', {}, colorByCluster
      ? '颜色区分 HDBSCAN 簇，菱形表示组外实体。选择一个点可回查原句。'
      : '每个点对应一条保留提及。选择图中的点，或从下方列表查看它在原文中的位置。'));
  return {wrap, dotIndex};
}

function renderMap({stage, mount, data, go, selected}) {
  const retrieval = data.retrieval;
  const points = retrieval.points;
  if (!points.length) {
    mount.replaceChildren(el('p', {class:'empty'}, '本次运行没有可显示的向量。'));
    return;
  }
  const clusterMode = stage === 'clustering';
  const pointById = indexBy(points, 'source_id');
  const normal = indexBy(data.normalization, 'id');
  const mentions = indexBy(data.mentions, 'id');
  const pairReview = indexBy(data.equivalence.pair_judgments, 'id');
  const concept = data.equivalence.concepts.find(c => c.id === selected);
  let current = pointById.has(selected) ? selected : concept?.source_mentions[0];
  if (!pointById.has(current)) current = retrieval.pairs[0]?.left || points[0].source_id;
  let activeCluster = 'all';
  let clusterButtons = [];
  const memberSelect = labelledSelect('选择提及', id => select(id));
  const visible = () => points.filter(p => activeCluster === 'all' || String(p.cluster_id) === activeCluster);
  const inspector = el('aside', {class: 'panel prep-inspector', 'aria-label':'投影点详情'});
  const chart = plot(points, retrieval.projection.explained_variance_ratio, clusterMode, id => select(id));

  function refreshMembers() {
    const options = visible();
    memberSelect.select.replaceChildren(...options.map(p => option(p.source_id, p.concept_name + ' · ' + p.source_id)));
    if (!options.some(p => p.source_id === current)) current = options[0].source_id;
    memberSelect.select.value = current;
  }

  function select(id) {
    if (!pointById.has(id)) return;
    current = id;
    memberSelect.select.value = id;
    for (const [pointId, dot] of chart.dotIndex) {
      const inactive = activeCluster !== 'all' && String(pointById.get(pointId).cluster_id) !== activeCluster;
      dot.classList.toggle('is-muted', inactive);
      dot.disabled = inactive;
      dot.setAttribute('aria-pressed', String(pointId === id));
    }
    for (const b of clusterButtons) b.setAttribute('aria-pressed', String(b.dataset.cluster === activeCluster));
    const p = pointById.get(id);
    const normalized = normal.get(id);
    const mention = mentions.get(id);
    const pairs = retrieval.pairs.filter(pair => pair.left === id || pair.right === id);
    const related = data.equivalence.concepts.find(c => c.source_mentions.includes(id));
    inspector.replaceChildren(
      el('div', {class:'prep-detail-key'}, (mention?.context_id === 'title' ? '标题' : '摘要') + ' · PMID ' + id.split(':')[0]),
      el('h3', {}, p.concept_name),
      el('p', {class:'prep-definition'}, p.description),
      keyValue([
        ['归一化类型', normalized.kind],
        ['文中写法', mention.mention],
        ...(clusterMode ? [['簇分配', clusterText(p.cluster_id)],
          ['成员概率', p.cluster_id < 0 ? '未分配到簇' : (p.probability * 100).toFixed(1) + '%']] :
          [['二维坐标', '(' + p.x.toFixed(3) + ', ' + p.y.toFixed(3) + ')'],
           ['别名', normalized.aliases.join('、') || '无']]),
      ]),
      el('div', {class:'prep-actions'}, button('查看原句', () => go('extraction', id)),
        ...(related ? [button('查看归并概念', () => go('deduplication', related.id))] : []))
    );
    const pairHeading = el('h4', {}, '同义候选配对', el('span', {class:'prep-count'}, String(pairs.length)));
    inspector.append(pairHeading);
    if (!pairs.length) inspector.append(el('p', {class:'muted'}, '此提及未召回候选配对。'));
    const list = el('ul', {class:'prep-pairs'});
    for (const pair of pairs) {
      const other = pointById.get(pair.left === id ? pair.right : pair.left);
      const pairId = pair.left + '|' + pair.right;
      const verdict = pairReview.get(pairId)?.applied_verdict;
      list.append(el('li', {}, button(other.concept_name, () => go('deduplication', pairId), 'prep-link'),
        el('span', {class:'prep-sim'}, '余弦 ' + pair.cosine_similarity.toFixed(3)),
        ...(verdict ? [status(VERDICTS[verdict] ?? verdict, verdict === 'equivalent' ? 'good' : verdict === 'uncertain' ? 'warn' : 'neutral')] : [])));
    }
    inspector.append(list);
  }

  const tool = el('div', {class:'prep-cluster-controls', 'aria-label':'按簇筛选'});
  if (clusterMode) {
    const clusters = retrieval.clustering.clusters.map(c => [String(c.cluster_id), c.member_ids.length]);
    if (retrieval.clustering.noise_ids.length) clusters.push(['-1', retrieval.clustering.noise_ids.length]);
    for (const [id, count] of [['all', points.length], ...clusters]) {
      const label = id === 'all' ? '全部' : clusterText(id);
      const b = button(label + ' (' + count + ')', () => {
        activeCluster = id;
        refreshMembers();
        select(current);
      }, 'prep-cluster-btn');
      b.dataset.cluster = id;
      if (id !== 'all') {
        const swatch = el('i', {class:'prep-swatch', 'aria-hidden':'true'});
        swatch.style.background = color(Number(id));
        if (id === '-1') swatch.classList.add('is-noise');
        b.prepend(swatch);
      }
      tool.append(b);
      clusterButtons.push(b);
    }
  }
  refreshMembers();
  select(current);
  const panel = el('section', {class:'panel prep-map-panel'},
    sectionTitle(clusterMode ? '簇分布' : '实体向量投影'),
    el('p', {class:'muted prep-description'}, clusterMode
      ? 'HDBSCAN 用于浏览分组。右侧的同义候选来自全局近邻或共享名称和别名。'
      : '名称、类型、定义和别名共同编码为向量。图中显示前两个主成分。'),
    ...(clusterMode ? [tool] : []),
    chart.wrap, memberSelect.wrapper);
  mount.replaceChildren(el('div', {class: 'entity-prep'},
    clusterMode ?
      statsLine([['聚类方法', retrieval.clustering.method], ['簇数', retrieval.clustering.clusters.length],
                 ['组外实体', retrieval.clustering.noise_ids.length]]) :
      statsLine([['向量模型', retrieval.embedding.model], ['向量维数', retrieval.embedding.dimensions],
                 ['实体提及', points.length], ['投影方法', retrieval.projection.method]]),
    el('div', {class: 'prep-columns'}, panel, inspector)));
}

function renderOrder({mount, data, go, selected}) {
  const order = data.conceptOrder;
  if (order?.status !== 'complete') {
    const messages = {
      invalid: '构树前的排序文件暂时无法读取。',
      stale: '排序结果与当前去重后的实体概念不一致，请重新运行排序。',
    };
    mount.replaceChildren(el('section', {class:'panel prep-order-panel'}, sectionTitle('构树前排序'),
      el('p', {class:'muted', role:'status'}, messages[order?.status] || '这批实体还没有构树前的排序结果。'),
      button('查看去重后的概念', () => go('deduplication'))));
    return;
  }

  const concepts = data.equivalence.concepts;
  const byId = indexBy(concepts, 'id');
  const mentions = indexBy(data.mentions, 'id');
  const rows = order.groups.flatMap(group => group.concept_ids.map((id, i) => (
    {concept: byId.get(id), rank: i + 1}
  )));
  const directConcept = byId.get(selected) || concepts.find(c => c.source_mentions.includes(selected));
  let current = directConcept?.id;
  let kind = 'all';
  let query = '';
  const body = el('tbody');
  const results = el('span', {class:'prep-counter', 'aria-live':'polite'});
  const inspector = el('aside', {class:'panel prep-inspector', 'aria-label':'排序概念详情'});
  const controls = labelledSelect('实体类型', value => {
    kind = value;
    update();
  });
  controls.select.append(option('all','全部类型'), ...order.groups.map(group => option(group.kind, group.kind)));
  const search = el('input', {type:'search', placeholder:'搜索名称、别名或 ID', 'aria-label':'搜索排序概念'});
  search.addEventListener('input', () => {
    query = search.value.trim().toLowerCase();
    update();
  });

  function show({concept, rank}) {
    const {id, name, kind, definition, aliases, source_mentions: sources} = concept;
    current = id;
    body.querySelectorAll('[data-node]').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.node === id)));
    inspector.replaceChildren(
      el('div', {class:'prep-detail-key'}, kind + ' · 组内序位 ' + rank),
      el('h3', {}, name),
      el('p', {class:'prep-definition'}, definition),
      keyValue([['概念标识', id], ['别名', aliases.join('、') || '无']]),
      el('div', {class:'prep-actions'}, button('查看去重后的概念', () => go('deduplication', id))),
    );
    if (sources.length) {
      const links = sources.map(sourceId => {
        const source = mentions.get(sourceId);
        const label = source ? source.mention + ' · ' + sourceId : sourceId;
        return button(label, () => go('extraction', sourceId), 'prep-link');
      });
      inspector.append(el('h4', {}, '文献提及'), el('div', {class:'prep-origin-mentions'}, ...links));
    }
  }
  function update() {
    const filtered = rows.filter(({concept}) => (kind === 'all' || concept.kind === kind) &&
      [concept.name, concept.kind, concept.id, ...concept.aliases].some(s => s.toLowerCase().includes(query)));
    results.textContent = filtered.length + ' / ' + rows.length + ' 个概念';
    body.replaceChildren();
    for (const row of filtered) {
      const b = button(row.concept.name, () => show(row), 'prep-node-button');
      b.dataset.node = row.concept.id;
      body.append(el('tr', {}, el('td', {class:'prep-rank'}, String(row.rank)),
        el('td', {}, b), el('td', {}, row.concept.kind)));
    }
    if (!filtered.length) {
      const cell = el('td', {colspan:'3', class:'empty'}, '没有符合条件的概念。');
      body.append(el('tr', {}, cell));
      inspector.replaceChildren(el('p', {class:'muted'}, '调整筛选后可查看概念详情。'));
      return;
    }
    show(filtered.find(row => row.concept.id === current) || filtered[0]);
  }
  const head = el('thead', {}, el('tr', {}, ...['组内序位','概念','实体类型'].map(t => el('th', {scope:'col'}, t))));
  const table = el('table', {class:'data-table prep-table', 'aria-label':'构树前的实体概念顺序'}, head, body);
  mount.replaceChildren(el('div', {class:'entity-prep'},
    statsLine([['实体概念', rows.length], ['实体类型', order.groups.length], ['排序方法', 'LLM']]),
    el('div', {class:'prep-columns'},
      el('section', {class:'panel prep-order-panel'}, sectionTitle('构树前排序'),
        el('p', {class:'muted prep-description'}, 'LLM 按名称和定义从宽到窄排序。序位在同一实体类型内有效，用来决定构树时先处理哪个概念。'),
        el('div', {class:'prep-order-controls'}, controls.wrapper,
          el('label', {class:'prep-field'}, el('span', {}, '搜索概念'), search)),
        results, el('div', {class:'prep-table-scroll'}, table)),
      inspector)));
  update();
}

export function renderPreparation(context) {
  if (context.stage === 'ordering') renderOrder(context);
  else renderMap(context);
}
