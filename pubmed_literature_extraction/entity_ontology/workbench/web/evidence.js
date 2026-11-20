import { el, status, articleLink, typeLabel } from './ui.js';

const decisions = {
  keep: ['保留', 'good'], review: ['待复核', 'warn'], outside_scope: ['超出范围', 'neutral'],
  equivalent: ['同义', 'good'], distinct: ['不同概念', 'neutral'], uncertain: ['不确定', 'warn']
};
const upstream = {
  retained_span: '保留原位置', linked_to_other_span: '归入其他位置的候选',
  filtered: '被抽取器过滤', unresolved_link: '未找到对应候选', not_run: '原流程未做核验',
  ontology_coverage: '抽取补漏找回'
};
const retrievalReasons = {
  shared_name_or_alias: '名称或别名重合', global_knn: '全局 KNN', global_top_two: '全局近邻', within_cluster: '同簇近邻'
};
const locations = {exact:'已定位', missing:'缺少坐标', mismatch:'坐标不一致'};
const co = (tag, cls, ...children) => el(tag, {class: cls}, ...children);
const small = text => co('span', 'ev-small', text);
const copy = text => co('p', 'ev-copy', text);
const badge = value => status(...(decisions[value] || [value, 'neutral']));
const fold = value => String(value ?? '').trim().toLowerCase();
const group = (title, ...children) => co('section', 'ev-group', el('h3', {}, title), ...children);

function action(label, handler, css = 'ev-link') {
  const button = co('button', css, label);
  button.type = 'button';
  button.addEventListener('click', handler);
  return button;
}

function selectControl(title, choices) {
  const select = el('select', {'aria-label': title});
  choices.forEach(([value, label]) => {
    const option = el('option', {}, label);
    option.value = value;
    select.append(option);
  });
  return select;
}

function searchControl(placeholder) {
  const field = el('input', {type: 'search', placeholder, 'aria-label': placeholder, autocomplete: 'off'});
  return field;
}

function field(control) {
  const label = control.type === 'search' ? '搜索记录' : control.getAttribute('aria-label');
  return co('label','ev-field',el('span',{},label),control);
}

function card(top, bottom, trailing) {
  const headline = co('span', 'ev-card-top', co('strong', '', top));
  if (trailing) headline.append(trailing);
  return [headline, co('span', 'ev-card-meta', bottom)];
}

function articleInfo(index, pmid) {
  const article = index.articles.get(pmid);
  return co('div', 'ev-paper', articleLink(pmid), co('span', 'ev-paper-title', article?.title || ''));
}

function markedQuote(text, from, to, span, className = 'ev-quote') {
  const points = Array.from(text);
  const quote = co('blockquote',className);
  if (from >= 0 && to > from && points.slice(from,to).join('') === span) {
    quote.append(points.slice(0,from).join(''),el('mark',{},span),points.slice(to).join(''));
  } else {
    quote.textContent = text;
  }
  return quote;
}

function fullAbstract(mention, index) {
  const article = index.articles.get(mention.pmid);
  if (!article?.abstract) return null;
  const start = Array.from('TITLE: ' + article.title + '\nABSTRACT: ').length;
  return el('details',{class:'ev-extra ev-abstract'},
    el('summary',{},'查看完整摘要'),
    markedQuote(article.abstract,
      mention.offset_status === 'exact' ? mention.char_start - start : -1,
      mention.char_end - start,mention.source_span,'ev-quote ev-abstract-text'));
}

function articleContext(mention, index) {
  let { context: text, context_start: start, context_end: end, context_id: where } = mention;
  // The upstream reader can split a decimal containing a space after the point.
  // Rejoin adjoining fragments only when their boundaries and source characters agree.
  const article = index.articles.get(mention.pmid);
  const source = article ? Array.from('TITLE: ' + article.title + '\nABSTRACT: ' + article.abstract) : [];
  const chunks = index.sentences.get(mention.pmid) || [];
  if (text && source.slice(start, end).join('') === text) {
    for (const c of chunks) {
      if (c.end < start && /^\s+$/.test(source.slice(c.end, start).join('')) &&
          /\d\.$/.test(c.text) && /^\d/.test(text)) {
        start = c.start;
        text = source.slice(start, end).join('');
        where = c.id + ' / ' + where;
      }
      if (c.start > end && /^\s+$/.test(source.slice(end, c.start).join('')) &&
          /\d\.$/.test(text) && /^\d/.test(c.text)) {
        end = c.end;
        text = source.slice(start, end).join('');
        where += ' / ' + c.id;
      }
    }
  }
  return {text, start, where};
}

function quoteBlock(mention, index) {
  const area = co('div', 'ev-evidence');
  if (mention.offset_status === 'exact') {
    const context = articleContext(mention, index);
    const from = mention.char_start - context.start;
    const to = mention.char_end - context.start;
    const quote = markedQuote(context.text,from,to,mention.source_span);
    area.append(small((context.where === 'title' ? '标题' : '摘要 ' + context.where) +
      ' · 原文坐标 [' + mention.char_start + ', ' + mention.char_end + ')'), quote);
  } else {
    area.append(status(locations[mention.offset_status] || '位置未确认', 'warn'),
      copy(mention.evidence_contexts.length ? '可以查看同一篇论文中的其他出现。' : '当前记录缺少可核对的原文位置。'));
  }
  if (mention.evidence_contexts.length) {
    const secondary = el('details', {class: 'ev-extra'});
    if (mention.offset_status !== 'exact') secondary.open = true;
    secondary.append(el('summary', {}, '同文参考句（' + mention.evidence_contexts.length + '）'));
    for (const evidence of mention.evidence_contexts) {
      secondary.append(small((evidence.id === 'title' ? '标题' : evidence.id) + ' · ' +
        (evidence.source === 'parenthetical_mention' ? '含括号释义或缩写' : '其他出现')),
        co('blockquote', 'ev-quote ev-alt-quote', evidence.text));
    }
    area.append(secondary);
  }
  return area;
}

function termButton(m, handler) {
  return action(m.mention + ' · ' + m.id, () => handler(m.id), 'ev-term-link');
}

function vDetails(m, index) {
  const boxes = co('div','ev-upstream');
  boxes.append(small(upstream[m.upstream_status] || m.upstream_status));
  if (m.upstream_filter_reasons.length) boxes.append(copy(m.upstream_filter_reasons.join(' · ')));
  for (const id of m.verification_ids) {
    const v = index.verifications.get(id);
    if (v) boxes.append(copy(v.mention + ' · ' + v.type + ' · ' + v.id));
  }
  return boxes;
}

function mentionInspector(m, index, stage, go) {
  const norm = index.norm.get(m.id);
  const concept = index.conceptByMention.get(m.id);
  const view = co('article','ev-record', small(m.id),
    el('h2', {class:'ev-title'}, m.mention), articleInfo(index, m.pmid));
  const evidence = group('原文证据', quoteBlock(m,index), fullAbstract(m,index));
  const assigned = co('div', 'ev-labeled-value', status(typeLabel(m.raw_type), 'neutral'));
  if (norm?.kind) assigned.append(co('span','ev-arrow','→'), status(typeLabel(norm.kind), 'neutral'));
  if (norm) assigned.append(badge(norm.status));
  if (stage === 'extraction') {
    view.append(evidence,group('抽取记录', co('div','ev-labeled-value',status(typeLabel(m.raw_type), 'neutral')),vDetails(m,index)));
    if (norm) view.append(group('概念化结果', assigned, copy(norm.reason),
      action('查看概念化判断', () => go('normalization',m.id))));
  } else {
    if (norm) {
      const decision = group('类型与处置', assigned, copy(norm.reason));
      if (norm.scope_note) decision.append(copy(norm.scope_note));
      if (norm.name && norm.status === 'keep') decision.append(
        co('div','ev-name',norm.name), copy(norm.definition));
      if (norm.aliases.length) decision.append(small('别名 · ' + norm.aliases.join('、')));
      view.append(decision);
    }
    view.append(evidence,action('回到原始提及',() => go('extraction',m.id)));
  }
  if (m.same_span_conflicts.length) {
    const conflicts = group('同一位置的类型冲突');
    for (const id of m.same_span_conflicts) {
      const other = index.mentions.get(id);
      if (other) conflicts.append(action(other.mention + ' · ' + other.raw_type + ' · ' + id,
        () => go(stage,id),'ev-term-link'));
    }
    view.append(conflicts);
  }
  if (concept) view.append(group('归并概念', action(concept.name, () => go('deduplication',concept.id),'ev-term-link')));
  return view;
}

// The toolbar lives outside the changing list so typing and keyboard focus are preserved.
function renderList({root, rows, initial, controls, match, label, title, details}) {
  const list = co('ul','ev-list');
  const count = el('span',{class:'ev-count','aria-live':'polite'});
  const detail = el('div',{class:'ev-detail',tabindex:-1});
  const reset = action('清除筛选',() => {
    controls.forEach(control => control.value = '');
    draw();
    controls[0]?.focus();
  },'ev-reset');
  const empty = co('p','ev-empty',rows.length ? '没有符合条件的记录。' : '本阶段没有记录。');
  const left = co('section','ev-browser',
    co('div','ev-list-heading',el('h2',{},title),count),
    co('div','ev-controls',...controls.map(field),reset),list);
  root.append(co('div','ev-layout',left, detail));
  let selected = initial;
  function choose(row, jump) {
    selected = row.id;
    for (const b of list.querySelectorAll('button')) {
      b.setAttribute('aria-pressed',String(b.dataset.id === row.id));
    }
    detail.replaceChildren(details(row));
    if (jump && window.matchMedia('(max-width: 900px)').matches) {
      detail.focus({preventScroll:true});
      detail.scrollIntoView({block:'start'});
    }
  }
  function draw() {
    const visible = rows.filter(match);
    count.textContent = String(visible.length) + ' / ' + rows.length;
    reset.hidden = controls.every(control => !control.value);
    list.replaceChildren();
    if (!visible.length) {
      list.append(el('li',{},empty));
      detail.replaceChildren(co('p','ev-detail-empty',rows.length ? '调整筛选后可查看记录。' : '本批次没有可显示的详情。'));
      return;
    }
    for (const row of visible) {
      const b = co('button','ev-row',...label(row));
      b.type = 'button';
      b.dataset.id = row.id;
      b.addEventListener('click',() => choose(row,true));
      list.append(el('li',{},b));
    }
    choose(visible.find(r => r.id===selected) || visible[0],false);
    const button = Array.from(list.querySelectorAll('button')).find(b => b.dataset.id === selected);
    if (button && list.scrollHeight > list.clientHeight) {
      const distance = button.getBoundingClientRect().top - list.getBoundingClientRect().top;
      if (distance < 0 || distance + button.offsetHeight > list.clientHeight) list.scrollTop += distance;
    }
  }
  controls.forEach(c => c.addEventListener(c.tagName === 'INPUT' ? 'input' : 'change',draw));
  draw();
}

function mentionView(root, stage, selected, index, data, go) {
  const rows = data.mentions;
  const isNorm = stage === 'normalization';
  const search = searchControl('搜索名称、缩写或 PMID');
  const papers = selectControl('文献', [['','全部文献'],...data.articles.map(p => [p.pmid,'PMID '+p.pmid])]);
  const counts = new Map();
  const stateFor = m => isNorm ? index.norm.get(m.id)?.status : m.offset_status;
  for (const m of rows) counts.set(stateFor(m),(counts.get(stateFor(m)) || 0)+1);
  const choices = isNorm ? ['keep','review','outside_scope'] : ['exact','missing','mismatch'];
  const textFor = v => isNorm ? decisions[v][0] : locations[v];
  const filter = selectControl(isNorm?'概念化去向':'原文定位', [['','全部状态'],
    ...choices.map(v=>[v,textFor(v)+' · '+(counts.get(v) || 0)])]);
  const recovered = rows.filter(m=>m.upstream_status==='ontology_coverage').length;
  const origin = !isNorm && recovered ? selectControl('提及来源', [['','全部来源'],
    ['native','原始抽取 · '+(rows.length-recovered)],['coverage','补漏 · '+recovered]]) : null;
  const concept = index.concepts.get(selected);
  const initial = concept?.source_mentions[0] || selected;
  const searchMatches = m => {
    const norm = index.norm.get(m.id);
    const c = index.conceptByMention.get(m.id);
    return fold([m.mention,m.raw_type,typeLabel(m.raw_type),m.id,
      norm?.name,norm?.kind,typeLabel(norm?.kind),...(norm?.aliases || []),c?.name,...(c?.aliases || [])].join(' ')).includes(fold(search.value));
  };
  renderList({root,rows,initial,controls:[search,papers,filter,...(origin?[origin]:[])],title:isNorm?'提及判断':'实体提及',
    match:m=>(!papers.value || m.pmid===papers.value) && (!filter.value || stateFor(m)===filter.value) &&
      (!origin?.value || (origin.value==='coverage') === (m.upstream_status==='ontology_coverage')) && searchMatches(m),
    label:m=>card(m.mention,m.id + ' · ' + typeLabel(m.raw_type),isNorm ? badge(stateFor(m)) :
      m.upstream_status==='ontology_coverage' ? status('补漏','good') :
        m.offset_status!=='exact' ? status(locations[m.offset_status] || '位置未确认','warn') : null),
    details:m=>mentionInspector(m,index,stage,go)});
}

function pairNames(pair, index) {
  return pair.id.split('|').map(id => index.mentions.get(id)?.mention || id);
}

function conceptInspector(c, index, go, openPair) {
  const section = co('article','ev-record',small(c.id),co('h2','ev-title',c.name),status(typeLabel(c.kind),'neutral'),copy(c.definition));
  if (c.aliases.length) section.append(small('别名 · '+c.aliases.join('、')));
  const sources=group('来源提及（'+c.source_mentions.length+'）');
  for (const id of c.source_mentions) {
    const m=index.mentions.get(id);
    if (m) sources.append(co('div','ev-source-item',articleInfo(index,m.pmid),
      termButton(m, sourceId=>go('extraction',sourceId)),quoteBlock(m,index)));
  }
  section.append(sources);
  const related=index.pairs.filter(p=>p.id.split('|').some(id=>c.source_mentions.includes(id)));
  if (related.length) {
    const pairs=group('相关配对判断（'+related.length+'）');
    for (const p of related) {
      pairs.append(action(pairNames(p,index).join(' / ')+' · '+decisions[p.applied_verdict][0],()=>openPair(p.id),'ev-term-link'));
    }
    section.append(pairs);
  }
  section.append(action('在层级中查看此概念',()=>go('tree',c.id)));
  return section;
}

function pairInspector(pair, index, go) {
  const selected = index.retrieval.get(pair.id);
  const box = co('article','ev-record',small('配对判断'),co('h2','ev-title',pairNames(pair,index).join(' / ')),
    badge(pair.applied_verdict),copy(pair.reason));
  if (pair.applied_verdict!==pair.verdict) box.append(copy('模型初判为'+decisions[pair.verdict][0]));
  if (selected) {
    const labels = selected.reasons.map(r=>retrievalReasons[r]||r).join('、');
    box.append(small('余弦相似度 '+selected.cosine_similarity.toFixed(3)+' · '+labels));
  }
  for (const id of pair.id.split('|')) {
    const m = index.mentions.get(id);
    const norm = index.norm.get(id);
    if (!m) continue;
    box.append(group(m.mention,articleInfo(index,m.pmid),status(typeLabel(norm?.kind || m.raw_type),'neutral'),
      quoteBlock(m,index),action('查看提及 '+id,()=>go('extraction',id))));
  }
  return box;
}

function dedupView(root, selected, index, data, go) {
  const concepts=data.equivalence.concepts;
  const judgments=index.pairs;
  const selectedPair=judgments.find(p=>p.id===selected || p.id.split('|').reverse().join('|')===selected);
  let mode=selectedPair?'pairs':'concepts';
  const menu=co('div','ev-mode');
  const content=co('div','ev-mode-panel');
  let focused=selectedPair?.id || index.conceptByMention.get(selected)?.id || selected;
  const tabs=[['concepts','归并概念 · '+concepts.length],['pairs','配对判断 · '+judgments.length]];
  function show(tab,id) {
    mode=tab;
    focused=id;
    content.replaceChildren();
    for (const btn of menu.children) btn.setAttribute('aria-pressed',String(btn.dataset.mode===mode));
    const search=searchControl(mode==='concepts'?'搜索概念或别名':'搜索配对名称或 PMID');
    if (mode==='concepts') {
      const kinds=[...new Set(concepts.map(c=>c.kind))].sort();
      const type=selectControl('概念类型',[['','全部类型'],...kinds.map(k=>[k,typeLabel(k)])]);
      renderList({root:content, rows:concepts,initial:focused,controls:[search,type],title:'规范概念',
        match:c=>(!type.value || c.kind===type.value)&&fold([c.name,c.kind,typeLabel(c.kind),...c.aliases,...c.source_mentions].join(' ')).includes(fold(search.value)),
        label:c=>card(c.name,typeLabel(c.kind)+' · '+c.source_mentions.length+' 条提及'),
        details:c=>conceptInspector(c,index,go,id=>show('pairs',id))});
    } else {
      const verdicts=['equivalent','distinct','uncertain'];
      const filter=selectControl('同义判断',[['','全部判断'],...verdicts.map(v=>[v,decisions[v][0]+' · '+judgments.filter(p=>p.applied_verdict===v).length])]);
      renderList({root:content,rows:judgments,initial:focused,controls:[search,filter],title:'候选配对',
        match:p=>(!filter.value || p.applied_verdict===filter.value) && fold([p.id,...pairNames(p,index)].join(' ')).includes(fold(search.value)),
        label:p=>card(pairNames(p,index).join(' / '),p.id.split('|').join(' · '),badge(p.applied_verdict)),
        details:p=>pairInspector(p,index,go)});
    }
  }
  for (const [key,label] of tabs) {
    const button=action(label,()=>show(key,null),'ev-mode-button');
    button.dataset.mode=key;
    menu.append(button);
  }
  root.append(menu,content);
  show(mode,focused);
}

export function renderEvidence({stage,mount,data,go,selected}) {
  mount.replaceChildren();
  const root=co('div','ev-stage');
  mount.append(root);
  const concepts=data.equivalence.concepts;
  const index={
    articles:new Map(data.articles.map(a=>[String(a.pmid),a])),
    mentions:new Map(data.mentions.map(m=>[m.id,m])),
    norm:new Map(data.normalization.map(r=>[r.id,r])),
    verifications:new Map(data.verification_candidates.map(r=>[r.id,r])),
    concepts:new Map(concepts.map(c=>[c.id,c])),
    conceptByMention:new Map(concepts.flatMap(c=>c.source_mentions.map(id=>[id,c]))),
    pairs:data.equivalence.pair_judgments,
    retrieval:new Map(data.retrieval.pairs.map(p=>[p.left+'|'+p.right,p])),
    sentences:new Map()
  };
  for (const [pmid] of index.articles) {
    const units=new Map();
    for (const m of data.mentions.filter(r=>r.pmid===pmid)) {
      if (m.context_id) units.set(m.context_id,{id:m.context_id,text:m.context,start:m.context_start,end:m.context_end});
      for (const c of m.evidence_contexts) units.set(c.id,c);
    }
    index.sentences.set(pmid,[...units.values()].sort((a,b)=>a.start-b.start));
  }
  if (stage==='deduplication') dedupView(root,selected,index,data,go);
  else mentionView(root,stage,selected,index,data,go);
}
