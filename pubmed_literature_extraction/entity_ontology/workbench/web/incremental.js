import { el, status, typeLabel } from './ui.js';

const actions = {
  new_under_root:['挂到类型根','neutral'], as_child:['挂到已有父类下','good'],
  as_parent:['成为已有概念的父类','good'], insert_between:['插入中间层','good'],
  needs_review:['待复核','warn'], equivalent_pending:['可能重复 · 待复核','warn'], rejected:['拒绝放置','bad'],
};
const supportLabels = {
  schema_type:['类型根','neutral'], dual_model_review:['模型复审支持','good'], go_informed_review:['GO 辅助复审','good'],
};
const relations = {narrower:'更具体',broader:'更宽',equivalent:'等价',unrelated:'未发现类包含',needs_review:'待复核',rejected:'复审未通过'};
const goStatuses = {MATCHED:'匹配',REFERENCE_ONLY:'仅供参考',NO_MATCH:'未找到匹配',UNCERTAIN:'证据不足',LOOKUP_FAILED:'查询失败'};
const reviewLabels = {true:'复审支持',false:'复审未通过',null:'证据不足'};

const button = (label,onClick,cls='btn') => el('button',{class:cls,type:'button',onClick},label);
const chip = (labels,key) => status(...(labels[key] || [key,'neutral']));

export function renderIncremental({mount,placement,data,go,selected}) {
  const {nodes,edges,roots,decisions,source_concepts:source} = placement;
  const placed = new Map(nodes.map(node => [node.id,node]));
  const all = new Map([...source,...nodes].map(node => [node.id,node]));
  const mentions = new Map(data.mentions.map(mention => [mention.id,mention]));
  const outcomes = new Map(decisions.map((decision,i) => [decision.concept_id,{decision,step:i+1}]));
  const parents = new Map(nodes.map(node => [node.id,[]]));
  const children = new Map(nodes.map(node => [node.id,[]]));
  for (const edge of edges) {
    parents.get(edge.child_id).push(edge);
    children.get(edge.parent_id).push(edge);
  }
  const unplaced = source.filter(node => !placed.has(node.id));
  const specific = edges.filter(edge => edge.support !== 'schema_type').length;
  const multiParents = nodes.filter(node => parents.get(node.id).length > 1).length;
  let mode = 'tree';
  let current = all.has(selected) ? selected : source.find(node=>placed.has(node.id))?.id;
  let query = '';
  let kind = '';
  const opened = new Set();
  const tabs = new Map();
  const search = el('input',{type:'search',class:'hw-search','aria-label':'按名称或类型搜索',placeholder:'名称、别名或类型'});
  const kindControl = el('select',{class:'hw-kind','aria-label':'实体类型'},
    el('option',{value:''},'全部类型'),
    Object.keys(roots).map(key=>el('option',{value:key},typeLabel(key))));
  const main = el('section',{class:'hw-main','aria-label':'层级浏览区'});
  const detail = el('aside',{class:'hw-detail','aria-label':'逐个放置详情',tabindex:-1});
  const metrics = [
    ['入树概念', source.length-unplaced.length],
    ['具体父子边', specific],
    ['多父类概念', multiParents],
    ['未挂接概念', unplaced.length],
  ];
  if (placement.go_reference_count) metrics.push(['GO 查询',placement.go_reference_count]);
  const toolbar = el('div',{class:'hw-toolbar'});
  const tabList = el('div',{class:'hw-tabs'});
  const views = [['tree','层级图'],['steps',`放置记录 · ${decisions.length}`],['unplaced',`未挂接 · ${unplaced.length}`]];
  for (const [id,label] of views) {
    const control = button(label,()=> { mode=id; refresh(); },'hw-tab');
    tabs.set(id,control);
    tabList.append(control);
  }
  toolbar.append(tabList);
  search.addEventListener('input',()=> { query=search.value.trim().toLowerCase(); renderMain(); });
  kindControl.addEventListener('change',()=> {
    kind=kindControl.value;
    if (kind && all.get(current)?.kind!==kind) {
      current = mode==='tree' && !query ? roots[kind] :
        (mode==='unplaced' ? unplaced : source).find(matches)?.id;
    }
    refresh();
  });
  mount.replaceChildren(el('div',{class:'hw'},
    el('dl',{class:'hw-metrics'},metrics.map(([label,value])=>el('div',{},el('dt',{},label),el('dd',{},String(value))))),
    el('div',{class:'hw-layout'},
      el('section',{class:'hw-browser'},toolbar,el('div',{class:'hw-filters'},
        el('label',{class:'hw-field'},el('span',{},'搜索概念'),search),
        el('label',{class:'hw-field'},el('span',{},'实体类型'),kindControl)),
      main),detail)));

  function display(id) {
    const row = all.get(id);
    if (!row) return id;
    return row.origin==='schema_root' ? typeLabel(row.kind) : row.name;
  }
  function matches(node) {
    return (!kind || kind===node.kind) &&
      [node.name,node.id,node.kind,typeLabel(node.kind),...(node.aliases||[])].join(' ').toLowerCase().includes(query);
  }
  function rememberPath(id) {
    const path = [id];
    let next = parents.get(id)?.[0];
    while (next && !path.includes(next.parent_id)) {
      path.unshift(next.parent_id);
      next=parents.get(next.parent_id)?.[0];
    }
    for (let i=1;i<path.length;i+=1) opened.add(path.slice(0,i).join('/'));
  }
  function focusDetail() {
    if (!window.matchMedia('(max-width: 900px)').matches) return;
    detail.focus({preventScroll:true});
    detail.scrollIntoView({block:'start'});
  }
  function jumpNode(id) {
    current=id;
    mode=placed.has(id)?'tree':'unplaced';
    query=''; search.value='';
    if (kind && all.get(id)?.kind!==kind) { kind=''; kindControl.value=''; }
    rememberPath(id);
    refresh();
    focusDetail();
  }
  function pick(id) {
    current=id;
    for (const row of main.querySelectorAll('[data-node]')) {
      const active = row.dataset.node === current;
      row.classList.toggle('hw-current',active);
      row.setAttribute('aria-pressed',String(active));
    }
    renderDetail();
    focusDetail();
  }
  function branch(id, path=[], edge=null) {
    const trail = [...path,id];
    const key = trail.join('/');
    const kids = children.get(id);
    const open = opened.has(key);
    const action = button('',()=>pick(id),'hw-node');
    action.dataset.node=id;
    action.setAttribute('aria-pressed',String(id===current));
    if (id===current) action.classList.add('hw-current');
    action.append(...[el('span',{class:'hw-node-name'},display(id)),
      edge && chip(supportLabels,edge.support),
      parents.get(id).length>1 && status(`${parents.get(id).length} 个父类`),
      kids.length>0 && el('span',{class:'hw-count'},String(kids.length))].filter(Boolean));
    let toggle = el('span',{class:'hw-branch-dot','aria-hidden':'true'});
    if (kids.length) {
      toggle = button(open?'▾':'▸',()=> {
        if (open) opened.delete(key);
        else opened.add(key);
        const scroll = main.querySelector('.hw-forest').scrollTop;
        renderMain();
        main.querySelector('.hw-forest').scrollTop = scroll;
        [...main.querySelectorAll('[data-branch]')].find(row=>row.dataset.branch===key)?.focus({preventScroll:true});
      },'hw-toggle');
      toggle.dataset.branch=key;
      toggle.setAttribute('aria-expanded',String(open));
      toggle.setAttribute('aria-label',(open?'折叠':'展开')+display(id));
    }
    const item = el('li',{class:'hw-branch'+(edge?'':' hw-root')},el('div',{class:'hw-branch-row'},toggle,action));
    if (open && kids.length) item.append(el('ul',{class:'hw-children'},kids.filter(row=>!trail.includes(row.child_id)).map(row=>branch(row.child_id,trail,row))));
    return item;
  }
  function renderMain() {
    for (const [name,control] of tabs) control.setAttribute('aria-pressed',String(name===mode));
    if (mode==='tree' && !query) {
      const shownRoots = Object.entries(roots).filter(([key])=>!kind || key===kind).map(([,id])=>id);
      main.replaceChildren(shownRoots.length ?
        el('ul',{class:'hw-forest'},shownRoots.map(id=>branch(id))) :
        el('p',{class:'hw-note hw-empty'},'这批文献没有已放置的概念。'));
      return;
    }
    const rows = (mode==='steps' ? decisions.map(row=>all.get(row.concept_id)) : mode==='unplaced' ? unplaced : [...all.values()]).filter(matches);
    const list = el('div',{class:'hw-flat'});
    for (const row of rows) {
      const outcome=outcomes.get(row.id);
      const ctl=button(el('span',{class:'hw-edge-names'},mode==='steps' && el('small',{class:'hw-sub'},String(outcome.step)),
        el('strong',{},display(row.id))),()=>pick(row.id),'hw-edge');
      ctl.dataset.node=row.id;
      ctl.setAttribute('aria-pressed',String(row.id===current));
      if (row.id===current) ctl.classList.add('hw-current');
      if (outcome) ctl.append(chip(actions,outcome.decision.action));
      list.append(ctl);
    }
    main.replaceChildren(rows.length ? list : el('p',{class:'hw-note hw-empty'},
      mode==='unplaced' && !unplaced.length ? '本批次的概念都已挂接。' : '没有符合条件的概念。'));
  }
  function links(title, entries, key) {
    if (!entries.length) return null;
    return el('div',{class:'hw-source'},el('h4',{},title),entries.map(edge=>
      el('div',{class:'hw-relation-link'},button(display(edge[key]),()=>jumpNode(edge[key]),'btn hw-link'),chip(supportLabels,edge.support))));
  }
  function showEvidence(reference) {
    if (!reference) return null;
    const item=reference.result;
    if (!item) return null;
    let sourceLink=null;
    try {
      const url=new URL(item.source);
      if (url.protocol==='https:' && url.hostname==='www.ebi.ac.uk') sourceLink=el('a',{href:url.href,target:'_blank',rel:'noreferrer'},item.go_id || 'OLS 记录');
    } catch { sourceLink=null; }
    return el('div',{class:'hw-source'},el('h4',{},'GO 参考'),status(goStatuses[item.status] || item.status),
      item.go_name && el('p',{lang:'en'},item.go_name),item.go_definition && el('p',{lang:'en'},item.go_definition),sourceLink,
      ...(item.ontology_relations||[]).map(row=>el('p',{},`${row.subject} · ${row.predicate} · ${row.object}`)),
      ...(reference.reviews||[]).map(row=>el('p',{},row.reason)));
  }
  function showComparisons(decision) {
    if (!decision.comparisons?.length) return null;
    const list=el('div',{class:'hw-comparisons'});
    for (const row of decision.comparisons) {
      list.append(el('section',{},
        el('div',{class:'hw-relation-link'},button(display(row.candidate_id),()=>jumpNode(row.candidate_id),'btn hw-link'),status(relations[row.result] || row.result)),
        row.proposal?.reason && el('p',{},row.proposal.reason),
        ...(row.reviews||[]).map(review=>el('p',{},reviewLabels[String(review.supported)]+' · '+review.reason)),
        ...(row.go_reviews||[]).map(review=>el('p',{},'GO 辅助'+reviewLabels[String(review.supported)]+' · '+review.reason))));
    }
    return el('details',{class:'hw-comparison-audit'},el('summary',{},`与 ${decision.comparisons.length} 个已有概念比较`),list);
  }
  function renderDetail() {
    const node=all.get(current);
    if (!node) {
      detail.replaceChildren(el('h3',{},'放置理由'),el('p',{},'选择概念查看父类、子类、模型判断和文献原句。'));
      return;
    }
    const record=outcomes.get(node.id);
    const decision=record?.decision;
    detail.replaceChildren(el('div',{class:'hw-detail-header'},
      el('p',{class:'hw-detail-id'},node.id),el('h3',{},display(node.id)),
      node.origin!=='schema_root' && el('p',{class:'hw-note'},typeLabel(node.kind)),
      node.origin!=='schema_root' && el('p',{lang:'en'},node.definition),
      record && el('div',{class:'hw-relation-link'},chip(actions,decision.action),el('span',{class:'hw-note'},`第 ${record.step} 次放置`)),
      ['needs_review','rejected','equivalent_pending'].includes(decision?.action) && decision.reason &&
        el('p',{class:'hw-rationale'},decision.reason)),
      ...[links('直接父类',parents.get(node.id)||[],'parent_id'),links('直接子类',children.get(node.id)||[],'child_id')].filter(Boolean));
    if (node.source_mentions?.length) detail.append(el('div',{class:'hw-source'},el('h4',{},'文献原句'),
      node.source_mentions.map(id=>button(mentions.get(id)?.mention || id,()=>go('extraction',id),'btn hw-link')),
      el('div',{class:'hw-evidence-path'},button('查看同义归并',()=>go('deduplication',node.id),'btn hw-link'),
        button('查看排序位置',()=>go('ordering',node.id),'btn hw-link'))));
    if (decision) {
      for (const [label,key] of [['新增连线','added_edges'],['移除的冗余连线','removed_edges']]) {
        if (decision[key]?.length) detail.append(el('div',{class:'hw-source'},el('h4',{},label),decision[key].map(row=>
          el('p',{},display(row.child_id)+' 属于 '+display(row.parent_id)))));
      }
      if (decision.generated_parent_reviews?.length) detail.append(el('div',{class:'hw-source'},
        el('h4',{},'拟议中间类复审'),decision.generated_parent_reviews.map(review=>
          el('p',{},reviewLabels[String(review.supported)]+' · '+review.reason))));
      detail.append(...[showComparisons(decision),showEvidence(decision.go_reference)].filter(Boolean));
    }
  }
  function refresh() { renderMain(); renderDetail(); }
  if (current) {
    mode=placed.has(current)?'tree':'unplaced';
    rememberPath(current);
  }
  refresh();
}
