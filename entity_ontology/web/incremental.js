import { el, status } from './ui.js';

const types = { Disease:'疾病', CellType:'细胞类型', Tissue:'组织', Metabolite:'代谢物', Protein:'蛋白质', Pathway:'通路', Gene:'基因' };
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

const button = (label,onClick,cls='btn') => el('button',{class:cls,type:'button',onClick},label);
const chip = (labels,key) => status(...(labels[key] || [key,'neutral']));

export function renderIncremental({mount,placement,go,selected}) {
  const {nodes,edges,roots,decisions,source_concepts:source} = placement;
  const placed = new Map(nodes.map(node => [node.id,node]));
  const all = new Map([...source,...nodes].map(node => [node.id,node]));
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
  let current = all.has(selected) ? selected : null;
  let query = '';
  const opened = new Set();
  const tabs = new Map();
  const search = el('input',{type:'search',class:'hw-search','aria-label':'按名称或类型搜索',placeholder:'搜索概念、别名或类型'});
  const main = el('section',{class:'hw-main','aria-label':'层级浏览区'});
  const detail = el('aside',{class:'hw-detail','aria-label':'逐个放置详情'});
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
  mount.replaceChildren(el('div',{class:'hw'},
    el('p',{class:'hw-summary'},`${source.length} 个概念中 ${source.length-unplaced.length} 个已放入候选层级，形成 ${specific} 条具体父子边。多父类节点 ${multiParents} 个，GO 查询 ${placement.go_reference_count} 次。`),
    toolbar,el('div',{class:'hw-filters'},search),el('div',{class:'hw-layout'},main,detail)));

  function display(id) {
    const row = all.get(id);
    if (!row) return id;
    return row.origin==='schema_root' ? (types[row.kind] || row.name) : row.name;
  }
  function matches(node) {
    return [node.name,node.kind,types[node.kind],...(node.aliases||[])].join(' ').toLowerCase().includes(query);
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
  function jumpNode(id) {
    current=id;
    mode=placed.has(id)?'tree':'unplaced';
    query=''; search.value='';
    rememberPath(id);
    refresh();
  }
  function pick(id,path,expand) {
    current=id;
    if (expand) {
      if (opened.has(path)) opened.delete(path);
      else opened.add(path);
    }
    refresh();
  }
  function branch(id, path=[], edge=null) {
    const trail = [...path,id];
    const key = trail.join('/');
    const kids = children.get(id);
    const open = opened.has(key);
    const action = button('',event=> {
      pick(id,key,kids.length);
      if (event.detail===0) [...main.querySelectorAll('[data-branch]')].find(row => row.dataset.branch===key)?.focus();
    },'hw-node');
    action.dataset.branch=key;
    if (kids.length) action.setAttribute('aria-expanded',String(open));
    if (id===current) action.classList.add('hw-current');
    action.append(...[el('span',{class:'hw-caret','aria-hidden':'true'},kids.length?(open?'▾':'▸'):'·'),
      el('span',{class:'hw-node-name'},display(id)),
      edge && chip(supportLabels,edge.support),
      parents.get(id).length>1 && status(`${parents.get(id).length} 个父类`),
      kids.length>0 && el('span',{class:'hw-count'},String(kids.length))].filter(Boolean));
    const item = el('li',{class:'hw-branch'+(edge?'':' hw-root')},action);
    if (open && kids.length) item.append(el('ul',{class:'hw-children'},kids.filter(row=>!trail.includes(row.child_id)).map(row=>branch(row.child_id,trail,row))));
    return item;
  }
  function renderMain() {
    for (const [name,control] of tabs) control.setAttribute('aria-pressed',String(name===mode));
    if (mode==='tree' && !query) {
      main.replaceChildren(el('p',{class:'hw-note'},'展开类型根查看子类。多个父类的概念可从不同分支打开。'),
        el('ul',{class:'hw-forest'},Object.values(roots).map(id=>branch(id))));
      if (!Object.keys(roots).length) main.append(el('p',{class:'empty'},'这批文献没有成功归并的概念。'));
      return;
    }
    const rows = (mode==='steps' ? decisions.map(row=>all.get(row.concept_id)) : mode==='unplaced' ? unplaced : [...all.values()]).filter(matches);
    const list = el('div',{class:'hw-flat'});
    for (const row of rows) {
      const outcome=outcomes.get(row.id);
      const ctl=button(el('span',{class:'hw-edge-names'},mode==='steps' && el('small',{class:'hw-sub'},String(outcome.step)),
        el('strong',{},display(row.id))),()=>pick(row.id),'hw-edge');
      if (row.id===current) ctl.classList.add('hw-current');
      if (outcome) ctl.append(chip(actions,outcome.decision.action));
      list.append(ctl);
    }
    main.replaceChildren(rows.length ? list : el('p',{class:'empty'},'没有匹配的概念。'));
  }
  function links(title, entries, key) {
    if (!entries.length) return null;
    return el('div',{class:'hw-source'},el('h4',{},title),entries.map(edge=>
      el('div',{class:'hw-relation-link'},button(display(edge[key]),()=>jumpNode(edge[key]),'btn hw-link'),chip(supportLabels,edge.support))));
  }
  function showEvidence(reference) {
    if (!reference) return null;
    const item=reference.result;
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
    if (!decision.comparisons.length) return null;
    const headings = {true:'模型复审支持',false:'模型复审未通过',null:'证据不足'};
    const list=el('div',{class:'hw-comparisons'});
    for (const row of decision.comparisons) {
      list.append(el('section',{},
        el('div',{class:'hw-relation-link'},button(display(row.candidate_id),()=>jumpNode(row.candidate_id),'btn hw-link'),status(relations[row.result] || row.result)),
        row.proposal?.reason && el('p',{},row.proposal.reason),
        ...row.reviews.map(review=>el('p',{},headings[String(review.supported)]+' · '+review.reason))));
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
    detail.replaceChildren(el('h3',{},display(node.id)),el('p',{class:'hw-note'},types[node.kind] || node.kind),
      node.origin!=='schema_root' && el('p',{lang:'en'},node.definition),
      record && el('div',{class:'hw-relation-link'},chip(actions,decision.action),el('span',{class:'hw-note'},`第 ${record.step} 次放置`)),
      links('直接父类',parents.get(node.id)||[],'parent_id'),links('直接子类',children.get(node.id)||[],'child_id'));
    if (node.source_mentions?.length) detail.append(el('div',{class:'hw-source'},el('h4',{},'文献原句'),
      node.source_mentions.map(id=>button(id,()=>go('extraction',id),'btn hw-link')),
      button('查看同义归并',()=>go('deduplication',node.id),'btn hw-link')));
    if (decision) {
      if (['needs_review','rejected','equivalent_pending'].includes(decision.action)) detail.append(el('p',{},decision.reason));
      for (const [label,key] of [['新增连线','added_edges'],['移除的冗余连线','removed_edges']]) {
        if (decision[key].length) detail.append(el('div',{class:'hw-source'},el('h4',{},label),decision[key].map(row=>
          el('p',{},display(row.child_id)+' 属于 '+display(row.parent_id)))));
      }
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
