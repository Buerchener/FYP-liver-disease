const state={token:null,overview:null,datasetId:'synthetic',dataset:null,runId:null,run:null,
 step:null,stepData:null,conceptId:null,view:'current',zoom:null,follow:true,player:null,polling:false,busy:false,viewVersion:0};
const $=id=>document.getElementById(id);
export const escapeHTML=value=>String(value??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
export const statusLabel=status=>({queued:'准备中',running:'运行中',pausing:'等待保存后暂停',paused:'已暂停',completed:'已完成',interrupted:'已中断，可恢复',failed:'启动失败'}[status]||'待开始');
export const isActive=status=>['queued','running','pausing'].includes(status);
export function stageOf(run,row,follow){
 if(follow&&isActive(run?.status))return {diagnosis:'A',induction:'B',validation:'C',diagnose:'A',propose:'B',validate:'C'}[run?.state?.waiting_for||run?.state?.phase]||null;
 return row?.stage||null;
}
export function counts(initial,current,eventCount){
 const original=initial?.frozen_input_concept_ids||[],present=new Set((current?.nodes||[]).map(n=>n.concept_id));
 return {original:original.length,added:Math.max(0,(current?.nodes?.length||0)-(initial?.nodes?.length||0)),retained:original.filter(i=>present.has(i)).length,events:eventCount||0};
}
export function pendingMarkup(taxonomy){
 const roots=new Set(taxonomy.predefined_roots),pending=new Set(taxonomy.unplaced_concept_ids);
 const button=node=>`<button class="pending-concept" data-concept="${escapeHTML(node.concept_id)}">${escapeHTML(node.concept_label)}</button>`;
 return `<div class="pending-preview">
   <h3>预定义类别</h3>
   <div class="pending-roots">${taxonomy.nodes.filter(node=>roots.has(node.concept_id)).map(button).join('')}</div>
   <h3>${pending.size} 个待归属概念</h3>
   <div class="pending-grid">${taxonomy.nodes.filter(node=>pending.has(node.concept_id)).map(button).join('')}</div>
 </div>`;
}
export function decisionMarkup(row,llmRef=row?.llm_ref){
 if(!row)return '<p class="empty-inline">选择概念或归纳步骤。</p>';
 const esc=escapeHTML;
 const meta=[];
 if(row.round||row.attempt)meta.push(['轮次 / 候选',`${row.round||'—'} / ${row.attempt||'—'}`]);
 if(row.confidence!=null)meta.push(['模型置信度',`${Math.round(row.confidence*100)}%`]);
 if(row.groups?.length)meta.push(['候选子群',row.groups.map(group=>`${group.group_id} (${group.child_ids.length})`).join('、')]);
 const added=row.added_labels?.length
   ? `<div class="concept-list" aria-label="涉及的中间概念">${row.added_labels.map(label=>`<span class="concept-chip">${esc(label)}</span>`).join('')}</div>`
   : '';
 const summary=row.action==='stop_level'&&row.summary==='当前层级无需增加中间概念。'
   ? '' : row.summary?.replace(/ 这不表示原结构已经最优。$/,'');
 return `<h3 class="${esc(row.tone)}">${esc(row.title)}</h3>
   ${row.parent_label?`<p class="detail-context">${esc(row.parent_label)}</p>`:''}
   ${summary?`<p class="decision-summary">${esc(summary)}</p>`:''}
   ${added}
   ${meta.length||row.reasoning?`<details class="detail-block"><summary>判定依据</summary>
     <dl>${meta.map(([label,value])=>`<div class="meta-row"><dt>${label}</dt><dd>${esc(value)}</dd></div>`).join('')}</dl>
     ${row.reasoning?`<blockquote class="reasoning">${esc(row.reasoning)}</blockquote>`:''}
   </details>`:''}
   ${llmRef?'<button class="text-button" id="raw-open">模型输入与输出</button>':''}`;
}

async function api(path,options={}){
 const headers=options.method?{'Content-Type':'application/json','X-Workbench-Token':state.token||''}:{};
 const response=await fetch(path,{...options,headers:{...headers,...options.headers}});
 if(!response.ok){let message='请求失败，请确认本地服务仍在运行。';try{message=(await response.json()).error||message;}catch{}throw new Error(message);}
 return options.svg?response.text():response.json();
}
function error(message){$('error').textContent=message;$('error').hidden=!message;}
function toast(message){$('toast').textContent=message;$('toast').hidden=false;clearTimeout(toast.timer);toast.timer=setTimeout(()=>$('toast').hidden=true,3500);}
function stopPlayer(){if(state.player)clearInterval(state.player);state.player=null;$('play').textContent='回放';}
function getSource(){return state.overview?.datasets.find(x=>x.id===state.datasetId);}
function selectedRow(){return state.run?.timeline?.find(row=>row.sequence===state.step)||null;}
function renderSources(){
 $('dataset').innerHTML=state.overview.datasets.map(row=>{
   const label=row.id==='synthetic'?'合成示例':row.label.replace(/ · 待准备$/,'');
   return `<option value="${escapeHTML(row.id)}">${escapeHTML(label)}</option>`;
 }).join('');
 $('dataset').value=state.datasetId;
 const source=getSource();if(!source)return;
 $('readiness').hidden=source.ready;
 const problems=[];
 if(source.missing_definitions)problems.push(`${source.missing_definitions} 个概念缺少定义`);
 if(source.unplaced_count)problems.push(`${source.unplaced_count} 个概念待归属`);
 $('readiness').textContent=source.ready?'':problems.length?problems.join('，')+'。':source.problem;
 $('mode').querySelector('option[value="mock"]').disabled=source.id!=='synthetic';
 if(source.id!=='synthetic')$('mode').value='live';
 renderControls();
}
function renderModel(){
 const model=state.overview?.model;if(!model)return;
 $('model-dot').classList.toggle('green',model.ready);
 $('model-label').textContent=model.ready?model.model:'模型连接';
 $('base-url').value=model.base_url;$('model-name').value=model.model;renderControls();
}
function renderControls(){
 const source=getSource(),running=Boolean(state.overview?.active_run),mode=$('mode').value;
 const missingModel=mode==='live'&&!state.overview?.model.ready;
 $('start').disabled=state.busy||running||!source?.ready||missingModel;
 $('start').textContent=mode==='mock'?'开始演示':'开始归纳';
 $('mode-help').hidden=!source?.ready||!missingModel;
 $('mode-help').textContent=missingModel?'请先配置模型连接。':'';
 $('pause').hidden=!state.runId||state.overview?.active_run!==state.runId;
 $('pause').disabled=state.busy||state.run?.status==='pausing';
 $('pause').textContent=state.run?.status==='pausing'?'正在等待保存…':'暂停实验';
 $('resume').hidden=!['paused','interrupted'].includes(state.run?.status)||!state.run?.state;
 $('resume').disabled=state.busy||running;
}
function renderHistory(){
 const rows=state.overview?.runs||[];
 $('history').innerHTML=rows.length?rows.map(row=>{
   const current=state.runId===row.id;
   return `<button class="history-item${current?' current':''}" data-run="${escapeHTML(row.id)}"${current?' aria-current="true"':''}>
     <span title="${escapeHTML(row.label)}">${escapeHTML(row.label)}</span><small>${statusLabel(row.status)}</small>
   </button>`;
 }).join(''):'<p class="empty-inline">暂无记录</p>';
}
function renderSummary(){
 const run=state.run,initial=run?.initial_taxonomy||state.dataset?.taxonomy,current=run?.taxonomy||initial;
 const source=getSource();
 const values=counts(initial,current,run?.event_count??state.dataset?.events.length);
 $('metric-added').textContent=initial?values.added:'—';
 $('metric-retained').textContent=initial?`${values.retained} / ${values.original}`:'—';$('metric-events').textContent=initial?values.events:'—';
 $('metric-evidence').textContent=initial?run?.evidence_count??state.dataset?.evidence.length??0:'—';
 const origin=run?.data_origin||run?.job?.data_origin||source?.data_origin;
 const mode=run?.mode||run?.job?.mode||$('mode').value;
 $('data-badge').className='source-tag';
 $('data-badge').hidden=!origin;
 $('data-badge').textContent=origin==='synthetic'?(mode==='mock'?'合成演示':'合成数据'):'真实文献';
 $('status').textContent=!run&&source&&!source.ready?'待准备':statusLabel(run?.status);
 $('status').className=run?.status||'';
 renderStage();
 const info=run?.state;
 $('progress').hidden=!isActive(run?.status);
 $('progress').textContent=info?`${info.parent_label||'归纳中'} · 已保存 ${info.sequence} 步 · ${info.llm_requests} 次${mode==='mock'?'模拟':'模型'}调用`:'正在准备数据…';
 $('export-graph').hidden=!run?.report;$('export-log').hidden=!run?.report;
 if(state.runId){$('export-graph').href=`/api/runs/${encodeURIComponent(state.runId)}/export/final_taxonomy.json`;$('export-log').href=`/api/runs/${encodeURIComponent(state.runId)}/export/change_log.json`;}
 renderControls();
}
function keepStepVisible(){
 const timeline=$('timeline'),selected=timeline.querySelector('.selected');
 if(!selected)return;
 const left=selected.offsetLeft,right=left+selected.offsetWidth;
 if(left<timeline.scrollLeft)timeline.scrollLeft=left-8;
 else if(right>timeline.scrollLeft+timeline.clientWidth)timeline.scrollLeft=right-timeline.clientWidth+8;
}
function renderTimeline(){
 const rows=state.run?.timeline||[];
 const timeline=$('timeline'),hadFocus=timeline.contains(document.activeElement);
 timeline.innerHTML=rows.length?rows.map(row=>{
   const selected=row.sequence===state.step;
   return `<button class="timeline-item ${escapeHTML(row.tone)}${selected?' selected':''}" data-step="${row.sequence}" aria-pressed="${selected}">
     <strong>${escapeHTML(row.title)}</strong><span class="step-parent" title="${escapeHTML(row.parent_label)}">${escapeHTML(row.parent_label)}</span>
   </button>`;
 }).join(''):'<p class="empty-inline">暂无归纳步骤</p>';
 const index=rows.findIndex(x=>x.sequence===state.step);
 $('previous').disabled=index<=0;$('next').disabled=index>=rows.length-1||!rows.length;
 $('play').disabled=!rows.length;$('latest').disabled=!rows.length;
 if(hadFocus)timeline.querySelector('.selected')?.focus({preventScroll:true});
 keepStepVisible();
}
function renderStage(){
 const stage=stageOf(state.run,selectedRow(),state.follow);
 for(const id of ['A','B','C']){
   const indicator=$('stage-'+id);
   indicator.classList.toggle('is-current',id===stage);
   if(id===stage)indicator.setAttribute('aria-current','step');
   else indicator.removeAttribute('aria-current');
 }
}
function renderDecision(){
 state.conceptId=null;
 highlightConcept();
 $('detail-heading').textContent='步骤详情';$('back-decision').hidden=true;
 const row=selectedRow();$('explanation').innerHTML=decisionMarkup(row,state.stepData?.llm_ref||row?.llm_ref);
 $('explanation').scrollTop=0;
 renderStage();
 const button=$('raw-open');if(button)button.addEventListener('click',showRaw);
}
function highlightConcept(){
 for(const node of $('graphs').querySelectorAll('[data-concept]')){
   const selected=node.dataset.concept===state.conceptId;
   node.classList.toggle('selected',selected);
   node.setAttribute('aria-pressed',selected);
 }
}
function zoom(){
 const width=state.zoom===null&&state.view!=='compare'?'max(100%, 720px)':`${(state.zoom??1)*100}%`;
 for(const root of [$('graph'),$('graph-compare')]){
   const graphic=root.querySelector('svg');
   if(graphic)graphic.style.width=width;
 }
}
function adjustZoom(delta){
 const pane=$('graph'),graphic=pane.querySelector('svg');
 if(!graphic)return;
 const style=getComputedStyle(pane);
 const width=pane.clientWidth-parseFloat(style.paddingLeft)-parseFloat(style.paddingRight);
 const current=state.zoom??graphic.getBoundingClientRect().width/Math.max(1,width);
 state.zoom=Math.min(4,Math.max(.5,current+delta));
 zoom();
}
function mountGraph(container,markup,label){
 container.innerHTML=markup;
 const graphic=container.querySelector('svg');
 if(!graphic)return;
 graphic.setAttribute('role','group');
 graphic.setAttribute('aria-label',label);
 const graphviz=graphic.querySelector('g.graph'),canvas=graphviz||graphic;
 for(const caption of canvas.querySelectorAll(':scope > text'))caption.remove();
 canvas.querySelector(':scope > [fill="#f4f7fb"]')?.remove();

 for(const node of graphic.querySelectorAll(graphviz?'g.node':':scope > g')){
   const title=node.querySelector(':scope > title');
   if(!title)continue;
   const lines=[...node.querySelectorAll('text')];
   if(/^\d+ supporting event\(s\)$/.test(lines.at(-1)?.textContent)){
     lines.pop().remove();
     const offset=graphviz?11:24-(lines.length-1)*8.5;
     for(const line of lines)line.setAttribute('dy',offset);
   }
   node.classList.add('concept-node');
   node.dataset.concept=title.textContent.split('\n')[0];
   node.setAttribute('role','button');
   node.setAttribute('tabindex','0');
   node.setAttribute('aria-label',lines.map(line=>line.textContent).join(' ')||node.dataset.concept);
 }
 const oversized=Array.from(graphic.querySelectorAll('.concept-node')).flatMap(node=>{
   const shape=node.querySelector('path, rect, polygon');
   if(!shape)return [];
   const width=shape.getBBox().width-16;
   return Array.from(node.querySelectorAll('text'))
     .filter(line=>width>0&&line.getComputedTextLength()>width)
     .map(line=>[line,width]);
 });
 for(const [line,width] of oversized){
   line.setAttribute('textLength',width);
   line.setAttribute('lengthAdjust','spacingAndGlyphs');
 }
 const box=graphic.getBBox(),padding=18;
 if(box.width&&box.height){
   graphic.setAttribute('viewBox',`${box.x-padding} ${box.y-padding} ${box.width+2*padding} ${box.height+2*padding}`);
   graphic.removeAttribute('width');
   graphic.removeAttribute('height');
 }
}
async function renderGraph(){
 const version=++state.viewVersion,runId=state.runId,step=state.step,view=state.view;
 const compare=Boolean(runId&&view==='compare');
 $('compare-pane').hidden=!compare;$('graphs').classList.toggle('compare',compare);
 $('graph-title').hidden=!compare;
 $('candidate-notice').hidden=view!=='candidate';
 for(const button of $('view-tabs').querySelectorAll('button')){
   const selected=button.dataset.view===view;
   button.classList.toggle('selected',selected);
   button.setAttribute('aria-pressed',selected);
   button.disabled=button.dataset.view!=='current'&&(!runId||(button.dataset.view==='candidate'&&!state.stepData?.candidate));
 }
 const pending=!runId&&state.dataset?.taxonomy.unplaced_concept_ids.length;
 $('zoom').hidden=Boolean(pending);
 if(pending){
   $('graph').innerHTML=pendingMarkup(state.dataset.taxonomy);
   $('graph').scrollTop=0;
   $('graph-compare').innerHTML='';
   highlightConcept();
   return;
 }
 const base=runId?`/api/runs/${encodeURIComponent(runId)}/graph`:`/api/datasets/${encodeURIComponent(state.datasetId)}/graph`;
 let queries=[],titles=[];
 if(!runId){queries=[''];titles=['初始结构 · 尚未运行'];}
 else if(view==='initial'){queries=['?view=initial'];titles=['初始结构'];}
 else if(view==='candidate'&&step){queries=[`?step=${step}&view=candidate`];titles=['候选结构 · 未提交'];}
 else if(compare){queries=step?[`?step=${step}&view=before`,`?step=${step}&view=after`]:['?view=initial','?view=current'];titles=step?['修改前','修改后']:['初始结构','当前结构'];}
 else{queries=[step?`?step=${step}&view=after`:'?view=current'];titles=['当前结构'];}
 try{
  const diagrams=await Promise.all(queries.map(query=>api(base+query,{svg:true})));
  if(version!==state.viewVersion)return;
  $('graph-title').textContent=titles[0];$('compare-title').textContent=titles[1]||'';
  mountGraph($('graph'),diagrams[0],titles[0]);
  mountGraph($('graph-compare'),diagrams[1]||'',titles[1]);
  highlightConcept();
  zoom();
 }catch(cause){if(version===state.viewVersion){$('graph').innerHTML='<p class="graph-empty">暂无分类图</p>';$('graph-compare').innerHTML='';if(!isActive(state.run?.status))error(cause.message);}}
}
async function selectStep(number,manual=true){
 if(!state.runId)return;
 if(manual){state.follow=false;stopPlayer();}
 state.step=number;const runId=state.runId;
 try{const detail=await api(`/api/runs/${encodeURIComponent(runId)}/steps/${number}`);
  if(state.runId!==runId||state.step!==number)return;
  state.stepData=detail;if(state.view==='candidate'&&!detail.candidate)state.view='current';
  renderTimeline();renderDecision();await renderGraph();
 }catch(cause){error(cause.message);}
}
async function refreshRun(){
 if(!state.runId)return;const id=state.runId,run=await api('/api/runs/'+encodeURIComponent(id));if(id!==state.runId)return;
 const previous=state.run?.state?.sequence;state.run=run;renderSummary();
 if(!run.state){renderControls();if(run.job?.message&&run.status==='failed')error(run.job.message);return;}
 if(['interrupted','failed'].includes(run.status)&&run.job?.message)error(run.job.message);
 if(previous!==run.state.sequence||state.step==null){
  if(state.follow&&run.timeline.length){await selectStep(run.timeline.at(-1).sequence,false);}
  else{renderTimeline();if(state.step==null)await renderGraph();}
 }
}
async function chooseRun(id){
 stopPlayer();state.viewVersion++;state.runId=id;state.run=null;state.step=null;state.stepData=null;state.follow=true;state.view='current';error('');
 renderHistory();renderTimeline();renderDecision();await refreshRun();renderHistory();
}
async function chooseDataset(id){
 stopPlayer();state.viewVersion++;state.datasetId=id;state.runId=null;state.run=null;state.step=null;state.stepData=null;state.view='current';error('');
 const result=await api('/api/datasets/'+encodeURIComponent(id));if(state.datasetId!==id||state.runId)return;
 state.dataset=result.dataset;renderSources();renderSummary();renderHistory();renderTimeline();renderDecision();await renderGraph();
}
async function refreshOverview(){
 const overview=await api('/api/overview');state.overview=overview;state.token=overview.token;
 $('server-label').textContent='本地服务已连接';$('server-dot').classList.add('green');renderSources();renderModel();renderHistory();
}
async function act(action){
 if(state.busy)return;state.busy=true;renderControls();error('');
 try{
  if(action==='start'){
   const input={dataset_id:state.datasetId,mode:$('mode').value,iterations_per_parent:Number($('iterations').value),candidate_attempts:Number($('attempts').value),max_depth:Number($('depth').value),llm_calls:Number($('calls').value)};
   const result=await api('/api/runs',{method:'POST',body:JSON.stringify(input)});await refreshOverview();await chooseRun(result.run_id);toast('实验已启动。');
  }else{
   await api(`/api/runs/${encodeURIComponent(state.runId)}/${action}`,{method:'POST',body:'{}'});await refreshOverview();await refreshRun();
   toast(action==='pause'?'会在当前步骤保存后暂停。':'从保存的位置继续运行。');
  }
 }catch(cause){error(cause.message);}finally{state.busy=false;renderControls();}
}
async function showEvidence(nodeId){
 const {runId,step,datasetId}=state;
 state.conceptId=nodeId;
 let detail;
 try{
  if(runId)detail=await api(`/api/runs/${encodeURIComponent(runId)}/evidence?concept_id=${encodeURIComponent(nodeId)}${step?`&step=${step}`:''}`);
  else{const data=state.dataset,node=data.taxonomy.nodes.find(n=>n.concept_id===nodeId);if(!node)return;detail={concept:node,events:data.events.filter(e=>node.supporting_event_ids.includes(e.event_id)),evidence:data.evidence.filter(e=>node.evidence.includes(e.evidence_id))};}
  if(state.runId!==runId||state.step!==step||state.datasetId!==datasetId||state.conceptId!==nodeId)return;
  const node=detail.concept,esc=escapeHTML;
  const evidence=new Map(detail.evidence.map(row=>[row.evidence_id,row]));
  const events=detail.events.map(event=>{
    const pmid=event.source_pmid&&/^\d+$/.test(event.source_pmid)
      ? `<small><a href="https://pubmed.ncbi.nlm.nih.gov/${event.source_pmid}/" target="_blank" rel="noopener">PMID ${event.source_pmid} ↗</a></small>`
      : '';
    const excerpts=event.evidence_ids.map(id=>{
      const record=evidence.get(id);
      if(!record)return '';
      const status={
        verified_exact:'',
        verified_whitespace:'原文空白有差异',
        ambiguous:'原文有多处匹配',
        unverified:'原句待定位',
        passage_only:'摘要级证据',
        missing:'缺少原句'
      }[record.alignment_status]??record.alignment_status;
      return `${record.text?`<blockquote>${esc(record.text)}</blockquote>`:''}${status?`<small>${esc(status)}</small>`:''}`;
    }).join('');
    return `<article class="evidence-event"><p>${esc(event.event_text)}</p>${pmid}${excerpts}</article>`;
  }).join('');
  const heading=$('detail-heading');
  heading.textContent='概念证据';
  $('back-decision').hidden=false;
  $('explanation').innerHTML=`<h3>${esc(node.concept_label)}</h3>
    <p class="decision-summary">${esc(node.concept_description||'待补定义。')}</p>
    <p class="detail-context">${detail.events.length} 个关联事件 · ${detail.evidence.length} 条证据</p>
    ${events}
    <details class="detail-block"><summary>概念 ID</summary><p class="muted small">${esc(node.concept_id)}</p></details>`;
  $('explanation').scrollTop=0;
  highlightConcept();
  heading.focus({preventScroll:true});
  if(matchMedia('(max-width: 690px)').matches)heading.scrollIntoView({block:'nearest'});
 }catch(cause){error(cause.message);}
}
function openGraphNode(event){
 const node=event.target.closest('[data-concept]');
 if(!node)return;
 if(event.type==='keydown'&&(node.tagName==='BUTTON'||!['Enter',' '].includes(event.key)))return;
 event.preventDefault();
 showEvidence(node.dataset.concept);
}
async function showRaw(){
 const ref=state.stepData?.llm_ref||selectedRow()?.llm_ref;if(!ref){toast('这一步没有对应的模型调用记录。');return;}
 try{const result=await api(`/api/runs/${encodeURIComponent(state.runId)}/llm/${encodeURIComponent(ref.split('/').at(-1))}`);$('raw-json').textContent=JSON.stringify(result,null,2);$('raw-dialog').showModal();}catch(cause){error(cause.message);}
}
async function poll(){
 if(state.polling||document.hidden||state.busy)return;state.polling=true;
 try{if(state.overview?.active_run||isActive(state.run?.status)){await refreshOverview();await refreshRun();}}
 catch(cause){$('server-label').textContent='本地服务连接中断';$('server-dot').classList.remove('green');error(cause.message);}
 finally{state.polling=false;}
}
async function init(){
 $('dataset').addEventListener('change',()=>chooseDataset($('dataset').value).catch(cause=>error(cause.message)));
 $('mode').addEventListener('change',()=>{renderControls();renderSummary();});
 for(const action of ['start','pause','resume'])$(action).addEventListener('click',()=>act(action));
 $('history').addEventListener('click',event=>{const button=event.target.closest('[data-run]');if(button)chooseRun(button.dataset.run).catch(cause=>error(cause.message));});
 $('refresh').addEventListener('click',()=>refreshOverview().catch(cause=>error(cause.message)));
 $('timeline').addEventListener('click',event=>{const button=event.target.closest('[data-step]');if(button)selectStep(Number(button.dataset.step));});
 $('view-tabs').addEventListener('click',event=>{const button=event.target.closest('[data-view]');if(!button||button.disabled)return;state.view=button.dataset.view;renderGraph();});
 for(const id of ['graph','graph-compare']){
   $(id).addEventListener('click',openGraphNode);
   $(id).addEventListener('keydown',openGraphNode);
 }
 $('back-decision').addEventListener('click',()=>{renderDecision();$('detail-heading').focus({preventScroll:true});});
 $('zoom-in').addEventListener('click',()=>adjustZoom(.25));
 $('zoom-out').addEventListener('click',()=>adjustZoom(-.25));
 $('zoom-reset').addEventListener('click',()=>{
   state.zoom=1;
   zoom();
   for(const pane of [$('graph'),$('graph-compare')])pane.scrollTo(0,0);
 });
 for(const [id,delta] of [['previous',-1],['next',1]])$(id).addEventListener('click',()=>{const rows=state.run?.timeline||[],index=rows.findIndex(x=>x.sequence===state.step);if(rows[index+delta])selectStep(rows[index+delta].sequence);});
 $('latest').addEventListener('click',()=>{stopPlayer();state.follow=true;const row=state.run?.timeline.at(-1);if(row)selectStep(row.sequence,false);});
 $('play').addEventListener('click',()=>{if(state.player){stopPlayer();return;}const rows=state.run?.timeline||[];if(!rows.length)return;state.follow=false;let index=rows.findIndex(x=>x.sequence===state.step);if(index<0||index===rows.length-1){index=0;selectStep(rows[0].sequence,false);}$('play').textContent='暂停回放';state.player=setInterval(()=>{index++;if(index>=rows.length){stopPlayer();return;}selectStep(rows[index].sequence,false);},2200);});
 $('guide-open').addEventListener('click',()=>$('guide-dialog').showModal());$('model-open').addEventListener('click',()=>{$('model-error').hidden=true;$('model-dialog').showModal();});
 document.querySelectorAll('[data-close]').forEach(button=>button.addEventListener('click',()=>{$(button.dataset.close).close();if(button.dataset.close==='model-dialog')$('api-key').value='';}));
 $('model-dialog').addEventListener('close',()=>$('api-key').value='');
 window.addEventListener('resize',keepStepVisible);
 $('model-form').addEventListener('submit',async event=>{event.preventDefault();try{const result=await api('/api/model',{method:'POST',body:JSON.stringify({base_url:$('base-url').value,model:$('model-name').value,api_key:$('api-key').value})});$('api-key').value='';state.overview.model=result;renderModel();$('model-dialog').close();toast('模型连接已保存。');}catch(cause){$('api-key').value='';$('model-error').textContent=cause.message;$('model-error').hidden=false;}});
 $('import-button').addEventListener('click',()=>$('import-file').click());$('import-file').addEventListener('change',async()=>{const file=$('import-file').files[0];if(!file)return;try{if(file.size>20*1024*1024)throw new Error('数据集不能超过 20 MB。');const input=JSON.parse(await file.text());const info=await api('/api/datasets',{method:'POST',body:JSON.stringify(input)});await refreshOverview();await chooseDataset(info.id);toast('数据集已导入并通过检查。');}catch(cause){error(cause.message);}finally{$('import-file').value='';}});
 try{await refreshOverview();const source=await api('/api/datasets/synthetic');state.dataset=source.dataset;renderSummary();const initial=state.overview.runs.find(row=>row.id===state.overview.active_run)||state.overview.runs.find(row=>row.status==='completed'&&row.mode==='mock');if(initial)await chooseRun(initial.id);else await renderGraph();}
 catch(cause){error('暂时无法连接本地服务。请双击项目中的「启动分类体系工作台.command」，并保持终端打开。');$('server-label').textContent='服务尚未连接';$('server-dot').classList.remove('green');}
 setInterval(poll,1000);
}
if(typeof document!=='undefined')init();
