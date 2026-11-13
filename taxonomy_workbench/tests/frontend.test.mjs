import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
const source=fs.readFileSync(new URL('../static/app.js',import.meta.url),'utf8');
const {escapeHTML,statusLabel,isActive,counts,decisionMarkup,stageOf,pendingMarkup}=await import('data:text/javascript;base64,'+Buffer.from(source).toString('base64'));

test('untrusted model labels and reasoning are escaped',()=>{
 const html=decisionMarkup({tone:'rejected',stage:'C',decision:'REJECT',title:'<img onerror=alert(1)>',summary:'<script>bad</script>',reasoning:'<svg onload=x>',added_labels:['<b>unsafe</b>'],parent_label:'A & B'});
 assert.ok(!html.includes('<script>'));assert.ok(!html.includes('<img'));assert.ok(html.includes('&lt;svg'));
 assert.equal(escapeHTML('A&B'),'A&amp;B');
});
test('retention is calculated from actual IDs rather than hard-coded 100 percent',()=>{
 const initial={frozen_input_concept_ids:['a','b'],nodes:[{concept_id:'r'},{concept_id:'a'},{concept_id:'b'}]};
 assert.deepEqual(counts(initial,{nodes:[{concept_id:'r'},{concept_id:'a'},{concept_id:'b'},{concept_id:'m'}]},2),{original:2,added:1,retained:2,events:2});
 assert.equal(counts(initial,{nodes:[{concept_id:'r'},{concept_id:'a'}]},2).retained,1);
});
test('pause, interruption and completion remain distinct',()=>{
 assert.equal(isActive('pausing'),true);assert.equal(isActive('paused'),false);assert.equal(isActive('completed'),false);
 assert.notEqual(statusLabel('completed'),statusLabel('paused'));assert.match(statusLabel('interrupted'),/恢复/);
});
test('active model stage takes precedence over the last saved step when following live',()=>{
 const run={status:'running',state:{phase:'validate',waiting_for:'validation'}};
 assert.equal(stageOf(run,{stage:'B'},true),'C');assert.equal(stageOf(run,{stage:'B'},false),'B');
 assert.equal(stageOf({status:'completed'},{stage:'A'},true),'A');
});
test('unplaced concepts remain explicitly unassigned and readable',()=>{
 const html=pendingMarkup({predefined_roots:['r'],unplaced_concept_ids:['a'],nodes:[{concept_id:'r',concept_label:'Root',concept_description:'Root definition'},{concept_id:'a',concept_label:'<unsafe>',concept_description:null}]});
 assert.match(html,/预定义类别/);assert.match(html,/1 个待归属概念/);assert.ok(html.includes('&lt;unsafe&gt;'));
 assert.equal([...html.matchAll(/data-concept=/g)].length,2);assert.ok(!html.includes('<svg'));
});
test('all application DOM IDs exist in the page',()=>{
 const html=fs.readFileSync(new URL('../static/index.html',import.meta.url),'utf8');
 const ids=new Set([...html.matchAll(/id="([^"]+)"/g)].map(x=>x[1]));
 for(const [,id] of source.matchAll(/\$\('([^']+)'\)/g)){if(id==='raw-open')continue;assert.ok(ids.has(id),'Missing ID: '+id);}
 assert.equal(ids.size,[...html.matchAll(/id="([^"]+)"/g)].length,'Duplicate DOM IDs');
});
