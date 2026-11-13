"""Dependency-free SVG DAGs and an offline audit viewer (no external scripts)."""
from collections import defaultdict
from html import escape
import json
import shutil
import subprocess
import textwrap
from .storage import read_json


def graphviz_svg(taxonomy, title):
    """Optional renderer. Quoted DOT labels never execute code or use HTML."""
    binary = shutil.which('dot')
    if not binary:
        return None
    quote = lambda value: json.dumps(str(value), ensure_ascii=False)
    lines = ['digraph taxonomy {',
        'graph [rankdir=TB, bgcolor="#f4f7fb", pad="0.35", nodesep="0.24", ranksep="0.65", labelloc=t, fontname="Helvetica", fontsize=18];',
        'node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=12, margin="0.16,0.12", color="#a4b6c7"];',
        'edge [color="#718ba1", arrowsize=0.65];',
        'label='+quote(title+'\nBlue: root | Green: retained input | Orange: induced intermediate')+';',
        '{ rank=same; '+'; '.join(quote(r) for r in taxonomy['predefined_roots'])+'; }']
    for node in taxonomy['nodes']:
        kind=node['origin_kind']
        color='#dbeafe' if kind=='predefined_root' else '#ffedd5' if kind=='induced_intermediate' else '#dcfce7'
        label='\n'.join(textwrap.wrap(node['concept_label'],28))+'\n\n%d supporting event(s)'%len(node['supporting_event_ids'])
        tooltip=node['concept_id']+' | '+(node['concept_description'] or '')
        lines.append('%s [label=%s, fillcolor=%s, tooltip=%s];'%(quote(node['concept_id']),quote(label),quote(color),quote(tooltip)))
    for edge in taxonomy['parent_child_edges']:
        lines.append('%s -> %s [style=%s];'%(quote(edge['parent_id']),quote(edge['child_id']),
                                            'dashed' if edge['status']=='provisional' else 'solid'))
    lines.append('}')
    try:
        result=subprocess.run([binary,'-Tsvg'],input='\n'.join(lines),capture_output=True,text=True,check=True,timeout=10)
        return result.stdout[result.stdout.index('<svg'):]
    except (OSError,subprocess.SubprocessError,ValueError):
        return None


def svg(taxonomy, title):
    rendered = graphviz_svg(taxonomy,title)
    if rendered is not None:
        return rendered
    return fallback_svg(taxonomy,title)


def fallback_svg(taxonomy, title):
    nodes = {n['concept_id']: n for n in taxonomy['nodes']}
    edges = taxonomy['parent_child_edges']
    indegree = dict.fromkeys(nodes, 0); children = defaultdict(list); depth = dict.fromkeys(nodes, 0)
    for edge in edges:
        children[edge['parent_id']].append(edge['child_id']); indegree[edge['child_id']] += 1
    pending = sorted(n for n in nodes if not indegree[n])
    for node in pending:
        for child in sorted(children[node]):
            depth[child] = max(depth[child], depth[node] + 1)
            indegree[child] -= 1
            if indegree[child] == 0:
                pending.append(child)
    layers = defaultdict(list)
    for nid in sorted(nodes):
        layers[depth[nid]].append(nid)
    width = max(900, max((len(row) for row in layers.values()), default=1) * 222 + 40)
    height = max(330, len(layers) * 145 + 100)
    positions = {}
    for level, row in layers.items():
        for i, nid in enumerate(row):
            positions[nid] = ((width - len(row) * 222) / 2 + i * 222 + 11, 80 + level * 145)
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d %d" role="img" aria-label="%s">' % (width, height, escape(title, quote=True)),
        '<defs><marker id="arrow" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto"><path d="M0,0 L0,6 L8,3 z" fill="#8398ad"/></marker></defs>',
        '<rect width="100%" height="100%" fill="#f4f7fb"/>',
        '<text x="24" y="32" font-size="20" font-family="Arial" fill="#162e46">%s</text>' % escape(title),
        '<text x="24" y="54" font-size="12" font-family="Arial" fill="#536d86">Blue: predefined root | Green: retained input | Orange: induced intermediate | Arrows: parent SUBSUMES child</text>']
    for edge in edges:
        x1, y1 = positions[edge['parent_id']]; x2, y2 = positions[edge['child_id']]
        parts.append('<path d="M%.1f,%.1f C%.1f,%.1f %.1f,%.1f %.1f,%.1f" fill="none" stroke="#8398ad" stroke-width="1.6" marker-end="url(#arrow)"%s/>' %
                     (x1+100,y1+84,x1+100,y1+110,x2+100,y2-25,x2+100,y2-3,
                      ' stroke-dasharray="5 3"' if edge['status'] == 'provisional' else ''))
    for nid, node in nodes.items():
        x, y = positions[nid]
        kind = node['origin_kind']
        fill = '#dbeafe' if kind == 'predefined_root' else '#ffedd5' if kind == 'induced_intermediate' else '#dcfce7'
        parts.append('<g><title>%s</title><rect x="%.1f" y="%.1f" width="200" height="84" rx="10" fill="%s" stroke="#a4b6c7"/>' %
            (escape(nid + '\n' + (node['concept_description'] or '')), x, y, fill))
        lines = textwrap.wrap(node['concept_label'], width=25)[:3]
        for k, line in enumerate(lines):
            parts.append('<text x="%.1f" y="%.1f" font-family="Arial" font-size="13" fill="#142d44">%s</text>' % (x+10,y+22+k*17,escape(line)))
        parts.append('<text x="%.1f" y="%.1f" font-family="Arial" font-size="10" fill="#52697b">%d supporting event(s)</text></g>' % (x+10,y+74,len(node['supporting_event_ids'])))
    parts.append('</svg>')
    return ''.join(parts)


def render_run(directory, state, report):
    initial = read_json(directory / 'initial_taxonomy.json')
    final = state['taxonomy']
    label = 'SYNTHETIC DATA / MOCK LLM' if report['data_origin'] == 'synthetic' else 'EXPERIMENTAL / NOT HUMAN-VALIDATED'
    (directory / 'initial.svg').write_text(svg(initial, label + ' — Initial'), encoding='utf-8')
    (directory / 'final.svg').write_text(svg(final, label + ' — ' + state['status']), encoding='utf-8')
    records = []
    for i in range(1, state['sequence'] + 1):
        folder = directory / 'steps' / ('%06d' % i)
        change = read_json(folder / 'change.json')
        if change['action'] in ('candidate_constructed', 'reject_candidate', 'accept_candidate', 'stop_level'):
            records.append({'label': '%03d · %s · %s' % (i,change['action'],change.get('parent_id','')),
                'change': change, 'svg': svg(read_json(folder / 'after.json'), label + ' — Step %d (committed)' % i),
                'candidate': svg(read_json(folder / 'candidate.json'), label + ' — Step %d (candidate only)' % i) if 'candidate_snapshot_ref' in change else None})
    payload = json.dumps({'initial': svg(initial,label+' — Initial'), 'final':svg(final,label+' — '+state['status']),
                          'records':records,'report':report},ensure_ascii=False).replace('<','\\u003c')
    html = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Event Taxonomy Induction Audit</title><style>body{font:15px system-ui;color:#183249;background:#edf3f8;margin:0}header,main{padding:24px;max-width:1600px;margin:auto}header{padding-bottom:8px}h1{margin:5px 0 10px;font-size:28px}.badge{display:inline-block;background:#fff0d1;color:#754710;padding:7px 12px;border-radius:6px}button,select{font:inherit;padding:9px;margin:4px;border:1px solid #aabccc;border-radius:7px;background:white}button{cursor:pointer}#canvas{background:white;border-radius:12px;overflow:auto;border:1px solid #c1d0dd}svg{min-width:900px;width:100%;display:block}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#102b40;color:#dceaf4;padding:18px;border-radius:10px;font-size:12px}p{line-height:1.6}summary{cursor:pointer;font-weight:600}#meta{color:#536d86}</style>
<header><div class="badge" id="badge"></div><h1>Event Taxonomy Induction</h1><p>Preserved fine concepts and event evidence · level-first refinement · shared-node DAG audit</p><p id="meta"></p></header>
<main><nav><button id="initial">Initial</button><button id="final">Final / current</button><select id="round"><option value="">Select a recorded step</option></select><button id="candidate" disabled>View uncommitted candidate</button></nav><div id="canvas"></div><details open><summary>Decision / preservation record</summary><pre id="detail"></pre></details><p>Hover a node to read its ID and definition. Rejected candidates are never committed. Mock decisions demonstrate control flow, not biomedical accuracy.</p></main>
<script type="application/json" id="data">PAYLOAD</script><script>
const d=JSON.parse(document.getElementById('data').textContent),canvas=document.getElementById('canvas'),detail=document.getElementById('detail'),select=document.getElementById('round'),candidate=document.getElementById('candidate');
document.getElementById('badge').textContent=d.report.data_origin==='synthetic'?'SYNTHETIC DATA + MOCK LLM':'EXPERIMENTAL — REVIEW REQUIRED';
document.getElementById('meta').textContent=`${d.report.initial_nodes} → ${d.report.final_nodes} nodes · ${d.report.accepted_candidates} accepted · ${d.report.rejected_candidates} rejected · ${d.report.original_concepts_preserved} original concepts preserved`;
function show(which){canvas.innerHTML=d[which];detail.textContent=JSON.stringify(d.report,null,2);candidate.disabled=true;select.value='';}
d.records.forEach((r,i)=>{let o=document.createElement('option');o.value=i;o.textContent=r.label;select.appendChild(o)});
document.getElementById('initial').onclick=()=>show('initial');document.getElementById('final').onclick=()=>show('final');
select.onchange=()=>{if(select.value===''){show('final');return}let r=d.records[+select.value];canvas.innerHTML=r.svg;detail.textContent=JSON.stringify(r.change,null,2);candidate.disabled=!r.candidate};
candidate.onclick=()=>{let r=d.records[+select.value];canvas.innerHTML=r.candidate;detail.textContent='UNCOMMITTED CANDIDATE\\n'+JSON.stringify(r.change,null,2)};show('final');
</script></html>'''.replace('PAYLOAD',payload)
    (directory / 'visualization.html').write_text(html, encoding='utf-8')
