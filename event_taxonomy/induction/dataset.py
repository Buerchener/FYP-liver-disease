"""Read Task 1 snapshots or an explicit dataset JSON; reject unresolved input contracts."""
import hashlib
import json
from pathlib import Path
from .graph import load_taxonomy, shortest_depths
from .storage import read_json


def load_dataset(path):
    path = Path(path)
    if path.is_file():
        return read_json(path)
    result = {'taxonomy': read_json(path / 'initial_taxonomy.json')}
    for name in ('events', 'evidence', 'documents'):
        with (path / (name + '.jsonl')).open(encoding='utf-8') as stream:
            result[name] = [json.loads(line) for line in stream if line.strip()]
    origins = {e['data_origin'] for e in result['events']}
    result['data_origin'] = next(iter(origins)) if len(origins) == 1 else 'mixed'
    return result


def _index(rows, key):
    indexed = {r[key]: r for r in rows}
    if len(indexed) != len(rows):
        raise ValueError('Duplicate dataset ID: ' + key)
    return indexed


def validate_dataset(dataset, require_semantic_freeze=True):
    taxonomy = load_taxonomy(dataset['taxonomy'])
    problems = []
    if taxonomy.unplaced_concept_ids:
        problems.append('%d input concepts lack a root path' % len(taxonomy.unplaced_concept_ids))
    if any(not isinstance(n.concept_description, str) or not n.concept_description.strip() for n in taxonomy.nodes):
        problems.append('one or more concept definitions are missing')
    if any(not isinstance(n.concept_label, str) or not n.concept_label.strip() for n in taxonomy.nodes):
        problems.append('one or more concept labels are missing')
    if require_semantic_freeze and taxonomy.input_freeze_status != 'semantic_input_frozen':
        problems.append('input semantic review is not frozen')
    if problems:
        raise ValueError('Induction input is not ready: ' + '; '.join(problems))
    if not taxonomy.predefined_roots or not taxonomy.frozen_input_concept_ids:
        raise ValueError('A nonempty root set and frozen input concept set are required')
    nodes = {n.concept_id: n for n in taxonomy.nodes}
    if set(shortest_depths(taxonomy)) != set(nodes):
        raise ValueError('Dataset contains disconnected taxonomy nodes')
    if any(n.concept_role != 'event_type' for n in taxonomy.nodes):
        raise ValueError('Topic concepts cannot silently become event types')
    events = _index(dataset['events'], 'event_id')
    evidence = _index(dataset['evidence'], 'evidence_id')
    docs = _index(dataset['documents'], 'document_id')
    synthetic = dataset['data_origin'] == 'synthetic'
    if not events or not evidence or not docs:
        raise ValueError('Dataset must carry event, evidence and document records')
    for doc in docs.values():
        if hashlib.sha256(doc['text'].encode()).hexdigest() != doc['text_sha256']:
            raise ValueError('Document text checksum mismatch')
        if synthetic and (doc.get('source_pmid') is not None or doc.get('data_origin') != 'synthetic'):
            raise ValueError('Synthetic documents must be labeled and cannot have a PMID')
    for ev in evidence.values():
        if ev['document_id'] not in docs:
            raise ValueError('Evidence references an unknown document')
        document = docs[ev['document_id']]['text']
        for span in ev['spans']:
            start, end = span['start'], span['end']
            if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(document):
                raise ValueError('Invalid evidence offset')
            if ' '.join(document[start:end].split()) != ' '.join((ev['text'] or '').split()):
                raise ValueError('Evidence does not match source span')
        if ev['alignment_status'] in ('verified_exact', 'verified_whitespace') and len(ev['spans']) != 1:
            raise ValueError('Verified evidence requires one unambiguous span')
        if ev['alignment_status'] == 'verified_exact':
            span = ev['spans'][0]
            if document[span['start']:span['end']] != ev['text']:
                raise ValueError('False exact-evidence flag')
    for event in events.values():
        if event['document_id'] not in docs or event['source_pmid'] != docs[event['document_id']]['source_pmid']:
            raise ValueError('Event/document PMID or identity mismatch')
        if not event['evidence_ids'] or any(e not in evidence or evidence[e]['document_id'] != event['document_id'] for e in event['evidence_ids']):
            raise ValueError('Event evidence cannot be resolved')
        if event.get('evidence_sentence') is not None and event['evidence_sentence'] not in docs[event['document_id']]['text']:
            raise ValueError('Event evidence sentence is not verbatim source text')
        if synthetic and (event['source_pmid'] is not None or event['data_origin'] != 'synthetic'):
            raise ValueError('Synthetic events must be labeled and cannot have a PMID')
        if not synthetic and event['data_origin'] == 'synthetic':
            raise ValueError('Synthetic and real data must be run separately')
    mappings = _index(taxonomy.concept_evidence_mapping, 'concept_id')
    if not set(mappings) <= set(nodes):
        raise ValueError('Evidence mapping references unknown concept')
    for node in taxonomy.nodes:
        if not set(node.supporting_event_ids) <= set(events) or not set(node.evidence) <= set(evidence):
            raise ValueError('Concept provenance cannot be resolved')
        if node.origin_kind != 'predefined_root' and not node.supporting_event_ids:
            raise ValueError('Non-root concept has no supporting events')
        supported_evidence = {eid for event_id in node.supporting_event_ids for eid in events[event_id]['evidence_ids']}
        if set(node.evidence) != supported_evidence:
            raise ValueError('Concept evidence differs from its supporting events')
        if node.concept_id in mappings:
            row = mappings[node.concept_id]
            if set(row['event_ids']) != set(node.supporting_event_ids) or set(row['evidence_ids']) != set(node.evidence):
                raise ValueError('Concept evidence mapping differs from node provenance')
        elif node.origin_kind != 'predefined_root':
            raise ValueError('Missing concept evidence mapping')
    seen = set()
    for link in taxonomy.instance_concept_links:
        pair = (link['event_id'], link['concept_id'])
        if pair in seen or pair[0] not in events or pair[1] not in nodes or link['relation'] != 'IS_INSTANCE_OF':
            raise ValueError('Invalid instance-concept link')
        seen.add(pair)
        if not set(link['evidence_ids']) <= set(events[pair[0]]['evidence_ids']):
            raise ValueError('Instance link has unrelated evidence')
    required = {(e, n.concept_id) for n in taxonomy.nodes if n.concept_id in taxonomy.frozen_input_concept_ids for e in n.supporting_event_ids}
    if not required <= seen:
        raise ValueError('Missing original event-concept associations')
    return taxonomy
