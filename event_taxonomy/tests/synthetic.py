"""Small SYNTHETIC fixture factory. No real literature, PMID or measured result."""
import hashlib
from ..models import (Document,Evidence,EventInstance,SourceRef,UniqueConcept,
                      Taxonomy,ParentChildEdge,stable_id,to_dict)


def dataset():
    roots = [('SYN-BIO','Biological Event','A biological process in a living system.'),
             ('SYN-CARE','Clinical Care Event','An action undertaken to diagnose or treat an individual.'),
             ('SYN-RESEARCH','Biomedical Research Event','An action explicitly undertaken to produce biomedical knowledge.')]
    leaves = [('ACT-A','Activation of gene A','Increased activity of gene A in a cell.'),
              ('ACT-B','Activation of gene B','Increased activity of gene B in a cell.'),
              ('INH-A','Inhibition of gene A','Decreased activity of gene A in a cell.'),
              ('INH-B','Inhibition of gene B','Decreased activity of gene B in a cell.'),
              ('PRO-A','Proliferation of cell population A','An increase in the number of cells in population A.'),
              ('PRO-B','Proliferation of cell population B','An increase in the number of cells in population B.'),
              ('IMG-A','Trial diagnostic imaging A','Diagnostic imaging of an individual using fictional protocol A as part of a research trial.'),
              ('IMG-B','Trial diagnostic imaging B','Diagnostic imaging of an individual using fictional protocol B as part of a research trial.')]
    nodes,documents,events,evidence,edges,mappings,links = [],[],[],[],[],[],[]
    for nid,label,description in roots:
        nodes.append(UniqueConcept(nid,label,description,[],[],nid,'synthetic/1',[],[],'event_type',
            origin_kind='predefined_root',validation_status='synthetic_fixture',description_provenance='synthetic_fixture'))
    for index,(suffix,label,description) in enumerate(leaves):
        nid,eid,did,vid = ['SYN-'+prefix+suffix for prefix in ('','EVENT-','DOC-','EVIDENCE-')]
        sentence = 'Synthetic example only: ' + description
        sha = hashlib.sha256(sentence.encode()).hexdigest()
        source = SourceRef('synthetic_fixture','synthetic-v1',sha,'/documents/'+str(index))
        documents.append(Document(did,None,'Synthetic Data — '+label,sentence,sha,'synthetic',[source]))
        evidence.append(Evidence(vid,did,sentence,[{'start':0,'end':len(sentence)}],'sentence','verified_exact',source))
        events.append(EventInstance(eid,description,None,did,sentence,[vid],eid,'synthetic-v1','synthetic_fixture',source,index,
                                   data_origin='synthetic',review_status='synthetic_fixture'))
        nodes.append(UniqueConcept(nid,label,description,[],[],nid,'synthetic/1',[eid],[vid],'event_type',
            validation_status='synthetic_fixture',description_provenance='synthetic_fixture'))
        parents = ['SYN-CARE','SYN-RESEARCH'] if suffix.startswith('IMG') else ['SYN-BIO']
        for parent in parents:
            edges.append(ParentChildEdge(stable_id('edge',parent,nid),parent,nid,'accepted','synthetic-fixture',
                                         'Synthetic fixture relationship, not a real result.','synthetic_fixture'))
        mappings.append({'concept_id':nid,'event_ids':[eid],'evidence_ids':[vid],'mapping_kind':'direct','via_child_ids':[]})
        links.append({'event_id':eid,'concept_id':nid,'relation':'IS_INSTANCE_OF','status':'synthetic_fixture',
                      'evidence_ids':[vid],'candidate_ids':[],'decision_ref':'synthetic-fixture'})
    taxonomy = Taxonomy('SYN-TAXONOMY',nodes,edges,[r[0] for r in roots],mappings,links,[],
                        sorted(n.concept_id for n in nodes if n.origin_kind=='input_concept'),'partial',
                        input_freeze_status='semantic_input_frozen')
    return {'data_origin':'synthetic','notice':'SYNTHETIC DATA. No real PMID, literature claim or experimental result.',
            'taxonomy':to_dict(taxonomy),'documents':[to_dict(x) for x in documents],
            'events':[to_dict(x) for x in events],'evidence':[to_dict(x) for x in evidence]}


def diagnosis(groups=None):
    return {'need_induction':bool(groups),'candidate_subgroups':groups or [],'structural_issues': ['Missing scoped process abstractions.'] if groups else [],
            'reasoning':'Synthetic fixture decision based on the supplied definitions.','confidence':0.95}


def group(name,labels):
    return {'group_id':name,'child_ids':{'$children':labels},'reasoning':'Synthetic concepts share the stated process subtype.'}


def proposal(items):
    return {'intermediates':[{'proposal_id':gid,'label':label,'description':description,'child_ids':{'$group':gid}} for gid,label,description in items],
            'reasoning':'Synthetic intermediate definitions summarize only the listed children.','confidence':0.95}


def validation(decision):
    return {'decision':decision,'edge_checks':{'$all_edge_checks':decision!='REJECT'},
            'improves_structure':decision!='REJECT','duplicates_existing_concepts':False,'remaining_subgroups':[],
            'reasoning':'Synthetic rejection: first label is too broad.' if decision=='REJECT' else 'Synthetic fixture accepts the scoped hierarchy; not independent biomedical validation.',
            'confidence':0.95}


def plan():
    rules = []
    def add(stage,parent,round_number,response,attempt=None):
        when={'parent_id':parent,'round':round_number}
        if attempt is not None:
            when['attempt']=attempt
        rules.append({'stage':stage,'when':when,'response':response})
    add('diagnosis','SYN-BIO',1,diagnosis([group('activation',['Activation of gene A','Activation of gene B']),
                                           group('inhibition',['Inhibition of gene A','Inhibition of gene B'])]))
    add('induction','SYN-BIO',1,proposal([('activation','Molecular process','A process involving a biological molecule.'),
                                          ('inhibition','Gene inhibition','A process that decreases gene activity.')]),1)
    add('validation','SYN-BIO',1,validation('REJECT'),1)
    add('induction','SYN-BIO',1,proposal([('activation','Gene activation','A process that increases gene activity.'),
                                          ('inhibition','Gene inhibition','A process that decreases gene activity.')]),2)
    add('validation','SYN-BIO',1,validation('ACCEPT_AND_CONTINUE'),2)
    add('diagnosis','SYN-BIO',2,diagnosis([group('regulation',['Gene activation','Gene inhibition']),
                                           group('proliferation',['Proliferation of cell population A','Proliferation of cell population B'])]))
    add('induction','SYN-BIO',2,proposal([('regulation','Gene regulation','A biological process that increases or decreases gene activity.'),
                                          ('proliferation','Cell proliferation','A biological process that increases the number of cells in a population.')]))
    add('validation','SYN-BIO',2,validation('ACCEPT_AND_STOP'))
    add('diagnosis','SYN-CARE',1,diagnosis([group('imaging',['Trial diagnostic imaging A','Trial diagnostic imaging B'])]))
    add('induction','SYN-CARE',1,proposal([('imaging','Trial diagnostic imaging','Diagnostic imaging of an individual as a prescribed component of a research trial.')]))
    add('validation','SYN-CARE',1,validation('ACCEPT_AND_STOP'))
    return {'data_origin':'synthetic','rules':rules,'default_diagnosis':diagnosis()}
