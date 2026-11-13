import io
import json
import os
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from event_taxonomy.models import ParentChildEdge, stable_id, to_dict
from event_taxonomy.taxonomy.dag import validate_dag
from event_taxonomy.induction.clustering import EmbeddingClusterer, LLMGuidedClusterer
from event_taxonomy.induction.config import read_config, validate_config
from event_taxonomy.induction.context import ContextBuilder
from event_taxonomy.induction.contracts import CandidateError, ContractError, ProviderError
from event_taxonomy.induction import contracts
from event_taxonomy.induction.dataset import load_dataset, validate_dataset
from event_taxonomy.induction.graph import (assert_preserved,build_candidate,children,load_taxonomy,
                                           local_signature,structure_signature)
from event_taxonomy.induction.journal import LLMJournal,strict_json
from event_taxonomy.induction.providers import MockProvider,OpenAICompatibleProvider
from event_taxonomy.induction.recursive import RecursiveRefiner,initial_state,frame
from event_taxonomy.induction.run import create_run,resume_run,verify_run
from event_taxonomy.induction.storage import (atomic_json,read_checkpoint,read_json,save_checkpoint,run_lock)
from event_taxonomy.tests.synthetic import dataset,plan,diagnosis,validation

PACKAGE = Path(__file__).resolve().parents[1]


def proposed(label='Gene activation', members=None):
    return {'intermediates':[{'proposal_id':'group','label':label,
        'description':'A biological process that increases activity of a gene.',
        'child_ids':members or ['SYN-ACT-A','SYN-ACT-B']}], 'reasoning':'Synthetic fixture proposal.','confidence':0.9}


class InductionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = read_config(PACKAGE/'configs/task2.mock.json')
        self.data = dataset(); self.taxonomy = load_taxonomy(self.data['taxonomy'])
        self.prompts = {s:Path(p).read_text() for s,p in self.config['prompts'].items()}

    def engine(self,config=None,fixture=None,mock_plan=None,subdir='engine'):
        data = fixture or self.data; config = config or self.config
        directory = self.root/subdir; directory.mkdir()
        provider = MockProvider(mock_plan or plan())
        journal = LLMJournal(directory/'llm',provider,self.prompts,config['stages'],config['limits']['llm_calls'])
        state = initial_state(load_taxonomy(data['taxonomy']),'test-fingerprint')
        save_checkpoint(directory/'checkpoint.json',state)
        engine = RecursiveRefiner(directory,state,journal,ContextBuilder(data,config['context']),LLMGuidedClusterer(),config)
        return engine,provider

    def changes(self,engine):
        return [read_json(engine.directory/'steps'/('%06d'%i)/'change.json') for i in range(1,engine.state['sequence']+1)]

    def test_level_first_continue_before_any_child(self):
        engine,provider = self.engine(); engine.run()
        log = self.changes(engine)
        accepts = [i for i,c in enumerate(log) if c['action']=='accept_candidate' and c['parent_id']=='SYN-BIO']
        first_descent = next(i for i,c in enumerate(log) if c['action']=='descend_to_child' and c['parent_id']=='SYN-BIO')
        self.assertEqual([log[i]['decision'] for i in accepts],['ACCEPT_AND_CONTINUE','ACCEPT_AND_STOP'])
        self.assertLess(accepts[1],first_descent)
        self.assertTrue(any(c['action']=='diagnose_and_group' and c['parent_id']=='SYN-BIO' for c in log[accepts[0]+1:accepts[1]]))
        self.assertEqual(engine.state['accepted_candidates'],3)
        self.assertEqual(engine.state['rejected_candidates'],1)

    def test_reject_is_transactional_and_alternative_is_attempted(self):
        engine,_ = self.engine(); engine.run()
        rejected = next(c for c in self.changes(engine) if c['action']=='reject_candidate')
        step = engine.directory/'steps'/('%06d'%rejected['sequence'])
        self.assertEqual(read_json(step/'before.json'),read_json(step/'after.json'))
        self.assertFalse(rejected['means_optimal'])
        self.assertTrue((step/'candidate.json').exists())
        self.assertGreater(engine.state['accepted_candidates'],0)

    def test_retry_exhaustion_still_traverses_children_and_other_roots(self):
        config = deepcopy(self.config); config['limits']['candidate_attempts']=1
        engine,provider = self.engine(config); engine.run()
        log=self.changes(engine)
        rejected=next(c for c in log if c['action']=='reject_candidate')
        self.assertEqual(rejected['level_stop']['reason'],'candidate_attempts_exhausted')
        self.assertTrue(any(c['action']=='enter_parent' and c['parent_id']=='SYN-ACT-A' for c in log))
        self.assertTrue(any(c['stage']=='induction' and c['cursor']['parent_id']=='SYN-CARE' for c in provider.calls))

    def test_no_induction_and_zero_child_leaf_stop(self):
        p={'rules':[],'default_diagnosis':diagnosis()}
        engine,provider=self.engine(mock_plan=p); engine.run()
        self.assertEqual(len(provider.calls),3)
        self.assertTrue(all(c['stage']=='diagnosis' for c in provider.calls))
        self.assertEqual(engine.state['taxonomy'],self.data['taxonomy'])
        self.assertTrue(any(c.get('reason')=='fewer_than_two_children' for c in self.changes(engine)))

    def test_iteration_cap_stops_parent_then_descends(self):
        config=deepcopy(self.config); config['limits']['iterations_per_parent']=1
        engine,_=self.engine(config); engine.run()
        log=self.changes(engine)
        stop=next(c for c in log if c.get('reason')=='maximum_parent_iterations')
        self.assertEqual(stop['parent_id'],'SYN-BIO')
        self.assertTrue(any(c['action']=='descend_to_child' and c['parent_id']=='SYN-BIO' for c in log[stop['sequence']:]))

    def test_depth_cap_does_not_end_other_roots(self):
        config=deepcopy(self.config); config['limits']['max_depth']=0
        engine,provider=self.engine(config); engine.run()
        self.assertEqual({c['cursor']['parent_id'] for c in provider.calls},{'SYN-BIO','SYN-CARE','SYN-RESEARCH'})
        self.assertTrue(any(c['action']=='skip_depth_limit' for c in self.changes(engine)))

    def test_call_budget_still_visits_unprocessed_nodes(self):
        config=deepcopy(self.config); config['limits']['llm_calls']=1
        engine,provider=self.engine(config); engine.run()
        self.assertEqual(len(provider.calls),1)
        self.assertEqual(engine.state['taxonomy'],self.data['taxonomy'])
        self.assertTrue(any(c.get('parent_id')=='SYN-RESEARCH' and c['action']=='stop_level' for c in self.changes(engine)))

    def test_context_limit_is_local_stop(self):
        config=deepcopy(self.config); config['context']['max_characters']=100
        engine,provider=self.engine(config); engine.run()
        self.assertEqual(provider.calls,[])
        self.assertTrue(any(c.get('reason','').startswith('ContextLimit') for c in self.changes(engine)))
        self.assertIn('SYN-IMG-B',engine.state['completed'])

    def test_new_node_budget_rejects_without_changing_input(self):
        config=deepcopy(self.config); config['limits']['new_nodes']=0
        engine,_=self.engine(config); engine.run()
        self.assertEqual(engine.state['taxonomy'],self.data['taxonomy'])
        self.assertTrue(any(c.get('reason')=='maximum_new_nodes' for c in self.changes(engine)))

    def test_repeated_local_state_stops_and_descends(self):
        engine,_=self.engine(); f=frame('SYN-BIO',0); f['phase']='diagnose'
        f['diagnosed_signatures']=[local_signature(self.taxonomy,'SYN-BIO')]
        engine.state['stack']=[f]; engine.state['pending_roots']=[]
        change=engine.step()
        self.assertEqual(change['reason'],'repeated_local_structure')
        self.assertEqual(engine.state['stack'][-1]['phase'],'descend')

    def test_same_rejected_candidate_is_not_revalidated(self):
        p=plan()
        original=next(r['response'] for r in p['rules'] if r['stage']=='induction' and r['when']=={'parent_id':'SYN-BIO','round':1,'attempt':1})
        for r in p['rules']:
            if r['stage']=='induction' and r['when'].get('parent_id')=='SYN-BIO' and r['when'].get('round')==1:
                r['response']=original; r['when'].pop('attempt',None)
        engine,provider=self.engine(mock_plan=p);engine.run()
        self.assertEqual(len([c for c in provider.calls if c['stage']=='validation' and c['cursor']['parent_id']=='SYN-BIO']),1)
        self.assertTrue(any(c.get('reason')=='repeated_rejected_candidate' for c in self.changes(engine)))

    def test_low_validation_confidence_cannot_commit(self):
        p=plan()
        for r in p['rules']:
            if r['stage']=='validation': r['response']['confidence']=0.1
        # Retry the same candidate triggers the duplicate-state guard, without another C call.
        for r in p['rules']:
            if r['stage']=='induction' and r['when'].get('attempt')==2: r['when'].pop('attempt')
        engine,_=self.engine(mock_plan=p);engine.run()
        self.assertEqual(engine.state['accepted_candidates'],0)

    def test_shared_nodes_skip_unchanged_context(self):
        engine,_=self.engine(mock_plan={'rules':[],'default_diagnosis':diagnosis()});engine.run()
        skips=[c['parent_id'] for c in self.changes(engine) if c['action']=='skip_unchanged_shared_context']
        self.assertIn('SYN-IMG-A',skips);self.assertIn('SYN-IMG-B',skips)
        final=load_taxonomy(engine.state['taxonomy'])
        self.assertTrue({'SYN-IMG-A','SYN-IMG-B'} <= set(children(final,'SYN-RESEARCH')))

    def test_shared_ancestor_context_change_is_revisited(self):
        data=deepcopy(self.data)
        data['taxonomy']['predefined_roots']=['SYN-RESEARCH','SYN-CARE','SYN-BIO']
        engine,_=self.engine(fixture=data);engine.run()
        stale=[c for c in self.changes(engine) if c['action']=='schedule_stale_dag_context']
        self.assertTrue(any(c['parent_id']=='SYN-RESEARCH' for c in stale))
        self.assertLess(engine.state['sequence'],150)

    def test_shared_first_deep_then_shallow_gets_remaining_depth_budget(self):
        data=deepcopy(self.data)
        pairs=[('SYN-BIO','SYN-ACT-A'),('SYN-ACT-A','SYN-ACT-B'),('SYN-ACT-B','SYN-INH-A'),
               ('SYN-CARE','SYN-INH-A'),('SYN-INH-A','SYN-INH-B'),('SYN-INH-A','SYN-PRO-A'),
               ('SYN-BIO','SYN-PRO-B'),('SYN-RESEARCH','SYN-IMG-A'),('SYN-RESEARCH','SYN-IMG-B')]
        data['taxonomy']['parent_child_edges']=[to_dict(ParentChildEdge(stable_id('edge',p,c),p,c,'accepted','fixture','Synthetic.','fixture')) for p,c in pairs]
        config=deepcopy(self.config);config['limits']['max_depth']=3
        engine,provider=self.engine(config,fixture=data,mock_plan={'rules':[],'default_diagnosis':diagnosis()});engine.run()
        depths=[c['cursor']['depth'] for c in provider.calls if c['cursor']['parent_id']=='SYN-INH-A']
        self.assertEqual(depths,[3,1])
        self.assertIn('SYN-INH-B',engine.state['completed'])

    def test_context_visit_cap_still_descends(self):
        engine,provider=self.engine();engine.state['context_visits']['SYN-BIO']=self.config['limits']['context_visits_per_node']
        engine.run()
        self.assertFalse(any(c['cursor']['parent_id']=='SYN-BIO' for c in provider.calls))
        self.assertIn('SYN-ACT-A',engine.state['completed'])

    def test_all_original_concepts_events_and_evidence_retained(self):
        engine,_=self.engine();engine.run();after=load_taxonomy(engine.state['taxonomy'])
        assert_preserved(self.taxonomy,after,accepted=True)
        validate_dataset(dict(self.data,taxonomy=engine.state['taxonomy']))
        self.assertEqual(self.taxonomy.instance_concept_links,after.instance_concept_links)
        old={n.concept_id:to_dict(n) for n in self.taxonomy.nodes}
        self.assertTrue(all(to_dict(n)==old[n.concept_id] for n in after.nodes if n.concept_id in old))

    def test_external_parent_and_outlier_edges_are_retained(self):
        response=proposed('Trial imaging',['SYN-IMG-A','SYN-IMG-B'])
        candidate,changes=build_candidate(self.taxonomy,'SYN-CARE',response,100)
        self.assertEqual(children(candidate,'SYN-RESEARCH'),['SYN-IMG-A','SYN-IMG-B'])
        self.assertEqual(children(candidate,'SYN-BIO'),children(self.taxonomy,'SYN-BIO'))

    def test_overlapping_subgroups_preserve_shared_child_and_deduplicate_evidence(self):
        response=proposed(members=['SYN-ACT-A','SYN-ACT-B'])
        response['intermediates'].append({'proposal_id':'other','label':'Processes involving gene B',
            'description':'A synthetic grouping for gene B-related activity changes.', 'child_ids':['SYN-ACT-B','SYN-INH-B']})
        candidate,changes=build_candidate(self.taxonomy,'SYN-BIO',response,100)
        self.assertEqual(sum(e.child_id=='SYN-ACT-B' for e in candidate.parent_child_edges),2)
        intermediates=[n for n in candidate.nodes if n.origin_kind=='induced_intermediate']
        follow=proposed('Regulatory process',[n.concept_id for n in intermediates]);follow['intermediates'][0]['description']='A synthetic union of the stated regulation subgroups.'
        next_candidate,_=build_candidate(candidate,'SYN-BIO',follow,100)
        top=next_candidate.nodes[-1]
        self.assertEqual(len(top.supporting_event_ids),3)
        self.assertEqual(len(top.evidence),3)
        assert_preserved(self.taxonomy,next_candidate)

    def test_unknown_non_direct_duplicate_and_self_references_rejected(self):
        for members in [['SYN-ACT-A','missing'],['SYN-ACT-A','SYN-IMG-A'],['SYN-ACT-A','SYN-ACT-A'],['SYN-BIO','SYN-ACT-A']]:
            with self.subTest(members=members),self.assertRaises(CandidateError):
                build_candidate(self.taxonomy,'SYN-BIO',proposed(members=members),100)

    def test_duplicate_label_alias_definition_and_group_rejected(self):
        for label in [' biological EVENT ','Activation of gene A']:
            with self.assertRaises(CandidateError): build_candidate(self.taxonomy,'SYN-BIO',proposed(label),100)
        self.taxonomy.nodes[0].aliases=['Gene activation']
        with self.assertRaises(CandidateError): build_candidate(self.taxonomy,'SYN-BIO',proposed(),100)
        self.taxonomy.nodes[0].aliases=[]
        response=proposed();response['intermediates'][0]['description']=self.taxonomy.nodes[0].concept_description
        with self.assertRaises(CandidateError): build_candidate(self.taxonomy,'SYN-BIO',response,100)
        response=proposed();second=deepcopy(response['intermediates'][0]);second['label']='Other';second['description']='Other definition';response['intermediates'].append(second)
        with self.assertRaises(CandidateError): build_candidate(self.taxonomy,'SYN-BIO',response,100)

    def test_cycle_and_dangling_candidates_cannot_commit(self):
        candidate,_=build_candidate(self.taxonomy,'SYN-BIO',proposed(),100)
        mid=candidate.nodes[-1].concept_id
        candidate.parent_child_edges.append(ParentChildEdge('bad','SYN-ACT-A',mid,'provisional','bad','Synthetic.','test'))
        with self.assertRaisesRegex(ValueError,'cycle'): assert_preserved(self.taxonomy,candidate)
        candidate.parent_child_edges[-1].child_id='missing'
        with self.assertRaisesRegex(ValueError,'Dangling'): assert_preserved(self.taxonomy,candidate)

    def test_modified_original_node_or_evidence_mapping_rejected(self):
        for field in ('concept_label','evidence'):
            candidate=deepcopy(self.taxonomy);setattr(candidate.nodes[3],field,'tamper' if field=='concept_label' else [])
            with self.assertRaises(CandidateError): assert_preserved(self.taxonomy,candidate)
        candidate=deepcopy(self.taxonomy);candidate.concept_evidence_mapping[0]['evidence_ids']=[]
        with self.assertRaises(CandidateError): assert_preserved(self.taxonomy,candidate)

    def test_strict_contracts_invalid_json_confidence_and_unknown_ids(self):
        for raw in ['{"x":1,"x":2}','{"x":NaN}','```json\n{}\n```']:
            with self.assertRaises(ContractError):strict_json(raw)
        response=diagnosis();response['confidence']=True
        with self.assertRaises(ContractError):contracts.diagnosis(response,set())
        response=diagnosis([{'group_id':'g','child_ids':['a','missing'],'reasoning':'Synthetic.'}])
        with self.assertRaises(ContractError):contracts.diagnosis(response,{'a','b'})

    def test_acceptance_requires_every_new_edge_and_improvement(self):
        edges=[{'parent_id':'p','child_id':'x'},{'parent_id':'x','child_id':'c'}]
        response=validation('ACCEPT_AND_STOP');response['edge_checks']=[]
        with self.assertRaises(ContractError):contracts.validation(response,edges,{'x'})
        response['edge_checks']=[dict(e,strict_subsumption=True,reasoning='Synthetic.') for e in edges]
        contracts.validation(response,edges,{'x'})
        response['improves_structure']=False
        with self.assertRaises(ContractError):contracts.validation(response,edges,{'x'})
        response['improves_structure']=True;response['remaining_subgroups']=[['x','y']]
        with self.assertRaises(ContractError):contracts.validation(response,edges,{'x','y'})

    def test_context_includes_shared_parents_ancestor_edges_and_evidence(self):
        builder=ContextBuilder(self.data,self.config['context'])
        context=builder.build(self.taxonomy,'SYN-CARE')
        self.assertEqual([n['concept_id'] for n in context['other_parent_nodes']],['SYN-RESEARCH'])
        leaf=builder.build(self.taxonomy,'SYN-IMG-A')
        self.assertEqual(len(leaf['ancestor_paths']),2)
        self.assertTrue(leaf['parent']['evidence_sample'][0]['text'].startswith('Synthetic'))
        config=deepcopy(self.config['context']);config['max_ancestor_paths']=1
        limited=ContextBuilder(self.data,config).build(self.taxonomy,'SYN-IMG-A')
        self.assertTrue(limited['ancestor_paths_truncated']);self.assertEqual(len(limited['ancestor_edges']),2)

    def test_embedding_outliers_determinism_and_derived_vectors(self):
        vectors={'SYN-ACT-A':[1,0],'SYN-ACT-B':[1,0.01],'SYN-PRO-A':[-1,0]}
        clusterer=EmbeddingClusterer(vectors,.95,20)
        context=ContextBuilder(self.data,self.config['context']).build(self.taxonomy,'SYN-BIO')
        groups,meta=clusterer.cluster(context,diagnosis(),self.taxonomy)
        self.assertEqual([g['child_ids'] for g in groups],[['SYN-ACT-A','SYN-ACT-B']])
        self.assertIn('SYN-PRO-A',meta['unassigned_child_ids']);self.assertIn('SYN-INH-A',meta['missing_vector_ids'])
        candidate,_=build_candidate(self.taxonomy,'SYN-BIO',proposed(),100)
        mid=candidate.nodes[-1].concept_id
        new_context=ContextBuilder(self.data,self.config['context']).build(candidate,'SYN-BIO')
        _,meta=clusterer.cluster(new_context,diagnosis(),candidate)
        self.assertEqual(meta['derived_vectors_from_unique_descendants'][mid],['SYN-ACT-A','SYN-ACT-B'])
        self.assertEqual(clusterer.cluster(context,diagnosis(),self.taxonomy)[0],groups)

    def test_embedding_invalid_vectors_rejected(self):
        for vectors in [{'a':[0,0]},{'a':[1],'b':[1,2]},{'a':[float('nan')]},{'a':[True]}]:
            with self.assertRaises(ValueError):EmbeddingClusterer(vectors)

    def test_missing_definitions_unplaced_or_unreviewed_preflight_fails(self):
        data=deepcopy(self.data);data['taxonomy']['nodes'][3]['concept_description']=None
        with self.assertRaisesRegex(ValueError,'definitions'):validate_dataset(data)
        data=deepcopy(self.data);data['taxonomy']['input_freeze_status']='prepared_snapshot_pending_semantic_review'
        with self.assertRaisesRegex(ValueError,'semantic review'):validate_dataset(data)
        validate_dataset(data,require_semantic_freeze=False)
        data=deepcopy(self.data);data['taxonomy']['parent_child_edges']=[]
        data['taxonomy']['unplaced_concept_ids']=data['taxonomy']['frozen_input_concept_ids']
        with self.assertRaisesRegex(ValueError,'lack a root path'):validate_dataset(data)

    def test_bad_evidence_and_synthetic_pmid_preflight_fails(self):
        data=deepcopy(self.data);data['documents'][0]['source_pmid']='123'
        with self.assertRaisesRegex(ValueError,'Synthetic'):validate_dataset(data)
        data=deepcopy(self.data);data['evidence'][0]['spans'][0]['end']=2
        with self.assertRaisesRegex(ValueError,'span'):validate_dataset(data)

    def test_invalid_model_result_is_rejected_without_mutation(self):
        engine,provider=self.engine()
        original=provider.generate
        def malformed(stage,*args):
            if stage=='induction':return {'content':'not json','finish_reason':'stop'}
            return original(stage,*args)
        provider.generate=malformed;engine.run()
        self.assertEqual(engine.state['taxonomy'],self.data['taxonomy'])
        self.assertEqual(engine.state['accepted_candidates'],0)

    def test_provider_failure_keeps_resumable_checkpoint(self):
        engine,provider=self.engine()
        while engine.state['stack']==[] or engine.state['stack'][-1]['phase']!='propose':engine.step()
        prior=read_checkpoint(engine.directory/'checkpoint.json')
        original=provider.generate
        provider.generate=lambda *args: (_ for _ in ()).throw(ProviderError('simulated interruption'))
        with self.assertRaises(ProviderError):engine.step()
        self.assertEqual(read_checkpoint(engine.directory/'checkpoint.json'),prior)
        provider.generate=original;engine.run()
        self.assertEqual(engine.state['accepted_candidates'],3)

    def test_response_durable_before_checkpoint_is_cached_on_resume(self):
        engine,provider=self.engine()
        while engine.state['stack']==[] or engine.state['stack'][-1]['phase']!='propose':engine.step()
        count=len(provider.calls);prior=read_checkpoint(engine.directory/'checkpoint.json')
        with patch('event_taxonomy.induction.recursive.save_checkpoint',side_effect=OSError('crash before commit')):
            with self.assertRaises(OSError):engine.step()
        self.assertEqual(read_checkpoint(engine.directory/'checkpoint.json'),prior)
        self.assertEqual(len(provider.calls),count+1)
        engine.step();self.assertEqual(len(provider.calls),count+1)
        engine.run();self.assertEqual(engine.state['accepted_candidates'],3)
        self.assertEqual(len(list((engine.directory/'steps').iterdir())),engine.state['sequence'])

    def test_pause_resume_matches_uninterrupted_run(self):
        full=create_run(PACKAGE/'configs/task2.mock.json',self.root,'full')
        paused=create_run(PACKAGE/'configs/task2.mock.json',self.root,'paused',step_limit=7)
        self.assertEqual(read_json(paused/'execution_status.json')['status'],'paused')
        resume_run(paused)
        for name in ('final_taxonomy.json','change_log.json','preservation_report.json'):
            self.assertEqual(read_json(full/name),read_json(paused/name))
        calls=len(list((paused/'llm').glob('*.json')));resume_run(paused)
        self.assertEqual(len(list((paused/'llm').glob('*.json'))),calls)
        with self.assertRaises(FileExistsError):create_run(PACKAGE/'configs/task2.mock.json',self.root,'paused')

    def test_resume_detects_frozen_input_prompt_or_checkpoint_tamper(self):
        for name in ('frozen/input_dataset.json','frozen/prompts.json','frozen/config.json','checkpoint.json'):
            run=create_run(PACKAGE/'configs/task2.mock.json',self.root,'tamper-'+Path(name).stem,step_limit=1)
            original=(run/name).read_text();value=json.loads(original)
            if name=='checkpoint.json':value['state']['sequence']=999
            else:value['tampered']=True
            atomic_json(run/name,value)
            with self.assertRaisesRegex(ValueError,'changed|checksum'):resume_run(run)
            (run/name).write_text(original)
            verify_run(run)

    def test_live_transport_parameters_and_secret_redaction_without_network(self):
        config={'kind':'openai_compatible','base_url':'https://example.invalid/v1',
                'api_key_env':'SYNTHETIC_TEST_KEY','timeout_seconds':10,'extra_body':{'enable_thinking':False}}
        provider=OpenAICompatibleProvider(config)
        body={'choices':[{'message':{'content':'{"safe":"synthetic-secret"}'},'finish_reason':'stop'}],
              'model':'test-model','usage':{'total_tokens':10}}
        response=io.BytesIO(json.dumps(body).encode())
        with patch.dict(os.environ,{'SYNTHETIC_TEST_KEY':'synthetic-secret'}),patch('urllib.request.urlopen',return_value=response) as transport:
            result=provider.generate('diagnosis','prompt',{'test':True},{'model':'test-model','temperature':.1,'max_tokens':300})
            sent=json.loads(transport.call_args[0][0].data)
        self.assertEqual(sent['response_format'],{'type':'json_object'})
        self.assertEqual(sent['temperature'],.1);self.assertFalse(sent['enable_thinking'])
        self.assertNotIn('synthetic-secret',json.dumps(result))

    def test_incomplete_response_is_not_accepted(self):
        engine,provider=self.engine()
        provider.generate=lambda *args:{'content':'{}','finish_reason':'length'}
        engine.run()
        self.assertEqual(engine.state['accepted_candidates'],0)
        self.assertTrue(any('Incomplete' in c.get('reason','') for c in self.changes(engine)))

    def test_config_limits_and_credentials_rejected(self):
        for value in (-1,True,0):
            config=deepcopy(self.config);config['limits']['llm_calls']=value
            with self.assertRaises(ValueError):validate_config(config)
        config=deepcopy(self.config);config['provider']['api_key']='never-save-this'
        with self.assertRaisesRegex(ValueError,'Credentials'):validate_config(config)

    def test_concurrent_resume_is_locked(self):
        self.root.mkdir(exist_ok=True)
        with run_lock(self.root):
            with self.assertRaisesRegex(ValueError,'already'):
                with run_lock(self.root):pass


if __name__=='__main__':unittest.main()
