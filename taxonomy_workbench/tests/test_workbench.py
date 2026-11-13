from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
from event_taxonomy.tests.synthetic import dataset
from event_taxonomy.induction.storage import atomic_json,read_checkpoint,read_json,save_checkpoint
from event_taxonomy.induction.recursive import RecursiveRefiner
from event_taxonomy.tests import test_induction
from taxonomy_workbench.server import Router,MAX_BODY
from taxonomy_workbench.service import APIError,Workbench
from taxonomy_workbench.viewmodel import explain


class WorkbenchTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.service=Workbench(self.root/'runs',self.root/'state');self.router=Router(self.service)
        self.service.model={'base_url':'','model':'','api_key':''}
        self.headers={'Host':'127.0.0.1:4318','Origin':'http://127.0.0.1:4318','Content-Type':'application/json','X-Workbench-Token':self.service.token}
        self.source=self.root/'dataset.json';atomic_json(self.source,dataset())
        patcher=patch.object(self.service,'sources',return_value={'synthetic':{'id':'synthetic','label':'Synthetic Data','path':self.source}})
        patcher.start();self.sources_patcher=patcher;self.addCleanup(patcher.stop)
        self.addCleanup(self.stop_process)

    def stop_process(self):
        if self.service.process:
            try:self.service.pause(self.service.active_id)
            except APIError:pass
            deadline=time.time()+15
            while self.service.active_id and time.time()<deadline:time.sleep(.02)
            if self.service.process:self.service.process.terminate();self.service.process.wait(timeout=5)

    def request(self,path,method='GET',value=None,headers=None,raw=None):
        response=self.router.dispatch(method,path,headers or self.headers,raw if raw is not None else json.dumps(value or {}).encode() if method=='POST' else b'')
        status,metadata,body=response
        return status,json.loads(body) if metadata['Content-Type'].startswith('application/json') else body

    def wait(self):
        deadline=time.time()+20
        while self.service.active_id and time.time()<deadline:time.sleep(.02)
        self.assertIsNone(self.service.active_id,'Worker did not finish')

    def test_static_and_overview_use_actual_dataset_counts(self):
        code,html=self.request('/');self.assertEqual(code,200);self.assertIn('层级归纳'.encode(),html)
        code,html=self.request('/experiments');self.assertEqual(code,200);self.assertIn('开始演示'.encode(),html)
        code,overview=self.request('/api/overview');self.assertEqual(code,200)
        self.assertEqual(overview['datasets'][0]['concept_count'],8)
        self.assertTrue(overview['datasets'][0]['ready'])
        self.assertNotIn('api_key',overview['model'])

    def test_end_to_end_start_poll_graph_steps_evidence_and_export(self):
        code,result=self.request('/api/runs','POST',{'dataset_id':'synthetic','mode':'mock'})
        self.assertEqual(code,202);rid=result['run_id'];self.wait()
        code,detail=self.request('/api/runs/'+rid);self.assertEqual(code,200);self.assertEqual(detail['status'],'completed')
        self.assertEqual(detail['state']['accepted'],3);self.assertEqual(detail['state']['rejected'],1)
        self.assertEqual(len(detail['taxonomy']['nodes']),16)
        reject=next(r for r in detail['timeline'] if r['action']=='reject_candidate')
        code,step=self.request(f'/api/runs/{rid}/steps/{reject["sequence"]}')
        self.assertEqual(step['before'],step['after']);self.assertEqual(len(step['candidate']['nodes']),13)
        code,graphic=self.request(f'/api/runs/{rid}/graph?step={reject["sequence"]}&view=candidate')
        self.assertEqual(code,200);self.assertIn(b'<svg',graphic)
        ref=step['llm_ref'];self.assertTrue(ref)
        code,call=self.request(f'/api/runs/{rid}/{ref}')
        self.assertEqual(code,200);self.assertEqual(call['request']['stage'],'validation')
        code,evidence=self.request(f'/api/runs/{rid}/evidence?concept_id=SYN-ACT-A')
        self.assertEqual(code,200);self.assertEqual(len(evidence['events']),1)
        self.assertIsNone(evidence['events'][0]['source_pmid'])
        code,result=self.request(f'/api/runs/{rid}/export/final_taxonomy.json')
        self.assertEqual(code,200);self.assertEqual(len(result['nodes']),16)
        code,_=self.request(f'/api/runs/{rid}/export/../../frozen/config.json');self.assertNotEqual(code,200)

    def test_cooperative_pause_then_resume_retains_same_result(self):
        rid=self.service.start({'dataset_id':'synthetic'})['run_id']
        self.service.pause(rid);self.wait()
        detail=self.service.detail(rid)
        self.assertEqual(detail['status'],'paused')
        saved=detail['state']['sequence'];self.assertLess(saved,82)
        self.service.resume(rid);self.wait()
        detail=self.service.detail(rid)
        self.assertEqual(detail['status'],'completed');self.assertEqual(detail['state']['accepted'],3)
        self.assertEqual(detail['report']['original_concepts_preserved'],8)

    def test_unready_data_cannot_start_or_be_relabelled_mock(self):
        data=dataset();data['taxonomy']['nodes'][3]['concept_description']=None;atomic_json(self.source,data)
        code,overview=self.request('/api/overview');self.assertFalse(overview['datasets'][0]['ready'])
        code,error=self.request('/api/runs','POST',{'dataset_id':'synthetic'})
        self.assertEqual(code,400);self.assertIn('定义',error['error'])
        self.assertIsNone(self.service.process)

    def test_host_origin_and_csrf_checks(self):
        for headers in [dict(self.headers,Host='attacker.example'),dict(self.headers,Origin='http://attacker.example'),
                        dict(self.headers,**{'X-Workbench-Token':'wrong'})]:
            code,_=self.request('/api/runs','POST',{'dataset_id':'synthetic'},headers)
            self.assertEqual(code,403)
        code,_=self.request('/api/runs','POST',{},dict(self.headers,**{'Content-Type':'text/plain'}))
        self.assertEqual(code,415)
        code,_=self.request('/api/overview','OPTIONS');self.assertEqual(code,405)

    def test_traversal_symlinks_and_future_steps_rejected(self):
        for path in ['/api/runs/%2e%2e','/api/runs/../../.env','/api/runs/x/llm/../../config.json','/.env','/state/configs/test.json']:
            code,_=self.request(path);self.assertIn(code,[400,403,404])
        outside=self.root/'outside';outside.mkdir();(self.service.runs/'symlink').symlink_to(outside,target_is_directory=True)
        with self.assertRaises(APIError):self.service.run_path('symlink')
        rid=self.service.start({'dataset_id':'synthetic'})['run_id'];self.wait()
        code,_=self.request(f'/api/runs/{rid}/steps/999999');self.assertEqual(code,404)

    def test_key_is_memory_only_and_configuration_does_not_call_model(self):
        key='synthetic-secret-for-test-only'
        code,response=self.request('/api/model','POST',{'base_url':'https://model.example/v1','model':'test-model','api_key':key})
        self.assertEqual(code,200);self.assertTrue(response['key_configured']);self.assertNotIn(key,json.dumps(response))
        self.assertFalse(list(self.service.state_root.rglob('*.json')))
        with patch.object(self.service,'_launch') as launch:
            result=self.service.start({'dataset_id':'synthetic','mode':'live'})
            self.assertTrue(launch.called)
        config=read_json(self.service.state_root/'configs'/(result['run_id']+'.json'))
        self.assertEqual(config['provider']['api_key_env'],'EVENT_TAXONOMY_API_KEY')
        self.assertNotIn(key,json.dumps(config))

    def test_model_configuration_preserves_key_when_blank_and_rejects_bad_urls(self):
        self.service.configure_model({'base_url':'https://model.example/v1','model':'m','api_key':'test-key'})
        self.service.configure_model({'base_url':'https://model.example/v1','model':'m2','api_key':''})
        self.assertEqual(self.service.model['api_key'],'test-key')
        for url in ['file:///tmp/secrets','https://user:secret@model.example/v1','https://model.example/v1?key=secret']:
            with self.assertRaises(APIError):self.service.configure_model({'base_url':url,'model':'m','api_key':''})

    def test_live_requires_connection_and_bad_limits_do_not_start(self):
        with self.assertRaises(APIError):self.service.start({'dataset_id':'synthetic','mode':'live'})
        for value in [0,1001,True,1.5]:
            with self.assertRaises(APIError):self.service.start({'dataset_id':'synthetic','llm_calls':value})
        self.assertIsNone(self.service.process)

    def test_import_preserves_full_dataset_and_rejects_bad_evidence(self):
        self.sources_patcher.stop()
        data=dataset();info=self.service.import_dataset(data)
        self.assertEqual(read_json(self.service.state_root/'datasets'/(info['id']+'.json')),data)
        data['evidence'][0]['spans'][0]['end']=1
        with self.assertRaises(APIError):self.service.import_dataset(data)

    def test_malformed_and_large_requests_rejected(self):
        for raw in [b'not json',b'[]',b'null']:
            code,_=self.request('/api/runs','POST',raw=raw);self.assertEqual(code,400)
        code,_=self.request('/api/runs','POST',raw=b'x'*(MAX_BODY+1));self.assertEqual(code,413)

    def test_unknown_model_response_is_not_exposed_as_success(self):
        rid='web-failed'
        self.service._launch(rid,[os.sys.executable,'-c','import sys;sys.exit(2)'],{'label':'test','mode':'mock','data_origin':'synthetic'})
        self.wait();detail=self.service.detail(rid)
        self.assertEqual(detail['status'],'failed');self.assertIsNone(detail['state'])

    def test_resume_detects_modified_frozen_parameters(self):
        rid=self.service.start({'dataset_id':'synthetic'})['run_id'];self.service.pause(rid);self.wait()
        path=self.service.run_path(rid)/'frozen/config.json';config=read_json(path);config['limits']['max_depth']=30;atomic_json(path,config)
        with self.assertRaises(APIError):self.service.resume(rid)

    def test_restart_marks_active_job_as_interrupted(self):
        atomic_json(self.service.job_path('web-crashed'),{'status':'running','mode':'mock'})
        self.assertEqual(self.service.job('web-crashed')['status'],'interrupted')

    def test_explanations_distinguish_continue_stop_and_reject(self):
        base={'sequence':1,'action':'accept_candidate','cursor':{'round':1,'attempt':1},'parent_id':'p','changes':{'added_nodes':[{'concept_label':'Middle'}]}}
        again=explain(dict(base,decision='ACCEPT_AND_CONTINUE'),{})
        done=explain(dict(base,decision='ACCEPT_AND_STOP'),{})
        reject=explain(dict(base,action='reject_candidate',reason='llm_rejected_candidate'),{})
        self.assertIn('还不向下递归',again['summary']);self.assertIn('再进入',done['summary']);self.assertIn('不表示',reject['summary'])


class PauseBoundaryTests(unittest.TestCase):
    def test_pause_is_checked_between_committed_transitions(self):
        fixture=test_induction.InductionTests('test_level_first_continue_before_any_child');fixture.setUp()
        try:
            engine,_=fixture.engine();counter={'n':0}
            def stop():counter['n']+=1;return counter['n']>3
            state=engine.run(stop_requested=stop)
            self.assertEqual(state['sequence'],3)
            self.assertEqual(read_checkpoint(engine.directory/'checkpoint.json'),state)
            engine.run();self.assertEqual(engine.state['accepted_candidates'],3)
        finally:fixture.doCleanups()


if __name__=='__main__':unittest.main()
