"""Local jobs and read-only experiment views. Model keys stay in process memory."""
from copy import deepcopy
from datetime import datetime
import os
from pathlib import Path
import re
import secrets
import subprocess
import sys
import threading
import urllib.parse

from event_taxonomy.induction.config import read_config,validate_config
from event_taxonomy.induction.dataset import load_dataset,validate_dataset
from event_taxonomy.induction.run import verify_run
from event_taxonomy.induction.storage import atomic_json,read_json,read_checkpoint
from event_taxonomy.induction.visualize import svg
from .viewmodel import PHASES,readiness_error,timeline

REPO=Path(__file__).resolve().parents[1]
PACKAGE=REPO/'event_taxonomy'
ID=re.compile(r'^[A-Za-z0-9][A-Za-z0-9_-]{0,119}$')


class APIError(ValueError):
    def __init__(self,message,status=400):super().__init__(message);self.status=status


def valid_id(value):
    if not isinstance(value,str) or not ID.fullmatch(value):raise APIError('记录标识不正确。')
    return value


class Workbench:
    def __init__(self,runs_root=None,state_root=None):
        self.runs=Path(runs_root or PACKAGE/'runs').resolve()
        self.state_root=Path(state_root or Path(__file__).parent/'state').resolve()
        self.state_root.mkdir(parents=True,exist_ok=True)
        self.runs.mkdir(parents=True,exist_ok=True)
        self.lock=threading.RLock();self.process=None;self.active_id=None;self._dataset_cache={}
        self.token=secrets.token_urlsafe(32)
        self.model={'base_url':os.environ.get('EVENT_TAXONOMY_BASE_URL',''),
                    'model':os.environ.get('EVENT_TAXONOMY_MODEL',''),
                    'api_key':os.environ.get('EVENT_TAXONOMY_API_KEY','')}

    def run_path(self,run_id):
        path=self.runs/valid_id(run_id)
        if path.is_symlink() or path.resolve().parent!=self.runs:raise APIError('不允许访问这个目录。',403)
        return path

    def job_path(self,run_id):return self.state_root/'jobs'/(valid_id(run_id)+'.json')

    def job(self,run_id):
        path=self.job_path(run_id)
        job=read_json(path) if path.exists() else {}
        if job.get('status') in ('running','queued','pausing') and run_id!=self.active_id:
            job=dict(job,status='interrupted',message='服务已重启，可从最后保存的位置恢复。')
        return job

    def sources(self):
        result={'synthetic':{'id':'synthetic','label':'合成演示 · 8 个细粒度概念',
                             'path':PACKAGE/'examples/synthetic/dataset.json'}}
        candidates=sorted(self.runs.glob('task1-*-final-*'))
        labels={'workbench':'网页已生成的概念','official':'原始 AutoSchemaKG','original_stage2':'原始 Stage 2','role_nested':'角色与嵌套事件'}
        for folder in candidates:
            if not (folder/'events.jsonl').exists():continue
            source_id='prepared-'+folder.name
            label=next((value for key,value in labels.items() if key in folder.name),folder.name)
            result[source_id]={'id':source_id,'label':label+' · 待准备','path':folder}
        for path in sorted((self.state_root/'datasets').glob('upload-*.json')):
            if path.is_symlink():continue
            result[path.stem]={'id':path.stem,'label':'已导入数据 · '+path.stem[-8:],'path':path}
        return result

    def dataset(self,source_id):
        source=self.sources().get(source_id)
        if not source:raise APIError('找不到所选数据。',404)
        path=source['path'];version=path.stat().st_mtime_ns
        if path.is_dir():version=max((path/name).stat().st_mtime_ns for name in ('initial_taxonomy.json','events.jsonl','evidence.jsonl','documents.jsonl'))
        key=(source_id,version)
        if key not in self._dataset_cache:
            self._dataset_cache={k:v for k,v in self._dataset_cache.items() if k[0]!=source_id}
            self._dataset_cache[key]=load_dataset(path)
        return self._dataset_cache[key]

    def dataset_info(self,source_id):
        data=self.dataset(source_id);taxonomy=data['taxonomy'];problem=None
        try:validate_dataset(data)
        except (ValueError,KeyError,TypeError) as error:problem=str(error)
        return {'id':source_id,'label':self.sources()[source_id]['label'],'data_origin':data['data_origin'],
                'concept_count':len(taxonomy['frozen_input_concept_ids']),'event_count':len(data['events']),
                'root_count':len(taxonomy['predefined_roots']),'unplaced_count':len(taxonomy['unplaced_concept_ids']),
                'missing_definitions':sum(not n.get('concept_description') for n in taxonomy['nodes']),
                'ready':problem is None,'problem':readiness_error(problem) if problem else None,'problem_detail':problem}

    def import_dataset(self,data):
        if not isinstance(data,dict):raise APIError('请导入完整的数据集 JSON。')
        try:validate_dataset(data)
        except (ValueError,KeyError,TypeError) as error:raise APIError(readiness_error(str(error))) from None
        source_id='upload-'+secrets.token_hex(6)
        atomic_json(self.state_root/'datasets'/(source_id+'.json'),data)
        return self.dataset_info(source_id)

    def model_public(self):
        return {'base_url':self.model['base_url'],'model':self.model['model'],
                'key_configured':bool(self.model['api_key']),
                'ready':all(bool(v) for v in self.model.values())}

    def configure_model(self,value):
        with self.lock:
            if self.active_id:raise APIError('请等待当前实验结束或暂停后，再更换模型连接。',409)
            if not isinstance(value,dict):raise APIError('模型设置格式不正确。')
            url=value.get('base_url','').strip();model=value.get('model','').strip()
            parsed=urllib.parse.urlsplit(url)
            if parsed.scheme not in ('https','http') or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
                raise APIError('请填写兼容接口的 base URL，不包含密钥或查询参数。')
            if not model or len(model)>150:raise APIError('请填写模型名称。')
            key=value.get('api_key','')
            if not isinstance(key,str) or len(key)>4096 or '\n' in key or '\r' in key:raise APIError('密钥格式不正确。')
            self.model={'base_url':url.rstrip('/'),'model':model,'api_key':key.strip() or self.model['api_key']}
            return self.model_public()

    def list_runs(self):
        ids={p.name for p in self.runs.iterdir() if p.is_dir() and not p.is_symlink() and (p/'checkpoint.json').exists() and ID.fullmatch(p.name)}
        ids.update(p.stem for p in (self.state_root/'jobs').glob('*.json') if ID.fullmatch(p.stem))
        result=[]
        for run_id in ids:
            path=self.run_path(run_id);job=self.job(run_id)
            try:
                if (path/'checkpoint.json').exists():
                    state=read_checkpoint(path/'checkpoint.json')
                    config=read_json(path/'frozen/config.json');data=read_json(path/'frozen/input_dataset.json')
                    status=job.get('status') or ('completed' if state['status']=='completed' else 'paused')
                    result.append({'id':run_id,'status':status,'data_origin':data['data_origin'],
                        'mode':'mock' if config['provider']['kind']=='mock' else 'live',
                        'steps':state['sequence'],'accepted':state['accepted_candidates'],
                        'mtime':(path/'checkpoint.json').stat().st_mtime,'label':job.get('label',run_id)})
                elif job:
                    result.append({'id':run_id,'status':job['status'],'data_origin':job.get('data_origin'),
                        'mode':job.get('mode'),'steps':0,'accepted':0,'mtime':self.job_path(run_id).stat().st_mtime,'label':job.get('label',run_id)})
            except (OSError,ValueError,KeyError,TypeError):continue
        return sorted(result,key=lambda row:row['mtime'],reverse=True)[:40]

    def overview(self):
        return {'token':self.token,'datasets':[self.dataset_info(i) for i in self.sources()],
                'runs':self.list_runs(),'active_run':self.active_id,'model':self.model_public()}

    def start(self,value):
        with self.lock:
            if self.active_id:raise APIError('已有实验正在运行，请等待完成或先暂停。',409)
            source_id=value.get('dataset_id');info=self.dataset_info(source_id)
            if not info['ready']:raise APIError(info['problem'])
            mode=value.get('mode','mock')
            if mode not in ('mock','live'):raise APIError('请选择演示或真实模型模式。')
            if mode=='mock' and source_id!='synthetic':raise APIError('固定 Mock 仅用于内置合成示例；导入数据请使用真实模型。')
            if mode=='live' and not self.model_public()['ready']:raise APIError('请先配置模型连接。')
            config=read_config(PACKAGE/'configs/task2.mock.json')
            for name,upper in [('iterations_per_parent',20),('candidate_attempts',10),('max_depth',30),('llm_calls',1000)]:
                val=value.get(name,config['limits'][name]);lower=0 if name=='max_depth' else 1
                if type(val) is not int or not lower<=val<=upper:raise APIError('实验参数超出允许范围：'+name)
                config['limits'][name]=val
            config['dataset']=str(self.sources()[source_id]['path'])
            if mode=='live':
                config['provider']={'kind':'openai_compatible','base_url':self.model['base_url'],
                    'api_key_env':'EVENT_TAXONOMY_API_KEY','timeout_seconds':120,'extra_body':{}}
                if urllib.parse.urlsplit(self.model['base_url']).hostname.endswith('aliyuncs.com'):
                    config['provider']['extra_body']['enable_thinking']=False
                for stage in config['stages'].values():stage['model']=self.model['model']
            validate_config(config)
            run_id='web-'+datetime.now().strftime('%Y%m%d-%H%M%S')+'-'+secrets.token_hex(3)
            config_path=self.state_root/'configs'/(run_id+'.json');atomic_json(config_path,config)
            command=[sys.executable,'-m','event_taxonomy','induce','--config',str(config_path),
                     '--output-parent',str(self.runs),'--run-id',run_id]
            if mode=='live':command.append('--live')
            self._launch(run_id,command,{'label':('合成演示' if info['data_origin']=='synthetic' else '真实数据实验')+' · '+datetime.now().strftime('%H:%M'),
                                       'mode':mode,'data_origin':info['data_origin']})
            return {'run_id':run_id}

    def _launch(self,run_id,command,metadata):
        pause=self.state_root/'controls'/(run_id+'.pause');pause.parent.mkdir(parents=True,exist_ok=True)
        pause.unlink(missing_ok=True)
        environment=dict(os.environ,EVENT_TAXONOMY_PAUSE_FILE=str(pause))
        if self.model['api_key']:environment['EVENT_TAXONOMY_API_KEY']=self.model['api_key']
        job=dict(metadata,status='queued',message='正在准备输入与实验快照。')
        atomic_json(self.job_path(run_id),job)
        try:process=subprocess.Popen(command,cwd=str(REPO),env=environment,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        except OSError:
            atomic_json(self.job_path(run_id),dict(job,status='failed',message='无法启动 Python 实验进程。'))
            raise APIError('无法启动实验进程。',500) from None
        self.process=process;self.active_id=run_id
        atomic_json(self.job_path(run_id),dict(job,status='running',message='正在运行归纳算法。'))
        key=environment.get('EVENT_TAXONOMY_API_KEY','')
        def wait():
            stdout,stderr=process.communicate()
            with self.lock:
                job=read_json(self.job_path(run_id));path=self.run_path(run_id)
                if process.returncode:
                    message=(stderr or stdout or '实验进程中断。')[-2000:]
                    if key:message=message.replace(key,'[REDACTED]')
                    job.update(status='interrupted' if (path/'checkpoint.json').exists() else 'failed',message=message)
                else:
                    state=read_checkpoint(path/'checkpoint.json')
                    job.update(status='completed' if state['status']=='completed' else 'paused',
                               message='本次实验已结束。' if state['status']=='completed' else '已保存当前位置，可继续运行。')
                atomic_json(self.job_path(run_id),job)
                if self.active_id==run_id:self.active_id=None;self.process=None
        threading.Thread(target=wait,daemon=True).start()

    def pause(self,run_id):
        with self.lock:
            valid_id(run_id)
            if run_id!=self.active_id:raise APIError('这个实验当前没有运行。',409)
            path=self.state_root/'controls'/(run_id+'.pause');path.touch()
            job=read_json(self.job_path(run_id));job.update(status='pausing',message='完成当前步骤并保存后暂停。')
            atomic_json(self.job_path(run_id),job)
            return {'status':'pausing'}

    def resume(self,run_id):
        with self.lock:
            if self.active_id:raise APIError('已有实验正在运行。',409)
            path=self.run_path(run_id)
            if not (path/'checkpoint.json').exists():raise APIError('没有可恢复的检查点。',404)
            try:state=verify_run(path)
            except (ValueError,OSError) as error:raise APIError('无法恢复：输入或算法版本已变化。可继续查看历史结果；原实验可使用其代码快照恢复。',409) from error
            if state['status']=='completed':raise APIError('这个实验已经结束，无需恢复。',409)
            config=read_json(path/'frozen/config.json');live=config['provider']['kind']!='mock'
            if live and not self.model['api_key']:raise APIError('恢复模型实验前，请提供 API 密钥。')
            command=[sys.executable,'-m','event_taxonomy','resume','--run',str(path)]
            if live:command.append('--live')
            old=self.job(run_id);data=read_json(path/'frozen/input_dataset.json')
            self._launch(run_id,command,{'label':old.get('label',run_id),'mode':'live' if live else 'mock','data_origin':data['data_origin']})
            return {'run_id':run_id}

    def detail(self,run_id):
        path=self.run_path(run_id);job=self.job(run_id)
        if not (path/'checkpoint.json').exists():
            if not job:raise APIError('找不到实验记录。',404)
            return {'id':run_id,'job':job,'status':job['status'],'state':None,'timeline':[]}
        state=read_checkpoint(path/'checkpoint.json');data=read_json(path/'frozen/input_dataset.json')
        config=read_json(path/'frozen/config.json');current=state['stack'][-1] if state['stack'] else {}
        by_id={n['concept_id']:n for n in state['taxonomy']['nodes']}
        status=job.get('status') or ('completed' if state['status']=='completed' else 'paused')
        active_request=None
        requests=sorted((path/'llm').glob('*.json'))
        if requests:
            last=read_json(requests[-1])
            if last.get('status')=='request_started':active_request=last['request']['stage']
        return {'id':run_id,'job':job,'status':status,'state':{'sequence':state['sequence'],
                    'accepted':state['accepted_candidates'],'rejected':state['rejected_candidates'],
                    'phase':current.get('phase'),'phase_label':PHASES.get(current.get('phase'),'遍历完成'),
                    'parent_label':by_id.get(current.get('parent_id'),{}).get('concept_label',''),
                    'round':current.get('round',0),'llm_requests':len(requests),'waiting_for':active_request},
                'data_origin':data['data_origin'],'mode':'mock' if config['provider']['kind']=='mock' else 'live',
                'taxonomy':state['taxonomy'],'initial_taxonomy':data['taxonomy'],
                'event_count':len(data['events']),'evidence_count':len(data['evidence']),
                'timeline':timeline(path,state),'report':read_json(path/'preservation_report.json') if (path/'preservation_report.json').exists() else None}

    def step(self,run_id,number):
        path=self.run_path(run_id);state=read_checkpoint(path/'checkpoint.json')
        if type(number) is not int or not 1<=number<=state['sequence']:raise APIError('这一步尚未提交。',404)
        folder=path/'steps'/('%06d'%number)
        result={name:read_json(folder/(name+'.json')) for name in ('before','after','change')}
        result['candidate']=read_json(folder/'candidate.json') if (folder/'candidate.json').exists() else None
        ref=result['change'].get('llm_ref')
        if not ref:
            cursor=result['change'].get('cursor',{})
            stage={'diagnose_and_group':'diagnosis','candidate_constructed':'induction','accept_candidate':'validation','reject_candidate':'validation'}.get(result['change']['action'])
            if result['change']['action']=='reject_candidate' and result['change'].get('phase')=='propose':stage='induction'
            matches=[]
            for candidate in (path/'llm').glob('*.json'):
                row=read_json(candidate)
                if row['request']['stage']==stage and row['request']['input']['cursor']==cursor:matches.append(candidate.name)
            if matches:ref='llm/'+sorted(matches)[-1]
        result['llm_ref']=ref
        return result

    def graph(self,run_id,view='current',number=None):
        path=self.run_path(run_id)
        if number is not None:
            detail=self.step(run_id,number)
            if view not in ('before','after','candidate') or detail[view] is None:raise APIError('没有这个结构视图。',404)
            taxonomy=detail[view]
        else:
            if view not in ('initial','current'):raise APIError('视图不正确。')
            taxonomy=read_json(path/'initial_taxonomy.json') if view=='initial' else read_checkpoint(path/'checkpoint.json')['taxonomy']
        return svg(taxonomy,'')

    def evidence(self,run_id,concept_id,number=None):
        path=self.run_path(run_id);data=read_json(path/'frozen/input_dataset.json')
        taxonomy=read_checkpoint(path/'checkpoint.json')['taxonomy']
        if number is not None:
            step=self.step(run_id,number)
            if step['candidate']:taxonomy=step['candidate']
        node=next((n for n in taxonomy['nodes'] if n['concept_id']==concept_id),None)
        if not node:raise APIError('找不到这个概念。',404)
        events=[e for e in data['events'] if e['event_id'] in node['supporting_event_ids']]
        evidence=[e for e in data['evidence'] if e['evidence_id'] in node['evidence']]
        return {'concept':node,'events':events,'evidence':evidence}

    def llm(self,run_id,name):
        if not re.fullmatch(r'\d{6}-(diagnosis|induction|validation)\.json',name):raise APIError('调用标识不正确。')
        path=self.run_path(run_id)/'llm'/name
        if not path.is_file():raise APIError('找不到模型记录。',404)
        return read_json(path)

    def export(self,run_id,name):
        if name not in ('initial_taxonomy.json','final_taxonomy.json','preservation_report.json','change_log.json'):raise APIError('不支持这个导出文件。',404)
        path=self.run_path(run_id)/name
        if not path.is_file():raise APIError('文件尚未生成，请等待保存完成。',409)
        return path.read_bytes()
