"""Loopback-only HTTP adapter. A pure Router is also testable without sockets."""
import json
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
import re
import urllib.parse
from .service import APIError,Workbench
from .pipeline import current_run,pipeline_view
from event_taxonomy.induction.visualize import svg

STATIC=Path(__file__).parent/'static'
MAX_BODY=20*1024*1024


class Router:
    def __init__(self,service,port=4318):
        self.service=service
        self.hosts={'127.0.0.1:'+str(port),'localhost:'+str(port)}

    def response(self,status,value,content_type='application/json; charset=utf-8'):
        body=value if isinstance(value,bytes) else json.dumps(value,ensure_ascii=False,allow_nan=False).encode()
        return status,{'Content-Type':content_type,'Cache-Control':'no-store','X-Content-Type-Options':'nosniff',
            'Referrer-Policy':'no-referrer','Content-Security-Policy':"default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'"},body

    def dispatch(self,method,target,headers,body=b''):
        headers={key.lower():value for key,value in headers.items()}
        try:
            if headers.get('host') not in self.hosts:raise APIError('不允许的主机。',403)
            origin=headers.get('origin')
            if origin and origin not in {'http://'+host for host in self.hosts}:raise APIError('不允许跨站调用。',403)
            if method not in ('GET','POST'):raise APIError('不支持这个请求方法。',405)
            if len(body)>MAX_BODY:raise APIError('文件过大，请使用不超过 20 MB 的数据集。',413)
            path=urllib.parse.unquote(urllib.parse.urlsplit(target).path)
            query=urllib.parse.parse_qs(urllib.parse.urlsplit(target).query)
            data={}
            if method=='POST':
                if headers.get('x-workbench-token')!=self.service.token:raise APIError('页面会话已失效，请刷新后重试。',403)
                if not headers.get('content-type','').startswith('application/json'):raise APIError('需要 JSON 请求。',415)
                try:data=json.loads(body)
                except (ValueError,UnicodeError):raise APIError('请求内容不是有效 JSON。') from None
                if not isinstance(data,dict):raise APIError('请求内容应为 JSON 对象。')
            if method=='GET' and path in ('/','/index.html','/experiments','/experiments.html',
                                         '/style.css','/app.js','/pipeline.css','/pipeline.js','/taxonomy-graph.js'):
                name={'/':'index.html','/experiments':'experiments.html'}.get(path,path[1:])
                mime={'html':'text/html','css':'text/css','js':'text/javascript'}[name.rsplit('.',1)[-1]]
                return self.response(200,(STATIC/name).read_bytes(),mime+'; charset=utf-8')
            if path=='/api/pipeline' and method=='GET':
                return self.response(200,pipeline_view(self.service.runs))
            if path=='/api/pipeline/export' and method=='GET':
                run=current_run(self.service.runs)
                if run is None:raise APIError('尚无已保存的层级归纳结果。',404)
                result=self.response(200,(run/'final_taxonomy.json').read_bytes())
                result[1]['Content-Disposition']='attachment; filename="event-taxonomy.json"'
                return result
            if path=='/api/overview' and method=='GET':return self.response(200,self.service.overview())
            if path=='/api/model' and method=='POST':return self.response(200,self.service.configure_model(data))
            if path=='/api/datasets' and method=='POST':return self.response(201,self.service.import_dataset(data))
            if path=='/api/runs' and method=='POST':return self.response(202,self.service.start(data))
            match=re.fullmatch(r'/api/datasets/([^/]+)',path)
            if match and method=='GET':return self.response(200,{'info':self.service.dataset_info(match[1]),'dataset':self.service.dataset(match[1])})
            match=re.fullmatch(r'/api/datasets/([^/]+)/graph',path)
            if match and method=='GET':return self.response(200,svg(self.service.dataset(match[1])['taxonomy'],'').encode(),'image/svg+xml')
            match=re.fullmatch(r'/api/runs/([^/]+)(?:/(.*))?',path)
            if match:
                run_id,tail=match.groups();step=None
                if 'step' in query:
                    if len(query['step'])!=1 or not query['step'][0].isdigit():raise APIError('步骤序号不正确。')
                    step=int(query['step'][0])
                if not tail and method=='GET':return self.response(200,self.service.detail(run_id))
                if tail=='pause' and method=='POST':return self.response(202,self.service.pause(run_id))
                if tail=='resume' and method=='POST':return self.response(202,self.service.resume(run_id))
                if tail=='graph' and method=='GET':return self.response(200,self.service.graph(run_id,query.get('view',['current'])[0],step).encode(),'image/svg+xml')
                if tail and tail.startswith('steps/') and method=='GET':return self.response(200,self.service.step(run_id,int(tail[6:])))
                if tail=='evidence' and method=='GET':return self.response(200,self.service.evidence(run_id,query.get('concept_id',[''])[0],step))
                if tail and tail.startswith('llm/') and method=='GET':return self.response(200,self.service.llm(run_id,tail[4:]))
                if tail and tail.startswith('export/') and method=='GET':
                    name=tail[7:];result=self.response(200,self.service.export(run_id,name))
                    result[1]['Content-Disposition']='attachment; filename="'+name+'"';return result
            raise APIError('找不到这个页面或接口。',404)
        except APIError as error:return self.response(error.status,{'error':str(error)})
        except FileNotFoundError:return self.response(404,{'error':'记录尚未生成或文件不存在。'})
        except (ValueError,KeyError,TypeError):return self.response(400,{'error':'数据格式不正确，或记录未通过完整性检查。'})
        except Exception:return self.response(500,{'error':'服务暂时无法完成请求，请查看本地运行状态后重试。'})


def make_server(service,port=4318):
    router=Router(service,port)
    class Handler(BaseHTTPRequestHandler):
        def handle_request(self):
            try:size=int(self.headers.get('Content-Length','0'))
            except ValueError:size=MAX_BODY+1
            if size<0 or size>MAX_BODY:
                result=router.response(413,{'error':'请求过大。'});self.close_connection=True
            else:
                self.connection.settimeout(15)
                body=self.rfile.read(size) if size else b''
                result=router.dispatch(self.command,self.path,dict(self.headers.items()),body)
            status,headers,payload=result
            self.send_response(status)
            for key,value in headers.items():self.send_header(key,value)
            self.send_header('Content-Length',str(len(payload)));self.end_headers()
            try:self.wfile.write(payload)
            except (BrokenPipeError,ConnectionResetError):pass
        do_GET=handle_request
        do_POST=handle_request
        do_OPTIONS=handle_request
        def log_message(self,*args):pass
    return ThreadingHTTPServer(('127.0.0.1',port),Handler)
