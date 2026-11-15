import argparse
from pathlib import Path
import threading
import webbrowser
from .pipeline import load_upward_result
from .server import make_server
from .service import Workbench


def main():
    parser=argparse.ArgumentParser(description='Event Taxonomy 中文实验工作台')
    parser.add_argument('--port',type=int,default=4318)
    parser.add_argument('--open',action='store_true',help='启动后打开浏览器')
    parser.add_argument('--hierarchy',type=Path,help='首页显示已完成的向上归纳结果 hierarchy.json')
    parser.add_argument('--runs-root',type=Path,help='现有实验记录目录')
    parser.add_argument('--state-root',type=Path,help='现有工作台状态目录')
    args=parser.parse_args()
    if not 1024<=args.port<=65535:parser.error('端口应在 1024–65535 之间。')
    if args.hierarchy is None:
        experiment=Path.home()/'Desktop'/'事件概念聚类实验'
        bundled=Path(__file__).resolve().parents[1]/'root_event_taxonomy'/'results'/'qwen3.8-flash-no-thinking-20261114'/'hierarchy.json'
        for candidate in (experiment/'向上归纳Qwen无思考'/'hierarchy.json', bundled,
                          experiment/'向上归纳正式'/'hierarchy.json'):
            if candidate.is_file():
                args.hierarchy=candidate
                break
    if args.hierarchy is not None:
        args.hierarchy=args.hierarchy.expanduser().resolve()
        try:
            load_upward_result(args.hierarchy)
        except (OSError,ValueError,KeyError,TypeError):
            parser.error('无法读取已完成的向上归纳结果，请检查 --hierarchy 指向的文件。')
    service=Workbench(args.runs_root,args.state_root)
    try:server=make_server(service,args.port,args.hierarchy)
    except OSError as error:parser.exit(1,'无法启动本地服务：'+str(error)+'\n请在本机终端运行启动脚本。\n')
    url='http://127.0.0.1:'+str(args.port)
    print('分类体系工作台：'+url,flush=True)
    print('关闭这个终端会停止页面服务。模型密钥仅保留在当前进程内。',flush=True)
    if args.open:threading.Timer(.4,lambda:webbrowser.open(url)).start()
    try:server.serve_forever()
    except KeyboardInterrupt:
        if service.active_id:
            try:service.pause(service.active_id)
            except ValueError:pass
    finally:server.server_close()


if __name__=='__main__':main()
