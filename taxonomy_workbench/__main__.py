import argparse
import threading
import webbrowser
from .server import make_server
from .service import Workbench


def main():
    parser=argparse.ArgumentParser(description='Event Taxonomy 中文实验工作台')
    parser.add_argument('--port',type=int,default=4318)
    parser.add_argument('--open',action='store_true',help='启动后打开浏览器')
    args=parser.parse_args()
    if not 1024<=args.port<=65535:parser.error('端口应在 1024–65535 之间。')
    service=Workbench()
    try:server=make_server(service,args.port)
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
