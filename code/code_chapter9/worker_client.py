"""用本机子进程隔离 Isaac Sim / PyTorch 依赖，NPZ 传数组、JSON 传命令。"""
import json
import os
from pathlib import Path
import select
import subprocess
import time
from config import ROOT


def isolated_environment(python):
    """供推理和显示子进程共用，不修改父进程的环境。"""
    env = os.environ.copy()
    # 不让 Isaac 的 Python/插件/动态库路径污染 Conda 子进程。
    for key in ('PYTHONPATH','PYTHONHOME','LD_LIBRARY_PATH','LD_PRELOAD','CARB_APP_PATH','EXP_PATH','ISAAC_PATH','PYTHONEXE'):
        env.pop(key,None)
    env['CONDA_PREFIX'] = str(python.parent.parent)
    env['PATH'] = str(python.parent)+':/usr/local/bin:/usr/bin:/bin'
    env['PYTHONNOUSERSITE'] = '1'
    (ROOT/'weights/ultralytics_config').mkdir(parents=True, exist_ok=True)
    env['YOLO_CONFIG_DIR'] = str(ROOT/'weights/ultralytics_config')
    env['HF_HOME'] = str(ROOT/'weights/hf_cache')
    env['XDG_CACHE_HOME'] = str(ROOT/'weights/cache')
    env['OMP_NUM_THREADS'] = '4'
    for key in ('QT_QPA_PLATFORM_PLUGIN_PATH','QT_QPA_FONTDIR','QT_PLUGIN_PATH'):
        env.pop(key, None)
    return env

class WorkerClient:
    def __init__(self, python=None, role='vision', options=None, log_dir=None):
        python = Path(python or ROOT/'.envs'/role/'bin/python').expanduser()
        if not python.is_absolute():
            # 子进程 cwd 固定为 ROOT；这里先按调用者当前目录解析，避免相对路径被误解。
            python = (Path.cwd() / python).resolve()
        if not python.is_file():
            raise FileNotFoundError(f'缺少独立环境解释器：{python}；运行 setup_env.sh {role}')
        log_dir = Path(log_dir or ROOT/'outputs'); log_dir.mkdir(parents=True,exist_ok=True)
        self.log_path = log_dir/(role+'_worker.log')
        self.log = self.log_path.open('w')
        env = isolated_environment(python)
        self.proc = subprocess.Popen([str(python),'-u',str(ROOT/'worker.py'),'--role',role,'--options',json.dumps(options or {})],
                                      stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,text=True,env=env,cwd=ROOT)

    def request(self, source, destination, timeout=240):
        msg = dict(input=str(Path(source).resolve()),output=str(Path(destination).resolve()))
        self.proc.stdin.write(json.dumps(msg)+'\n'); self.proc.stdin.flush()
        deadline = time.monotonic()+timeout
        while time.monotonic()<deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f'推理进程退出 {self.proc.returncode}，查看 {self.log_path}')
            ready,_,_ = select.select([self.proc.stdout],[],[],.2)
            if not ready: continue
            line = self.proc.stdout.readline()
            if not line: continue
            reply = json.loads(line)
            if not reply.get('ok'):
                raise RuntimeError(f"推理失败：{reply.get('error')}；查看 {self.log_path}")
            return Path(reply['output'])
        self.close()
        raise TimeoutError(f'推理超时 {timeout}s，查看 {self.log_path}')

    def close(self):
        if self.proc.poll() is None:
            self.proc.terminate()
            try: self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill(); self.proc.wait()
        self.log.close()

    def __enter__(self): return self
    def __exit__(self,*_): self.close()
