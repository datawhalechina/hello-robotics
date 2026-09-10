"""独立四宫格窗口。只读可视化，不向规划/控制反馈结果；GUI 不阻塞推理。

左上实时 RGB，右上最近 YOLO，左下最近 SAM2，右下同帧点云/抓取投影。
绿色仅表示几何筛选通过，黄色才是通过 IK/路径规划后选中的抓取。
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import numpy as np
from config import ROOT


def write_json(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False))
    tmp.replace(path)


def save_live(path, rgb, sim_time):
    tmp = path.with_suffix('.tmp.npz')
    np.savez(tmp, rgb=rgb, sim_time=sim_time, wall_time=time.time())
    tmp.replace(path)


class PerceptionViewer:
    """Isaac 侧发布器：不导入 OpenCV/Qt；文件原子替换，无队列积压。"""
    def __init__(self, python, run, backend):
        if not os.environ.get('DISPLAY'):
            raise RuntimeError('--show-perception 需要桌面 DISPLAY；无桌面可离线导出 viewer.py --save')
        from worker_client import isolated_environment
        self.root = Path(run).resolve()/'viewer'
        self.root.mkdir(parents=True, exist_ok=True)
        # 明确开始新会话，不沿用旧的 ready/stop 文件。
        for name in ('ready.json', 'stop'):
            (self.root/name).unlink(missing_ok=True)
        self.state = dict(stage='starting', backend=backend, completed=0, round=0)
        self.last_rgb = 0.
        self.update()
        self.log = (self.root/'viewer.log').open('w')
        self.proc = subprocess.Popen([str(Path(python).resolve()),str(ROOT/'viewer.py'),'--run',str(Path(run).resolve()),'--follow'],
                                     env=isolated_environment(Path(python).resolve()),stdout=self.log,stderr=self.log)
        for _ in range(150):
            if (self.root/'ready.json').exists(): return
            if self.proc.poll() is not None: break
            time.sleep(.05)
        self.close()
        raise RuntimeError(f'显示窗口启动失败；查看 {self.root}/viewer.log（检查 DISPLAY/X11 权限）')

    def update(self, **values):
        self.state.update(values)
        write_json(self.root/'state.json', self.state)

    def publish_rgb(self, sensor, sim_time):
        if time.monotonic()-self.last_rgb < .12 or not self.is_running(): return
        self.last_rgb = time.monotonic()
        frame = sensor.get_current_frame()
        rgba = frame.get('rgb', frame.get('rgba'))
        if rgba is None or not np.asarray(rgba).size: return
        rgb = np.asarray(rgba)[...,:3]
        if rgb.dtype != np.uint8: rgb=np.clip(rgb*255,0,255).astype(np.uint8)
        save_live(self.root/'live.npz', rgb, sim_time)

    def is_running(self):
        return self.proc.poll() is None

    def close(self):
        if self.is_running():
            (self.root/'stop').touch()
            try: self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.proc.terminate()
                try: self.proc.wait(timeout=2)
                except subprocess.TimeoutExpired: self.proc.kill(); self.proc.wait()
        self.log.close()


def project(points, K, T_world_camera):
    """世界点投影到 RGB，返回像素和正深度标记；不使用物体真值。"""
    p=(np.asarray(points)-T_world_camera[:3,3]) @ T_world_camera[:3,:3]
    valid=np.isfinite(p).all(1)&(p[:,2]>.01)
    uv=(p @ K.T)[:,:2]/np.maximum(p[:,2,None],.01)
    return np.clip(uv,-100000,100000).astype(np.int32),valid


def draw_grasp(image, T, width, K, camera, color, thickness=1):
    import cv2
    # 抓取坐标 X 闭合、+Z 向指尖；U 形线仅用于可视化，不替代实测碰撞体。
    local=np.array([[-width/2,0,0],[-width/2,0,-.045],
                    [width/2,0,-.045],[width/2,0,0],[0,0,-.045],[0,0,-.075]])
    pixels,valid=project(local @ T[:3,:3].T+T[:3,3],K,camera)
    for a,b in ((0,1),(1,2),(2,3),(4,5)):
        if valid[a] and valid[b]: cv2.line(image,tuple(pixels[a]),tuple(pixels[b]),color,thickness,cv2.LINE_AA)


def render_dashboard(live, frame=None, result=None, report=None, state=None):
    """纯渲染函数，输出 BGR 图；可单元测试或无桌面导出，不创建窗口。"""
    import cv2
    state=state or {}; report=report or {}
    blank=np.zeros((480,640,3),np.uint8)
    current=cv2.cvtColor(live,cv2.COLOR_RGB2BGR) if live is not None else blank.copy()
    rgb=cv2.cvtColor(frame['rgb'],cv2.COLOR_RGB2BGR) if frame is not None else blank.copy()
    boxes,masks,grasps=rgb.copy(),rgb.copy(),(rgb*.4).astype(np.uint8)
    if result is not None and frame is not None:
        palette=((50,190,255),(255,180,50),(100,230,90),(200,80,220),(90,210,210),(210,180,240))
        for i,(box,label) in enumerate(zip(result['boxes'],result['labels'])):
            x1,y1,x2,y2=np.asarray(box,int);color=palette[i%len(palette)]
            cv2.rectangle(boxes,(x1,y1),(x2,y2),color,2)
            cv2.putText(boxes,str(label),(x1,max(18,y1-5)),cv2.FONT_HERSHEY_SIMPLEX,.5,color,1,cv2.LINE_AA)
            mask=result['masks'][i].astype(bool)
            masks[mask]=(masks[mask]*.45+np.array(color)*.55).astype(np.uint8)
            contours,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(masks,contours,-1,color,1)
        K,camera=frame['K'],frame['T_world_camera']
        for key in (k for k in result if k.startswith('object_') and k!='object_ids'):
            pts=result[key][::3]
            uv,valid=project(pts,K,camera)
            h,w=grasps.shape[:2]; valid&=(uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h)
            grasps[uv[valid,1],uv[valid,0]]=(160,110,50)
        for T,width in zip(result.get('preview_raw_transforms',[])[:24],result.get('preview_raw_widths',[])[:24]):
            draw_grasp(grasps,T,width,K,camera,(125,125,125))
        for T,width in zip(result['transforms'][:12],result['widths'][:12]):
            draw_grasp(grasps,T,width,K,camera,(60,240,60),2)
        if state.get('selected'):
            g=state['selected'];draw_grasp(grasps,np.array(g['T']),g['width'],K,camera,(0,230,255),3)
    panels=[]
    titles=['LIVE RGB (updates on simulation steps)','YOLO: latest inference snapshot',
            'SAM2: box-prompted instance masks',
            f"{state.get('backend',report.get('backend','grasp'))}: cloud + grasps (camera projection)"]
    for title,img in zip(titles,(current,boxes,masks,grasps)):
        tile=cv2.resize(img,(640,480));cv2.rectangle(tile,(0,0),(640,30),(25,25,25),-1)
        cv2.putText(tile,title,(8,21),cv2.FONT_HERSHEY_SIMPLEX,.48,(245,245,245),1,cv2.LINE_AA)
        panels.append(tile)
    canvas=np.vstack([np.hstack(panels[:2]),np.hstack(panels[2:])])
    footer=np.zeros((84,1280,3),np.uint8)
    stamp=f"snapshot sim={float(frame['sim_time']):.2f}s" if frame is not None else 'waiting for first inference'
    age=f" | received {max(0,time.time()-state['snapshot_time']):.1f}s ago" if state.get('snapshot_time') else ''
    lines=[f"round={state.get('round',0)} completed={state.get('completed',0)} | {state.get('stage','replay')} | {stamp}{age}",
           f"raw={report.get('raw_candidates','-')} filtered={report.get('accepted_candidates','-')} | gray: raw (up to 24), green: filtered (up to 12), yellow: selected IK plan",
           'Inference panels are frozen snapshots, NOT continuously replanned. Closing this viewer does NOT stop the robot. ESC/Q: close viewer.']
    for i,text in enumerate(lines):cv2.putText(footer,text,(8,22+26*i),cv2.FONT_HERSHEY_SIMPLEX,.49,(230,230,230),1,cv2.LINE_AA)
    return np.vstack([canvas,footer])


def load_snapshot(folder):
    folder=Path(folder)
    with np.load(folder/'rgbd.npz',allow_pickle=False) as f:frame=dict(f)
    with np.load(folder/'result.npz',allow_pickle=False) as f:result=dict(f)
    report=json.loads((folder/'perception.json').read_text())
    return frame,result,report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--follow',action='store_true',help='跟随主进程状态/实时 RGB；独立重放不需要')
    p.add_argument('--pick',default='pick_00',help='独立查看旧报告时的轮次')
    p.add_argument('--save',type=Path,help='不创建窗口，只导出四宫格图片')
    p.add_argument('--test-seconds',type=float,default=0,help='GUI 测试时自动关闭；默认一直显示')
    args=p.parse_args()
    import cv2
    root=args.run/'viewer';root.mkdir(parents=True,exist_ok=True)
    state={};frame=result=report=None;last=None
    if not args.follow:
        state=dict(snapshot=str((args.run/args.pick).resolve()),stage='offline replay')
    if args.save:
        frame,result,report=load_snapshot(args.run/args.pick)
        args.save.parent.mkdir(parents=True,exist_ok=True)
        if not cv2.imwrite(str(args.save),render_dashboard(frame['rgb'],frame,result,report,dict(backend=report['backend'],stage='offline replay'))):
            raise RuntimeError('图片写入失败')
        return
    if not os.environ.get('DISPLAY'):p.error('没有 DISPLAY；使用 --save 导出图片')
    window='Chapter 9 | RGB / YOLO / SAM2 / Grasp'
    cv2.namedWindow(window,cv2.WINDOW_NORMAL);cv2.resizeWindow(window,1280,1044)
    started=time.monotonic()
    try:
        while not (args.follow and (root/'stop').exists()):
            if args.follow and (root/'state.json').exists():state=json.loads((root/'state.json').read_text())
            if state.get('snapshot') and state['snapshot']!=last:
                frame,result,report=load_snapshot(state['snapshot']);last=state['snapshot']
            live=frame['rgb'] if frame is not None else None
            if args.follow and (root/'live.npz').exists():
                with np.load(root/'live.npz',allow_pickle=False) as f:live=f['rgb']
            canvas=render_dashboard(live,frame,result,report,state)
            cv2.imshow(window,canvas)
            key=cv2.waitKey(80)&0xff
            if not (root/'ready.json').exists():write_json(root/'ready.json',dict(pid=os.getpid()))
            if key in (27,ord('q')) or cv2.getWindowProperty(window,cv2.WND_PROP_VISIBLE)<1:break
            if args.test_seconds and time.monotonic()-started>args.test_seconds:break
    finally:cv2.destroyAllWindows()


if __name__=='__main__':main()
