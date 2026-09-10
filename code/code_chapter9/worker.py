"""常驻推理进程；stdout 仅输出协议 JSON，所有库日志重定向 stderr。"""
import argparse
import contextlib
from dataclasses import asdict
import json
from pathlib import Path
import sys
import traceback
import numpy as np
from config import ROOT, GripperGeometry, PlannerConfig, TRAY_EXCLUSION_HALF_SIZE
from pointcloud import unproject, clean_cloud, fit_table, estimate_pose, write_ply
from grasp_planner import Grasp, geometric_grasps, filter_grasps


def save_grasps(path, grasps, **arrays):
    np.savez_compressed(path, transforms=np.array([g.T for g in grasps]).reshape(-1,4,4),
                        widths=np.array([g.width for g in grasps]),scores=np.array([g.score for g in grasps]),
                        object_ids=np.array([g.object_id for g in grasps],dtype=int),
                        sources=np.array([g.source for g in grasps],dtype=str),**arrays)


def load_grasps(data):
    return [Grasp(T,float(w),float(s),int(i),str(src)) for T,w,s,i,src in
            zip(data['transforms'],data['widths'],data['scores'],data['object_ids'],data['sources'])]


class VisionPipeline:
    def __init__(self, **options):
        from perception import DetectorSegmenter
        self.backend = options.pop('backend','geometry')
        self.grasp_python = options.pop('grasp_python',None)
        self.device = options.get('device','cuda:0')
        self.detector = DetectorSegmenter(**options)
        self.grasp_worker = None

    def run(self, source, destination):
        from perception import annotate
        from worker_client import WorkerClient
        import cv2
        cfg = PlannerConfig()
        cal = json.loads((ROOT/'assets/g2_gripper.json').read_text())
        geom = GripperGeometry(**cal['geometry'])
        with np.load(source,allow_pickle=False) as f: frame = dict(f)
        rgb = frame['rgb']; K = frame['K']; T = frame['T_world_camera']
        full = clean_cloud(unproject(frame['depth'],K,T,stride=2))
        bounds = frame['workspace']
        keep = (full[:,:2]>=bounds[0]).all(1)&(full[:,:2]<=bounds[1]).all(1)
        scene = full[keep & (full[:,2] > float(frame['table_z'])-.045) & (full[:,2]<float(frame['table_z'])+.40)]
        plane = fit_table(scene,float(frame['table_z']))
        detections = self.detector(rgb)
        folder = destination.parent
        folder.mkdir(parents=True,exist_ok=True)
        annotate(rgb,detections,folder/'detections.jpg')
        write_ply(folder/'scene.ply',scene)
        arrays = {}; metadata = []; candidates = []
        for i,d in enumerate(detections):
            try:
                mask = cv2.erode(d['mask'].astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)
                points = clean_cloud(unproject(frame['depth'],K,T,mask))
                valid = (points@plane[:3]+plane[3]>.007)&(points[:,:2]>=bounds[0]).all(1)&(points[:,:2]<=bounds[1]).all(1)
                points = points[valid]
                # 托盘 ROI 不再作为待抓取区；区域是任务先验，不是物体真值。
                if 'place_xy' in frame and len(points) and np.all(abs(np.median(points[:,:2],axis=0)-frame['place_xy']) < TRAY_EXCLUSION_HALF_SIZE):
                    raise ValueError('位于已放置区域')
                if len(points)<cfg.min_points: raise ValueError('有效物体点太少')
                pose = estimate_pose(points,plane)
                arrays[f'object_{i}'] = points
                write_ply(folder/f'object_{i}_{d["label"]}.ply',points)
                if self.backend == 'geometry':
                    gs,_ = geometric_grasps(points,plane,i,geom)
                    candidates.extend(gs)
                metadata.append(dict(id=i,label=d['label'],confidence=d['confidence'],sam_score=d['sam_score'],
                                     points=len(points),center=pose.center.tolist(),rotation=pose.rotation.tolist(),
                                     extents=pose.extents.tolist(),yaw_observable=pose.yaw_observable))
            except ValueError as exc:
                metadata.append(dict(id=i,label=d['label'],rejected=str(exc)))
        if self.backend == 'graspgenx' and arrays:
            if self.grasp_worker is None:
                self.grasp_worker = WorkerClient(self.grasp_python,role='graspgenx',options=dict(device=self.device),log_dir=folder)
            source_x = folder/'graspgenx_input.npz'; output_x = folder/'graspgenx_raw.npz'
            np.savez_compressed(source_x,**arrays)
            self.grasp_worker.request(source_x,output_x,timeout=300)
            with np.load(output_x,allow_pickle=False) as f: candidates = load_grasps(f)
        grasps = filter_grasps(candidates,scene,plane,geom,cfg)
        preview=sorted(candidates,key=lambda g:g.score,reverse=True)[:48]
        save_grasps(destination,grasps,scene=scene,plane=plane,
                    preview_raw_transforms=np.array([g.T for g in preview]).reshape(-1,4,4),
                    preview_raw_widths=np.array([g.width for g in preview]),
                    labels=np.array([d['label'] for d in detections],dtype=str),
                    boxes=np.array([d['box'] for d in detections]).reshape(-1,4),
                    masks=np.array([d['mask'] for d in detections],dtype=bool).reshape(-1,*rgb.shape[:2]),
                    frame_id=frame['frame_id'],sim_time=frame['sim_time'],**arrays)
        report = dict(backend=self.backend,objects=metadata,raw_candidates=len(candidates),accepted_candidates=len(grasps),
                      grasps=[dict(object_id=g.object_id,score=g.score,width=g.width,T=g.T.tolist(),source=g.source) for g in grasps])
        (folder/'perception.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))

    def close(self):
        if self.grasp_worker: self.grasp_worker.close()


def main():
    p=argparse.ArgumentParser(); p.add_argument('--role',choices=['vision','graspgenx'],default='vision'); p.add_argument('--options',default='{}')
    args=p.parse_args(); protocol=sys.stdout; engine=None
    import signal
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    try:
        for line in sys.stdin:
            try:
                request=json.loads(line)
                source,dest=Path(request['input']),Path(request['output'])
                with contextlib.redirect_stdout(sys.stderr):
                    if engine is None:
                        if args.role=='vision': engine=VisionPipeline(**json.loads(args.options))
                        else:
                            from graspgenx_backend import GraspGenXBackend
                            engine=GraspGenXBackend(**json.loads(args.options))
                    if args.role=='vision': engine.run(source,dest)
                    else:
                        candidates=[]
                        with np.load(source,allow_pickle=False) as data:
                            for key in data.files:
                                if key.startswith('object_'): candidates.extend(engine.generate(data[key],int(key.split('_')[1])))
                        save_grasps(dest,candidates)
                reply=dict(ok=True,output=str(dest))
            except Exception as exc:
                traceback.print_exc(file=sys.stderr); reply=dict(ok=False,error=str(exc))
            protocol.write(json.dumps(reply)+'\n'); protocol.flush()
    finally:
        if engine and hasattr(engine,'close'): engine.close()

if __name__=='__main__': main()
