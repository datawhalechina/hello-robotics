"""主入口：检测 → SAM2 → RGB-D 点云 → 抓取候选 → IK → 物理抓取/放置。"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import traceback
import numpy as np
from config import ROOT, OBJECTS, SimulationConfig, GripperGeometry
from worker_client import WorkerClient
from worker import load_grasps


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--headless',action='store_true')
    p.add_argument('--objects',default='apple,orange,lemon,soap,can,wood_block',help='最多6个；--list-objects 查看15种')
    p.add_argument('--list-objects',action='store_true')
    p.add_argument('--backend',choices=['geometry','graspgenx'],default='geometry')
    p.add_argument('--detector',choices=['auto','world','yolo'],default='auto')
    p.add_argument('--weights',type=Path)
    p.add_argument('--confidence',type=float,default=.18)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--vision-python',type=Path,default=ROOT/'.envs/vision/bin/python')
    p.add_argument('--grasp-python',type=Path,default=ROOT/'.envs/graspgenx/bin/python')
    counts=p.add_mutually_exclusive_group()
    counts.add_argument('--max-picks',type=int,default=3)
    counts.add_argument('--pick-all',action='store_true',help='按本批物体数逐个抓放，每次重新感知；失败仍停止')
    p.add_argument('--show-perception',action='store_true',help='独立窗口显示实时RGB和最近检测/分割/抓取快照，需要桌面')
    p.add_argument('--perception-only',action='store_true',help='只检测、分割、估计抓取，不执行动作')
    p.add_argument('--calibrate-only',action='store_true')
    p.add_argument('--output',type=Path)
    p.add_argument('--seed',type=int,default=9)
    p.add_argument('--keep-open',action='store_true')
    return p.parse_args()


def main():
    args=parse_args()
    if args.list_objects:
        for s in OBJECTS: print(f'{s.name:12s} {s.prompt:30s} {s.size}')
        return 0
    if args.max_picks<1 or not 0<args.confidence<1: raise ValueError('非法次数/置信度')
    names=args.objects.split(',')
    if len(set(names))!=len(names): raise ValueError('同一批物体名不能重复')
    out=args.output or ROOT/'outputs'/datetime.now().strftime('%Y%m%d_%H%M%S')
    out.mkdir(parents=True,exist_ok=True)
    (out/'run_config.json').write_text(json.dumps(vars(args),default=str,ensure_ascii=False,indent=2))
    weights=args.weights
    detector=args.detector
    if detector=='auto':
        if weights is None and (ROOT/'weights/chapter9_yolo.pt').is_file(): weights=ROOT/'weights/chapter9_yolo.pt'
        detector='yolo' if weights is not None else 'world'
    if detector=='yolo' and weights is None: raise ValueError('--detector yolo 需要 --weights 检测器路径')
    options=dict(backend=args.backend,detector=detector,device=args.device,confidence=args.confidence,
                 weights=str(weights.resolve()) if weights else None,grasp_python=str(args.grasp_python.resolve()))
    from simulation import G2Simulation
    from controller import PickController, PlanningError, ExecutionError
    sim=None; worker=None; viewer=None; records=[]; status=0
    try:
        sim=G2Simulation(SimulationConfig(headless=args.headless),names)
        cal=sim.calibrate_gripper(ROOT/'assets/g2_gripper.json')
        (out/'calibration.json').write_text(json.dumps(cal,indent=2))
        if args.calibrate_only: return 0
        sim.scene.randomize(np.random.default_rng(args.seed))
        for _ in range(90): sim.step()
        if args.show_perception:
            from viewer import PerceptionViewer
            viewer=PerceptionViewer(args.vision_python,out,args.backend)
            sim.frame_observer=lambda: viewer.publish_rgb(sim.camera.sensor,sim.ticks*sim.cfg.physics_dt)
        controller=PickController(sim,GripperGeometry(**cal['geometry']))
        np.savez_compressed(out/'robot_state.npz',T_world_base=sim.T_world_base,
                            right=sim.arm.get_joint_positions(),left=sim.left.get_joint_positions(),
                            bounds=sim.scene.bounds,place=sim.scene.place,table_z=sim.scene.table_top)
        worker=WorkerClient(args.vision_python,options=options,log_dir=out)
        def observe(folder):
            folder.mkdir(parents=True,exist_ok=True)
            frame=sim.camera.capture()
            np.savez_compressed(folder/'rgbd.npz',**frame,table_z=sim.scene.table_top,workspace=sim.scene.bounds,place_xy=sim.scene.place[:2])
            if viewer: viewer.update(stage='inference: YOLO -> SAM2 -> '+args.backend+' (physics paused)')
            worker.request(folder/'rgbd.npz',folder/'result.npz',timeout=420)
            if viewer:
                import time
                viewer.update(snapshot=str(folder.resolve()),snapshot_time=time.time(),stage='IK / path planning')
            with np.load(folder/'result.npz',allow_pickle=False) as f: return dict(f)
        for pick in range(len(names) if args.pick_all else args.max_picks):
            if viewer: viewer.update(round=pick+1,selected=None)
            folder=out/f'pick_{pick:02d}'
            data=observe(folder)
            np.savez_compressed(folder/'robot_state.npz',T_world_base=sim.T_world_base,
                                right=sim.arm.get_joint_positions(),left=sim.left.get_joint_positions(),
                                bounds=sim.scene.bounds,place=sim.scene.place,table_z=sim.scene.table_top)
            grasps=load_grasps(data)
            print(f'[第{pick+1}轮] 检测{len(data["labels"])}个，候选{len(grasps)}个',flush=True)
            report=json.loads((folder/'perception.json').read_text())
            for item in report['objects']:
                if 'rejected' in item:
                    print(f"[感知过滤] {item['label']}: {item['rejected']}",flush=True)
            print(f"[候选统计] 生成 {report['raw_candidates']} → 保留 {report['accepted_candidates']}",flush=True)
            if args.perception_only:
                records.append(dict(mode='perception-only',detections=len(data['labels']),candidates=len(grasps))); break
            if not grasps:
                status=3; records.append(dict(status='no_grasp')); print('没有安全抓取候选；查看 detections.jpg/perception.json，不执行盲抓。',flush=True); break
            plan=None; rejected=[]
            # 每个实例先试若干高分候选，避免总是被同一不可达物体占满尝试预算。
            for g in grasps:
                try:
                    plan=controller.plan(g,data['scene'],data['plane'],data[f'object_{g.object_id}']); break
                except PlanningError as exc:
                    rejected.append(dict(object_id=g.object_id,reason=str(exc)))
            (folder/'planning.json').write_text(json.dumps(dict(rejected=rejected,planned=plan is not None),indent=2,ensure_ascii=False))
            if plan is None:
                records.append(dict(status='no_reachable_collision_free_plan',rejected=rejected)); status=3; break
            def recheck(g):
                fresh=observe(folder/'recheck')
                label=str(data['labels'][g.object_id])
                old_center=np.median(data[f'object_{g.object_id}'],axis=0)
                for i,newlabel in enumerate(fresh['labels']):
                    key=f'object_{i}'
                    if str(newlabel)==label and key in fresh:
                        if np.linalg.norm(np.median(fresh[key],axis=0)-old_center)<.015: return True
                return False
            try:
                if viewer: viewer.update(selected=dict(T=plan.grasp.T.tolist(),width=plan.grasp.width),selected_folder=str(folder.resolve()))
                result=controller.execute(plan,recheck=recheck,
                    on_stage=(lambda stage: viewer.update(stage=stage)) if viewer else None)
                result.update(status='success' if result['placed'] else 'place_failed',
                              detected_label=str(data['labels'][plan.grasp.object_id]),backend=args.backend)
                records.append(result)
                if viewer: viewer.update(completed=sum(r.get('status')=='success' for r in records))
                print('[执行结果]',result,flush=True)
                after=sim.camera.capture()
                np.savez_compressed(folder/'after.npz',**after)
                from PIL import Image
                Image.fromarray(after['rgb']).save(folder/'after.jpg')
                if not result['placed']: status=2; break
            except ExecutionError as exc:
                # 跟踪/物理验证失败后不自动进行未经重新规划的恢复动作。
                records.append(dict(status='execution_stopped',reason=str(exc))); status=2
                print('[已停止]',str(exc),flush=True); break
        if viewer: viewer.update(stage=f'finished (exit={status})')
        if args.keep_open and (not args.headless or viewer is not None):
            while sim.app.is_running() and (not args.headless or viewer.is_running()): sim.step()
    except BaseException as exc:
        traceback.print_exc(); status=1
        records.append(dict(status='error',reason=str(exc)))
    finally:
        (out/'summary.json').write_text(json.dumps(dict(records=records,exit_status=status),indent=2,ensure_ascii=False))
        print(f'[报告] {out}',flush=True)
        if viewer:
            viewer.update(stage=f'finished (exit={status})')
            sim.frame_observer=None
            viewer.close()
        if worker: worker.close()
        if sim: sim.close(status)
    return status

if __name__=='__main__': raise SystemExit(main())
