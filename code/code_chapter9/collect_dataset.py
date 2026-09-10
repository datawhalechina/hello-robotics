"""用仿真语义真值生成 YOLO 检测标注（仅训练数据，不参与在线抓取）。"""
import argparse
import json
from pathlib import Path
import traceback
import numpy as np
from config import ROOT, OBJECTS, SimulationConfig


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--objects',default='apple,orange,lemon,tomato,potato')
    p.add_argument('--samples',type=int,default=200)
    p.add_argument('--seed',type=int,default=9)
    p.add_argument('--headless',action='store_true')
    p.add_argument('--output',type=Path,default=ROOT/'data/detection')
    args=p.parse_args()
    if args.samples<5: raise ValueError('samples 至少5')
    from simulation import G2Simulation
    from camera import look_at_optical
    from kinematics import matrix_to_quaternion
    sim=None
    try:
        sim=G2Simulation(SimulationConfig(headless=args.headless),args.objects.split(','))
        sim.camera.sensor.add_semantic_segmentation_to_frame()
        names=[s.name for s in OBJECTS]; rng=np.random.default_rng(args.seed)
        for split in ('train','val'):
            for kind in ('images','labels'): (args.output/kind/split).mkdir(parents=True,exist_ok=True)
        prefix='_'.join(args.objects.split(','))+f'_s{args.seed}'
        for idx in range(args.samples):
            split='val' if idx%5==0 else 'train'
            stem=f'{prefix}_{idx:05d}'
            image=args.output/'images'/split/(stem+'.jpg')
            if image.exists(): raise FileExistsError(f'不会覆盖已采集图片 {image}；更换 seed 或 output')
            sim.scene.randomize(rng,spread=.040)
            target=sim.arm_to_world([.57,.57,-.20])+rng.uniform(-.025,.025,3)
            eye=target+np.array([.25,-.05,.78])+rng.uniform(-.035,.035,3)
            sim.camera.sensor.set_world_pose(eye,matrix_to_quaternion(look_at_optical(eye,target)),camera_axes='ros')
            for _ in range(50): sim.step()
            frame=sim.camera.capture()
            seg=sim.camera.sensor.get_current_frame().get('semantic_segmentation')
            if not isinstance(seg,dict) or 'info' not in seg:
                raise RuntimeError('无语义标注帧；检查本机 Isaac Sim annotator 接口')
            labels=seg['info']['idToLabels']; pixels=np.asarray(seg['data']).squeeze()
            h,w=frame['rgb'].shape[:2]; lines=[]
            for identity,item in labels.items():
                name=item.get('class','') if isinstance(item,dict) else str(item)
                if name not in names: continue
                yy,xx=np.where(pixels==int(identity))
                if len(xx)<40: continue
                x1,x2,y1,y2=int(xx.min()),int(xx.max())+1,int(yy.min()),int(yy.max())+1
                lines.append(f'{names.index(name)} {(x1+x2)/(2*w):.7f} {(y1+y2)/(2*h):.7f} {(x2-x1)/w:.7f} {(y2-y1)/h:.7f}')
            if not lines: raise RuntimeError(f'未找到物体语义标注：{labels}')
            from PIL import Image
            Image.fromarray(frame['rgb']).save(image,quality=94)
            (args.output/'labels'/split/(stem+'.txt')).write_text('\n'.join(lines)+'\n')
            if idx%20==0: print(f'[采集] {idx}/{args.samples}, {len(lines)} objects',flush=True)
        # 相对路径便于整章移动；Ultralytics path 由训练脚本临时写绝对路径。
        (args.output/'dataset.json').write_text(json.dumps(dict(names=names,source='Isaac Sim semantic segmentation',
            validation='held-out random poses/camera jitter in the same procedural scene; not real-world generalization'),indent=2))
        print('[采集完成]',args.output,flush=True)
    except BaseException:
        traceback.print_exc(); raise
    finally:
        if sim: sim.close()

if __name__=='__main__': main()
