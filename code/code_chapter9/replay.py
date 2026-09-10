"""不用启动 Isaac Sim，从保存的 RGB-D 重跑视觉与抓取生成。"""
import argparse
from pathlib import Path
from config import ROOT

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('frame',type=Path)
    p.add_argument('--backend',choices=['geometry','graspgenx'],default='geometry')
    p.add_argument('--detector',choices=['auto','world','yolo'],default='auto')
    p.add_argument('--weights',type=Path); p.add_argument('--device',default='cuda:0')
    p.add_argument('--output',type=Path,default=ROOT/'outputs/replay')
    args=p.parse_args()
    if args.detector == 'auto':
        if args.weights is None and (ROOT/'weights/chapter9_yolo.pt').is_file():
            args.weights=ROOT/'weights/chapter9_yolo.pt'
        args.detector='yolo' if args.weights else 'world'
    if args.detector == 'yolo' and args.weights is None:
        p.error('--detector yolo 需要 --weights')
    from worker import VisionPipeline
    pipeline=VisionPipeline(backend=args.backend,detector=args.detector,weights=args.weights,device=args.device)
    args.output.mkdir(parents=True,exist_ok=True)
    try: pipeline.run(args.frame,args.output/'result.npz')
    finally: pipeline.close()
    print(args.output/'perception.json')

if __name__=='__main__': main()
