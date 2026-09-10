"""在独立视觉环境训练小 YOLO 检测器，让简化仿真物体拥有可靠类别。"""
import argparse
import json
import os
from pathlib import Path
import shutil
from config import ROOT


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=ROOT/'data/detection')
    p.add_argument('--epochs',type=int,default=40)
    p.add_argument('--batch',type=int,default=16)
    p.add_argument('--device',default='0')
    p.add_argument('--overwrite',action='store_true')
    args=p.parse_args()
    destination=ROOT/'weights/chapter9_yolo.pt'
    if destination.exists() and not args.overwrite: raise FileExistsError('已有 chapter9_yolo.pt，使用 --overwrite 显式替换')
    info=json.loads((args.data/'dataset.json').read_text())
    for split in ('train','val'):
        if len(list((args.data/'images'/split).glob('*.jpg')))<10: raise ValueError(f'{split} 图片太少，请先采集数据')
    spec=dict(path=str(args.data.resolve()),train='images/train',val='images/val',names=info['names'])
    import yaml
    config=args.data/'dataset.yaml'; config.write_text(yaml.safe_dump(spec))
    (ROOT/'weights/ultralytics_config').mkdir(parents=True, exist_ok=True)
    os.environ['YOLO_CONFIG_DIR']=str(ROOT/'weights/ultralytics_config')
    from ultralytics import YOLO
    model=YOLO(str(ROOT/'weights/yolov8n.pt'))
    model.train(data=str(config),epochs=args.epochs,batch=args.batch,imgsz=640,device=args.device,workers=2,
                project=str(ROOT/'outputs/detector_training'),name='chapter9',exist_ok=False,
                seed=9,deterministic=True,patience=15,amp=False,plots=True)
    best=Path(model.trainer.best)
    if not best.is_file(): raise RuntimeError('训练未生成 best.pt')
    shutil.copy2(best,destination)
    (ROOT/'weights/detector_training.json').write_text(json.dumps(dict(source=str(best.relative_to(ROOT)),
        data=str(args.data),epochs=args.epochs,names=info['names'],validation=info['validation']),indent=2))
    # 重训是显式替换，同步校验清单，避免 --verify 仍检查旧模型的哈希。
    from download_weights import sha256
    manifest = ROOT/'weights/manifest.json'
    entries = json.loads(manifest.read_text()) if manifest.exists() else []
    entries = [item for item in entries if item['path'] != 'weights/chapter9_yolo.pt']
    entries.append(dict(path='weights/chapter9_yolo.pt', source='local training; detector_training.json',
                        bytes=destination.stat().st_size, sha256=sha256(destination)))
    manifest.write_text(json.dumps(entries, indent=2))
    print('检测器已保存：',destination)

if __name__=='__main__': main()
