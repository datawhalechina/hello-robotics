"""联网下载显式权重到本章，保留 SHA256 和来源清单。可重复执行。"""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request
from config import ROOT

# 固定本章验证过的官方快照，避免 main 更新改变模型。
GRASPGENX_REVISION='7c834043c11a11417e31d6d5ea9355801e40a2c1'

FILES={
    'yolov8s-worldv2.pt':'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s-worldv2.pt',
    'yolov8n.pt':'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt',
    'sam2.1_hiera_small.pt':'https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt',
}

def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def download(url,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    if not path.is_file():
        tmp=path.with_suffix(path.suffix+'.part')
        print('Downloading',url,flush=True)
        urllib.request.urlretrieve(url,tmp)
        if tmp.stat().st_size<1024: raise RuntimeError(f'下载文件过小：{tmp}')
        tmp.replace(path)
    return dict(url=url,path=str(path.relative_to(ROOT)),bytes=path.stat().st_size,
                sha256=sha256(path))


def main():
    p=argparse.ArgumentParser(); p.add_argument('--graspgenx',action='store_true'); p.add_argument('--skip-clip',action='store_true'); p.add_argument('--verify',action='store_true',help='离线校验已有 manifest，不下载'); args=p.parse_args()
    manifest=ROOT/'weights/manifest.json'
    if args.verify:
        for item in json.loads(manifest.read_text()):
            path=ROOT/item['path']
            if not path.is_file() or sha256(path)!=item['sha256']:
                raise RuntimeError(f'权重校验失败：{path}')
            print('OK',item['path'])
        return
    # 基础下载重跑时保留之前已登记的可选模型，不让校验范围悄悄缩小。
    existing = json.loads(manifest.read_text()) if manifest.exists() else []
    report = []
    def save_manifest():
        merged = {item['path']: item for item in existing}
        merged.update({item['path']: item for item in report})
        manifest.write_text(json.dumps(list(merged.values()), indent=2))
    trained=ROOT/'weights/chapter9_yolo.pt'
    if trained.is_file():
        report.append(dict(path=str(trained.relative_to(ROOT)),source='local training; detector_training.json',
                           bytes=trained.stat().st_size,sha256=sha256(trained)))
    for name,url in FILES.items(): report.append(download(url,ROOT/'weights'/name))
    if not args.skip_clip:
        import clip
        # 官方 CLIP URL 内含 SHA256，clip.load 也会校验。
        report.append(download(clip.clip._MODELS['ViT-B/32'],ROOT/'weights/clip/ViT-B-32.pt'))
    # 即使可选的大模型下载失败，基础权重也有可核验清单。
    save_manifest()
    if args.graspgenx:
        import os
        os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT','180')
        from huggingface_hub import snapshot_download
        snapshot_download('adithyamurali/GraspGenXModel',revision=GRASPGENX_REVISION,local_dir=ROOT/'weights/graspgenx',allow_patterns=['release/**','LICENSE*','README*'])
        for f in (ROOT/'weights/graspgenx/release').rglob('*'):
            if f.is_file(): report.append(dict(repo='adithyamurali/GraspGenXModel',revision=GRASPGENX_REVISION,path=str(f.relative_to(ROOT)),bytes=f.stat().st_size,sha256=sha256(f)))
    save_manifest()

if __name__=='__main__': main()
