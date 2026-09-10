"""YOLO 框提示 SAM2；只依赖 RGB，不读取场景实例真值。"""
from pathlib import Path
import numpy as np
from config import ROOT, OBJECTS

class DetectorSegmenter:
    def __init__(self, weights=None, detector='world', device='cuda:0', confidence=.18):
        import torch
        from ultralytics import YOLO, YOLOWorld
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        self.device, self.confidence = device, confidence
        if device.startswith('cuda') and not torch.cuda.is_available():
            raise RuntimeError('视觉环境没有可用 CUDA；可用 --device cpu 测试，但会很慢')
        weights = Path(weights or ROOT/'weights/yolov8s-worldv2.pt')
        sam = ROOT/'weights/sam2.1_hiera_small.pt'
        for path in (weights,sam):
            if not path.is_file():
                raise FileNotFoundError(f'缺少权重 {path}；先运行 download_weights.py')
        if detector == 'world':
            # ultralytics 的文本编码器是 CLIP；将缓存明确放进本章，而不是 ~/.cache。
            import clip
            original_load = clip.load
            def load_local(*args, **kwargs):
                kwargs.setdefault('download_root',str(ROOT/'weights/clip'))
                return original_load(*args,**kwargs)
            clip.load = load_local
            try:
                self.detector = YOLOWorld(str(weights))
                self.detector.set_classes([s.prompt for s in OBJECTS])
            finally:
                clip.load = original_load
            self.names = [s.name for s in OBJECTS]
        else:
            self.detector = YOLO(str(weights))
            self.names = [self.detector.names[i] for i in range(len(self.detector.names))]
        self.sam = SAM2ImagePredictor(build_sam2('configs/sam2.1/sam2.1_hiera_s.yaml',str(sam),device=device))

    def __call__(self, rgb):
        import cv2
        import torch
        # ndarray 输入 Ultralytics 为 BGR，SAM2 为 RGB，不能传反。
        result = self.detector.predict(np.ascontiguousarray(rgb[...,::-1]), conf=self.confidence,
                                       iou=.45, imgsz=640, device=self.device, verbose=False)[0]
        detections = []
        with torch.inference_mode():
            self.sam.set_image(rgb)
            for b in result.boxes:
                box = b.xyxy[0].cpu().numpy()
                masks, scores, _ = self.sam.predict(box=box, multimask_output=True)
                mask = np.asarray(masks[int(np.argmax(scores))],bool)
                if float(np.max(scores)) < .60:
                    continue
                h,w = mask.shape
                x1,y1,x2,y2 = np.rint(box).astype(int)
                roi = np.zeros_like(mask)
                roi[max(0,y1-3):min(h,y2+4),max(0,x1-3):min(w,x2+4)] = True
                mask &= roi
                n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8),8)
                if n < 2:
                    continue
                mask = labels == (1+np.argmax(stats[1:,cv2.CC_STAT_AREA]))
                if mask.sum() < 60 or mask.sum() > h*w*.25:
                    continue
                if any(np.sum(mask & d['mask'])/max(1,np.sum(mask | d['mask']))>.65 for d in detections):
                    continue
                cls = int(b.cls.item())
                detections.append(dict(box=box,mask=mask,label=self.names[cls],confidence=float(b.conf.item()),
                                       sam_score=float(np.max(scores))))
        return detections


def annotate(rgb, detections, path):
    import cv2
    canvas = rgb.copy()
    rng = np.random.default_rng(9)
    for i,d in enumerate(detections):
        color = rng.integers(60,250,3)
        mask = d['mask']
        canvas[mask] = (.6*canvas[mask]+.4*color).astype(np.uint8)
        x1,y1,x2,y2 = np.rint(d['box']).astype(int)
        cv2.rectangle(canvas,(x1,y1),(x2,y2),tuple(map(int,color)),2)
        cv2.putText(canvas,f"{i}:{d['label']} {d['confidence']:.2f}",(x1,max(14,y1-5)),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1)
    if not cv2.imwrite(str(path),canvas[...,::-1]):
        raise OSError(f'写图片失败 {path}')
