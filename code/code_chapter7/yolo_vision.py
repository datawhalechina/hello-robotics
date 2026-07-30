"""Ultralytics YOLO 的简洁封装：检测框、类别、置信度和实例掩码。"""

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class Detection:
    xyxy: tuple[int, int, int, int]
    confidence: float
    class_id: int
    class_name: str


@dataclass(frozen=True)
class YOLOPrediction:
    image_shape: tuple[int, int]
    detections: tuple[Detection, ...]
    masks: np.ndarray | None = None  # (N, H, W)，取值范围 0~1

    @property
    def has_masks(self) -> bool:
        return self.masks is not None and len(self.masks) == len(self.detections)


class YOLOVision:
    """一次加载模型，多次调用 predict()。支持 detect 和 segment 权重。"""

    def __init__(self, model_path: str | Path) -> None:
        path = Path(model_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(
                f"找不到 YOLO 模型：{path}\n"
                "检测示例使用仓库根目录 yolo26s.pt；分割示例使用 yolo26s-seg.pt 权重。"
            )

        from ultralytics import YOLO

        print(f"[YOLOVision] 正在加载模型：{path}", flush=True)
        self.model_path = path
        self.model = YOLO(str(path))

    def predict(
        self,
        image_bgr: np.ndarray,
        confidence: float = 0.25,
        iou: float = 0.45,
        image_size: int = 640,
    ) -> YOLOPrediction:
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence 必须位于 [0, 1]")
        if not 0.0 <= iou <= 1.0:
            raise ValueError("iou 必须位于 [0, 1]")

        image = np.ascontiguousarray(image_bgr)
        result = self.model.predict(
            source=image,
            conf=confidence,
            iou=iou,
            imgsz=image_size,
            verbose=False,
        )[0]

        height, width = image.shape[:2]
        detections = self._read_detections(result, width, height)
        masks = self._read_masks(result, width, height, len(detections))
        return YOLOPrediction((height, width), tuple(detections), masks)

    @staticmethod
    def _class_name(names: Mapping | Sequence, class_id: int) -> str:
        if isinstance(names, Mapping):
            return str(names.get(class_id, class_id))
        if 0 <= class_id < len(names):
            return str(names[class_id])
        return str(class_id)

    def _read_detections(self, result, width: int, height: int) -> list[Detection]:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.detach().cpu().numpy()
        scores = boxes.conf.detach().cpu().numpy()
        class_ids = boxes.cls.detach().cpu().numpy().astype(int)
        detections = []
        for box, score, class_id in zip(xyxy, scores, class_ids):
            x1, y1, x2, y2 = box.tolist()
            clipped = (
                int(np.clip(round(x1), 0, max(0, width - 1))),
                int(np.clip(round(y1), 0, max(0, height - 1))),
                int(np.clip(round(x2), 0, max(0, width - 1))),
                int(np.clip(round(y2), 0, max(0, height - 1))),
            )
            detections.append(
                Detection(
                    xyxy=clipped,
                    confidence=float(score),
                    class_id=int(class_id),
                    class_name=self._class_name(result.names, int(class_id)),
                )
            )
        return detections

    @staticmethod
    def _read_masks(
        result,
        width: int,
        height: int,
        detection_count: int,
    ) -> np.ndarray | None:
        if result.masks is None or result.masks.data is None:
            return None
        masks = result.masks.data.detach().cpu().numpy().astype(np.float32)
        if masks.ndim != 3 or masks.shape[0] != detection_count:
            return None
        if masks.shape[1:] != (height, width):
            masks = np.stack(
                [cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR) for mask in masks]
            )
        return masks


def draw_detections(image_bgr: np.ndarray, prediction: YOLOPrediction) -> np.ndarray:
    """使用 OpenCV 绘制框、类别和置信度，避免依赖模型内部绘图。"""
    output = image_bgr.copy()
    for detection in prediction.detections:
        x1, y1, x2, y2 = detection.xyxy
        color = class_color(detection.class_id)
        label = f"{detection.class_name} {detection.confidence:.2f}"
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        (text_width, text_height), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
        )
        top = max(0, y1 - text_height - 8)
        cv2.rectangle(output, (x1, top), (x1 + text_width + 6, y1), color, -1)
        cv2.putText(
            output,
            label,
            (x1 + 3, max(text_height + 1, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return output


def class_color(class_id: int) -> tuple[int, int, int]:
    """根据类别编号生成稳定、醒目的 BGR 颜色。"""
    hue = int((class_id * 47 + 29) % 180)
    hsv = np.uint8([[[hue, 210, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return tuple(int(value) for value in bgr)
