"""YOLO-World 开放词汇检测、颜色校验和 RGB-D 目标定位。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import cv2
import numpy as np

try:
    from .config import PerceptionConfig, TARGET_COLORS, YOLO_WORLD_CLASSES
    from .rgbd_geometry import depth_pixels_to_world
except ImportError:
    from config import PerceptionConfig, TARGET_COLORS, YOLO_WORLD_CLASSES
    from rgbd_geometry import depth_pixels_to_world


@dataclass(frozen=True)
class TargetDetection:
    color: str
    confidence: float
    bbox: tuple[int, int, int, int]
    world_xyz: tuple[float, float, float]

    def summary(self) -> dict:
        return {
            "color": self.color,
            "confidence": round(self.confidence, 3),
            "world_xy": [round(self.world_xyz[0], 3), round(self.world_xyz[1], 3)],
        }


def _class_name(names: Mapping | Sequence, class_id: int) -> str:
    if isinstance(names, Mapping):
        return str(names.get(class_id, class_id))
    return str(names[class_id]) if 0 <= class_id < len(names) else str(class_id)


def _color_from_text(text: str) -> str | None:
    lower = text.lower()
    return next((color for color in TARGET_COLORS if color in lower), None)


def _dominant_hsv_color(image_bgr: np.ndarray, bbox) -> str | None:
    """YOLO 负责找物体，HSV 只用于纠正红/蓝/黄标签。"""
    x1, y1, x2, y2 = bbox
    width, height = x2 - x1, y2 - y1
    x1 += max(1, int(width * 0.18))
    x2 -= max(1, int(width * 0.18))
    y1 += max(1, int(height * 0.18))
    y2 -= max(1, int(height * 0.18))
    crop = image_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    vivid = (saturation > 90) & (value > 65)
    scores = {
        "red": int(np.count_nonzero(vivid & ((hue < 10) | (hue > 170)))),
        "blue": int(np.count_nonzero(vivid & (hue >= 95) & (hue <= 135))),
        "yellow": int(np.count_nonzero(vivid & (hue >= 18) & (hue <= 38))),
    }
    color, score = max(scores.items(), key=lambda item: item[1])
    return color if score >= max(20, int(crop.shape[0] * crop.shape[1] * 0.04)) else None


def _hsv_mask(image_bgr: np.ndarray, color: str) -> np.ndarray:
    """提取目标颜色像素，远距离检测框包含背景时定位更准确。"""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    vivid = (saturation > 80) & (value > 55)
    if color == "red":
        return vivid & ((hue < 12) | (hue > 168))
    if color == "blue":
        return vivid & (hue >= 92) & (hue <= 138)
    if color == "yellow":
        return vivid & (hue >= 16) & (hue <= 42)
    return np.zeros(hue.shape, dtype=bool)


def target_world_position(
    camera, image_bgr: np.ndarray, depth: np.ndarray, bbox, color: str, tolerance: float
) -> np.ndarray | None:
    """优先用目标颜色像素反投影；颜色像素不足时退回检测框深度。"""
    x1, y1, x2, y2 = bbox
    mask = _hsv_mask(image_bgr, color)
    box_mask = np.zeros(mask.shape, dtype=bool)
    box_mask[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
    box_mask &= np.isfinite(depth) & (depth > 0.20) & (depth < 8.0)
    if np.count_nonzero(box_mask) >= 8:
        points = depth_pixels_to_world(camera, depth, box_mask)
        if len(points) >= 5:
            return np.median(points, axis=0)
    return bbox_world_position(camera, depth, bbox, tolerance)


def bbox_world_position(camera, depth: np.ndarray, bbox, tolerance: float) -> np.ndarray | None:
    """取框中心较近的一簇深度点，求目标世界坐标中位数。"""
    x1, y1, x2, y2 = bbox
    height, width = depth.shape
    x1, x2 = np.clip([x1, x2], 0, width - 1)
    y1, y2 = np.clip([y1, y2], 0, height - 1)
    if x2 <= x1 or y2 <= y1:
        return None

    margin_x = max(2, int((x2 - x1) * 0.25))
    margin_y = max(2, int((y2 - y1) * 0.20))
    x1i, x2i = x1 + margin_x, x2 - margin_x
    y1i, y2i = y1 + margin_y, y2 - margin_y
    if x2i <= x1i or y2i <= y1i:
        return None

    inner = depth[y1i:y2i, x1i:x2i]
    valid_values = inner[np.isfinite(inner) & (inner > 0.20) & (inner < 8.0)]
    if len(valid_values) < 8:
        return None

    # 背景通常比目标远；较低分位数更接近目标表面。
    surface_depth = float(np.quantile(valid_values, 0.30))
    mask = np.zeros(depth.shape, dtype=bool)
    region = np.isfinite(inner) & (np.abs(inner - surface_depth) <= tolerance)
    mask[y1i:y2i, x1i:x2i] = region
    points = depth_pixels_to_world(camera, depth, mask)
    if len(points) < 5:
        return None
    return np.median(points, axis=0)


class YOLOWorldDetector:
    """YOLO-World 检测框 + 深度反投影，输出可导航的世界坐标。"""

    def __init__(self, model_path, config: PerceptionConfig = PerceptionConfig()) -> None:
        from ultralytics import YOLO

        self.config = config
        print(f"[YOLO-World] 加载模型：{model_path}", flush=True)
        self.model = YOLO(str(model_path))
        self.model.set_classes(YOLO_WORLD_CLASSES)
        print(f"[YOLO-World] 开放词汇类别：{YOLO_WORLD_CLASSES}", flush=True)

    def detect(self, image_bgr: np.ndarray, depth: np.ndarray, camera) -> list[TargetDetection]:
        result = self.model.predict(
            source=np.ascontiguousarray(image_bgr),
            conf=self.config.confidence,
            iou=self.config.iou,
            imgsz=self.config.image_size,
            verbose=False,
        )[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []

        height, width = image_bgr.shape[:2]
        detections = []
        for box, confidence, class_id in zip(
            boxes.xyxy.detach().cpu().numpy(),
            boxes.conf.detach().cpu().numpy(),
            boxes.cls.detach().cpu().numpy().astype(int),
        ):
            x1, y1, x2, y2 = box.tolist()
            bbox = (
                int(np.clip(round(x1), 0, width - 1)),
                int(np.clip(round(y1), 0, height - 1)),
                int(np.clip(round(x2), 0, width - 1)),
                int(np.clip(round(y2), 0, height - 1)),
            )
            yolo_color = _color_from_text(_class_name(result.names, int(class_id)))
            color = _dominant_hsv_color(image_bgr, bbox) or yolo_color
            if color not in TARGET_COLORS:
                continue
            world = target_world_position(
                camera, image_bgr, depth, bbox, color, self.config.depth_tolerance
            )
            if world is None:
                continue
            detections.append(
                TargetDetection(
                    color=color,
                    confidence=float(confidence),
                    bbox=bbox,
                    world_xyz=tuple(float(value) for value in world),
                )
            )

        # 每种颜色只保留置信度最高的一个框。
        best = {}
        for detection in detections:
            if detection.color not in best or detection.confidence > best[detection.color].confidence:
                best[detection.color] = detection
        return list(best.values())


def draw_detections(image_bgr: np.ndarray, detections) -> np.ndarray:
    output = image_bgr.copy()
    colors = {"red": (30, 30, 240), "blue": (240, 80, 30), "yellow": (20, 220, 240)}
    for detection in detections:
        x1, y1, x2, y2 = detection.bbox
        color = colors[detection.color]
        label = (
            f"{detection.color} {detection.confidence:.2f} "
            f"({detection.world_xyz[0]:.2f}, {detection.world_xyz[1]:.2f})"
        )
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            output, label, (x1, max(22, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX,
            0.52, color, 2, cv2.LINE_AA,
        )
    return output


def make_observation_board(observations: dict[str, tuple[TargetDetection, np.ndarray]]) -> np.ndarray:
    """把扫描过程中分散出现的目标拼成一张图交给 VLM。"""
    tiles = []
    for color in TARGET_COLORS:
        if color not in observations:
            continue
        detection, image = observations[color]
        x1, y1, x2, y2 = detection.bbox
        pad = 18
        crop = image[max(0, y1 - pad): y2 + pad, max(0, x1 - pad): x2 + pad]
        if crop.size == 0:
            continue
        crop = cv2.resize(crop, (240, 200), interpolation=cv2.INTER_AREA)
        tile = np.full((240, 240, 3), 245, dtype=np.uint8)
        tile[:200] = crop
        cv2.putText(
            tile, f"candidate: {color}", (12, 228), cv2.FONT_HERSHEY_SIMPLEX,
            0.62, (20, 20, 20), 2, cv2.LINE_AA,
        )
        tiles.append(tile)
    if not tiles:
        raise RuntimeError("没有可生成观察板的目标图像")
    return np.hstack(tiles)
