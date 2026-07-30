"""语义分割工具：传统 HSV 颜色分割与 YOLO 掩码语义化。"""

from dataclasses import dataclass

import cv2
import numpy as np

try:
    from .yolo_vision import YOLOPrediction, class_color
except ImportError:
    from yolo_vision import YOLOPrediction, class_color


@dataclass(frozen=True)
class ColorRule:
    class_id: int
    name: str
    # 一个类别可有多个 HSV 区间，例如红色跨越 0/179 边界。
    ranges: tuple[tuple[tuple[int, int, int], tuple[int, int, int]], ...]


DEFAULT_COLOR_RULES = (
    ColorRule(1, "red", (((0, 100, 60), (10, 255, 255)), ((170, 100, 60), (179, 255, 255)))),
    ColorRule(2, "green", (((35, 70, 50), (85, 255, 255)),)),
    ColorRule(3, "blue", (((90, 70, 40), (135, 255, 255)),)),
    ColorRule(4, "yellow", (((18, 90, 70), (35, 255, 255)),)),
)


class HSVSemanticSegmenter:
    """传统颜色语义分割，适合讲解 HSV、形态学与逐像素分类。"""

    def __init__(
        self,
        rules: tuple[ColorRule, ...] = DEFAULT_COLOR_RULES,
        morphology_kernel: int = 5,
    ) -> None:
        if morphology_kernel <= 0 or morphology_kernel % 2 == 0:
            raise ValueError("morphology_kernel 必须是正奇数")
        self.rules = rules
        self.kernel = np.ones((morphology_kernel, morphology_kernel), np.uint8)

    @property
    def class_names(self) -> dict[int, str]:
        return {0: "background", **{rule.class_id: rule.name for rule in self.rules}}

    @property
    def class_colors(self) -> dict[int, tuple[int, int, int]]:
        # BGR，与类别含义保持直观一致。
        defaults = {1: (0, 0, 255), 2: (0, 255, 0), 3: (255, 0, 0), 4: (0, 255, 255)}
        return {rule.class_id: defaults.get(rule.class_id, class_color(rule.class_id)) for rule in self.rules}

    def predict(self, image_bgr: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        semantic_map = np.zeros(image_bgr.shape[:2], dtype=np.int32)
        for rule in self.rules:
            mask = np.zeros(image_bgr.shape[:2], dtype=np.uint8)
            for lower, upper in rule.ranges:
                mask = cv2.bitwise_or(
                    mask,
                    cv2.inRange(hsv, np.asarray(lower), np.asarray(upper)),
                )
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)
            semantic_map[mask > 0] = rule.class_id
        return semantic_map


def yolo_to_semantic_map(
    prediction: YOLOPrediction,
    mask_threshold: float = 0.5,
) -> tuple[np.ndarray, dict[int, str]]:
    """把 YOLO 实例掩码合并成语义图；0 保留为 background。

    YOLO 类别编号从 0 开始，因此语义图中使用 class_id + 1。
    重叠区域由置信度更高的实例覆盖。
    """
    if not 0.0 <= mask_threshold <= 1.0:
        raise ValueError("mask_threshold 必须位于 [0, 1]")

    semantic_map = np.zeros(prediction.image_shape, dtype=np.int32)
    class_names = {0: "background"}

    # 分割模型在当前画面没有识别到目标时，boxes 和 masks 都为空。
    # 这是正常结果，应返回全背景语义图，而不是误判为加载了检测模型。
    if not prediction.detections:
        return semantic_map, class_names
    if not prediction.has_masks:
        raise ValueError("当前 YOLO 权重没有输出掩码，请使用 yolo26s-seg.pt 分割模型")
    scores = np.asarray([item.confidence for item in prediction.detections])
    # 低置信度先写，高置信度后写，从而覆盖重叠区域。
    for index in np.argsort(scores):
        detection = prediction.detections[int(index)]
        semantic_id = detection.class_id + 1
        semantic_map[prediction.masks[index] >= mask_threshold] = semantic_id
        class_names[semantic_id] = detection.class_name
    return semantic_map, class_names


def colorize_semantic_map(
    semantic_map: np.ndarray,
    class_colors: dict[int, tuple[int, int, int]] | None = None,
) -> np.ndarray:
    """把类别编号图转换为 BGR 彩色图；背景保持黑色。"""
    if semantic_map.ndim != 2:
        raise ValueError("semantic_map 必须是二维数组")
    color = np.zeros((*semantic_map.shape, 3), dtype=np.uint8)
    for semantic_id in np.unique(semantic_map):
        semantic_id = int(semantic_id)
        if semantic_id == 0:
            continue
        # semantic_id 通常为 YOLO class_id + 1。
        value = class_color(semantic_id - 1)
        if class_colors is not None:
            value = class_colors.get(semantic_id, value)
        color[semantic_map == semantic_id] = value
    return color


def overlay_semantic(
    image_bgr: np.ndarray,
    semantic_map: np.ndarray,
    alpha: float = 0.45,
    class_colors: dict[int, tuple[int, int, int]] | None = None,
) -> np.ndarray:
    """把语义颜色半透明叠加到原图，背景区域不染色。"""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha 必须位于 [0, 1]")
    if image_bgr.shape[:2] != semantic_map.shape:
        raise ValueError("图像与 semantic_map 尺寸不一致")

    colored = colorize_semantic_map(semantic_map, class_colors)
    blended = cv2.addWeighted(image_bgr, 1.0 - alpha, colored, alpha, 0.0)
    output = image_bgr.copy()
    foreground = semantic_map != 0
    output[foreground] = blended[foreground]
    return output


def draw_legend(
    image_bgr: np.ndarray,
    class_names: dict[int, str],
    semantic_map: np.ndarray,
    class_colors: dict[int, tuple[int, int, int]] | None = None,
) -> np.ndarray:
    """只为当前画面中出现的类别绘制图例。"""
    output = image_bgr.copy()
    present = [int(value) for value in np.unique(semantic_map) if int(value) != 0]
    for row, semantic_id in enumerate(present):
        y = 24 + row * 25
        color = class_color(semantic_id - 1)
        if class_colors is not None:
            color = class_colors.get(semantic_id, color)
        cv2.rectangle(output, (8, y - 14), (25, y + 3), color, -1)
        cv2.putText(
            output,
            class_names.get(semantic_id, str(semantic_id)),
            (32, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return output
