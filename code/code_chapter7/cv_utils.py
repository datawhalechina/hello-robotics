"""OpenCV 基础处理：灰度、滤波、边缘、二值化、轮廓与结果拼图。"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class CVResult:
    gray: np.ndarray
    blurred: np.ndarray
    edges: np.ndarray
    binary: np.ndarray
    color_mask: np.ndarray
    contours: np.ndarray


class BasicCVProcessor:
    """把常见 OpenCV 操作组织成一条便于学习的处理流水线。"""

    def __init__(
        self,
        blur_kernel: int = 5,
        canny_low: int = 80,
        canny_high: int = 160,
        min_contour_area: float = 300.0,
    ) -> None:
        if blur_kernel <= 0 or blur_kernel % 2 == 0:
            raise ValueError("blur_kernel 必须是正奇数")
        if not 0 <= canny_low < canny_high <= 255:
            raise ValueError("Canny 阈值应满足 0 <= low < high <= 255")
        self.blur_kernel = blur_kernel
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.min_contour_area = min_contour_area

    def process(self, image_bgr: np.ndarray) -> CVResult:
        image = ensure_bgr_uint8(image_bgr)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (self.blur_kernel, self.blur_kernel), 0)
        edges = cv2.Canny(blurred, self.canny_low, self.canny_high)
        _, binary = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # 示例颜色分割：提取“饱和度较高”的区域，不依赖某一种具体物体颜色。
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        color_mask = cv2.inRange(hsv, (0, 80, 40), (179, 255, 255))
        kernel = np.ones((3, 3), dtype=np.uint8)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)

        contour_image = image.copy()
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_contour_area:
                continue
            x, y, width, height = cv2.boundingRect(contour)
            cv2.rectangle(contour_image, (x, y), (x + width, y + height), (0, 255, 0), 2)
            cv2.putText(
                contour_image,
                f"area={area:.0f}",
                (x, max(18, y - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

        return CVResult(gray, blurred, edges, binary, color_mask, contour_image)


def ensure_bgr_uint8(image: np.ndarray) -> np.ndarray:
    """校验并统一为连续内存的 uint8 BGR 图像。"""
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"期望 HxWx3 BGR 图像，实际形状为 {array.shape}")
    if np.issubdtype(array.dtype, np.floating):
        scale = 255.0 if array.size and float(np.nanmax(array)) <= 1.0 else 1.0
        array = np.nan_to_num(array, nan=0.0, posinf=255.0, neginf=0.0)
        array = np.clip(array * scale, 0.0, 255.0).astype(np.uint8)
    else:
        array = np.clip(array, 0, 255).astype(np.uint8, copy=False)
    return np.ascontiguousarray(array)


def _to_bgr(image: np.ndarray) -> np.ndarray:
    return image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


def _labeled(image: np.ndarray, title: str, size: tuple[int, int]) -> np.ndarray:
    panel = cv2.resize(_to_bgr(image), size, interpolation=cv2.INTER_AREA)
    cv2.rectangle(panel, (0, 0), (size[0], 28), (0, 0, 0), -1)
    cv2.putText(
        panel, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
        0.55, (255, 255, 255), 1, cv2.LINE_AA,
    )
    return panel


def make_cv_panel(image_bgr: np.ndarray, result: CVResult) -> np.ndarray:
    """生成 2x3 教学拼图，便于对比每一步的效果。"""
    image = ensure_bgr_uint8(image_bgr)
    height, width = image.shape[:2]
    cell_width = min(480, width)
    cell_height = max(180, int(cell_width * height / width))
    size = (cell_width, cell_height)
    panels = [
        _labeled(image, "1 Original", size),
        _labeled(result.gray, "2 Gray", size),
        _labeled(result.blurred, "3 Gaussian blur", size),
        _labeled(result.binary, "4 Otsu threshold", size),
        _labeled(result.edges, "5 Canny edges", size),
        _labeled(result.color_mask, "6 HSV color mask", size),
        _labeled(result.contours, "7 Contours", size),
        _labeled(image, "Input for next stage", size),
    ]
    return np.vstack((np.hstack(panels[:4]), np.hstack(panels[4:])))


class ImageWindow:
    """OpenCV 窗口；无 GUI 时自动降级，不影响保存结果。"""

    def __init__(self, name: str, enabled: bool = True) -> None:
        self.name = name
        self.enabled = enabled
        if enabled:
            try:
                cv2.namedWindow(name, cv2.WINDOW_NORMAL)
            except Exception as exc:
                print(f"[OpenCV] 无法创建窗口，将只保存图像：{exc}")
                self.enabled = False

    def show(self, image: np.ndarray) -> bool:
        """显示图像；按 q 或 Esc 返回 False。"""
        if not self.enabled:
            return True
        cv2.imshow(self.name, image)
        key = cv2.waitKey(1) & 0xFF
        return key not in (ord("q"), 27)

    def close(self) -> None:
        if self.enabled:
            cv2.destroyWindow(self.name)


def save_image(path: str | Path, image: np.ndarray) -> Path:
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), image):
        raise OSError(f"图像保存失败：{output}")
    print(f"[OpenCV] 已保存：{output}", flush=True)
    return output
