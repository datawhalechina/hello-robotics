"""G2 RGB 相机的创建、采集与颜色格式转换。"""

import numpy as np

try:
    from .config import CameraConfig
except ImportError:
    from config import CameraConfig


def rgba_to_rgb(image: np.ndarray) -> np.ndarray:
    """把 Isaac Sim 的 RGB/RGBA 图像统一转换为 uint8 RGB。"""
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] < 3:
        raise ValueError(f"期望 HxWx3/4 图像，实际形状为 {array.shape}")

    rgb = array[..., :3]
    if np.issubdtype(rgb.dtype, np.floating):
        scale = 255.0 if rgb.size and float(np.nanmax(rgb)) <= 1.0 else 1.0
        rgb = np.nan_to_num(rgb, nan=0.0, posinf=255.0, neginf=0.0)
        rgb = np.clip(rgb * scale, 0.0, 255.0).astype(np.uint8)
    else:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8, copy=False)
    return np.ascontiguousarray(rgb)


def rgb_to_bgr(image_rgb: np.ndarray) -> np.ndarray:
    """RGB 转 OpenCV 常用的 BGR。"""
    import cv2

    return cv2.cvtColor(rgba_to_rgb(image_rgb), cv2.COLOR_RGB2BGR)


def rgba_to_bgr(image: np.ndarray) -> np.ndarray:
    """Isaac Sim RGB/RGBA 图像直接转 BGR。"""
    return rgb_to_bgr(image)


class G2RGBCamera:
    """对 Isaac Sim Camera 的轻量封装。"""

    def __init__(self, config: CameraConfig, frequency: int = 30) -> None:
        if frequency <= 0:
            raise ValueError("相机 frequency 必须大于 0")

        from isaacsim.sensors.camera import Camera

        self.config = config
        self.sensor = Camera(
            prim_path=config.prim_path,
            name=f"chapter7_{config.name}_camera",
            frequency=frequency,
            resolution=config.resolution,
        )
        self.sensor.initialize()
        self.sensor.add_rgb_to_frame()
        print(
            f"[G2RGBCamera] 已初始化 {config.name}："
            f"{config.prim_path}, resolution={config.resolution}",
            flush=True,
        )

    def capture_rgb(self) -> np.ndarray | None:
        rgba = self.sensor.get_rgba()
        if rgba is None or np.asarray(rgba).size == 0:
            return None
        return rgba_to_rgb(rgba)

    def capture_bgr(self) -> np.ndarray | None:
        rgb = self.capture_rgb()
        if rgb is None:
            return None
        return rgb_to_bgr(rgb)

    def wait_for_bgr(self, simulation, max_steps: int = 60) -> np.ndarray:
        """推进仿真直到取得第一帧，失败时给出明确错误。"""
        for _ in range(max_steps):
            simulation.step(render=True)
            image = self.capture_bgr()
            if image is not None:
                return image
        raise RuntimeError("相机在等待期间没有返回有效图像，请检查相机 prim 路径")
