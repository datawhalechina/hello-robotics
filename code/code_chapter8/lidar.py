"""在 G2 base_link 下添加双 RTX LiDAR，并读取/发布点云。"""

from dataclasses import dataclass

import numpy as np

try:
    from .config import LIDAR_MOUNTS, LidarConfig, LidarMount
    from .pointcloud import PointCloud, PointCloudProcessor
except ImportError:
    from config import LIDAR_MOUNTS, LidarConfig, LidarMount
    from pointcloud import PointCloud, PointCloudProcessor


SCAN_BUFFER_ANNOTATOR = "IsaacCreateRTXLidarScanBuffer"


@dataclass
class LidarHandle:
    mount: LidarMount
    sensor: object


class DualRtxLidar:
    """双 OS1 的最小封装。

    创建顺序只有四步：LidarRtx -> initialize -> attach_annotator -> 逐帧读取。
    """

    def __init__(
        self,
        config: LidarConfig = LidarConfig(),
        mounts=LIDAR_MOUNTS,
        show_visual: bool = False,
        publish_raw_ros: bool = True,
    ) -> None:
        from isaacsim.sensors.rtx import LidarRtx

        self.config = config
        self.processor = PointCloudProcessor(config)
        self.handles: list[LidarHandle] = []
        self._writers = []
        for mount in mounts:
            sensor = LidarRtx(
                prim_path=mount.prim_path,
                name=mount.name,
                config_file_name=config.model,
                variant=config.profile,
                translation=np.asarray(mount.translation, dtype=np.float64),
                orientation=np.asarray(mount.orientation_wxyz, dtype=np.float64),
            )
            sensor.initialize()
            sensor.attach_annotator(
                SCAN_BUFFER_ANNOTATOR,
                outputIntensity=True,
                enablePerFrameOutput=True,
            )
            if show_visual:
                sensor.enable_visualization()
            self.handles.append(LidarHandle(mount, sensor))

        if publish_raw_ros:
            self._attach_raw_ros_writers()
        if show_visual:
            self._attach_debug_writers()

    def capture(self) -> tuple[PointCloud, dict[str, PointCloud]]:
        clouds = {}
        for handle in self.handles:
            frame = handle.sensor.get_current_frame()
            scan = frame.get(SCAN_BUFFER_ANNOTATOR, {})
            clouds[handle.mount.name] = self.processor.process(scan, handle.mount)
        return PointCloud.concatenate(list(clouds.values())), clouds

    def _attach_raw_ros_writers(self) -> None:
        try:
            import omni.replicator.core as rep

            for handle in self.handles:
                writer = rep.writers.get("RtxLidarROS2PublishPointCloud")
                writer.initialize(
                    topicName=handle.mount.topic.lstrip("/"),
                    frameId=handle.mount.frame_id,
                )
                writer.attach([handle.sensor.get_render_product_path()])
                self._writers.append(writer)
        except Exception as exc:
            print(f"[DualRtxLidar] 原始 ROS2 点云发布未启用：{exc}", flush=True)

    def _attach_debug_writers(self) -> None:
        try:
            import omni.replicator.core as rep

            for handle in self.handles:
                writer = rep.writers.get("RtxLidarDebugDrawPointCloud")
                writer.attach([handle.sensor.get_render_product_path()])
                self._writers.append(writer)
        except Exception as exc:
            print(f"[DualRtxLidar] 雷达调试绘制未启用：{exc}", flush=True)
