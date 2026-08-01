"""使用 ROS 2 Marker 在 RViz 显示障碍物、原始路径、优化路径和机械臂。"""

from typing import Sequence

import numpy as np

try:
    from .arm_model import G2PlanningModel
    from .config import BoxObstacle
except ImportError:
    from arm_model import G2PlanningModel
    from config import BoxObstacle


class PlanningRvizPublisher:
    def __init__(self, model: G2PlanningModel, enabled: bool = True) -> None:
        self.enabled = enabled
        self.model = model
        self.node = None
        if not enabled:
            return
        try:
            import rclpy
            from visualization_msgs.msg import Marker, MarkerArray
            from rclpy.node import Node
            from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

            if not rclpy.ok():
                rclpy.init(args=None)
            self.rclpy = rclpy
            self.Marker = Marker
            self.MarkerArray = MarkerArray
            self.node = Node("chapter11_1_planning_visualizer")
            # TRANSIENT_LOCAL 会保存最近的可视化消息：即使 RViz 稍晚打开，
            # 也仍能看到场景、轨迹和头部深度体素。
            qos = QoSProfile(
                depth=10,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            )
            self.publisher = self.node.create_publisher(
                MarkerArray, "/chapter11/planning_markers", qos
            )
            self.octomap_publisher = self.node.create_publisher(
                Marker, "/chapter11/octomap_voxels", qos
            )
        except Exception as exc:
            self.enabled = False
            print(f"[RViz] ROS 2 不可用，仅在 Isaac Sim 中显示：{exc}", flush=True)

    @staticmethod
    def _to_world(point: Sequence[float]) -> np.ndarray:
        """把 G2 arm_base_link 坐标转换为 RViz 的 ROS Z-up world 坐标。

        G2 规划坐标：x 前、y 下、z 左；RViz world：x 前、y 左、z 上。
        等价于绕 x 轴旋转 -90 度：world = [x, z, -y]。
        """
        x, y, z = np.asarray(point, dtype=np.float64)
        return np.array([x, z, -y], dtype=np.float64)

    def _marker(self, marker_id, marker_type, color, scale, namespace):
        from visualization_msgs.msg import Marker

        marker = Marker()
        # Marker 直接发布到 RViz 固定坐标系，避免系统中其他 robot_state_publisher
        # 对 arm_base_link 重复指定父坐标系时造成 TF 树冲突。
        marker.header.frame_id = "world"
        marker.ns = namespace
        marker.id = marker_id
        marker.type = marker_type
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x, marker.scale.y, marker.scale.z = scale
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = (*color, 0.92)
        return marker

    def publish_scene(self, obstacles: Sequence[BoxObstacle]) -> None:
        if not self.enabled:
            return
        from visualization_msgs.msg import Marker

        array = self.MarkerArray()
        for index, box in enumerate(obstacles):
            # 绕 x 轴旋转后 y/z 轴互换；盒子仍可用轴对齐 CUBE 表示。
            world_size = (box.size[0], box.size[2], box.size[1])
            marker = self._marker(index, Marker.CUBE, box.color, world_size, "scene")
            world_center = self._to_world(box.center)
            marker.pose.position.x, marker.pose.position.y, marker.pose.position.z = world_center
            array.markers.append(marker)
        self.publisher.publish(array)
        self.rclpy.spin_once(self.node, timeout_sec=0.0)

    def publish_octomap(self, voxel_centers: np.ndarray, voxel_size: float) -> None:
        """用 CUBE_LIST 显示头部深度生成的占用体素。

        自编规划器内部直接使用体素 AABB，不依赖 MoveIt/octomap 库。这里把同一批
        未膨胀体素发布到 RViz，因此显示结果和碰撞检测器使用的障碍地图一致。
        """
        if not self.enabled:
            return
        from geometry_msgs.msg import Point

        marker = self._marker(
            0,
            self.Marker.CUBE_LIST,
            (0.20, 0.65, 0.95),
            (voxel_size, voxel_size, voxel_size),
            "head_depth_octomap",
        )
        marker.color.a = 0.62
        for center in np.asarray(voxel_centers, dtype=np.float64):
            point = Point()
            point.x, point.y, point.z = self._to_world(center)
            marker.points.append(point)
        self.octomap_publisher.publish(marker)
        self.rclpy.spin_once(self.node, timeout_sec=0.0)

    def publish_paths(
        self,
        raw_path: Sequence[Sequence[float]],
        optimized_path: Sequence[Sequence[float]],
    ) -> None:
        if not self.enabled:
            return
        from geometry_msgs.msg import Point
        from visualization_msgs.msg import Marker

        array = self.MarkerArray()
        for marker_id, (path, color, namespace) in enumerate(
            ((raw_path, (0.95, 0.25, 0.08), "raw_path"),
             (optimized_path, (0.05, 0.85, 0.95), "optimized_path")),
            start=100,
        ):
            marker = self._marker(marker_id, Marker.LINE_STRIP, color, (0.014, 0.0, 0.0), namespace)
            for q in path:
                position = self._to_world(self.model.forward(q).position)
                point = Point()
                point.x, point.y, point.z = position
                marker.points.append(point)
            array.markers.append(marker)
        self.publisher.publish(array)
        self.rclpy.spin_once(self.node, timeout_sec=0.0)

    def publish_arm(self, commanded: np.ndarray, actual: np.ndarray) -> None:
        if not self.enabled:
            return
        from geometry_msgs.msg import Point
        from visualization_msgs.msg import Marker

        array = self.MarkerArray()
        for marker_id, (q, color, namespace) in enumerate(
            ((commanded, (0.1, 0.75, 1.0), "commanded_arm"),
             (actual, (0.95, 0.95, 0.95), "actual_arm")),
            start=200,
        ):
            marker = self._marker(marker_id, Marker.LINE_STRIP, color, (0.035, 0.0, 0.0), namespace)
            for xyz in self.model.link_points(q)[1:]:
                point = Point()
                point.x, point.y, point.z = self._to_world(xyz)
                marker.points.append(point)
            array.markers.append(marker)
        self.publisher.publish(array)
        self.rclpy.spin_once(self.node, timeout_sec=0.0)

    def close(self) -> None:
        if self.node is not None:
            self.node.destroy_node()
