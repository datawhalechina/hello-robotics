"""发布双雷达点云、地图、轨迹和完整 G2 TF 树，供 RViz2 显示。"""

import math

import numpy as np

try:
    from .config import LIDAR_MOUNTS
    from .geometry import Pose2D
except ImportError:
    from config import LIDAR_MOUNTS
    from geometry import Pose2D


class MappingRos2Publisher:
    """集中管理第八章 ROS2/RViz 输出。

    TF 结构与 examples 中的完整案例保持一致：
    map -> base_link -> G2 各关节/连杆，并额外发布 base_link -> 双 LiDAR。
    """

    def __init__(self, map_frame: str = "map", base_frame: str = "base_link") -> None:
        self.map_frame = map_frame
        self.base_frame = base_frame
        self.enabled = False
        self.node = None
        self.base_prim = None
        self.robot_tf_links = []
        self.UsdGeom = None
        try:
            import rclpy
            from builtin_interfaces.msg import Time
            from geometry_msgs.msg import PoseStamped, TransformStamped
            from nav_msgs.msg import OccupancyGrid, Path
            from rclpy.node import Node
            from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
            from sensor_msgs.msg import PointCloud2, PointField
            from std_msgs.msg import Header
            from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster

            if not rclpy.ok():
                rclpy.init(args=None)
            self.rclpy = rclpy
            self.Time = Time
            self.PoseStamped = PoseStamped
            self.TransformStamped = TransformStamped
            self.OccupancyGrid = OccupancyGrid
            self.Path = Path
            self.PointCloud2 = PointCloud2
            self.PointField = PointField
            self.Header = Header
            self.node = Node("chapter8_mapping_publisher")

            latched = QoSProfile(depth=1)
            latched.reliability = ReliabilityPolicy.RELIABLE
            latched.durability = DurabilityPolicy.TRANSIENT_LOCAL
            self.cloud_pub = self.node.create_publisher(PointCloud2, "/map_cloud", latched)
            self.fused_pub = self.node.create_publisher(PointCloud2, "/lidar/fused", 1)
            self.grid_pub = self.node.create_publisher(OccupancyGrid, "/map", latched)
            self.raw_path_pub = self.node.create_publisher(Path, "/trajectory/odom", latched)
            self.optimized_path_pub = self.node.create_publisher(Path, "/trajectory/optimized", latched)
            self.tf_pub = TransformBroadcaster(self.node)
            self.static_tf_pub = StaticTransformBroadcaster(self.node)
            self.enabled = True
            self.publish_lidar_static_tf(sim_time=0.0)
            print(
                "[ROS2] 点云: /lidar/left/points /lidar/right/points /lidar/fused /map_cloud",
                flush=True,
            )
            print("[ROS2] TF: map -> base_link -> G2 links + lidar_left/lidar_right", flush=True)
        except Exception as exc:
            print(f"[ROS2] 发布器未启用，仍会保存离线地图：{exc}", flush=True)

    def register_robot_tf_tree(self, usd_stage, robot_prim_path: str) -> None:
        """从 G2 USD 的 Joint 关系自动注册各连杆 TF。"""
        if not self.enabled or usd_stage is None:
            return
        from pxr import UsdGeom, UsdPhysics

        self.UsdGeom = UsdGeom
        robot_prim_path = robot_prim_path.rstrip("/")
        robot_prefix = robot_prim_path + "/"
        base_path = f"{robot_prim_path}/{self.base_frame}"
        self.base_prim = usd_stage.GetPrimAtPath(base_path)
        if not self.base_prim or not self.base_prim.IsValid():
            self.base_prim = None
            print(f"[ROS2 TF] 未找到 {base_path}，map -> base_link 使用传入位姿", flush=True)

        self.robot_tf_links = []
        seen_children = {self.base_frame, *(mount.frame_id for mount in LIDAR_MOUNTS)}
        for prim in usd_stage.Traverse():
            if not prim.IsA(UsdPhysics.Joint):
                continue
            joint = UsdPhysics.Joint(prim)
            body1_targets = joint.GetBody1Rel().GetTargets()
            if not body1_targets:
                continue
            child_prim = usd_stage.GetPrimAtPath(body1_targets[0])
            if not child_prim or not child_prim.IsValid():
                continue
            child_path = str(child_prim.GetPath())
            if child_path != robot_prim_path and not child_path.startswith(robot_prefix):
                continue

            child_frame = child_prim.GetName()
            if child_frame in seen_children:
                continue
            body0_targets = joint.GetBody0Rel().GetTargets()
            if body0_targets:
                parent_prim = usd_stage.GetPrimAtPath(body0_targets[0])
                if not parent_prim or not parent_prim.IsValid():
                    continue
                parent_frame = parent_prim.GetName()
            else:
                parent_prim = None
                parent_frame = self.map_frame
            if parent_frame == child_frame:
                continue
            self.robot_tf_links.append((parent_frame, child_frame, parent_prim, child_prim))
            seen_children.add(child_frame)

        print(f"[ROS2 TF] 已注册 {len(self.robot_tf_links)} 个 G2 关节/连杆坐标系", flush=True)

    def publish_lidar_static_tf(self, sim_time: float | None = None) -> None:
        if not self.enabled:
            return
        stamp = self._stamp(sim_time)
        transforms = []
        for mount in LIDAR_MOUNTS:
            msg = self.TransformStamped()
            msg.header.stamp = stamp
            msg.header.frame_id = self.base_frame
            msg.child_frame_id = mount.frame_id
            msg.transform.translation.x = float(mount.translation[0])
            msg.transform.translation.y = float(mount.translation[1])
            msg.transform.translation.z = float(mount.translation[2])
            w, x, y, z = mount.orientation_wxyz
            msg.transform.rotation.w = float(w)
            msg.transform.rotation.x = float(x)
            msg.transform.rotation.y = float(y)
            msg.transform.rotation.z = float(z)
            transforms.append(msg)
        self.static_tf_pub.sendTransform(transforms)
        self.rclpy.spin_once(self.node, timeout_sec=0.0)

    def publish_dynamic_tree(self, pose: Pose2D, sim_time: float | None = None) -> None:
        """发布 map -> base_link 以及 G2 当前各关节/连杆 TF。"""
        if not self.enabled:
            return
        stamp = self._stamp(sim_time)
        transforms = [self._base_transform(stamp, pose)]
        for parent_frame, child_frame, parent_prim, child_prim in self.robot_tf_links:
            transform = self._relative_prim_transform(
                stamp, parent_frame, child_frame, parent_prim, child_prim
            )
            if transform is not None:
                transforms.append(transform)
        self.tf_pub.sendTransform(transforms)
        self.rclpy.spin_once(self.node, timeout_sec=0.0)

    def publish_base_tf(self, pose: Pose2D, sim_time: float | None = None) -> None:
        """兼容旧调用；现在会同时发布完整机器人 TF 树。"""
        self.publish_dynamic_tree(pose, sim_time)

    def publish_fused_cloud(self, xyzi: np.ndarray, sim_time: float | None = None) -> None:
        if self.enabled:
            self.fused_pub.publish(self._pointcloud_message(xyzi, self.base_frame, sim_time))

    def publish_map_cloud(self, xyzi: np.ndarray, sim_time: float | None = None) -> None:
        if self.enabled:
            self.cloud_pub.publish(self._pointcloud_message(xyzi, self.map_frame, sim_time))

    def publish_grid(self, mapper, sim_time: float | None = None) -> None:
        if not self.enabled or mapper is None:
            return
        _, data, width, height, origin = mapper.to_arrays()
        msg = self.OccupancyGrid()
        msg.header = self._header(self.map_frame, sim_time)
        msg.info.resolution = float(mapper.config.resolution)
        msg.info.width = int(width)
        msg.info.height = int(height)
        msg.info.origin.position.x = float(origin[0])
        msg.info.origin.position.y = float(origin[1])
        msg.info.origin.orientation.w = 1.0
        msg.data = data.astype(np.int8).tolist()
        self.grid_pub.publish(msg)

    def publish_paths(
        self,
        raw: list[Pose2D],
        optimized: list[Pose2D],
        sim_time: float | None = None,
    ) -> None:
        if not self.enabled:
            return
        self.raw_path_pub.publish(self._path_message(raw, sim_time))
        self.optimized_path_pub.publish(self._path_message(optimized, sim_time))

    def _base_transform(self, stamp, pose: Pose2D):
        msg = self.TransformStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = self.map_frame
        msg.child_frame_id = self.base_frame
        msg.transform.translation.x = float(pose.x)
        msg.transform.translation.y = float(pose.y)
        msg.transform.rotation.z = math.sin(pose.yaw / 2.0)
        msg.transform.rotation.w = math.cos(pose.yaw / 2.0)
        return msg

    def _relative_prim_transform(self, stamp, parent_frame, child_frame, parent_prim, child_prim):
        if self.UsdGeom is None or child_prim is None or not child_prim.IsValid():
            return None
        try:
            matrix = self.UsdGeom.Xformable(child_prim).ComputeLocalToWorldTransform(0.0)
            if parent_prim is not None and parent_prim.IsValid():
                parent_matrix = self.UsdGeom.Xformable(parent_prim).ComputeLocalToWorldTransform(0.0)
                matrix = matrix * parent_matrix.GetInverse()
            translation = matrix.ExtractTranslation()
            orientation = matrix.ExtractRotationQuat()
            msg = self.TransformStamped()
            msg.header.stamp = stamp
            msg.header.frame_id = parent_frame
            msg.child_frame_id = child_frame
            msg.transform.translation.x = float(translation[0])
            msg.transform.translation.y = float(translation[1])
            msg.transform.translation.z = float(translation[2])
            msg.transform.rotation.x = float(orientation.imaginary[0])
            msg.transform.rotation.y = float(orientation.imaginary[1])
            msg.transform.rotation.z = float(orientation.imaginary[2])
            msg.transform.rotation.w = float(orientation.real)
            return msg
        except Exception:
            return None

    def _pointcloud_message(self, xyzi: np.ndarray, frame_id: str, sim_time: float | None):
        points = np.asarray(xyzi, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] < 3:
            points = np.empty((0, 4), dtype=np.float32)
        elif points.shape[1] == 3:
            points = np.column_stack((points, np.zeros((len(points),), dtype=np.float32)))
        else:
            points = np.ascontiguousarray(points[:, :4], dtype=np.float32)
        msg = self.PointCloud2()
        msg.header = self._header(frame_id, sim_time)
        msg.height = 1
        msg.width = int(len(points))
        msg.is_dense = False
        msg.is_bigendian = False
        msg.point_step = 16
        msg.row_step = 16 * msg.width
        msg.fields = [
            self.PointField(name="x", offset=0, datatype=self.PointField.FLOAT32, count=1),
            self.PointField(name="y", offset=4, datatype=self.PointField.FLOAT32, count=1),
            self.PointField(name="z", offset=8, datatype=self.PointField.FLOAT32, count=1),
            self.PointField(name="intensity", offset=12, datatype=self.PointField.FLOAT32, count=1),
        ]
        msg.data = points.tobytes()
        return msg

    def _path_message(self, poses: list[Pose2D], sim_time: float | None):
        msg = self.Path()
        msg.header = self._header(self.map_frame, sim_time)
        for pose in poses:
            item = self.PoseStamped()
            item.header = msg.header
            item.pose.position.x = float(pose.x)
            item.pose.position.y = float(pose.y)
            item.pose.orientation.z = math.sin(pose.yaw / 2.0)
            item.pose.orientation.w = math.cos(pose.yaw / 2.0)
            msg.poses.append(item)
        return msg

    def _header(self, frame_id: str, sim_time: float | None):
        header = self.Header()
        header.stamp = self._stamp(sim_time)
        header.frame_id = frame_id
        return header

    def _stamp(self, sim_time: float | None):
        if sim_time is None:
            return self.node.get_clock().now().to_msg()
        seconds = max(0.0, float(sim_time))
        whole = int(seconds)
        stamp = self.Time()
        stamp.sec = whole
        stamp.nanosec = min(999_999_999, int((seconds - whole) * 1e9))
        return stamp

    def close(self) -> None:
        if self.node is not None:
            self.node.destroy_node()
            self.node = None
