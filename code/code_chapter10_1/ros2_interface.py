"""第十章 ROS 2 / RViz 接口。

算法本身不依赖 ROS；本文件只负责消息转换、TF、目标订阅和可视化。
"""

from __future__ import annotations

import math
from typing import Sequence

try:
    from .costmap import OccupancyGrid2D
    from .geometry import CircleObstacle, Pose2D, Velocity2D
except ImportError:
    from costmap import OccupancyGrid2D
    from geometry import CircleObstacle, Pose2D, Velocity2D


class NavigationRos2Interface:
    def __init__(self, node_name: str = "chapter10_1_navigation") -> None:
        self.enabled = False
        self.node = None
        self._goal: Pose2D | None = None
        try:
            import rclpy
            from geometry_msgs.msg import Point, PoseStamped, TransformStamped, Twist
            from nav_msgs.msg import OccupancyGrid, Odometry, Path
            from rclpy.node import Node
            from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
            from rosgraph_msgs.msg import Clock
            from sensor_msgs.msg import LaserScan
            from std_msgs.msg import ColorRGBA, String
            from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster
            from visualization_msgs.msg import Marker, MarkerArray

            if not rclpy.ok():
                rclpy.init(args=None)
            self.rclpy = rclpy
            self.PoseStamped = PoseStamped
            self.TransformStamped = TransformStamped
            self.Twist = Twist
            self.OccupancyGrid = OccupancyGrid
            self.Odometry = Odometry
            self.Path = Path
            self.LaserScan = LaserScan
            self.Clock = Clock
            self.Point = Point
            self.ColorRGBA = ColorRGBA
            self.String = String
            self.Marker = Marker
            self.MarkerArray = MarkerArray
            self.node = Node(node_name)

            latched = QoSProfile(depth=1)
            latched.reliability = ReliabilityPolicy.RELIABLE
            latched.durability = DurabilityPolicy.TRANSIENT_LOCAL
            self.map_pub = self.node.create_publisher(OccupancyGrid, "/map", latched)
            self.global_path_pub = self.node.create_publisher(Path, "/plan/global", latched)
            self.local_path_pub = self.node.create_publisher(Path, "/plan/local", 10)
            self.executed_path_pub = self.node.create_publisher(Path, "/plan/executed", 10)
            self.odom_pub = self.node.create_publisher(Odometry, "/odom", 20)
            self.scan_pub = self.node.create_publisher(LaserScan, "/scan", 10)
            self.command_pub = self.node.create_publisher(Twist, "/teaching_cmd_vel", 10)
            self.marker_pub = self.node.create_publisher(MarkerArray, "/navigation/markers", 10)
            self.status_pub = self.node.create_publisher(String, "/navigation/status", 10)
            self.clock_pub = self.node.create_publisher(Clock, "/clock", 10)
            self.tf_pub = TransformBroadcaster(self.node)
            self.static_tf_pub = StaticTransformBroadcaster(self.node)
            self.node.create_subscription(PoseStamped, "/goal_pose", self._goal_callback, 10)
            self.executed_path: list[Pose2D] = []
            self._publish_static_transforms()
            self.enabled = True
            print("[ROS2] RViz 话题已启用；可用 2D Goal Pose 发布 /goal_pose", flush=True)
        except Exception as exc:
            print(f"[ROS2] 接口未启用，导航仍可在 Isaac Sim 中运行：{exc}", flush=True)

    def _goal_callback(self, message) -> None:
        quaternion = message.pose.orientation
        yaw = math.atan2(
            2.0 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y),
            1.0 - 2.0 * (quaternion.y * quaternion.y + quaternion.z * quaternion.z),
        )
        self._goal = Pose2D(message.pose.position.x, message.pose.position.y, yaw)
        print(f"[ROS2] 收到 RViz 目标：({self._goal.x:.2f}, {self._goal.y:.2f}, {yaw:.2f})")

    def take_goal(self) -> Pose2D | None:
        goal, self._goal = self._goal, None
        return goal

    def spin_once(self) -> None:
        if self.enabled:
            self.rclpy.spin_once(self.node, timeout_sec=0.0)

    def publish_map(self, grid: OccupancyGrid2D, sim_time: float) -> None:
        if not self.enabled:
            return
        message = self.OccupancyGrid()
        message.header.frame_id = "map"
        message.header.stamp = self._stamp(sim_time)
        message.info.resolution = grid.resolution
        message.info.width = grid.width
        message.info.height = grid.height
        message.info.origin.position.x = grid.origin_x
        message.info.origin.position.y = grid.origin_y
        message.info.origin.orientation.w = 1.0
        message.data = grid.data.reshape(-1).astype(int).tolist()
        self.map_pub.publish(message)

    def publish_pose(
        self, sim_time: float, pose: Pose2D, velocity: Velocity2D
    ) -> None:
        """先发布仿真时钟、动态 TF 和里程计。

        该函数应以控制频率调用，并且必须早于同时间戳的传感器消息。
        """
        if not self.enabled:
            return
        self._publish_clock(sim_time)
        self._publish_pose_tf_odom(sim_time, pose, velocity)

    def publish_visualization(
        self,
        sim_time: float,
        pose: Pose2D,
        global_path: Sequence[Pose2D],
        local_path: Sequence[Pose2D],
        target: Pose2D | None,
        goal: Pose2D | None,
        dynamic_obstacles: Sequence[CircleObstacle],
        status: str,
    ) -> None:
        """发布路径和标记等低频 RViz 数据，不重复发布 TF。"""
        if not self.enabled:
            return
        self.global_path_pub.publish(self._path_message(global_path, sim_time))
        self.local_path_pub.publish(self._path_message(local_path, sim_time))
        if not self.executed_path or math.hypot(
            pose.x - self.executed_path[-1].x, pose.y - self.executed_path[-1].y
        ) >= 0.04:
            self.executed_path.append(Pose2D(pose.x, pose.y, pose.yaw))
        self.executed_path_pub.publish(self._path_message(self.executed_path, sim_time))
        self.marker_pub.publish(self._markers(sim_time, target, goal, dynamic_obstacles))
        text = self.String()
        text.data = status
        self.status_pub.publish(text)

    def publish_state(
        self,
        sim_time: float,
        pose: Pose2D,
        velocity: Velocity2D,
        global_path: Sequence[Pose2D],
        local_path: Sequence[Pose2D],
        target: Pose2D | None,
        goal: Pose2D | None,
        dynamic_obstacles: Sequence[CircleObstacle],
        status: str,
    ) -> None:
        """兼容旧示例：依次发布位姿和可视化数据。"""
        self.publish_pose(sim_time, pose, velocity)
        self.publish_visualization(
            sim_time, pose, global_path, local_path, target, goal, dynamic_obstacles, status
        )

    def publish_scan(
        self,
        sim_time: float,
        ranges: Sequence[float],
        angle_min: float,
        angle_max: float,
        range_min: float,
        range_max: float,
    ) -> None:
        if not self.enabled:
            return
        scan = self.LaserScan()
        scan.header.frame_id = "base_scan"
        scan.header.stamp = self._stamp(sim_time)
        scan.angle_min = angle_min
        scan.angle_max = angle_max
        scan.angle_increment = (angle_max - angle_min) / max(1, len(ranges) - 1)
        scan.scan_time = 0.1
        # 本例的所有射线由同一时刻、同一位姿计算，并非逐束扫描。
        # 若填写 0.1/N，RViz 会请求“未来 0.1 秒”的 TF，从而产生 extrapolation。
        scan.time_increment = 0.0
        scan.range_min = range_min
        scan.range_max = range_max
        scan.ranges = list(ranges)
        self.scan_pub.publish(scan)

    def publish_command(self, command: Velocity2D) -> None:
        if not self.enabled:
            return
        message = self.Twist()
        message.linear.x = command.vx
        message.linear.y = command.vy
        message.angular.z = command.wz
        self.command_pub.publish(message)

    def close(self) -> None:
        if self.enabled and self.node is not None:
            self.node.destroy_node()
            if self.rclpy.ok():
                self.rclpy.try_shutdown()
            self.enabled = False

    def _publish_clock(self, seconds: float) -> None:
        message = self.Clock()
        message.clock = self._stamp(seconds)
        self.clock_pub.publish(message)

    def _publish_static_transforms(self) -> None:
        """发布不会随时间变化的 map->odom 与 base_link->base_scan。"""
        stamp = self.node.get_clock().now().to_msg()

        map_to_odom = self.TransformStamped()
        map_to_odom.header.frame_id = "map"
        map_to_odom.child_frame_id = "odom"
        map_to_odom.header.stamp = stamp
        map_to_odom.transform.rotation.w = 1.0

        base_to_scan = self.TransformStamped()
        base_to_scan.header.frame_id = "base_link"
        base_to_scan.child_frame_id = "base_scan"
        base_to_scan.header.stamp = stamp
        base_to_scan.transform.translation.z = 0.30
        base_to_scan.transform.rotation.w = 1.0

        self.static_tf_pub.sendTransform([map_to_odom, base_to_scan])

    def _publish_pose_tf_odom(self, seconds: float, pose: Pose2D, velocity: Velocity2D) -> None:
        stamp = self._stamp(seconds)
        quaternion_z, quaternion_w = math.sin(pose.yaw / 2.0), math.cos(pose.yaw / 2.0)
        transform = self.TransformStamped()
        transform.header.frame_id = "odom"
        transform.child_frame_id = "base_link"
        transform.header.stamp = stamp
        transform.transform.translation.x = pose.x
        transform.transform.translation.y = pose.y
        transform.transform.rotation.z = quaternion_z
        transform.transform.rotation.w = quaternion_w
        self.tf_pub.sendTransform(transform)

        odometry = self.Odometry()
        odometry.header.frame_id = "odom"
        odometry.child_frame_id = "base_link"
        odometry.header.stamp = stamp
        odometry.pose.pose.position.x = pose.x
        odometry.pose.pose.position.y = pose.y
        odometry.pose.pose.orientation.z = quaternion_z
        odometry.pose.pose.orientation.w = quaternion_w
        odometry.twist.twist.linear.x = velocity.vx
        odometry.twist.twist.linear.y = velocity.vy
        odometry.twist.twist.angular.z = velocity.wz
        self.odom_pub.publish(odometry)

    def _path_message(self, path: Sequence[Pose2D], seconds: float):
        message = self.Path()
        message.header.frame_id = "map"
        message.header.stamp = self._stamp(seconds)
        for pose in path:
            item = self.PoseStamped()
            item.header = message.header
            item.pose.position.x = pose.x
            item.pose.position.y = pose.y
            item.pose.orientation.z = math.sin(pose.yaw / 2.0)
            item.pose.orientation.w = math.cos(pose.yaw / 2.0)
            message.poses.append(item)
        return message

    def _markers(
        self,
        seconds: float,
        target: Pose2D | None,
        goal: Pose2D | None,
        dynamic_obstacles: Sequence[CircleObstacle],
    ):
        markers = self.MarkerArray()
        entries = []
        if goal is not None:
            entries.append(("goal", goal.x, goal.y, 0.22, (0.1, 0.9, 0.2, 0.95)))
        if target is not None:
            entries.append(("lookahead", target.x, target.y, 0.14, (1.0, 0.9, 0.1, 0.95)))
        for index, obstacle in enumerate(dynamic_obstacles):
            entries.append((f"dynamic_{index}", obstacle.x, obstacle.y, obstacle.radius, (0.1, 0.4, 1.0, 0.75)))
        for index, (name, x, y, radius, color) in enumerate(entries):
            marker = self.Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self._stamp(seconds)
            marker.ns = "chapter10_1"
            marker.id = index
            marker.type = self.Marker.CYLINDER
            marker.action = self.Marker.ADD
            marker.pose.position.x = x
            marker.pose.position.y = y
            marker.pose.position.z = 0.08
            marker.pose.orientation.w = 1.0
            marker.scale.x = 2.0 * radius
            marker.scale.y = 2.0 * radius
            marker.scale.z = 0.16
            marker.color.r, marker.color.g, marker.color.b, marker.color.a = color
            marker.lifetime.sec = 1
            markers.markers.append(marker)
        return markers

    def _stamp(self, seconds: float):
        stamp = self.node.get_clock().now().to_msg()
        stamp.sec = int(seconds)
        stamp.nanosec = int((seconds - int(seconds)) * 1e9)
        return stamp
