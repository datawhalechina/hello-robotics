"""Isaac Sim 与 Nav2 的最小 ROS 2 桥接器。"""

from __future__ import annotations

import math
import time

try:
    from .config import NavigationConfig, Pose2D, SensorConfig, Velocity2D
except ImportError:
    from config import NavigationConfig, Pose2D, SensorConfig, Velocity2D


def make_standoff_goal(robot: Pose2D, target_xy, distance: float) -> Pose2D:
    """在目标前方保留安全距离，并让机器人最终朝向目标。"""
    target_x, target_y = float(target_xy[0]), float(target_xy[1])
    dx, dy = target_x - robot.x, target_y - robot.y
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return Pose2D(robot.x, robot.y, robot.yaw)
    ratio = min(max(0.0, distance), max(0.0, length - 0.05)) / length
    goal_x, goal_y = target_x - dx * ratio, target_y - dy * ratio
    return Pose2D(goal_x, goal_y, math.atan2(target_y - goal_y, target_x - goal_x))


class Nav2Bridge:
    """发布传感器/里程计，接收 cmd_vel，并异步发送 NavigateToPose。"""

    def __init__(
        self,
        sensor_config: SensorConfig = SensorConfig(),
        nav_config: NavigationConfig = NavigationConfig(),
    ) -> None:
        import rclpy
        from action_msgs.msg import GoalStatus
        from geometry_msgs.msg import PoseStamped, TransformStamped, Twist
        from nav2_msgs.action import NavigateToPose
        from nav_msgs.msg import Odometry
        from rclpy.action import ActionClient
        from rosgraph_msgs.msg import Clock
        from sensor_msgs.msg import LaserScan
        from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster

        self.rclpy = rclpy
        self.GoalStatus = GoalStatus
        self.PoseStamped = PoseStamped
        self.TransformStamped = TransformStamped
        self.NavigateToPose = NavigateToPose
        self.Odometry = Odometry
        self.Clock = Clock
        self.LaserScan = LaserScan
        self.sensor_config = sensor_config
        self.nav_config = nav_config

        if not rclpy.ok():
            rclpy.init(args=None)
        self.node = rclpy.create_node("chapter13_g2_vlm_bridge")
        self.clock_pub = self.node.create_publisher(Clock, "/clock", 10)
        self.odom_pub = self.node.create_publisher(Odometry, "/odom", 10)
        self.scan_pub = self.node.create_publisher(LaserScan, "/scan", 10)
        self.tf_pub = TransformBroadcaster(self.node)
        self.static_tf_pub = StaticTransformBroadcaster(self.node)
        self.node.create_subscription(Twist, "/cmd_vel", self._cmd_callback, 10)
        self.action = ActionClient(self.node, NavigateToPose, "/navigate_to_pose")

        self.command = Velocity2D()
        self._send_future = None
        self._goal_handle = None
        self._result_future = None
        self.result_status = None
        self.distance_remaining = math.inf
        self._last_feedback_time = -math.inf
        self._publish_static_transforms()

    def _cmd_callback(self, message) -> None:
        self.command = Velocity2D(
            float(message.linear.x), float(message.linear.y), float(message.angular.z)
        )

    def spin_once(self) -> Velocity2D:
        self.rclpy.spin_once(self.node, timeout_sec=0.0)
        self._update_action_state()
        return self.command

    def nav2_ready(self) -> bool:
        return bool(self.action.server_is_ready() or self.action.wait_for_server(timeout_sec=0.0))

    def send_goal(self, goal: Pose2D) -> None:
        if not self.nav2_ready():
            raise RuntimeError("Nav2 navigate_to_pose action 尚未就绪")
        message = self.NavigateToPose.Goal()
        message.pose = self.PoseStamped()
        message.pose.header.frame_id = "map"
        message.pose.header.stamp = self.node.get_clock().now().to_msg()
        message.pose.pose.position.x = goal.x
        message.pose.pose.position.y = goal.y
        message.pose.pose.orientation.z = math.sin(goal.yaw / 2.0)
        message.pose.pose.orientation.w = math.cos(goal.yaw / 2.0)
        self.result_status = None
        self._send_future = self.action.send_goal_async(
            message, feedback_callback=self._feedback
        )
        print(
            f"[Nav2] 发送目标：x={goal.x:.2f}, y={goal.y:.2f}, yaw={goal.yaw:.2f}",
            flush=True,
        )

    @property
    def navigation_done(self) -> bool:
        return self.result_status is not None

    @property
    def navigation_succeeded(self) -> bool:
        return self.result_status == self.GoalStatus.STATUS_SUCCEEDED

    def _update_action_state(self) -> None:
        if self._send_future is not None and self._send_future.done():
            self._goal_handle = self._send_future.result()
            self._send_future = None
            if self._goal_handle is None or not self._goal_handle.accepted:
                print("[Nav2] 目标被拒绝", flush=True)
                self.result_status = self.GoalStatus.STATUS_ABORTED
            else:
                print("[Nav2] 目标已接受", flush=True)
                self._result_future = self._goal_handle.get_result_async()
        if self._result_future is not None and self._result_future.done():
            result = self._result_future.result()
            self.result_status = (
                result.status if result is not None else self.GoalStatus.STATUS_ABORTED
            )
            self._result_future = None
            print(f"[Nav2] 导航结束，状态码：{self.result_status}", flush=True)

    def _feedback(self, feedback_message) -> None:
        self.distance_remaining = float(feedback_message.feedback.distance_remaining)
        now = time.monotonic()
        if now - self._last_feedback_time >= 1.0:
            self._last_feedback_time = now
            print(f"[Nav2] 剩余距离：{self.distance_remaining:.2f} m", flush=True)

    def publish(self, sim_time: float, pose: Pose2D, velocity: Velocity2D, ranges) -> None:
        stamp = self._stamp(sim_time)
        clock = self.Clock()
        clock.clock = stamp
        self.clock_pub.publish(clock)

        qz, qw = math.sin(pose.yaw / 2.0), math.cos(pose.yaw / 2.0)
        transform = self.TransformStamped()
        transform.header.frame_id = "odom"
        transform.child_frame_id = "base_link"
        transform.header.stamp = stamp
        transform.transform.translation.x = pose.x
        transform.transform.translation.y = pose.y
        transform.transform.rotation.z = qz
        transform.transform.rotation.w = qw
        self.tf_pub.sendTransform(transform)

        odometry = self.Odometry()
        odometry.header.frame_id = "odom"
        odometry.child_frame_id = "base_link"
        odometry.header.stamp = stamp
        odometry.pose.pose.position.x = pose.x
        odometry.pose.pose.position.y = pose.y
        odometry.pose.pose.orientation.z = qz
        odometry.pose.pose.orientation.w = qw
        odometry.twist.twist.linear.x = velocity.vx
        odometry.twist.twist.linear.y = velocity.vy
        odometry.twist.twist.angular.z = velocity.wz
        odometry.pose.covariance[0] = 0.0025
        odometry.pose.covariance[7] = 0.0025
        odometry.pose.covariance[35] = 0.004
        self.odom_pub.publish(odometry)

        scan = self.LaserScan()
        scan.header.frame_id = "base_scan"
        scan.header.stamp = stamp
        scan.angle_min = -math.pi
        scan.angle_max = math.pi
        scan.angle_increment = 2.0 * math.pi / max(1, len(ranges))
        scan.scan_time = 1.0 / self.nav_config.lidar_hz
        scan.time_increment = scan.scan_time / max(1, len(ranges))
        scan.range_min = self.sensor_config.lidar_min_range
        scan.range_max = self.sensor_config.lidar_max_range
        scan.ranges = [float(value) for value in ranges]
        self.scan_pub.publish(scan)

    def _publish_static_transforms(self) -> None:
        transforms = []
        map_to_odom = self.TransformStamped()
        map_to_odom.header.frame_id = "map"
        map_to_odom.child_frame_id = "odom"
        map_to_odom.transform.rotation.w = 1.0
        transforms.append(map_to_odom)

        base_to_scan = self.TransformStamped()
        base_to_scan.header.frame_id = "base_link"
        base_to_scan.child_frame_id = "base_scan"
        base_to_scan.transform.translation.z = 0.34
        base_to_scan.transform.rotation.w = 1.0
        transforms.append(base_to_scan)
        self.static_tf_pub.sendTransform(transforms)

    def _stamp(self, seconds: float):
        stamp = self.node.get_clock().now().to_msg()
        stamp.sec = int(seconds)
        stamp.nanosec = int((seconds - int(seconds)) * 1e9)
        return stamp

    def close(self) -> None:
        self.node.destroy_node()
        if self.rclpy.ok():
            self.rclpy.try_shutdown()
