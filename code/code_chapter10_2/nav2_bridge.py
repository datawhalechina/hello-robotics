"""示例 2A：Isaac Sim 与 ROS 2 Nav2 的最小移动底盘桥接器。

发布：/clock、/odom、/scan、map->odom->base_link->base_scan TF
订阅：/cmd_vel（Nav2 velocity_smoother 输出）

本文件只负责仿真和接口，不实现规划算法；规划、控制、恢复由 Nav2 完成。
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import math
from pathlib import Path
import sys

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from code_chapter4.base_controller import G2BaseController
from code_chapter4.config import ControlLimits, RobotGeometry
from code_chapter4.kinematics import SwerveKinematics

try:
    from .build_map import build_navigation_map
    from .config import SimulationConfig
    from .config import Velocity2D
    from .simulation import G2NavigationSimulation
except ImportError:
    from build_map import build_navigation_map
    from config import SimulationConfig
    from config import Velocity2D
    from simulation import G2NavigationSimulation


class Nav2IsaacBridge:
    def __init__(self) -> None:
        import rclpy
        from geometry_msgs.msg import TransformStamped, Twist
        from nav_msgs.msg import Odometry
        from rclpy.node import Node
        from rosgraph_msgs.msg import Clock
        from sensor_msgs.msg import LaserScan
        from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster

        if not rclpy.ok():
            rclpy.init(args=None)
        self.rclpy = rclpy
        self.TransformStamped = TransformStamped
        self.Odometry = Odometry
        self.Clock = Clock
        self.LaserScan = LaserScan
        self.node = Node("chapter10_2_isaac_nav2_bridge")
        self.odom_pub = self.node.create_publisher(Odometry, "/odom", 30)
        self.scan_pub = self.node.create_publisher(LaserScan, "/scan", 10)
        self.clock_pub = self.node.create_publisher(Clock, "/clock", 20)
        self.tf_pub = TransformBroadcaster(self.node)
        self.static_tf_pub = StaticTransformBroadcaster(self.node)
        self.node.create_subscription(Twist, "/cmd_vel", self._command_callback, 10)
        self.command = Velocity2D()
        self.sim_time = 0.0
        self.last_command_time = -math.inf
        self._publish_static_transforms()
        print("[Nav2 Bridge] 订阅 /cmd_vel，发布 /odom、/scan、/clock 和 TF", flush=True)

    def _command_callback(self, message) -> None:
        self.command = Velocity2D(message.linear.x, message.linear.y, message.angular.z)
        self.last_command_time = self.sim_time

    def spin_once(self, sim_time: float) -> Velocity2D:
        self.sim_time = sim_time
        self.rclpy.spin_once(self.node, timeout_sec=0.0)
        if sim_time - self.last_command_time > 0.5:
            return Velocity2D()
        return self.command

    def publish(self, sim_time, pose, velocity, ranges, angle_min, angle_max) -> None:
        stamp = self._stamp(sim_time)
        clock = self.Clock()
        clock.clock = stamp
        self.clock_pub.publish(clock)

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
        # 仿真真值里程计仍给出小的非零协方差，符合消息语义。
        odometry.pose.covariance[0] = 0.0025
        odometry.pose.covariance[7] = 0.0025
        odometry.pose.covariance[35] = 0.004
        self.odom_pub.publish(odometry)

        scan = self.LaserScan()
        scan.header.frame_id = "base_scan"
        scan.header.stamp = stamp
        scan.angle_min = angle_min
        scan.angle_max = angle_max
        scan.angle_increment = (angle_max - angle_min) / max(1, len(ranges) - 1)
        scan.scan_time = 0.1
        scan.time_increment = scan.scan_time / max(1, len(ranges))
        scan.range_min = 0.12
        scan.range_max = 8.0
        scan.ranges = list(ranges)
        self.scan_pub.publish(scan)

    def close(self) -> None:
        self.node.destroy_node()
        if self.rclpy.ok():
            self.rclpy.try_shutdown()

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
        base_to_scan.transform.translation.z = 0.30
        base_to_scan.transform.rotation.w = 1.0
        transforms.append(base_to_scan)
        self.static_tf_pub.sendTransform(transforms)

    def _stamp(self, seconds: float):
        stamp = self.node.get_clock().now().to_msg()
        stamp.sec = int(seconds)
        stamp.nanosec = int((seconds - int(seconds)) * 1e9)
        return stamp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--no-dynamic-obstacle", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sim_config = replace(
        SimulationConfig(),
        headless=args.headless,
        dynamic_obstacle=not args.no_dynamic_obstacle,
    )
    raw_map = build_navigation_map()
    simulation = G2NavigationSimulation(sim_config, enable_ros2=True)
    kinematics = SwerveKinematics(RobotGeometry().wheel_positions, RobotGeometry().wheel_radius)
    base = G2BaseController(simulation.robot, kinematics, ControlLimits())
    bridge = Nav2IsaacBridge()
    scan_angles = [(-math.pi + index * 2.0 * math.pi / 359.0) for index in range(360)]
    publish_steps = max(1, sim_config.physics_hz // 20)
    scan_steps = max(1, sim_config.physics_hz // 10)
    ranges = [8.0] * len(scan_angles)
    command = Velocity2D()

    print("\n=== 第十章 Nav2 专用版：Isaac Sim / Nav2 桥接器 ===")
    print("下一步请在另一个终端启动 g2_chapter10_2_nav，再在 RViz 中发送目标。")
    try:
        step = 0
        while simulation.is_running():
            if step % publish_steps == 0:
                command = bridge.spin_once(simulation.sim_time)
            base.set_velocity(command.vx, command.vy, command.wz)
            base.update(sim_config.physics_dt)
            pose = simulation.get_pose2d()
            if step % scan_steps == 0:
                sensor_map = raw_map.copy()
                sensor_map.add_circles(simulation.dynamic_obstacles())
                ranges = sensor_map.ray_cast(pose, scan_angles, 0.12, 8.0)
            if step % publish_steps == 0:
                bridge.publish(simulation.sim_time, pose, command, ranges, -math.pi, math.pi)
            simulation.step()
            step += 1
    finally:
        base.stop()
        bridge.close()
        simulation.close()


if __name__ == "__main__":
    main()
