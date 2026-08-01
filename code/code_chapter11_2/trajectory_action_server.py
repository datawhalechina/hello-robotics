"""Isaac 端 ROS 2 话题桥。

为什么这里不直接在 Isaac Python 中实现 FollowJointTrajectory action？
Isaac Sim 使用 Python 3.11，而 ROS 2 Humble 的 ``control_msgs`` Python 模块按
Python 3.10 编译，二者不能安全混用。因此：

- 本文件只使用 Isaac ROS Bridge 自带的 ``rclpy/sensor_msgs/std_msgs``；
- C++ 节点 ``isaac_controller_bridge`` 实现标准 FollowJointTrajectory action；
- 两端通过简单的 JointState 命令/反馈话题连接。

这样既保留了 MoveIt 标准控制器接口，也避免 Python ABI 冲突。
"""

from __future__ import annotations

import threading
from typing import Sequence

import numpy as np

try:
    from .config import GRIPPER_JOINT_NAMES, RIGHT_ARM_JOINT_NAMES
except ImportError:
    from config import GRIPPER_JOINT_NAMES, RIGHT_ARM_JOINT_NAMES


class IsaacTrajectoryTopicBridge:
    """在 Isaac articulation 与 ROS 2 C++ 控制器桥之间传递关节命令。"""

    def __init__(self, articulation, simulation, perception=None) -> None:
        import rclpy
        from rclpy.executors import MultiThreadedExecutor
        from rclpy.node import Node
        from sensor_msgs.msg import JointState, PointCloud2
        from std_msgs.msg import Bool

        if not rclpy.ok():
            rclpy.init(args=None)
        self.rclpy = rclpy
        self.node = Node("chapter11_isaac_topic_bridge")
        self.articulation = articulation
        self.simulation = simulation
        self.perception = perception
        self.names = list(articulation.dof_names)
        self.arm_indices = np.asarray(
            [self.names.index(name) for name in RIGHT_ARM_JOINT_NAMES], dtype=np.int64
        )
        self.gripper_indices = np.asarray(
            [self.names.index(name) for name in GRIPPER_JOINT_NAMES], dtype=np.int64
        )

        self.joint_pub = self.node.create_publisher(
            JointState, "/chapter11/isaac_joint_states", 30
        )
        self.cloud_pub = self.node.create_publisher(
            PointCloud2, "/chapter11/head/depth_points", 2
        )
        self.node.create_subscription(
            JointState, "/chapter11/isaac_joint_command", self._arm_command_callback, 30
        )
        self.node.create_subscription(
            Bool, "/chapter11/gripper_open", self._gripper_callback, 10
        )
        self.node.create_subscription(
            Bool, "/chapter11/attach_red", self._attach_callback, 10
        )

        self._lock = threading.Lock()
        self._target = None
        self._gripper_open = True
        self._sim_time = 0.0
        self._last_joint_publish = -1.0
        self._last_cloud_publish = -1.0

        self.executor = MultiThreadedExecutor(num_threads=2)
        self.executor.add_node(self.node)
        self.thread = threading.Thread(target=self.executor.spin, daemon=True)
        self.thread.start()
        print("[Bridge] Isaac 关节命令/反馈话题已启动", flush=True)

    def _arm_command_callback(self, message) -> None:
        if tuple(message.name) != RIGHT_ARM_JOINT_NAMES or len(message.position) != 7:
            self.node.get_logger().error("收到的关节命令名称、顺序或长度不正确")
            return
        target = np.asarray(message.position, dtype=np.float64)
        if not np.all(np.isfinite(target)):
            self.node.get_logger().error("收到非法关节命令")
            return
        with self._lock:
            self._target = target.copy()

    def _gripper_callback(self, message) -> None:
        self._gripper_open = bool(message.data)

    def _attach_callback(self, message) -> None:
        self.simulation.set_red_attached(bool(message.data))

    def _command_arm(self, positions: Sequence[float]) -> None:
        from isaacsim.core.utils.types import ArticulationAction

        self.articulation.apply_action(
            ArticulationAction(
                joint_positions=np.asarray(positions, dtype=np.float64),
                joint_indices=self.arm_indices,
            )
        )

    def _command_gripper(self) -> None:
        from isaacsim.core.utils.types import ArticulationAction

        target = (
            np.array([0.68, -0.68], dtype=np.float64)
            if self._gripper_open
            else np.array([0.03, -0.03], dtype=np.float64)
        )
        self.articulation.apply_action(
            ArticulationAction(joint_positions=target, joint_indices=self.gripper_indices)
        )

    def _publish_joint_states(self) -> None:
        if self._sim_time - self._last_joint_publish < 1.0 / 30.0:
            return
        from sensor_msgs.msg import JointState

        positions = np.asarray(self.articulation.get_joint_positions(), dtype=np.float64)
        velocities = np.asarray(self.articulation.get_joint_velocities(), dtype=np.float64)
        message = JointState()
        message.header.stamp = self.node.get_clock().now().to_msg()
        message.name = list(RIGHT_ARM_JOINT_NAMES)
        message.position = positions[self.arm_indices].tolist()
        message.velocity = velocities[self.arm_indices].tolist()
        self.joint_pub.publish(message)
        self._last_joint_publish = self._sim_time

    def _publish_perception(self) -> None:
        if self.perception is None:
            return
        if self._sim_time - self._last_cloud_publish >= 1.0 / self.perception.config.map_publish_hz:
            # 即使本帧读取失败也更新时间戳，避免以 120 Hz 重试并刷屏。
            self._last_cloud_publish = self._sim_time
            try:
                from sensor_msgs.msg import PointCloud2, PointField
                points = self.perception.head_obstacle_points()
                if len(points):
                    message = PointCloud2()
                    # 点云已经转换到 arm_base_link。使用零时间戳让 MoveIt 自滤波
                    # 查询最新机器人 TF，避免点云时间比 robot_state_publisher 快几毫秒
                    # 时出现 Missing transform 并把夹爪写进 OctoMap。
                    message.header.stamp.sec = 0
                    message.header.stamp.nanosec = 0
                    message.header.frame_id = "arm_base_link"
                    message.height, message.width = 1, len(points)
                    message.fields = [
                        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
                        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
                        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
                    ]
                    message.is_bigendian = False
                    message.point_step = 12
                    message.row_step = 12 * len(points)
                    message.is_dense = True
                    message.data = np.ascontiguousarray(points, dtype=np.float32).tobytes()
                    self.cloud_pub.publish(message)
            except Exception as exc:
                self.node.get_logger().warn(f"头部深度点云发布失败: {exc}")

    def update(self, dt: float) -> None:
        self._sim_time += dt
        with self._lock:
            target = None if self._target is None else self._target.copy()
        if target is not None:
            self._command_arm(target)
        self._command_gripper()
        self._publish_joint_states()
        self._publish_perception()

    def close(self) -> None:
        self.executor.shutdown()
        self.node.destroy_node()
        self.thread.join(timeout=2.0)
        if self.rclpy.ok():
            self.rclpy.shutdown()


# 兼容前一个教学草稿中的类名，后续代码应优先使用新名称。
IsaacTrajectoryActionServer = IsaacTrajectoryTopicBridge
