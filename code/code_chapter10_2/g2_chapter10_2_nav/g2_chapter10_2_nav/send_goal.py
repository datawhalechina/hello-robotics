"""从命令行向 Nav2 NavigateToPose action 发送目标。"""

import argparse
import math
import time

import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from rclpy.node import Node


class GoalClient(Node):
    def __init__(self) -> None:
        super().__init__("chapter10_2_nav2_goal_client")
        self.client = ActionClient(self, NavigateToPose, "navigate_to_pose")
        self._last_feedback_time = -math.inf

    def send(self, x: float, y: float, yaw: float, startup_timeout: float = 30.0) -> bool:
        """等待 Nav2 激活后发送目标，并一直等待最终结果。"""
        self.get_logger().info("等待 Nav2 navigate_to_pose action...")
        deadline = time.monotonic() + startup_timeout
        if not self.client.wait_for_server(timeout_sec=startup_timeout):
            self.get_logger().error("Nav2 action 未就绪，请检查 bringup 和 lifecycle 状态")
            return False

        # Action server 可能已经创建，但 BT Navigator 仍处于 lifecycle activating。
        # 这种短暂阶段会拒绝目标，因此在启动超时内低频重试。
        handle = None
        while True:
            future = self.client.send_goal_async(
                self._make_goal(x, y, yaw), feedback_callback=self._feedback
            )
            rclpy.spin_until_future_complete(self, future)
            handle = future.result()
            if handle is not None and handle.accepted:
                break
            if time.monotonic() >= deadline:
                break
            self.get_logger().warn("Nav2 尚未激活，1 秒后重试目标")
            time.sleep(1.0)

        if handle is None or not handle.accepted:
            self.get_logger().error("目标被拒绝；请检查 BT Navigator lifecycle 状态")
            return False

        self.get_logger().info("目标已接受")
        result_future = handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        result = result_future.result()
        if result is None:
            self.get_logger().error("未收到导航结果")
            return False
        self.get_logger().info(f"导航结束，action 状态码：{result.status}")
        return result.status == GoalStatus.STATUS_SUCCEEDED

    def _make_goal(self, x: float, y: float, yaw: float):
        goal = NavigateToPose.Goal()
        goal.pose = PoseStamped()
        goal.pose.header.frame_id = "map"
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = x
        goal.pose.pose.position.y = y
        goal.pose.pose.orientation.z = math.sin(yaw / 2.0)
        goal.pose.pose.orientation.w = math.cos(yaw / 2.0)
        return goal

    def _feedback(self, feedback_message) -> None:
        now = time.monotonic()
        if now - self._last_feedback_time < 0.5:
            return
        self._last_feedback_time = now
        feedback = feedback_message.feedback
        self.get_logger().info(
            f"剩余距离：{feedback.distance_remaining:.2f} m，"
            f"恢复次数：{feedback.number_of_recoveries}"
        )


def main(args=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--x", type=float, default=3.6)
    parser.add_argument("--y", type=float, default=3.2)
    parser.add_argument("--yaw", type=float, default=1.57)
    parser.add_argument("--startup-timeout", type=float, default=30.0)
    known, ros_args = parser.parse_known_args(args)
    rclpy.init(args=ros_args)
    node = GoalClient()
    try:
        success = node.send(known.x, known.y, known.yaw, known.startup_timeout)
        if not success:
            raise SystemExit(1)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
