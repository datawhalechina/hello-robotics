"""Main demo ROS2 node: orchestrates navigation, manipulation, and depth capture.

Supports both auto-run and manual service-based control.
"""

import numpy as np

import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
from std_srvs.srv import Trigger
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped, Twist, Vector3, Quaternion
from sensor_msgs.msg import PointCloud2, Image
from tf2_ros import TransformBroadcaster
from cv_bridge import CvBridge

from g2_robot.controllers.base import BaseController
from g2_robot.controllers.arm import ArmController, PickController, PickState
from g2_robot.perception.depth_sensor import DepthSensor


class DemoPhase:
    WARMUP = "warmup"
    NAVIGATE = "navigate"
    DEPTH_PRE = "depth_pre_pick"
    PICK = "pick"
    DEPTH_POST = "depth_post_pick"
    IDLE = "idle"
    DONE = "done"


class DemoNode(Node):
    """ROS2 node that orchestrates the G2 robot demo.

    Capabilities:
        - Navigation to a target position
        - Object pick-up via arm manipulation
        - Depth camera capture and point cloud publishing

    Modes:
        - Auto-run: executes the full demo sequence on startup
        - Manual: trigger each action via ROS2 services
    """

    def __init__(self, sim_setup, ros_bridge, config):
        super().__init__("g2_demo_node")

        self.sim_setup = sim_setup
        self.ros_bridge = ros_bridge
        self.config = config

        # Physics state
        self.physics_dt = 1.0 / config["sim"]["physics_step"]
        self.step_count = 0

        # --- Publishers ---
        self.odom_pub = self.create_publisher(Odometry, "/odom", 10)
        self.pointcloud_pub = self.create_publisher(PointCloud2, "/demo/depth_pointcloud", 10)
        self.depth_colormap_pub = self.create_publisher(Image, "/demo/depth_colormap", 10)

        # TF broadcaster for odom -> base_link
        self.odom_tf_broadcaster = TransformBroadcaster(self)

        # CvBridge for image conversion
        self.bridge = CvBridge()

        # --- Services ---
        self.nav_srv = self.create_service(Trigger, "/demo/navigate_to", self._handle_navigate)
        self.pick_srv = self.create_service(Trigger, "/demo/pick_object", self._handle_pick)
        self.depth_srv = self.create_service(Trigger, "/demo/capture_depth", self._handle_depth)

        # --- Controllers ---
        self.base_controller = BaseController(sim_setup.articulation, self.physics_dt)

        # Initialize velocity-based wheel control for /cmd_vel
        chassis_cfg = config.get("chassis", {})
        self.base_controller.init_velocity_mode(
            max_linear_speed=chassis_cfg.get("max_linear_speed", 1.0),
            max_angular_speed=chassis_cfg.get("max_angular_speed", 2.0),
            max_wheel_speed=chassis_cfg.get("max_wheel_speed", 20.0),
            cmd_timeout=chassis_cfg.get("cmd_timeout", 0.5),
        )

        # /cmd_vel subscriber
        self.cmd_vel_sub = self.create_subscription(
            Twist, "/cmd_vel", self._cmd_vel_callback, 10,
        )

        arm_controller = ArmController(
            articulation=sim_setup.articulation,
            arm="right",
            ik_solver=sim_setup.ik_solvers["right"],
            ruckig=sim_setup.ruckig,
            gripper=sim_setup.gripper,
        )
        self.pick_controller = PickController(arm_controller)

        depth_cfg = config["depth_camera"]
        self.depth_sensor = DepthSensor(
            robot_interface=ros_bridge.robot_interface,
            camera_id=depth_cfg["camera_id"],
            camera_prim=depth_cfg["camera_prim"],
            resolution=depth_cfg["resolution"],
            max_depth=depth_cfg.get("max_depth", 5.0),
        )

        # --- Auto-run state ---
        self.auto_run = config["demo"].get("auto_run", True)
        self.warmup_steps = config["demo"].get("warmup_steps", 240)
        self.phase = DemoPhase.WARMUP if self.auto_run else DemoPhase.IDLE

        self.get_logger().info("DemoNode initialized")
        self.get_logger().info(f"  Auto-run: {self.auto_run}")
        self.get_logger().info("  Subscribers: /cmd_vel")
        self.get_logger().info("  Services: /demo/navigate_to, /demo/pick_object, /demo/capture_depth")

    # ----------------------------------------------------------------
    # Physics tick (called from main loop)
    # ----------------------------------------------------------------
    def physics_tick(self, step_size):
        """Called every physics step from the main simulation loop."""
        self.step_count += 1

        # Tick ROS2 sensor publishers
        current_time = self.step_count * self.physics_dt
        self.ros_bridge.tick(current_time, self.step_count)

        # Process ROS2 callbacks
        rclpy.spin_once(self, timeout_sec=0)

        # Navigation / velocity step
        if self.base_controller.active:
            done, dist = self.base_controller.navigation_step()
        else:
            self.base_controller.velocity_step()

        # Always publish odometry
        if self.step_count % 2 == 0:
            self._publish_odometry()

        # Manipulation step
        if self.pick_controller.is_active:
            self.pick_controller.step()

        # Auto-run state machine
        if self.auto_run:
            self._auto_run_tick()

    # ----------------------------------------------------------------
    # Auto-run state machine
    # ----------------------------------------------------------------
    def _auto_run_tick(self):
        if self.phase == DemoPhase.WARMUP:
            if self.step_count >= self.warmup_steps:
                self.get_logger().info("=== Demo: Starting auto-run sequence ===")
                self.depth_sensor.compute_intrinsics()
                self._start_navigation()
                self.phase = DemoPhase.NAVIGATE

        elif self.phase == DemoPhase.NAVIGATE:
            if not self.base_controller.active:
                self.get_logger().info("=== Demo: Navigation complete, capturing depth ===")
                self.phase = DemoPhase.DEPTH_PRE
                self._depth_warmup_counter = 0

        elif self.phase == DemoPhase.DEPTH_PRE:
            self._depth_warmup_counter += 1
            if self._depth_warmup_counter >= 60:
                self._capture_and_publish_depth("pre_pick")
                self.get_logger().info("=== Demo: Starting pick sequence ===")
                self._start_pick()
                self.phase = DemoPhase.PICK

        elif self.phase == DemoPhase.PICK:
            if not self.pick_controller.is_active:
                if self.pick_controller.state == PickState.DONE:
                    self.get_logger().info("=== Demo: Pick complete, capturing final depth ===")
                    self.phase = DemoPhase.DEPTH_POST
                    self._depth_warmup_counter = 0
                else:
                    self.get_logger().warn("=== Demo: Pick failed ===")
                    self.phase = DemoPhase.DONE

        elif self.phase == DemoPhase.DEPTH_POST:
            self._depth_warmup_counter += 1
            if self._depth_warmup_counter >= 60:
                self._capture_and_publish_depth("post_pick")
                self.get_logger().info("=== Demo: Full sequence completed! ===")
                self.phase = DemoPhase.DONE
                self.auto_run = False

    # ----------------------------------------------------------------
    # /cmd_vel callback
    # ----------------------------------------------------------------
    def _cmd_vel_callback(self, msg):
        """Handle incoming /cmd_vel Twist messages."""
        self.base_controller.set_velocity_command(msg.linear.x, msg.angular.z)

    # ----------------------------------------------------------------
    # Service handlers (for manual control)
    # ----------------------------------------------------------------
    def _handle_navigate(self, request, response):
        if self.base_controller.active:
            response.success = False
            response.message = "Navigation already in progress"
        else:
            self._start_navigation()
            response.success = True
            response.message = "Navigation started"
        return response

    def _handle_pick(self, request, response):
        if self.pick_controller.is_active:
            response.success = False
            response.message = "Pick already in progress"
        elif self.base_controller.active:
            response.success = False
            response.message = "Wait for navigation to complete first"
        else:
            self._start_pick()
            response.success = True
            response.message = "Pick sequence started"
        return response

    def _handle_depth(self, request, response):
        result = self._capture_and_publish_depth("manual")
        if result:
            response.success = True
            response.message = f"Depth captured: {result['pointcloud'].shape[0]} points"
        else:
            response.success = False
            response.message = "Failed to capture depth"
        return response

    # ----------------------------------------------------------------
    # Action starters
    # ----------------------------------------------------------------
    def _start_navigation(self):
        nav_cfg = self.config["navigation"]
        target = nav_cfg["target_position"]
        self.base_controller.navigate_to(
            target_x=target[0],
            target_y=target[1],
            target_yaw=nav_cfg["target_yaw"],
            linear_speed=nav_cfg["linear_speed"],
            angular_speed=nav_cfg["angular_speed"],
            pos_tolerance=nav_cfg["position_tolerance"],
            yaw_tolerance=nav_cfg["yaw_tolerance"],
        )

    def _start_pick(self):
        manip_cfg = self.config["manipulation"]
        scene_cfg = self.config["scene"]

        object_pos = scene_cfg.get("object_position", [-4.03, -0.032, 0.81])

        self.pick_controller.start_pick(
            object_position=object_pos,
            grasp_orientation=manip_cfg["grasp_orientation"],
            pre_grasp_offset=manip_cfg["pre_grasp_offset"],
            grasp_offset=manip_cfg["grasp_offset"],
            lift_height=manip_cfg["lift_height"],
        )

    # ----------------------------------------------------------------
    # Depth capture and publishing
    # ----------------------------------------------------------------
    def _capture_and_publish_depth(self, label):
        result = self.depth_sensor.capture_and_process()
        if result is None:
            return None

        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = "head_front_camera"

        # Publish PointCloud2
        pc_msg = DepthSensor.create_pointcloud2_msg(result["pointcloud"], header)
        self.pointcloud_pub.publish(pc_msg)

        # Publish depth colormap
        try:
            colormap = DepthSensor.create_depth_colormap(
                result["depth_image"], self.depth_sensor.max_depth
            )
            img_msg = self.bridge.cv2_to_imgmsg(colormap, encoding="bgr8")
            img_msg.header = header
            self.depth_colormap_pub.publish(img_msg)
        except Exception as e:
            self.get_logger().warn(f"Failed to publish depth colormap: {e}")

        # Save point cloud to file
        output_dir = self.config["depth_camera"].get("output_dir", "/tmp/ros2_demo_depth/")
        ply_path = f"{output_dir}/pointcloud_{label}.ply"
        self.depth_sensor.save_pointcloud_ply(result["pointcloud"], ply_path)

        return result

    # ----------------------------------------------------------------
    # Odometry publishing
    # ----------------------------------------------------------------
    def _publish_odometry(self):
        pos, quat, lin_vel, ang_vel = self.base_controller.get_odometry()

        odom = Odometry()
        odom.header.stamp = self.get_clock().now().to_msg()
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base_link"

        odom.pose.pose.position.x = float(pos[0])
        odom.pose.pose.position.y = float(pos[1])
        odom.pose.pose.position.z = float(pos[2])
        odom.pose.pose.orientation.w = float(quat[0])
        odom.pose.pose.orientation.x = float(quat[1])
        odom.pose.pose.orientation.y = float(quat[2])
        odom.pose.pose.orientation.z = float(quat[3])

        odom.twist.twist.linear.x = float(lin_vel[0])
        odom.twist.twist.linear.y = float(lin_vel[1])
        odom.twist.twist.linear.z = float(lin_vel[2])
        odom.twist.twist.angular.x = float(ang_vel[0])
        odom.twist.twist.angular.y = float(ang_vel[1])
        odom.twist.twist.angular.z = float(ang_vel[2])

        self.odom_pub.publish(odom)

        # TF: odom -> base_link
        tf = TransformStamped()
        tf.header = odom.header
        tf.child_frame_id = "base_link"
        tf.transform.translation = Vector3(
            x=float(pos[0]), y=float(pos[1]), z=float(pos[2])
        )
        tf.transform.rotation = Quaternion(
            w=float(quat[0]), x=float(quat[1]), y=float(quat[2]), z=float(quat[3])
        )
        self.odom_tf_broadcaster.sendTransform(tf)
