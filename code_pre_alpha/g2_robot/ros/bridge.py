"""ROS2 bridge: manages RobotInterface lifecycle for joint states, TF, and camera publishing."""

import omni.usd
from g2_robot.ros.robot_interface import RobotInterface
from g2_robot.core.constants import G2_CAMERAS


class ROSBridge:
    """Manages the ROS2 sensor publishing node (RobotInterface)."""

    def __init__(self):
        self.robot_interface = None

    def setup(self, articulation, cameras=None):
        """Initialize the RobotInterface ROS2 node.

        Args:
            articulation: SingleArticulation instance
            cameras: dict of {prim_path: [width, height, hz]} or None for defaults
        """
        if cameras is None:
            cameras = G2_CAMERAS

        stage = omni.usd.get_context().get_stage()

        self.robot_interface = RobotInterface()

        # Register joint state publisher
        self.robot_interface.register_joint_state(articulation)

        # Register TF broadcasting
        self.robot_interface.register_robot_tf(stage, "/genie")

        # Register cameras
        for prim_path, params in cameras.items():
            width, height = params[0], params[1]
            hz = params[2] if len(params) > 2 else 30
            every_n_frame = max(1, 120 // hz)
            self.robot_interface.register_camera(prim_path, [width, height], every_n_frame)

        print(f"[ROSBridge] RobotInterface initialized with {len(cameras)} cameras")
        return self.robot_interface

    def tick(self, current_time: float, step_index: int):
        """Tick the ROS2 publishers (call each physics step)."""
        if self.robot_interface:
            self.robot_interface.tick(current_time, step_index)
