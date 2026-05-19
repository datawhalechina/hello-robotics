"""
Unified base/chassis controller for the G2 robot.

Provides multiple control modes for moving the robot base:

  Mode 1: Pose-based navigation (set_world_pose)
    - Smooth incremental movement at physics rate
    - WARNING: Causes arm jitter during simultaneous arm control.
      set_world_pose resets the entire articulation root pose each step,
      which conflicts with arm joint position targets set via apply_action.
      Use this mode only when arm movement is not required simultaneously.

  Mode 2: Velocity-based wheel control (apply_action with joint velocities)
    - Receives (v_x, omega) commands (e.g. from Nav2 /cmd_vel)
    - Swerve inverse kinematics: computes each wheel's steering angle + spin speed
    - Compatible with simultaneous arm control (disjoint joint indices)
    - Requires floating base (robot.usda, not robot_fix.usda)

"""

import math
import time

import numpy as np


def yaw_to_quat(yaw):
    """Convert yaw angle (radians) to quaternion [w, x, y, z]."""
    return np.array([math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)])


def quat_to_yaw(quat):
    """Extract yaw from quaternion [w, x, y, z]."""
    w, x, y, z = quat
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def normalize_angle(angle):
    """Normalize angle to [-pi, pi]."""
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


class BaseController:
    """Unified base/chassis controller for the G2 robot.

    WARNING on set_world_pose mode:
        When using navigate_to() / navigation_step(), the robot base is moved
        by calling articulation.set_world_pose() each physics step. This causes
        PhysX to reset the entire articulation root transform, which conflicts
        with arm joint targets set via apply_action(). The symptom is arm jitter
        during base movement that stops when the base stops.

        For simultaneous arm + base control, consider using velocity-based
        wheel control instead (not yet implemented).

    Args:
        articulation: SingleArticulation for the robot
        physics_dt: physics timestep in seconds (e.g., 1/120)
    """

    def __init__(self, articulation, physics_dt):
        self.articulation = articulation
        self.physics_dt = physics_dt

        # Navigation state
        self.active = False
        self.target_x = 0.0
        self.target_y = 0.0
        self.target_yaw = 0.0
        self.linear_speed = 0.3
        self.angular_speed = 0.5
        self.pos_tolerance = 0.02
        self.yaw_tolerance = 0.05

        # Odometry state
        self.linear_vel = np.zeros(3)
        self.angular_vel = np.zeros(3)

        # Velocity control state (initialized by init_velocity_mode)
        self._velocity_mode_initialized = False
        self._steering_indices = None
        self._spin_indices = None
        self._cmd_vx = 0.0
        self._cmd_omega = 0.0
        self._cmd_timestamp = 0.0
        self._cmd_timeout = 0.5

    # ── Mode 1: Pose-based navigation (set_world_pose) ──────────────

    def navigate_to(self, target_x, target_y, target_yaw,
                    linear_speed=0.3, angular_speed=0.5,
                    pos_tolerance=0.02, yaw_tolerance=0.05):
        """Start navigating to a target pose using set_world_pose.

        WARNING: This mode conflicts with arm control. See class docstring.
        """
        self.target_x = target_x
        self.target_y = target_y
        self.target_yaw = target_yaw
        self.linear_speed = linear_speed
        self.angular_speed = angular_speed
        self.pos_tolerance = pos_tolerance
        self.yaw_tolerance = yaw_tolerance
        self.active = True
        print(f"[BaseController] Navigating to ({target_x:.2f}, {target_y:.2f}, yaw={target_yaw:.2f})")

    def navigation_step(self):
        """Execute one navigation step. Call each physics tick.

        Returns:
            (done: bool, distance_remaining: float)
        """
        if not self.active:
            return True, 0.0

        pos, quat = self.articulation.get_world_pose()
        cur_x, cur_y, cur_z = float(pos[0]), float(pos[1]), float(pos[2])
        cur_yaw = quat_to_yaw(quat)

        dx = self.target_x - cur_x
        dy = self.target_y - cur_y
        distance = math.sqrt(dx * dx + dy * dy)
        yaw_error = normalize_angle(self.target_yaw - cur_yaw)

        # Phase 1: Rotate to face target direction
        if distance > self.pos_tolerance:
            desired_yaw = math.atan2(dy, dx)
            face_error = normalize_angle(desired_yaw - cur_yaw)

            if abs(face_error) > self.yaw_tolerance:
                rot_step = self.angular_speed * self.physics_dt
                rot_step = min(rot_step, abs(face_error))
                new_yaw = cur_yaw + math.copysign(rot_step, face_error)
                new_quat = yaw_to_quat(new_yaw)
                self.articulation.set_world_pose(
                    position=np.array([cur_x, cur_y, cur_z]),
                    orientation=new_quat,
                )
                self.linear_vel = np.zeros(3)
                self.angular_vel = np.array([0.0, 0.0, math.copysign(self.angular_speed, face_error)])
                return False, distance

            # Phase 2: Move toward target
            move_step = self.linear_speed * self.physics_dt
            move_step = min(move_step, distance)
            ratio = move_step / distance
            new_x = cur_x + dx * ratio
            new_y = cur_y + dy * ratio
            new_quat = yaw_to_quat(cur_yaw)
            self.articulation.set_world_pose(
                position=np.array([new_x, new_y, cur_z]),
                orientation=new_quat,
            )
            self.linear_vel = np.array([
                dx / distance * self.linear_speed,
                dy / distance * self.linear_speed,
                0.0,
            ])
            self.angular_vel = np.zeros(3)
            return False, distance

        # Phase 3: Final yaw alignment
        if abs(yaw_error) > self.yaw_tolerance:
            rot_step = self.angular_speed * self.physics_dt
            rot_step = min(rot_step, abs(yaw_error))
            new_yaw = cur_yaw + math.copysign(rot_step, yaw_error)
            new_quat = yaw_to_quat(new_yaw)
            self.articulation.set_world_pose(
                position=np.array([cur_x, cur_y, cur_z]),
                orientation=new_quat,
            )
            self.linear_vel = np.zeros(3)
            self.angular_vel = np.array([0.0, 0.0, math.copysign(self.angular_speed, yaw_error)])
            return False, distance

        # Done
        self.active = False
        self.linear_vel = np.zeros(3)
        self.angular_vel = np.zeros(3)
        print("[BaseController] Reached target")
        return True, 0.0

    # ── Odometry ────────────────────────────────────────────────────

    def get_odometry(self):
        """Get current odometry data for ROS2 publishing.

        Returns:
            (position, orientation_quat, linear_vel, angular_vel)
        """
        pos, quat = self.articulation.get_world_pose()
        return pos, quat, self.linear_vel, self.angular_vel

    def get_world_pose(self):
        """Get current world pose.

        Returns:
            (position, orientation_quat)
        """
        return self.articulation.get_world_pose()

    # ── Mode 2: Velocity-based wheel control (apply_action) ────────

    def init_velocity_mode(self, max_linear_speed=1.0, max_angular_speed=2.0,
                           max_wheel_speed=20.0, cmd_timeout=0.5):
        """Initialize velocity-based swerve wheel control.

        Uses swerve inverse kinematics to compute each wheel's steering angle
        and spin speed from (v_x, omega). Applied via apply_action on disjoint
        joint indices. Compatible with simultaneous arm control.

        Requires:
            - robot.usda (floating base, not robot_fix.usda)

        Args:
            max_linear_speed: max forward speed (m/s)
            max_angular_speed: max turning rate (rad/s)
            max_wheel_speed: max wheel angular velocity (rad/s)
            cmd_timeout: zero velocity if no command received within this time (s)
        """
        from g2_robot.core.constants import (
            WHEEL_STEERING_JOINT_NAMES, WHEEL_SPIN_JOINT_NAMES,
            WHEEL_RADIUS, WHEEL_BASE, TRACK_WIDTH, find_joint_indices,
        )

        dof_names = self.articulation.dof_names
        self._steering_indices = find_joint_indices(dof_names, WHEEL_STEERING_JOINT_NAMES)
        self._spin_indices = find_joint_indices(dof_names, WHEEL_SPIN_JOINT_NAMES)

        # Wheel positions relative to robot center: LF, LR, RF, RR
        half_wb = WHEEL_BASE / 2.0
        half_tw = TRACK_WIDTH / 2.0
        self._wheel_positions = [
            (+half_wb, +half_tw),   # LF
            (-half_wb, +half_tw),   # LR
            (+half_wb, -half_tw),   # RF
            (-half_wb, -half_tw),   # RR
        ]
        self._wheel_radius = WHEEL_RADIUS
        self._max_linear_speed = max_linear_speed
        self._max_angular_speed = max_angular_speed
        self._max_wheel_speed = max_wheel_speed

        self._cmd_timeout = cmd_timeout
        self._cmd_timestamp = 0.0
        self._cmd_vx = 0.0
        self._cmd_omega = 0.0
        self._velocity_mode_initialized = True

        print(f"[BaseController] Velocity mode initialized (swerve IK)")
        print(f"  Steering indices: {self._steering_indices}")
        print(f"  Spin indices:     {self._spin_indices}")
        print(f"  Wheel positions:  {self._wheel_positions}")

    def set_velocity_command(self, vx, omega):
        """Set velocity command from /cmd_vel.

        Args:
            vx: forward linear velocity (m/s)
            omega: angular velocity around Z (rad/s)
        """
        self._cmd_vx = vx
        self._cmd_omega = omega
        self._cmd_timestamp = time.time()

    def velocity_step(self):
        """Execute one velocity control step. Call each physics tick.

        Computes each wheel's steering angle and spin speed from (v_x, omega)
        using swerve inverse kinematics, then applies via apply_action.
        """
        if not self._velocity_mode_initialized:
            return

        from isaacsim.core.utils.types import ArticulationAction

        # Watchdog: zero velocity if no command received recently
        if time.time() - self._cmd_timestamp > self._cmd_timeout:
            vx, omega = 0.0, 0.0
        else:
            vx = np.clip(self._cmd_vx, -self._max_linear_speed, self._max_linear_speed)
            omega = np.clip(self._cmd_omega, -self._max_angular_speed, self._max_angular_speed)

        # Swerve inverse kinematics: for each wheel compute steering angle + spin speed
        steering_angles = []
        spin_speeds = []
        for wx, wy in self._wheel_positions:
            # Velocity vector at this wheel's position
            vx_w = vx - omega * wy
            vy_w = omega * wx
            speed = math.sqrt(vx_w * vx_w + vy_w * vy_w) / self._wheel_radius

            if speed > 0.01:
                angle = math.atan2(vy_w, vx_w)
                # Optimize: if angle > 90° from forward, flip direction to minimize steering
                if angle > math.pi / 2:
                    angle -= math.pi
                    speed = -speed
                elif angle < -math.pi / 2:
                    angle += math.pi
                    speed = -speed
            else:
                angle = 0.0
                speed = 0.0

            steering_angles.append(angle)
            spin_speeds.append(np.clip(speed, -self._max_wheel_speed, self._max_wheel_speed))

        # Apply steering angles (position control)
        self.articulation.apply_action(ArticulationAction(
            joint_positions=np.array(steering_angles),
            joint_indices=self._steering_indices,
        ))

        # Apply spin speeds (velocity control)
        self.articulation.apply_action(ArticulationAction(
            joint_velocities=np.array(spin_speeds),
            joint_indices=self._spin_indices,
        ))

        # Update odometry velocities
        self.linear_vel = np.array([vx, 0.0, 0.0])
        self.angular_vel = np.array([0.0, 0.0, omega])

    def stop(self):
        """Immediately stop all wheel motion."""
        if not self._velocity_mode_initialized:
            return

        from isaacsim.core.utils.types import ArticulationAction

        self._cmd_vx = 0.0
        self._cmd_omega = 0.0

        self.articulation.apply_action(ArticulationAction(
            joint_positions=np.array([0.0] * len(self._steering_indices)),
            joint_indices=self._steering_indices,
        ))
        self.articulation.apply_action(ArticulationAction(
            joint_velocities=np.array([0.0] * len(self._spin_indices)),
            joint_indices=self._spin_indices,
        ))

        self.linear_vel = np.zeros(3)
        self.angular_vel = np.zeros(3)
