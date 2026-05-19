"""
G2 Robot Constants

Central definitions for joint names, posture targets, camera configs,
and TF tree joint sets used across the entire g2_robot package.
"""

# ── Body/Head joints fixed during IK solving ────────────────────────
FIXED_JOINT_NAMES = [
    "idx01_body_joint1", "idx02_body_joint2", "idx03_body_joint3",
    "idx04_body_joint4", "idx05_body_joint5",
    "idx11_head_joint1", "idx12_head_joint2", "idx13_head_joint3",
]

FIXED_JOINT_TARGETS = {
    "idx01_body_joint1": 0.0,
    "idx02_body_joint2": 0.0,
    "idx03_body_joint3": 0.261,   # body lean forward ~15deg
    "idx04_body_joint4": 0.0,
    "idx05_body_joint5": 0.0,
    "idx11_head_joint1": 0.0,
    "idx12_head_joint2": 0.0,
    "idx13_head_joint3": 0.174,   # head tilt
}

# ── Arm joint names (7-DOF each) ────────────────────────────────────
RIGHT_ARM_JOINT_NAMES = [
    "idx61_arm_r_joint1", "idx62_arm_r_joint2", "idx63_arm_r_joint3",
    "idx64_arm_r_joint4", "idx65_arm_r_joint5", "idx66_arm_r_joint6",
    "idx67_arm_r_joint7",
]

LEFT_ARM_JOINT_NAMES = [
    "idx21_arm_l_joint1", "idx22_arm_l_joint2", "idx23_arm_l_joint3",
    "idx24_arm_l_joint4", "idx25_arm_l_joint5", "idx26_arm_l_joint6",
    "idx27_arm_l_joint7",
]

# ── Chassis wheel joints (4 swerve modules × 2 joints each) ───────
WHEEL_STEERING_JOINT_NAMES = [
    "idx111_chassis_lwheel_front_joint1",
    "idx121_chassis_lwheel_rear_joint1",
    "idx131_chassis_rwheel_front_joint1",
    "idx141_chassis_rwheel_rear_joint1",
]

WHEEL_SPIN_JOINT_NAMES = [
    "idx112_chassis_lwheel_front_joint2",
    "idx122_chassis_lwheel_rear_joint2",
    "idx132_chassis_rwheel_front_joint2",
    "idx142_chassis_rwheel_rear_joint2",
]

# Interleaved: [LF_steer, LF_spin, LR_steer, LR_spin, RF_steer, RF_spin, RR_steer, RR_spin]
WHEEL_ALL_JOINT_NAMES = [
    "idx111_chassis_lwheel_front_joint1", "idx112_chassis_lwheel_front_joint2",
    "idx121_chassis_lwheel_rear_joint1",  "idx122_chassis_lwheel_rear_joint2",
    "idx131_chassis_rwheel_front_joint1", "idx132_chassis_rwheel_front_joint2",
    "idx141_chassis_rwheel_rear_joint1",  "idx142_chassis_rwheel_rear_joint2",
]

# ── Robot physical parameters (from URDF) ──────────────────────────
WHEEL_RADIUS = 0.07       # meters
WHEEL_BASE = 0.46         # front-rear axle distance (meters)
TRACK_WIDTH = 0.436       # left-right wheel distance (meters)

# ── Camera configurations: {prim_path: [width, height, hz]} ─────────
G2_CAMERAS = {
    "/genie/head_link3/head_front_Camera": [640, 400, 30],
    "/genie/gripper_l_base_link/Left_Camera": [1280, 1056, 30],
    "/genie/gripper_r_base_link/Right_Camera": [1280, 1056, 30],
}

# ── Dynamic TF joint names (published at runtime) ───────────────────
MAP_DYNAMIC_TF_NAMES = {
    "idx01_body_joint1", "idx02_body_joint2", "idx03_body_joint3",
    "idx04_body_joint4", "idx05_body_joint5",
    "idx11_head_joint1", "idx12_head_joint2", "idx13_head_joint3",
    "idx21_arm_l_joint1", "idx61_arm_r_joint1",
    "idx22_arm_l_joint2", "idx62_arm_r_joint2",
    "idx23_arm_l_joint3", "idx63_arm_r_joint3",
    "idx24_arm_l_joint4", "idx64_arm_r_joint4",
    "idx25_arm_l_joint5", "idx65_arm_r_joint5",
    "idx26_arm_l_joint6", "idx66_arm_r_joint6",
    "idx27_arm_l_joint7", "idx67_arm_r_joint7",
    "idx31_gripper_l_inner_joint1", "idx41_gripper_l_outer_joint1",
    "idx71_gripper_r_inner_joint1", "idx81_gripper_r_outer_joint1",
    "idx32_gripper_l_inner_joint3", "idx42_gripper_l_outer_joint3",
    "idx72_gripper_r_inner_joint3", "idx82_gripper_r_outer_joint3",
    "idx33_gripper_l_inner_joint4", "idx43_gripper_l_outer_joint4",
    "idx73_gripper_r_inner_joint4", "idx83_gripper_r_outer_joint4",
    "idx54_gripper_l_inner_joint0", "idx53_gripper_l_outer_joint0",
    "idx94_gripper_r_inner_joint0", "idx93_gripper_r_outer_joint0",
    "idx111_chassis_lwheel_front_joint1", "idx112_chassis_lwheel_front_joint2",
    "idx131_chassis_rwheel_front_joint1", "idx132_chassis_rwheel_front_joint2",
    "idx121_chassis_lwheel_rear_joint1", "idx122_chassis_lwheel_rear_joint2",
    "idx141_chassis_rwheel_rear_joint1", "idx142_chassis_rwheel_rear_joint2",
}


def find_joint_indices(dof_names, joint_names):
    """Find DOF indices for a list of joint names.

    Args:
        dof_names: list of all DOF names from articulation.dof_names
        joint_names: list of joint names to find

    Returns:
        list of integer indices
    """
    indices = []
    for name in joint_names:
        for i, dn in enumerate(dof_names):
            if dn == name:
                indices.append(i)
                break
    return indices
