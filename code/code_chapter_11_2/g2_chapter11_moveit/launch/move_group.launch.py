"""启动 G2 右臂 MoveIt 2、robot_state_publisher 和 RViz。"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder
from ament_index_python.packages import get_package_share_directory


def build_config():
    return (
        MoveItConfigsBuilder("g2_right_arm", package_name="g2_chapter11_moveit")
        .robot_description(file_path="urdf/g2_right_arm.urdf.xacro")
        .robot_description_semantic(file_path="srdf/g2_right_arm.srdf")
        .robot_description_kinematics(file_path="config/kinematics.yaml")
        .joint_limits(file_path="config/joint_limits.yaml")
        .trajectory_execution(file_path="config/moveit_controllers.yaml")
        .planning_pipelines(default_planning_pipeline="ompl", pipelines=["ompl"], load_all=False)
        .sensors_3d(file_path="config/sensors_3d.yaml")
        .to_moveit_configs()
    )


def generate_launch_description():
    package_dir = get_package_share_directory("g2_chapter11_moveit")
    moveit_config = build_config()
    use_rviz = LaunchConfiguration("use_rviz")

    controller_bridge = Node(
        package="g2_chapter11_moveit",
        executable="isaac_controller_bridge",
        output="screen",
    )
    move_group = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[moveit_config.to_dict(), {
            "publish_robot_description_semantic": True,
            "octomap_frame": "arm_base_link",
            "octomap_resolution": 0.04,
        }],
        arguments=["--ros-args", "--log-level", "info"],
    )
    state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[moveit_config.robot_description],
    )
    static_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="chapter11_world_to_arm_base",
        arguments=[
            "--x", "0", "--y", "0", "--z", "0",
            "--roll", "-1.5707963267948966", "--pitch", "0", "--yaw", "0",
            "--frame-id", "world", "--child-frame-id", "arm_base_link",
        ],
        output="log",
    )
    rviz = Node(
        condition=IfCondition(use_rviz),
        package="rviz2",
        executable="rviz2",
        name="chapter11_moveit_rviz",
        arguments=["-d", os.path.join(package_dir, "rviz", "chapter11_moveit.rviz")],
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            moveit_config.planning_pipelines,
        ],
        output="screen",
    )
    return LaunchDescription([
        DeclareLaunchArgument("use_rviz", default_value="true"),
        static_tf, state_publisher, controller_bridge, move_group, rviz,
    ])
