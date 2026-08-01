"""只启动红色物体抓取客户端；move_group 和 Isaac bridge 应已运行。"""

from launch import LaunchDescription
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    moveit_config = (
        MoveItConfigsBuilder("g2_right_arm", package_name="g2_chapter11_moveit")
        .robot_description(file_path="urdf/g2_right_arm.urdf.xacro")
        .robot_description_semantic(file_path="srdf/g2_right_arm.srdf")
        .robot_description_kinematics(file_path="config/kinematics.yaml")
        .joint_limits(file_path="config/joint_limits.yaml")
        .trajectory_execution(file_path="config/moveit_controllers.yaml")
        .planning_pipelines(default_planning_pipeline="ompl", pipelines=["ompl"], load_all=False)
        .to_moveit_configs()
    )
    pick = Node(
        package="g2_chapter11_moveit",
        executable="pick_red_moveit",
        output="screen",
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            moveit_config.joint_limits,
            moveit_config.planning_pipelines,
        ],
    )
    return LaunchDescription([pick])
