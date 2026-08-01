"""启动静态地图、Nav2 导航服务器和第十章 RViz 配置。"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    package_dir = get_package_share_directory("g2_chapter10_2_nav")
    nav2_dir = get_package_share_directory("nav2_bringup")
    params = LaunchConfiguration("params_file")
    map_yaml = LaunchConfiguration("map")
    use_rviz = LaunchConfiguration("use_rviz")

    map_server = Node(
        package="nav2_map_server",
        executable="map_server",
        name="map_server",
        output="screen",
        parameters=[params, {"yaml_filename": map_yaml, "use_sim_time": True}],
    )
    map_lifecycle = Node(
        package="nav2_lifecycle_manager",
        executable="lifecycle_manager",
        name="lifecycle_manager_map",
        output="screen",
        parameters=[
            {"use_sim_time": True},
            {"autostart": True},
            {"node_names": ["map_server"]},
        ],
    )
    navigation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(nav2_dir, "launch", "navigation_launch.py")),
        launch_arguments={
            "use_sim_time": "true",
            "autostart": "true",
            "params_file": params,
            "use_composition": "False",
        }.items(),
    )
    rviz = Node(
        condition=IfCondition(use_rviz),
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=["-d", os.path.join(package_dir, "config", "chapter10_2_navigation.rviz")],
        parameters=[{"use_sim_time": True}],
        output="screen",
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "params_file",
                default_value=os.path.join(package_dir, "config", "nav2_params.yaml"),
            ),
            DeclareLaunchArgument(
                "map",
                default_value=os.path.join(package_dir, "maps", "chapter10_2_map.yaml"),
            ),
            DeclareLaunchArgument("use_rviz", default_value="true"),
            map_server,
            map_lifecycle,
            navigation,
            rviz,
        ]
    )
