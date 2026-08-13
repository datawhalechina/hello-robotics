"""不创建 ROS 包，直接启动地图、Nav2 和 RViz。"""

from pathlib import Path
import sys

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription, LaunchService
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


CHAPTER_DIR = Path(__file__).resolve().parents[1]


def _parse_overrides(arguments):
    """支持 run_nav2.sh use_rviz:=false 这种 ROS launch 写法。"""
    return dict(item.split(":=", 1) for item in arguments if ":=" in item)


def generate_launch_description(overrides=None):
    overrides = overrides or {}
    nav2_dir = Path(get_package_share_directory("nav2_bringup"))
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
            {"use_sim_time": True, "autostart": True, "node_names": ["map_server"]}
        ],
    )
    navigation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(str(nav2_dir / "launch/navigation_launch.py")),
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
        arguments=["-d", str(CHAPTER_DIR / "config/chapter13_navigation.rviz")],
        parameters=[{"use_sim_time": True}],
        output="screen",
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "params_file", default_value=overrides.get("params_file", str(CHAPTER_DIR / "config/nav2_params.yaml"))
            ),
            DeclareLaunchArgument(
                "map", default_value=overrides.get("map", str(CHAPTER_DIR / "maps/chapter13_map.yaml"))
            ),
            DeclareLaunchArgument("use_rviz", default_value=overrides.get("use_rviz", "true")),
            map_server,
            map_lifecycle,
            navigation,
            rviz,
        ]
    )


if __name__ == "__main__":
    service = LaunchService()
    service.include_launch_description(generate_launch_description(_parse_overrides(sys.argv[1:])))
    raise SystemExit(service.run())
