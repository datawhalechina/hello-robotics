cmake_minimum_required(VERSION 3.8)
project(g2_chapter11_moveit)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(control_msgs REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(trajectory_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(moveit_msgs REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)
find_package(rclcpp REQUIRED)
find_package(shape_msgs REQUIRED)
find_package(std_msgs REQUIRED)

add_executable(isaac_controller_bridge src/isaac_controller_bridge.cpp)
ament_target_dependencies(isaac_controller_bridge
  control_msgs rclcpp rclcpp_action sensor_msgs trajectory_msgs)
target_compile_features(isaac_controller_bridge PUBLIC cxx_std_17)

add_executable(pick_red_moveit
  src/pick_red.cpp
  src/planning_scene.cpp
  src/local_replanner.cpp)
target_include_directories(pick_red_moveit PUBLIC include)
ament_target_dependencies(pick_red_moveit
  geometry_msgs moveit_msgs moveit_ros_planning_interface rclcpp shape_msgs std_msgs)
target_compile_features(pick_red_moveit PUBLIC cxx_std_17)

install(TARGETS isaac_controller_bridge pick_red_moveit DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY config launch rviz srdf urdf DESTINATION share/${PROJECT_NAME})
install(DIRECTORY include/ DESTINATION include)
ament_package()
