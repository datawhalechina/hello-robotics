#include "g2_chapter11_moveit/local_replanner.hpp"
#include "g2_chapter11_moveit/planning_scene.hpp"

#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/bool.hpp>

#include <chrono>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

namespace
{
// 本案例中红色物块位姿已知：中心为 (0.56, 0.535, -0.43)，坐标系 arm_base_link。
// 下列关节目标由该已知位姿离线求逆解得到，便于把重点放在 MoveIt 规划与执行流程。
const std::vector<double> kHome{0.0, -0.35, 0.0, -1.10, 0.0, 0.35, 0.0};
const std::vector<double> kPreGrasp{
  -0.302439188, -0.360194634, -0.248441966, -1.500835580,
   0.034931271,  0.243289332,  0.106226869};
const std::vector<double> kGrasp{
  -0.455824868, -0.664226570, -0.329120534, -1.148956005,
   0.024455005,  0.283677832,  0.088931390};
const std::vector<double> kLift{
  -0.215750212, -0.152865235, -0.205355940, -1.691462625,
   0.044111527,  0.150545802,  0.114957112};

void publishBool(const rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr& pub, bool value)
{
  std_msgs::msg::Bool message;
  message.data = value;
  for (int i = 0; i < 5; ++i) {
    pub->publish(message);
    rclcpp::sleep_for(60ms);
  }
}
}  // namespace

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  auto node = rclcpp::Node::make_shared("chapter11_pick_red_moveit");
  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);
  std::thread spinner([&executor]() { executor.spin(); });

  moveit::planning_interface::MoveGroupInterface move_group(node, "right_arm");
  moveit::planning_interface::PlanningSceneInterface scene;
  auto gripper_pub = node->create_publisher<std_msgs::msg::Bool>(
    "/chapter11/gripper_open", 10);
  auto attach_pub = node->create_publisher<std_msgs::msg::Bool>(
    "/chapter11/attach_red", 10);

  move_group.setMaxVelocityScalingFactor(0.55);
  move_group.setMaxAccelerationScalingFactor(0.45);
  move_group.setNumPlanningAttempts(4);
  move_group.allowReplanning(true);
  move_group.setReplanAttempts(4);
  move_group.setReplanDelay(0.15);

  RCLCPP_INFO(node->get_logger(), "等待 Isaac /joint_states 与轨迹 action server");
  rclcpp::sleep_for(3s);
  g2_chapter11_moveit::addTeachingScene(scene);
  rclcpp::sleep_for(1s);
  publishBool(gripper_pub, true);

  RCLCPP_INFO(
    node->get_logger(),
    "红色物块位姿已知：中心 (0.560, 0.535, -0.430)，不使用右夹爪相机和视觉伺服");

  bool ok = g2_chapter11_moveit::planAndExecuteWithLocalReplan(
    move_group, kHome, "回到 HOME");
  ok = ok && g2_chapter11_moveit::planAndExecuteWithLocalReplan(
    move_group, kPreGrasp, "绕过阻挡物到预抓取位姿");

  if (ok) {
    // 预抓取阶段目标仍是 Planning Scene 中的碰撞物体。最后接近时允许夹爪接触它。
    g2_chapter11_moveit::removeRedForFinalApproach(scene);
    rclcpp::sleep_for(250ms);
    ok = g2_chapter11_moveit::planAndExecuteWithLocalReplan(
      move_group, kGrasp, "按已知物块位姿运动到抓取点");
  }

  if (ok) {
    publishBool(gripper_pub, false);
    rclcpp::sleep_for(900ms);
    if (!g2_chapter11_moveit::attachRedToGripper(scene)) {
      RCLCPP_ERROR(node->get_logger(), "无法在 MoveIt 规划场景中附着红色物体");
      ok = false;
    }
    publishBool(attach_pub, ok);
    rclcpp::sleep_for(300ms);
    if (ok) {
      ok = g2_chapter11_moveit::planAndExecuteWithLocalReplan(
        move_group, kLift, "夹持后抬升");
    }
  }

  if (ok) {
    RCLCPP_INFO(node->get_logger(), "红色物体抓取完成，规划轨迹已避开黄色阻挡物");
  } else {
    RCLCPP_ERROR(node->get_logger(), "抓取流程失败，请检查 /joint_states、action server 和规划场景");
  }
  executor.cancel();
  spinner.join();
  rclcpp::shutdown();
  return ok ? 0 : 1;
}
