#include "g2_chapter11_moveit/local_replanner.hpp"

#include <rclcpp/rclcpp.hpp>

namespace g2_chapter11_moveit
{
bool planAndExecuteWithLocalReplan(
  moveit::planning_interface::MoveGroupInterface& move_group,
  const std::vector<double>& joint_target, const std::string& label)
{
  // MoveIt 中机械臂“局部避障”的常用基础做法：每次从当前实测状态重新规划，
  // 并在快速规划失败时切换到更充分的采样规划器。规划场景更新后可直接重用。
  const std::vector<std::string> planners = {
    "RRTConnectkConfigDefault", "RRTstarkConfigDefault", "PRMkConfigDefault"};
  for (std::size_t attempt = 0; attempt < planners.size(); ++attempt) {
    move_group.setStartStateToCurrentState();
    move_group.setJointValueTarget(joint_target);
    move_group.setPlannerId(planners[attempt]);
    move_group.setPlanningTime(attempt == 0 ? 4.0 : 7.0);
    moveit::planning_interface::MoveGroupInterface::Plan plan;
    const bool planned =
      move_group.plan(plan) == moveit::core::MoveItErrorCode::SUCCESS;
    if (!planned) {
      RCLCPP_WARN(rclcpp::get_logger("chapter11_local_replanner"),
                  "%s: %s 未找到路径，更新当前状态后重试",
                  label.c_str(), planners[attempt].c_str());
      continue;
    }
    RCLCPP_INFO(rclcpp::get_logger("chapter11_local_replanner"),
                "%s: %s 规划成功，共 %zu 个轨迹点",
                label.c_str(), planners[attempt].c_str(),
                plan.trajectory_.joint_trajectory.points.size());
    if (move_group.execute(plan) == moveit::core::MoveItErrorCode::SUCCESS) {
      return true;
    }
    RCLCPP_WARN(rclcpp::get_logger("chapter11_local_replanner"),
                "%s: 执行失败，从当前反馈状态重新规划", label.c_str());
  }
  return false;
}
}  // namespace g2_chapter11_moveit
