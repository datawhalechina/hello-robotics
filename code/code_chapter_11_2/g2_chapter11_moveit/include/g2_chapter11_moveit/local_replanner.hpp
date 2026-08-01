#pragma once

#include <moveit/move_group_interface/move_group_interface.h>
#include <string>
#include <vector>

namespace g2_chapter11_moveit
{
bool planAndExecuteWithLocalReplan(
  moveit::planning_interface::MoveGroupInterface& move_group,
  const std::vector<double>& joint_target, const std::string& label);
}
