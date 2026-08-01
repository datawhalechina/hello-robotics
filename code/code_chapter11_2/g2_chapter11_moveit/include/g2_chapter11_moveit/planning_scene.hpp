#pragma once

#include <moveit/planning_scene_interface/planning_scene_interface.h>

namespace g2_chapter11_moveit
{
void addTeachingScene(moveit::planning_interface::PlanningSceneInterface& scene);
void removeRedForFinalApproach(moveit::planning_interface::PlanningSceneInterface& scene);
bool attachRedToGripper(moveit::planning_interface::PlanningSceneInterface& scene);
}
