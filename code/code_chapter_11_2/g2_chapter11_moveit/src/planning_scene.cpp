#include "g2_chapter11_moveit/planning_scene.hpp"

#include <moveit_msgs/msg/attached_collision_object.hpp>
#include <moveit_msgs/msg/collision_object.hpp>
#include <shape_msgs/msg/solid_primitive.hpp>

#include <array>
#include <string>
#include <vector>

namespace g2_chapter11_moveit
{
namespace
{
moveit_msgs::msg::CollisionObject makeBox(
  const std::string& id, const std::array<double, 3>& center,
  const std::array<double, 3>& size, const std::string& frame = "arm_base_link")
{
  moveit_msgs::msg::CollisionObject object;
  object.header.frame_id = frame;
  object.id = id;
  shape_msgs::msg::SolidPrimitive primitive;
  primitive.type = primitive.BOX;
  primitive.dimensions = {size[0], size[1], size[2]};
  geometry_msgs::msg::Pose pose;
  pose.orientation.w = 1.0;
  pose.position.x = center[0];
  pose.position.y = center[1];
  pose.position.z = center[2];
  object.primitives.push_back(primitive);
  object.primitive_poses.push_back(pose);
  object.operation = object.ADD;
  return object;
}
}  // namespace

void addTeachingScene(moveit::planning_interface::PlanningSceneInterface& scene)
{
  // 桌子、阻挡物和非目标物体由 G2 头部深度点云生成的 OctoMap 提供。
  // 这里只保留抓取目标，便于最后接近时移除并在抓取后附着。
  scene.applyCollisionObject(
    makeBox("red_object", {0.56, 0.535, -0.43}, {0.075, 0.075, 0.075}));
}

void removeRedForFinalApproach(moveit::planning_interface::PlanningSceneInterface& scene)
{
  // 预抓取之前红色物体是普通障碍；进入最后几厘米时允许夹爪接触目标。
  moveit_msgs::msg::CollisionObject remove;
  remove.header.frame_id = "arm_base_link";
  remove.id = "red_object";
  remove.operation = remove.REMOVE;
  scene.applyCollisionObject(remove);
}

bool attachRedToGripper(moveit::planning_interface::PlanningSceneInterface& scene)
{
  // 抓住后将物体改为附着碰撞体，后续抬升规划会连同物体一起避障。
  moveit_msgs::msg::AttachedCollisionObject attached;
  attached.link_name = "gripper_r_center_link";
  attached.touch_links = {"gripper_r_center_link"};
  attached.object = makeBox(
    "red_object", {0.0, 0.0, 0.0}, {0.075, 0.075, 0.075},
    "gripper_r_center_link");
  return scene.applyAttachedCollisionObject(attached);
}
}  // namespace g2_chapter11_moveit
