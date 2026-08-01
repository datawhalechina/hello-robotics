#include <control_msgs/action/follow_joint_trajectory.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <trajectory_msgs/msg/joint_trajectory_point.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

namespace g2_chapter11_moveit
{
class IsaacControllerBridge : public rclcpp::Node
{
public:
  using Follow = control_msgs::action::FollowJointTrajectory;
  using GoalHandle = rclcpp_action::ServerGoalHandle<Follow>;

  IsaacControllerBridge()
  : Node("chapter11_isaac_controller_bridge")
  {
    joint_names_ = {
      "idx61_arm_r_joint1", "idx62_arm_r_joint2", "idx63_arm_r_joint3",
      "idx64_arm_r_joint4", "idx65_arm_r_joint5", "idx66_arm_r_joint6",
      "idx67_arm_r_joint7"};
    command_pub_ = create_publisher<sensor_msgs::msg::JointState>(
      "/chapter11/isaac_joint_command", 30);
    joint_state_pub_ = create_publisher<sensor_msgs::msg::JointState>("/joint_states", 30);
    state_sub_ = create_subscription<sensor_msgs::msg::JointState>(
      "/chapter11/isaac_joint_states", 30,
      std::bind(&IsaacControllerBridge::stateCallback, this, std::placeholders::_1));
    action_server_ = rclcpp_action::create_server<Follow>(
      this, "/right_arm_controller/follow_joint_trajectory",
      std::bind(&IsaacControllerBridge::handleGoal, this, std::placeholders::_1,
                std::placeholders::_2),
      std::bind(&IsaacControllerBridge::handleCancel, this, std::placeholders::_1),
      std::bind(&IsaacControllerBridge::handleAccepted, this, std::placeholders::_1));
    RCLCPP_INFO(get_logger(),
                "FollowJointTrajectory C++ 桥已启动，等待 Isaac 关节反馈");
  }

private:
  static double seconds(const builtin_interfaces::msg::Duration& value)
  {
    return static_cast<double>(value.sec) + 1e-9 * static_cast<double>(value.nanosec);
  }

  void stateCallback(const sensor_msgs::msg::JointState::SharedPtr message)
  {
    if (message->name != joint_names_ || message->position.size() != joint_names_.size()) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 3000,
        "忽略名称、顺序或长度不匹配的 Isaac 关节反馈");
      return;
    }
    {
      std::lock_guard<std::mutex> guard(state_mutex_);
      current_positions_ = message->position;
      have_state_ = true;
    }
    auto filtered = *message;
    filtered.header.stamp = now();
    joint_state_pub_->publish(filtered);
  }

  rclcpp_action::GoalResponse handleGoal(
    const rclcpp_action::GoalUUID&,
    std::shared_ptr<const Follow::Goal> goal)
  {
    if (busy_.load()) {
      RCLCPP_WARN(get_logger(), "已有轨迹正在执行，拒绝新目标");
      return rclcpp_action::GoalResponse::REJECT;
    }
    if (goal->trajectory.joint_names != joint_names_ || goal->trajectory.points.empty()) {
      RCLCPP_ERROR(get_logger(), "轨迹关节名称/顺序错误，或轨迹为空");
      return rclcpp_action::GoalResponse::REJECT;
    }
    return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
  }

  rclcpp_action::CancelResponse handleCancel(const std::shared_ptr<GoalHandle>)
  {
    return rclcpp_action::CancelResponse::ACCEPT;
  }

  void handleAccepted(const std::shared_ptr<GoalHandle> goal_handle)
  {
    std::thread(&IsaacControllerBridge::execute, this, goal_handle).detach();
  }

  bool readState(std::vector<double>& output)
  {
    std::lock_guard<std::mutex> guard(state_mutex_);
    if (!have_state_) {
      return false;
    }
    output = current_positions_;
    return true;
  }

  void publishCommand(const std::vector<double>& positions)
  {
    sensor_msgs::msg::JointState command;
    command.header.stamp = now();
    command.name = joint_names_;
    command.position = positions;
    command_pub_->publish(command);
  }

  static std::vector<double> interpolate(
    const std::vector<trajectory_msgs::msg::JointTrajectoryPoint>& points,
    double elapsed)
  {
    const auto first_time = seconds(points.front().time_from_start);
    if (elapsed <= first_time || points.size() == 1) {
      return points.front().positions;
    }
    for (std::size_t i = 1; i < points.size(); ++i) {
      const double right_time = seconds(points[i].time_from_start);
      if (elapsed <= right_time) {
        const double left_time = seconds(points[i - 1].time_from_start);
        const double denominator = std::max(1e-9, right_time - left_time);
        const double ratio = std::clamp((elapsed - left_time) / denominator, 0.0, 1.0);
        std::vector<double> result(points[i].positions.size());
        for (std::size_t joint = 0; joint < result.size(); ++joint) {
          result[joint] = points[i - 1].positions[joint] + ratio *
            (points[i].positions[joint] - points[i - 1].positions[joint]);
        }
        return result;
      }
    }
    return points.back().positions;
  }

  void finishWithError(
    const std::shared_ptr<GoalHandle>& goal_handle, int32_t code,
    const std::string& text)
  {
    auto result = std::make_shared<Follow::Result>();
    result->error_code = code;
    result->error_string = text;
    goal_handle->abort(result);
    busy_.store(false);
  }

  void execute(const std::shared_ptr<GoalHandle> goal_handle)
  {
    busy_.store(true);
    const auto points = goal_handle->get_goal()->trajectory.points;
    for (int wait = 0; rclcpp::ok() && wait < 500; ++wait) {
      std::vector<double> state;
      if (readState(state)) {
        break;
      }
      if (wait == 499) {
        finishWithError(goal_handle, Follow::Result::INVALID_GOAL,
                        "5 秒内未收到 Isaac 关节反馈");
        return;
      }
      std::this_thread::sleep_for(10ms);
    }

    const double final_time = seconds(points.back().time_from_start);
    const auto begin = std::chrono::steady_clock::now();
    rclcpp::Rate rate(100.0);
    while (rclcpp::ok()) {
      if (goal_handle->is_canceling()) {
        auto result = std::make_shared<Follow::Result>();
        result->error_code = Follow::Result::SUCCESSFUL;
        result->error_string = "轨迹已取消";
        goal_handle->canceled(result);
        busy_.store(false);
        return;
      }
      const double elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - begin).count();
      const auto desired = interpolate(points, elapsed);
      publishCommand(desired);

      std::vector<double> actual;
      if (readState(actual)) {
        auto feedback = std::make_shared<Follow::Feedback>();
        feedback->header.stamp = now();
        feedback->joint_names = joint_names_;
        feedback->desired.positions = desired;
        feedback->actual.positions = actual;
        feedback->error.positions.resize(joint_names_.size());
        for (std::size_t i = 0; i < joint_names_.size(); ++i) {
          feedback->error.positions[i] = desired[i] - actual[i];
        }
        goal_handle->publish_feedback(feedback);
      }
      if (elapsed >= final_time) {
        break;
      }
      rate.sleep();
    }

    const auto final_target = points.back().positions;
    const auto settle_begin = std::chrono::steady_clock::now();
    while (rclcpp::ok()) {
      publishCommand(final_target);
      std::vector<double> actual;
      if (readState(actual)) {
        double maximum_error = 0.0;
        for (std::size_t i = 0; i < joint_names_.size(); ++i) {
          maximum_error = std::max(maximum_error, std::abs(final_target[i] - actual[i]));
        }
        if (maximum_error < 0.10) {
          auto result = std::make_shared<Follow::Result>();
          result->error_code = Follow::Result::SUCCESSFUL;
          goal_handle->succeed(result);
          busy_.store(false);
          return;
        }
      }
      const double settling = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - settle_begin).count();
      if (settling > 8.0) {
        finishWithError(goal_handle, Follow::Result::GOAL_TOLERANCE_VIOLATED,
                        "Isaac 8 秒内最终关节跟踪误差仍超过 0.10 rad");
        return;
      }
      rate.sleep();
    }
    finishWithError(goal_handle, Follow::Result::INVALID_GOAL, "ROS 2 已关闭");
  }

  std::vector<std::string> joint_names_;
  std::vector<double> current_positions_;
  bool have_state_{false};
  std::mutex state_mutex_;
  std::atomic<bool> busy_{false};
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr state_sub_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr command_pub_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr joint_state_pub_;
  rclcpp_action::Server<Follow>::SharedPtr action_server_;
};
}  // namespace g2_chapter11_moveit

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<g2_chapter11_moveit::IsaacControllerBridge>());
  rclcpp::shutdown();
  return 0;
}
