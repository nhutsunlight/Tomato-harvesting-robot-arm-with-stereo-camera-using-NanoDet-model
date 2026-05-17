#pragma once

#include <chrono>
#include <geometry_msgs/msg/pose.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <vector>

namespace control_action
{

struct ActionRequests
{
    template <typename Controller, typename ClientPtr>
    static bool ensureActionServerReady(Controller& self,
                                        ClientPtr& client,
                                        bool& ready_cache,
                                        const char* action_name)
    {
        if (ready_cache) {
            return true;
        }

        if (!client->wait_for_action_server(std::chrono::seconds(5))) {
            RCLCPP_ERROR(self.get_logger(), "Action server '%s' không khả dụng!", action_name);
            return false;
        }

        ready_cache = true;
        return true;
    }

    template <typename Controller, typename Future>
    static bool waitForGoalResponse(Controller& self,
                                    Future& future_goal,
                                    const char* action_name)
    {
        if (future_goal.wait_for(std::chrono::seconds(2)) != std::future_status::ready) {
            RCLCPP_ERROR(self.get_logger(),
                         "Timeout waiting for '%s' goal response", action_name);
            return false;
        }
        return true;
    }

    template <typename Controller>
    static void callMoveToHome(Controller& self,
                               const std::vector<double>& joint_positions,
                               size_t id,
                               size_t pass_permit = 0)
    {
        if (!ensureActionServerReady(
                self, self.move_to_home_client_, self.home_action_ready_, "move_to_home")) {
            return;
        }

        auto goal_msg = typename Controller::MoveToHome::Goal();
        goal_msg.joint_positions = joint_positions;
        goal_msg.id = id;
        goal_msg.pass_permit = pass_permit;

        auto send_goal_options =
            typename rclcpp_action::Client<typename Controller::MoveToHome>::SendGoalOptions();

        if ((self.bypass && id != 9) || (self.pass_all_ && id != 9)) {
            return;
        }

        auto future_goal = self.move_to_home_client_->async_send_goal(goal_msg, send_goal_options);
        if (!waitForGoalResponse(self, future_goal, "move_to_home")) {
            self.home_action_ready_ = false;
            return;
        }
        auto goal_handle = future_goal.get();
        if (!goal_handle) {
            RCLCPP_ERROR(self.get_logger(), "Gửi action goal thất bại!");
            return;
        }

        auto future_result = self.move_to_home_client_->async_get_result(goal_handle);
        auto result = future_result.get();
        if (result.code == rclcpp_action::ResultCode::SUCCEEDED) {
            RCLCPP_INFO(self.get_logger(), "Move to home thành công: %s",
                        result.result->message.c_str());
        } else {
            RCLCPP_ERROR(self.get_logger(), "Move to home thất bại!");
            self.setOctomapCollision(true);
            self.callMoveRobot(
                self.offsetPose(self.jointStatesToPose(joint_positions), 0.0, 0.5, 0.0),
                self.jointStatesToPose(joint_positions),
                id,
                2);
            self.setOctomapCollision(false);
        }
    }

    template <typename Controller>
    static void callMoveRobot(Controller& self,
                              const geometry_msgs::msg::Pose& start_pose,
                              const geometry_msgs::msg::Pose& target_pose,
                              size_t id,
                              size_t mode)
    {
        if (self.pass_all_ || self.bypass) {
            return;
        }

        if (!ensureActionServerReady(
                self, self.move_client_, self.move_action_ready_, "robot_move_action")) {
            return;
        }

        auto goal_msg = typename Controller::MoveRobot::Goal();
        goal_msg.mode = mode;
        goal_msg.id = id;
        goal_msg.start_pose = start_pose;
        goal_msg.target_pose = target_pose;

        auto send_goal_options =
            typename rclcpp_action::Client<typename Controller::MoveRobot>::SendGoalOptions();
        auto future_goal = self.move_client_->async_send_goal(goal_msg, send_goal_options);
        if (!waitForGoalResponse(self, future_goal, "robot_move_action")) {
            self.move_action_ready_ = false;
            return;
        }
        auto goal_handle = future_goal.get();
        if (!goal_handle) {
            RCLCPP_ERROR(self.get_logger(), "Gửi action goal thất bại!");
            return;
        }

        auto future_result = self.move_client_->async_get_result(goal_handle);
        auto result = future_result.get();
        if (result.code == rclcpp_action::ResultCode::SUCCEEDED) {
            RCLCPP_INFO(self.get_logger(), "Move robot thành công: %s",
                        result.result->message.c_str());
        } else {
            RCLCPP_ERROR(self.get_logger(), "Move robot thất bại!");
            if (id == 1 || id == 30) {
                self.bypass = true;
            } else if (id == 3) {
                self.callMoveRobot(target_pose, self.offsetPose(target_pose, 0.0, 0.0, 0.0), 30, 0);
            }
        }
    }

    template <typename Controller>
    static void sendGripperCommand(Controller& self,
                                   double position,
                                   size_t id,
                                   size_t pass_permit = 0)
    {
        if ((self.pass_all_ || self.bypass) && !(self.bypass && id == 8)) {
            return;
        }

        if (!ensureActionServerReady(
                self, self.gripper_client_, self.gripper_action_ready_, "gripper_action")) {
            return;
        }

        while (rclcpp::ok()) {
            auto goal_msg = typename Controller::GripperControl::Goal();
            goal_msg.position = position;
            goal_msg.id = id;
            goal_msg.pass_permit = pass_permit;

            auto goal_handle_future = self.gripper_client_->async_send_goal(goal_msg);
            if (!waitForGoalResponse(self, goal_handle_future, "gripper_action")) {
                self.gripper_action_ready_ = false;
                return;
            }
            auto goal_handle = goal_handle_future.get();
            if (!goal_handle) {
                RCLCPP_ERROR(self.get_logger(), "Gửi lệnh gripper thất bại!");
                return;
            }

            auto result_future = self.gripper_client_->async_get_result(goal_handle);
            auto result = result_future.get();
            if (result.code == rclcpp_action::ResultCode::SUCCEEDED) {
                RCLCPP_INFO(self.get_logger(), "Gripper điều khiển thành công.");
                break;
            }

            RCLCPP_ERROR(self.get_logger(), "Gripper thất bại");
            self.resendGripperCommand();
        }
    }

    template <typename Controller>
    static void resendGripperCommand(Controller& self)
    {
        self.callMoveRobot(self.target_pose, self.target_pose, 1, 0);
        self.sendGripperCommand(0.8, 2000);
    }
};

}  // namespace control_action
