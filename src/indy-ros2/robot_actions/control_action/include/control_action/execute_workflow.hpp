#pragma once

#include <chrono>
#include <memory>
#include <rclcpp/rclcpp.hpp>

namespace control_action
{

struct ExecuteWorkflow
{
    template <typename Controller>
    static void execute(Controller& self,
                        const std::shared_ptr<typename Controller::GoalHandleControlRobot> goal_handle)
    {
        self.success_count = 0;
        auto timeout = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (!self.target_ready_ || !self.time_recieved_ || !self.obs_ready ||
               !self.config_received_) {
            if (std::chrono::steady_clock::now() > timeout) {
                auto result = std::make_shared<typename Controller::ControlRobot::Result>();
                result->success = false;
                result->message = "Timeout waiting for target/time";
                RCLCPP_ERROR(self.get_logger(), "Execute timeout!");
                goal_handle->abort(result);
                self.target_ready_ = false;
                self.time_recieved_ = false;
                self.obs_ready = false;
                self.publish_signal(false);
                return;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }

        auto feedback = std::make_shared<typename Controller::ControlRobot::Feedback>();
        auto result = std::make_shared<typename Controller::ControlRobot::Result>();
        self.captureTargetBaseTransform();
        self.rebuildTargetPoseList();

        for (size_t i = 0; i < self.target_position_.size(); i++) {
            self.applyOctomapForIdx(static_cast<int>(i));
            self.pass_all_ = false;
            self.bypass = false;
            self.publisher_callback(true, self.now().seconds(), true, self.mul_mode_);
            std::this_thread::sleep_for(std::chrono::milliseconds(100));

            RCLCPP_INFO(self.get_logger(),
                        "[Loop Debug] i=%zu / total=%zu | mul_mode=%s | time=%.3f",
                        i,
                        self.target_position_.size(),
                        self.mul_mode_ ? "true" : "false",
                        self.now().seconds());

            self.posecheck_and_recompute(self.target_position_[i], self.home_position_, i);
            self.target_pose = self.targetPositionToBasePose(i);

            RCLCPP_INFO(self.get_logger(),
                        "Updated target: x=%.2f y=%.2f z=%.2f r=%.2f p=%.2f y=%.2f",
                        self.target_position_[i][0],
                        self.target_position_[i][1],
                        self.target_position_[i][2],
                        self.target_position_[i][3],
                        self.target_position_[i][4],
                        self.target_position_[i][5]);

            self.test_pose = self.offsetPose(self.target_pose, 0.0, self.offset_distance_, 0.0);

            if (!self.refreshPlanningScene()) {
                RCLCPP_ERROR(self.get_logger(),
                             "Failed to refresh planning scene, skip iteration %zu", i);
                continue;
            }

            if (self.pose_check &&
                !self.checkCollisionAtTarget(self.test_pose) &&
                !self.checkCollisionAtTarget(self.target_pose)) {
                RCLCPP_INFO(self.get_logger(), "DEBUG CONSUME: CAN REACH TO TARGET");
                self.ws_check = true;
            } else {
                RCLCPP_ERROR(self.get_logger(), "DEBUG CONSUME: UNABLE TO REACH TO TARGET");
                self.ws_check = false;
            }

            self.obs_check_1 =
                self.checkCollisionAtTarget(
                    self.offsetPose(self.target_pose, self.y_offset_distance_, 0.0, 0.0));
            self.obs_check_2 =
                self.checkCollisionAtTarget(
                    self.offsetPose(self.target_pose, 0.0, 0.0, self.offset_angle_));
            self.obs_check_3 =
                self.checkCollisionAtTarget(
                    self.offsetPose(self.target_pose, 0.0, 0.0, -self.offset_angle_));

            if (!self.obs_check_1) {
                self.next_pose =
                    self.offsetPose(self.target_pose, self.y_offset_distance_, 0.0, 0.0);
                RCLCPP_ERROR(self.get_logger(), "CASE 1: Y-Offset");
            } else if (!self.obs_check_2) {
                self.next_pose =
                    self.offsetPose(self.target_pose, 0.0, 0.0, self.offset_angle_);
                RCLCPP_ERROR(self.get_logger(), "CASE 2: Z-Offset");
            } else if (!self.obs_check_3) {
                self.next_pose =
                    self.offsetPose(self.target_pose, 0.0, 0.0, -self.offset_angle_);
                RCLCPP_ERROR(self.get_logger(), "CASE 3: -Z-Offset");
            }

            if (!self.ws_check) {
                continue;
            }

            if (!self.go_home_) {
                self.callMoveToHome(self.home_position_, 9);
                self.go_home_ = true;
            }

            feedback->progress = 0.0;
            goal_handle->publish_feedback(feedback);

            self.callMoveRobot(
                self.offsetPose(self.target_pose, 0.0, self.offset_distance_, 0.0),
                self.target_pose,
                1,
                2);

            self.sendGripperCommand(0.8, 2);
            feedback->progress = 0.10;
            goal_handle->publish_feedback(feedback);

            //self.setOctomapCollision(true);
            self.callMoveRobot(
                self.target_pose,
                self.offsetPose(self.target_pose, 0.0, self.offset_distance_, 0.0),
                3,
                1);

            feedback->progress = 0.25;
            goal_handle->publish_feedback(feedback);

            self.sendGripperCommand(0.0, 4);

            feedback->progress = 0.40;
            goal_handle->publish_feedback(feedback);

            self.callMoveRobot(
                self.offsetPose(self.target_pose, 0.0, self.offset_distance_, 0.0),
                self.next_pose,
                5,
                1);
            //self.setOctomapCollision(false);
            self.checkCollisionAtTarget(self.next_pose);

            feedback->progress = 0.55;
            goal_handle->publish_feedback(feedback);

            self.callMoveToHome(self.drop_position_, 6);
            feedback->progress = 0.70;
            goal_handle->publish_feedback(feedback);

            self.sendGripperCommand(0.8, 7);
            feedback->progress = 0.85;
            goal_handle->publish_feedback(feedback);

            self.sendGripperCommand(0.0, 8);
            feedback->progress = 0.90;
            goal_handle->publish_feedback(feedback);

            self.callMoveToHome(self.home_position_, 9);
            feedback->progress = 1.0;
            goal_handle->publish_feedback(feedback);

            result->message = "Robot come to home!";
            RCLCPP_INFO(self.get_logger(), "time: %f", self.now().seconds());
            result->success = true;
            goal_handle->succeed(result);
            self.success_count++;
            break;
        }

        self.time_publisher(self.now().seconds());
        if (self.mul_mode_) {
            self.publisher_callback(true, 0.0, false, true);
            self.publisher_callback(false, self.now().seconds(), true, self.mul_mode_);
        } else {
            self.publisher_callback(true, 0.0, false);
            self.publisher_callback(false, self.now().seconds());
        }

        self.is_robot_moving_ = false;
        self.target_ready_ = false;
        self.time_recieved_ = false;
        self.obs_ready = false;
        if (self.success_count == 0) {
            self.publish_skip_signal(true);
            self.publish_move_signal(true);
        }
        self.publish_signal(false);
    }
};

}  // namespace control_action
