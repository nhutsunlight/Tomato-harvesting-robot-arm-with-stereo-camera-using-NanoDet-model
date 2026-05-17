#pragma once

#include <chrono>
#include <future>
#include <memory>
#include <vector>

#include <geometry_msgs/msg/pose.hpp>
#include <moveit/collision_detection/collision_common.h>
#include <moveit/planning_scene/planning_scene.h>
#include <moveit/robot_state/robot_state.h>
#include <moveit_msgs/msg/planning_scene_components.hpp>
#include <moveit_msgs/srv/get_planning_scene.hpp>
#include <rclcpp/rclcpp.hpp>

namespace control_action
{

struct ControlChecks
{
    template <typename Controller>
    static bool PosesCheck(Controller& self,
                           const geometry_msgs::msg::Pose& input_pose,
                           const std::vector<double>& joint_values)
    {
        const auto robot_model = self.move_group_interface_->getRobotModel();
        if (!robot_model) {
            RCLCPP_ERROR(self.get_logger(), "Robot model is null.");
            return false;
        }

        moveit::core::RobotState robot_state(robot_model);
        const auto* joint_model_group =
            robot_state.getJointModelGroup(self.move_group_interface_->getName());
        if (!joint_model_group) {
            RCLCPP_ERROR(self.get_logger(), "Joint model group '%s' not found.",
                         self.move_group_interface_->getName().c_str());
            return false;
        }

        if (joint_values.size() != joint_model_group->getVariableCount()) {
            RCLCPP_ERROR(self.get_logger(),
                         "Mismatch joint count: expected %zu, got %zu.",
                         static_cast<size_t>(joint_model_group->getVariableCount()),
                         joint_values.size());
            return false;
        }

        robot_state.setJointGroupPositions(joint_model_group, joint_values);
        const bool found_ik = robot_state.setFromIK(joint_model_group, input_pose, 0.1);

        if (found_ik || !self.recompute) {
            RCLCPP_INFO(self.get_logger(), "Pose is reachable from provided joint state.");
        } else {
            RCLCPP_WARN(self.get_logger(), "Pose is NOT reachable from provided joint state.");
        }

        return found_ik;
    }

    template <typename Controller>
    static void posecheck_and_recompute(Controller& self,
                                        const std::array<double, 6>& test_position,
                                        const std::vector<double>& test_joint_values,
                                        std::size_t idx)
    {
        self.test_position_ref = test_position;
        self.test_position_ref_offset = self.test_position_ref;
        self.found_test_ik = false;
        self.recompute = true;

        const int max_iterations = 10;
        int iteration = 0;

        while (rclcpp::ok() && iteration < max_iterations) {
            iteration++;
            self.compute_offset_position(
                self.test_position_ref[0], self.test_position_ref[1], self.test_position_ref[2],
                self.test_position_ref[3], self.test_position_ref[4], self.test_position_ref[5],
                self.object_offset_,
                self.test_position_ref_offset[0],
                self.test_position_ref_offset[1],
                self.test_position_ref_offset[2]);
            self.test_position_ref_offset[3] = self.test_position_ref[3];
            self.test_position_ref_offset[4] = self.test_position_ref[4];
            self.input_test_pose = self.transformToBaseFrame(self.test_position_ref_offset);
            self.found_test_ik = self.PosesCheck(self.input_test_pose, test_joint_values);

            if (self.found_test_ik) {
                RCLCPP_INFO(self.get_logger(), "IK found at iteration %d", iteration);
                self.target_position_[idx][3] = self.test_position_ref[3];
                self.target_position_[idx][4] = self.test_position_ref[4];
                break;
            }

            if (!self.recompute) {
                RCLCPP_WARN(self.get_logger(),
                            "Recompute converged but IK still not found for idx %zu", idx);
                break;
            }

            RCLCPP_WARN(self.get_logger(), "Recompute for pose idx %zu, iteration %d",
                        idx, iteration);
            auto [test_roll, test_pitch] = self.computeRollPitchFromXYZ(
                self.test_position_ref[0], self.test_position_ref[1], self.test_position_ref[2],
                self.test_position_ref[3], self.test_position_ref[4]);
            self.test_position_ref[3] = test_roll;
            self.test_position_ref[4] = test_pitch;
        }

        if (iteration >= max_iterations) {
            RCLCPP_ERROR(self.get_logger(), "Max recompute iterations reached for idx %zu", idx);
        }

        self.pose_check = self.found_test_ik;
        self.recompute = false;
    }

    template <typename Controller>
    static bool refreshPlanningScene(Controller& self)
    {
        using GetPlanningScene = moveit_msgs::srv::GetPlanningScene;

        if (!self.planning_scene_client_->wait_for_service(std::chrono::seconds(3))) {
            RCLCPP_ERROR(self.get_logger(), "Service /get_planning_scene not available");
            return false;
        }

        auto request = std::make_shared<GetPlanningScene::Request>();
        request->components.components =
            moveit_msgs::msg::PlanningSceneComponents::SCENE_SETTINGS |
            moveit_msgs::msg::PlanningSceneComponents::ROBOT_STATE |
            moveit_msgs::msg::PlanningSceneComponents::WORLD_OBJECT_GEOMETRY |
            moveit_msgs::msg::PlanningSceneComponents::ALLOWED_COLLISION_MATRIX |
            moveit_msgs::msg::PlanningSceneComponents::OCTOMAP;

        std::promise<GetPlanningScene::Response::SharedPtr> promise;
        auto fut = promise.get_future();
        self.planning_scene_client_->async_send_request(
            request,
            [&promise](rclcpp::Client<GetPlanningScene>::SharedFuture f) {
                promise.set_value(f.get());
            });

        if (fut.wait_for(std::chrono::seconds(3)) != std::future_status::ready) {
            RCLCPP_ERROR(self.get_logger(), "Timeout getting planning scene");
            return false;
        }

        auto response = fut.get();
        auto robot_model = self.move_group_interface_->getRobotModel();

        std::lock_guard<std::mutex> lock(self.scene_mutex_);
        self.cached_scene_ = std::make_shared<planning_scene::PlanningScene>(robot_model);
        self.cached_scene_->setPlanningSceneDiffMsg(response->scene);
        self.scene_valid_ = true;
        return true;
    }

    template <typename Controller>
    static bool checkCollisionAtTarget(Controller& self,
                                       const geometry_msgs::msg::Pose& target_pose)
    {
        std::lock_guard<std::mutex> lock(self.scene_mutex_);
        if (!self.scene_valid_ || !self.cached_scene_) {
            RCLCPP_WARN(self.get_logger(), "Scene not ready");
            return false;
        }

        auto robot_model = self.move_group_interface_->getRobotModel();
        moveit::core::RobotState robot_state(robot_model);
        robot_state.setToDefaultValues();

        const moveit::core::JointModelGroup* jmg =
            robot_model->getJointModelGroup("indy_manipulator");

        bool ik_found = robot_state.setFromIK(jmg, target_pose, 1.0);
        if (!ik_found) {
            RCLCPP_WARN(self.get_logger(), "IK not found for target pose");
            return false;
        }
        robot_state.update();

        collision_detection::CollisionRequest collision_request;
        collision_detection::CollisionResult collision_result;
        collision_request.contacts = true;
        collision_request.max_contacts = 10;

        self.cached_scene_->checkCollision(collision_request, collision_result, robot_state);

        if (collision_result.collision) {
            RCLCPP_WARN(self.get_logger(), "Collision detected at target pose!");
            for (const auto& contact : collision_result.contacts) {
                RCLCPP_WARN(self.get_logger(), "  Contact: %s <-> %s",
                            contact.first.first.c_str(), contact.first.second.c_str());
            }
            return true;
        }

        RCLCPP_INFO(self.get_logger(), "No collision at target pose.");
        return false;
    }
};

}  // namespace control_action
