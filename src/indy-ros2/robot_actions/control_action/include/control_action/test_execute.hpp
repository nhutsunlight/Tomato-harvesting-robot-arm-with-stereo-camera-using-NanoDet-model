#pragma once

#include <array>
#include <chrono>
#include <cmath>
#include <memory>
#include <rclcpp/rclcpp.hpp>
#include <vector>

namespace control_action
{

inline double poseDistance(const std::array<double, 6>& a,
                           const std::array<double, 6>& b)
{
    const double dx = a[0] - b[0];
    const double dy = a[1] - b[1];
    const double dz = a[2] - b[2];
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

struct ClusterEntry
{
    std::array<double, 6> pose;
    size_t original_idx;
};

inline std::vector<std::vector<ClusterEntry>> clusterByDistance(
    const std::vector<std::array<double, 6>>& poses,
    double threshold = 0.15)
{
    std::vector<std::vector<ClusterEntry>> clusters;
    std::vector<bool> assigned(poses.size(), false);

    for (size_t i = 0; i < poses.size(); ++i) {
        if (assigned[i]) {
            continue;
        }

        std::vector<ClusterEntry> cluster;
        cluster.push_back({poses[i], i});
        assigned[i] = true;

        for (size_t j = i + 1; j < poses.size(); ++j) {
            if (!assigned[j] && poseDistance(poses[i], poses[j]) <= threshold) {
                cluster.push_back({poses[j], j});
                assigned[j] = true;
            }
        }
        clusters.push_back(cluster);
    }

    return clusters;
}

struct ValidTarget
{
    geometry_msgs::msg::Pose pose;
    size_t original_idx;
};

struct TestExecuteWorkflow
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

        constexpr double kClusterDistanceMeters = 0.15;
        auto clusters = clusterByDistance(self.target_position_, kClusterDistanceMeters);

        RCLCPP_INFO(self.get_logger(),
                    "[Cluster] %zu raw targets -> %zu cluster(s) (threshold=%.3f m)",
                    self.target_position_.size(), clusters.size(), kClusterDistanceMeters);

        for (size_t ci = 0; ci < clusters.size(); ++ci) {
            const auto& cluster = clusters[ci];
            RCLCPP_INFO(self.get_logger(), "[Cluster %zu/%zu] %zu member(s)",
                        ci + 1, clusters.size(), cluster.size());

            bool cluster_started = false;
            ValidTarget previous_target;
            int step_id = 3;

            for (size_t mi = 0; mi < cluster.size(); ++mi) {
                const auto& raw = cluster[mi].pose;
                const size_t orig_i = cluster[mi].original_idx;

                self.applyOctomapForIdx(static_cast<int>(orig_i));
                self.pass_all_ = false;
                self.bypass = false;
                self.publisher_callback(true, self.now().seconds(), true, self.mul_mode_);
                std::this_thread::sleep_for(std::chrono::milliseconds(100));

                RCLCPP_INFO(self.get_logger(),
                            "[Loop Debug] i=%zu / total=%zu | mul_mode=%s | time=%.3f",
                            orig_i, cluster.size(),
                            self.mul_mode_ ? "true" : "false",
                            self.now().seconds());

                self.posecheck_and_recompute(raw, self.home_position_, orig_i);
                const auto& checked_target = self.target_position_[orig_i];
                self.target_pose = self.targetPositionToBasePose(orig_i);

                RCLCPP_INFO(self.get_logger(),
                            "Updated target: x=%.2f y=%.2f z=%.2f r=%.2f p=%.2f y=%.2f",
                            checked_target[0], checked_target[1], checked_target[2],
                            checked_target[3], checked_target[4], checked_target[5]);

                if (!self.refreshPlanningScene()) {
                    RCLCPP_ERROR(self.get_logger(),
                                 "Failed to refresh planning scene, skip iteration %zu",
                                 orig_i);
                    continue;
                }

                if (self.pose_check && !self.checkCollisionAtTarget(self.target_pose)) {
                    RCLCPP_INFO(self.get_logger(), "DEBUG CONSUME: CAN REACH TO TARGET");
                    self.ws_check = true;
                } else {
                    RCLCPP_ERROR(self.get_logger(), "DEBUG CONSUME: UNABLE TO REACH TO TARGET");
                    self.ws_check = false;
                }

                if (!self.ws_check) {
                    continue;
                }

                const ValidTarget current_target{self.target_pose, orig_i};
                if (!cluster_started) {
                    if (!self.go_home_) {
                        self.callMoveToHome(self.home_position_, 9);
                        self.go_home_ = true;
                    }

                    feedback->progress = 0.0;
                    goal_handle->publish_feedback(feedback);

                    self.callMoveRobot(
                        self.offsetPose(current_target.pose, 0.0, self.offset_distance_, 0.0),
                        current_target.pose,
                        1,
                        2);

                    //self.setOctomapCollision(true);
                    self.sendGripperCommand(0.8, 2);
                    feedback->progress = 0.10;
                    goal_handle->publish_feedback(feedback);

                    self.sendGripperCommand(0.0, step_id++);
                    //self.setOctomapCollision(false);

                    previous_target = current_target;
                    cluster_started = true;
                    continue;
                }

                //self.setOctomapCollision(true);
                self.callMoveRobot(previous_target.pose, current_target.pose, step_id++, 1);
                self.sendGripperCommand(0.8, step_id++);
                self.sendGripperCommand(0.0, step_id++);
                //self.setOctomapCollision(false);

                const float ratio =
                    static_cast<float>(mi + 1) / static_cast<float>(cluster.size());
                feedback->progress = 0.10f + 0.15f * ratio;
                goal_handle->publish_feedback(feedback);

                previous_target = current_target;
            }

            if (!cluster_started) {
                RCLCPP_WARN(self.get_logger(),
                            "[Cluster %zu] No reachable targets - trying next cluster",
                            ci + 1);
                continue;
            }

            self.callMoveToHome(self.home_position_, step_id++);
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
