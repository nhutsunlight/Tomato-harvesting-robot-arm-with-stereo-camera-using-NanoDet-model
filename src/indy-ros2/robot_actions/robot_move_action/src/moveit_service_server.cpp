#include <moveit/move_group_interface/move_group_interface.h>
#include <atomic>
#include <chrono>
#include <cmath>
#include <functional>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <memory>
#include <limits>
#include <moveit_msgs/msg/constraints.hpp>
#include <moveit_msgs/msg/orientation_constraint.hpp>
#include <rclcpp/rclcpp.hpp>
#include <string>
#include <thread>
#include <utility>
#include <vector>
#include <array>
#include <moveit/planning_scene/planning_scene.h>
#include <moveit/trajectory_processing/iterative_time_parameterization.h>
#include <moveit/robot_trajectory/robot_trajectory.h>

#include "robot_move_action/action/move_robot.hpp"
#include "rclcpp_action/rclcpp_action.hpp"
#include "connect_msgs/msg/connect_msg.hpp"

using moveit::planning_interface::MoveGroupInterface;

class MoveItController : public rclcpp::Node {
public:
    MoveItController() : Node("moveit_controller") {
        action_server_ = rclcpp_action::create_server<MoveRobot>(
            this, "robot_move_action",
            std::bind(&MoveItController::handle_goal,     this, std::placeholders::_1, std::placeholders::_2),
            std::bind(&MoveItController::handle_cancel,   this, std::placeholders::_1),
            std::bind(&MoveItController::handle_accepted, this, std::placeholders::_1));

        RCLCPP_INFO(get_logger(), "MoveIt Action Server started.");
    }

    void initialize() {
        move_group_ = std::make_unique<MoveGroupInterface>(shared_from_this(), "indy_manipulator");
        move_group_->setPlanningTime(2.0);
        move_group_->setPlannerId("RRTConnectkConfigDefault");
        move_group_->setMaxVelocityScalingFactor(1.0);
        move_group_->setMaxAccelerationScalingFactor(1.0);
        robot_model_ = move_group_->getRobotModel();
        jmg_         = robot_model_->getJointModelGroup("indy_manipulator");
    }

private:
    static constexpr double CARTESIAN_VALIDATION_FRACTION = 0.95;
    static constexpr double CARTESIAN_MIN_FRACTION = 0.99;
    static constexpr double CARTESIAN_FULL_FRACTION = 0.999;
    static constexpr double MAX_CARTESIAN_START_ERROR_M = 0.02;

    using MoveRobot           = robot_move_action::action::MoveRobot;
    using GoalHandleMoveRobot = rclcpp_action::ServerGoalHandle<MoveRobot>;

    std::unique_ptr<MoveGroupInterface>         move_group_;
    moveit::core::RobotModelConstPtr            robot_model_;
    const moveit::core::JointModelGroup*        jmg_ = nullptr;
    rclcpp_action::Server<MoveRobot>::SharedPtr action_server_;

    size_t                   mode_ = 0;
    geometry_msgs::msg::Pose target_pose_;
    geometry_msgs::msg::Pose start_pose_;
    geometry_msgs::msg::Pose next_pose_;
    std::atomic<bool>        is_reset_{false};
    const bool avoid_collisions = true;

    // =========================================================
    rclcpp_action::GoalResponse handle_goal(
        const rclcpp_action::GoalUUID&,
        std::shared_ptr<const MoveRobot::Goal> goal)
    {
        if (is_reset_) return rclcpp_action::GoalResponse::REJECT;

        mode_ = goal->mode;
        switch (mode_) {
            case 0:
                target_pose_ = goal->target_pose;
                return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
            case 1:
                start_pose_  = goal->start_pose;
                target_pose_ = goal->target_pose;
                return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
            case 2:
                target_pose_ = goal->target_pose;
                next_pose_   = goal->start_pose;
                return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
            default:
                return rclcpp_action::GoalResponse::REJECT;
        }
    }

    rclcpp_action::CancelResponse handle_cancel(const std::shared_ptr<GoalHandleMoveRobot>&) {
        return rclcpp_action::CancelResponse::ACCEPT;
    }

    // ✅ Dùng detach thread — execute đồng bộ bên trong, không dùng promise
    void handle_accepted(const std::shared_ptr<GoalHandleMoveRobot> goal_handle) {
        std::thread{[this, goal_handle]() { execute(goal_handle); }}.detach();
    }

    // =========================================================
    void execute(const std::shared_ptr<GoalHandleMoveRobot> goal_handle) {
        if (mode_ == 1)
            moveStraightCartesian(start_pose_, target_pose_, goal_handle);
        else
            executePlan(target_pose_, goal_handle);
    }

    // ✅ Execute đồng bộ — không dùng thread/promise/future
    bool executeSync(const MoveGroupInterface::Plan& plan) {
        auto code = move_group_->execute(plan);
        if (code != moveit::core::MoveItErrorCode::SUCCESS) {
            RCLCPP_ERROR(get_logger(), "Execute failed, code=%d", code.val);
            return false;
        }
        return true;
    }

    // =========================================================
    std::pair<bool, MoveGroupInterface::Plan> planMotion() {
        MoveGroupInterface::Plan plan;
        bool ok = static_cast<bool>(move_group_->plan(plan));
        return {ok, plan};
    }

    bool findConsistentIK(
        const geometry_msgs::msg::Pose& t1,
        const geometry_msgs::msg::Pose& t2,
        std::vector<double>& out_joints)
    {
        constexpr int    MAX_ATTEMPTS  = 10;
        constexpr double SCORE_THRESH  = 0.01;

        double best_score = std::numeric_limits<double>::max();
        bool   found      = false;

        moveit::core::RobotState seed(robot_model_);
        moveit::core::RobotState state_t1(robot_model_);
        moveit::core::RobotState state_t2(robot_model_);
        planning_scene::PlanningScene planning_scene(robot_model_);

        for (int i = 0; i < MAX_ATTEMPTS; ++i) {
            seed.setToRandomPositions(jmg_);
            if (!seed.setFromIK(jmg_, t1, 0.05)) continue;

            state_t1 = seed;
            if (!state_t1.satisfiesBounds(jmg_) ||
                planning_scene.isStateColliding(state_t1, jmg_->getName())) {
                continue;
            }

            std::vector<double> j1;
            state_t1.copyJointGroupPositions(jmg_, j1);

            state_t2.setJointGroupPositions(jmg_, j1);
            if (!state_t2.setFromIK(jmg_, t2, 0.05)) continue;
            if (!state_t2.satisfiesBounds(jmg_) ||
                planning_scene.isStateColliding(state_t2, jmg_->getName())) {
                continue;
            }

            std::vector<double> j2;
            state_t2.copyJointGroupPositions(jmg_, j2);

            double score = 0.0;
            for (size_t k = 0; k < j1.size(); ++k) {
                double d = j1[k] - j2[k];
                score += d * d;
            }
            if (score < best_score) {
                best_score = score;
                out_joints = j1;
                found      = true;
                if (score < SCORE_THRESH) break;
            }
        }

        RCLCPP_INFO(get_logger(), "ConsistentIK: found=%d score=%.4f", found, best_score);
        return found;
    }

    bool planWithJointTarget(
        const std::vector<double>& joints,
        MoveGroupInterface::Plan& plan)
    {
        move_group_->setJointValueTarget(joints);

        for (int i = 0; i < 5; ++i) {
            auto [ok, p] = planMotion();
            if (ok && !p.trajectory_.joint_trajectory.points.empty()) {
                plan = p;
                return true;
            }
        }

        return false;
    }

    bool validateCartesian(
        const std::vector<double>& joints,
        const geometry_msgs::msg::Pose& target)
    {
        moveit::core::RobotState start_state(robot_model_);
        start_state.setJointGroupPositions(jmg_, joints);
        move_group_->setStartState(start_state);

        moveit_msgs::msg::RobotTrajectory traj;
        double fraction = move_group_->computeCartesianPath({target}, 0.01, 0.0, traj, moveit_msgs::msg::Constraints(),
        avoid_collisions);
        move_group_->setStartStateToCurrentState();

        RCLCPP_INFO(get_logger(), "Cartesian validation: %.1f%%", fraction * 100.0);
        return fraction >= CARTESIAN_VALIDATION_FRACTION;
    }


bool computeCartesianPlan(
    const geometry_msgs::msg::Pose& from,
    const geometry_msgs::msg::Pose& to,
    MoveGroupInterface::Plan& plan,
    double& best_fraction)
{
    constexpr std::array<double, 2> EEF_STEPS = {0.005, 0.01};

    best_fraction = 0.0;

    moveit_msgs::msg::RobotTrajectory best_traj;

    // Lấy trạng thái robot hiện tại làm điểm xuất phát.
    auto current_state = move_group_->getCurrentState(2.0);
    if (!current_state) {
        RCLCPP_ERROR(get_logger(),
                     "Cannot get current robot state");
        return false;
    }

    // Kiểm tra khoảng cách giữa pose hiện tại và from.
    // Nếu from là điểm xuất phát mong muốn, hãy xác nhận
    // nó gần với TCP thực tế trước khi lập kế hoạch.
    const auto* tcp_link =
        current_state->getLinkModel("tcp0");

    if (!tcp_link) {
        RCLCPP_ERROR(get_logger(),
                     "Link tcp0 not found");
        return false;
    }

    const Eigen::Isometry3d actual_tf =
        current_state->getGlobalLinkTransform(tcp_link);

    const double dx = actual_tf.translation().x()
                    - from.position.x;
    const double dy = actual_tf.translation().y()
                    - from.position.y;
    const double dz = actual_tf.translation().z()
                    - from.position.z;

    const double start_error =
        std::sqrt(dx * dx + dy * dy + dz * dz);

    RCLCPP_INFO(get_logger(),
                "Cartesian start error: %.4f m",
                start_error);

    if (start_error > MAX_CARTESIAN_START_ERROR_M) {
        RCLCPP_WARN(
            get_logger(),
            "Cartesian start pose is too far from current TCP: %.4f m (max %.4f m)",
            start_error,
            MAX_CARTESIAN_START_ERROR_M);
        return false;
    }

    for (const double eef_step : EEF_STEPS) {
        move_group_->setStartState(*current_state);

        moveit_msgs::msg::RobotTrajectory traj;

        const std::vector<geometry_msgs::msg::Pose> waypoints = {from, to};

        const double fraction =
            move_group_->computeCartesianPath(
                waypoints,
                eef_step,
                1.5,  // jump threshold; kiểm tra trên robot thực tế
                traj,
                moveit_msgs::msg::Constraints(),
                avoid_collisions);

        RCLCPP_INFO(
            get_logger(),
            "Cartesian: step=%.3f, fraction=%.1f%%, points=%zu",
            eef_step,
            fraction * 100.0,
            traj.joint_trajectory.points.size());

        if (fraction > best_fraction &&
            !traj.joint_trajectory.points.empty()) {
            best_fraction = fraction;
            best_traj = traj;
        }

        // Đã có đường đi gần như hoàn chỉnh thì dừng.
        if (fraction >= CARTESIAN_FULL_FRACTION) {
            break;
        }
    }

    if (best_fraction < CARTESIAN_MIN_FRACTION ||
        best_traj.joint_trajectory.points.empty()) {
        RCLCPP_WARN(
            get_logger(),
            "Cartesian insufficient: best=%.1f%%",
            best_fraction * 100.0);
        return false;
    }

    robot_trajectory::RobotTrajectory rt(
        robot_model_, "indy_manipulator");

    rt.setRobotTrajectoryMsg(*current_state, best_traj);

    trajectory_processing::IterativeParabolicTimeParameterization iptp;

    if (!iptp.computeTimeStamps(rt, 0.3, 0.3)) {
        RCLCPP_ERROR(get_logger(),
                     "Time parameterization failed");
        return false;
    }

    rt.getRobotTrajectoryMsg(best_traj);
    plan.trajectory_ = best_traj;

    return true;
}


    // =========================================================
    void moveStraightCartesian(
        const geometry_msgs::msg::Pose& from,
        const geometry_msgs::msg::Pose& to,
        const std::shared_ptr<GoalHandleMoveRobot> goal_handle)
    {
        auto result = std::make_shared<MoveRobot::Result>();
        move_group_->clearPoseTargets();
        move_group_->clearPathConstraints();

        MoveGroupInterface::Plan plan;
        double fraction = 0.0;
        bool   plan_ok  = computeCartesianPlan(from, to, plan, fraction);

        if (!plan_ok) {
            RCLCPP_WARN(get_logger(), "Cartesian failed, falling back to pose planning");
            move_group_->setPoseTarget(to);
            for (int i = 0; i < 5 && !plan_ok; ++i) {
                auto [ok, p] = planMotion();
                if (ok && !p.trajectory_.joint_trajectory.points.empty()) {
                    plan    = p;
                    plan_ok = true;
                }
            }
        }

        if (!plan_ok) {
            RCLCPP_ERROR(get_logger(), "All planning failed");
            result->success = false;
            goal_handle->abort(result);
            return;
        }

        // ✅ Execute đồng bộ — không promise
        if (executeSync(plan)) {
            result->success = true;
            goal_handle->succeed(result);
        } else {
            result->success = false;
            goal_handle->abort(result);
        }
    }

    void executePlan(
        const geometry_msgs::msg::Pose& pose,
        const std::shared_ptr<GoalHandleMoveRobot> goal_handle)
    {
        auto result = std::make_shared<MoveRobot::Result>();
        move_group_->clearPoseTargets();
        move_group_->clearPathConstraints();

        MoveGroupInterface::Plan plan;
        bool plan_ok = false;

        if (mode_ == 2) {
            std::vector<double> best_joints;
            if (findConsistentIK(pose, next_pose_, best_joints)) {
                const bool cart_ok = validateCartesian(best_joints, next_pose_);
                if (!cart_ok) {
                    RCLCPP_WARN(get_logger(),
                                "Cartesian validation failed, but keeping joint-target plan "
                                "to avoid falling back to pose sampling.");
                }

                plan_ok = planWithJointTarget(best_joints, plan);
            }
        }

        if (!plan_ok) {
            if (mode_ == 2) {
                RCLCPP_ERROR(get_logger(),
                             "Joint-target planning failed in mode 2; not falling back to pose "
                             "sampling because it can produce an invalid goal state with octomap.");
                result->success = false;
                goal_handle->abort(result);
                return;
            }

            move_group_->setPoseTarget(pose);
            for (int i = 0; i < 1 && !plan_ok; ++i) {
                auto [ok, p] = planMotion();
                if (ok && !p.trajectory_.joint_trajectory.points.empty()) {
                    plan    = p;
                    plan_ok = true;
                }
            }
        }

        if (!plan_ok) {
            RCLCPP_ERROR(get_logger(), "All plan attempts failed");
            result->success = false;
            goal_handle->abort(result);
            return;
        }

        // ✅ Execute đồng bộ — không promise
        if (executeSync(plan)) {
            result->success = true;
            goal_handle->succeed(result);
        } else {
            result->success = false;
            goal_handle->abort(result);
        }
    }
};

int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);

    auto node = std::make_shared<MoveItController>();
    node->initialize();

    // ✅ MultiThreadedExecutor: cần thiết vì execute() chạy trong detach thread
    // nhưng vẫn cần executor để xử lý MoveIt service callbacks trong lúc execute
    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(node->get_node_base_interface());

    std::thread spin_thread([&executor]() { executor.spin(); });
    spin_thread.join();

    rclcpp::shutdown();
    return 0;
}
