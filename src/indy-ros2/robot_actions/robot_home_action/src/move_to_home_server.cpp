#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include "robot_home_action/action/move_to_home.hpp"
#include <future>
#include <atomic>

using moveit::planning_interface::MoveGroupInterface;

class MoveToHomeServer : public rclcpp::Node
{
public:
    using MoveToHome          = robot_home_action::action::MoveToHome;
    using GoalHandleMoveToHome = rclcpp_action::ServerGoalHandle<MoveToHome>;

    MoveToHomeServer() : Node("move_to_home_server")
    {
        action_server_ = rclcpp_action::create_server<MoveToHome>(
            this, "move_to_home",
            std::bind(&MoveToHomeServer::handle_goal,     this, std::placeholders::_1, std::placeholders::_2),
            std::bind(&MoveToHomeServer::handle_cancel,   this, std::placeholders::_1),
            std::bind(&MoveToHomeServer::handle_accepted, this, std::placeholders::_1));

        RCLCPP_INFO(this->get_logger(), "MoveToHome action server started.");
    }

    void initialize() {
        move_group_ = std::make_unique<MoveGroupInterface>(shared_from_this(), "indy_manipulator");

        // ── Set planning params 1 lần ──
        move_group_->setPlanningTime(10.0);
        move_group_->setPlannerId("RRTConnectkConfigDefault");
        move_group_->setMaxVelocityScalingFactor(1.0);
        move_group_->setMaxAccelerationScalingFactor(1.0);
    }

private:
    std::unique_ptr<MoveGroupInterface>           move_group_;
    rclcpp_action::Server<MoveToHome>::SharedPtr  action_server_;

    std::vector<double> target_joints_;

    // ── Goal / Cancel / Accept ───────────────────────────────────────────
    rclcpp_action::GoalResponse handle_goal(
        const rclcpp_action::GoalUUID&,
        std::shared_ptr<const MoveToHome::Goal> goal)
    {
        if (goal->joint_positions.empty() && goal->pass_permit == 0) {
            RCLCPP_WARN(this->get_logger(), "Empty joint_positions, rejecting.");
            return rclcpp_action::GoalResponse::REJECT;
        }
        if (goal->pass_permit != 0) {
            RCLCPP_WARN(this->get_logger(), "Reset active, rejecting.");
            return rclcpp_action::GoalResponse::REJECT;
        }
        return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
    }

    rclcpp_action::CancelResponse handle_cancel(const std::shared_ptr<GoalHandleMoveToHome>&) {
        RCLCPP_WARN(this->get_logger(), "Goal canceled.");
        return rclcpp_action::CancelResponse::ACCEPT;
    }

    void handle_accepted(const std::shared_ptr<GoalHandleMoveToHome> goal_handle) {
        std::thread{[this, goal_handle]() { execute(goal_handle); }}.detach();
    }

    // ── Safe execute với timeout, fix "Promise already satisfied" ────────
    bool executeWithTimeout(
        const MoveGroupInterface::Plan& plan,
        std::chrono::seconds timeout = std::chrono::seconds(60))
    {
        auto promise     = std::make_shared<std::promise<bool>>();
        auto future      = promise->get_future();
        auto promise_set = std::make_shared<std::atomic<bool>>(false);

        std::thread([this, plan, promise, promise_set]() {
            bool ok = (move_group_->execute(plan) == moveit::core::MoveItErrorCode::SUCCESS);
            bool expected = false;
            if (promise_set->compare_exchange_strong(expected, true))
                promise->set_value(ok);
        }).detach();

        if (future.wait_for(timeout) != std::future_status::ready) {
            bool expected = false;
            promise_set->compare_exchange_strong(expected, true);
            RCLCPP_ERROR(this->get_logger(), "Execute timeout!");
            return false;
        }
        return future.get();
    }

    // ── Execute ──────────────────────────────────────────────────────────
    void execute(const std::shared_ptr<GoalHandleMoveToHome> goal_handle)
    {
        auto result = std::make_shared<MoveToHome::Result>();
        const auto& goal = goal_handle->get_goal();

        if (goal->joint_positions.empty()) {
            result->success = false;
            result->message = "Empty joint positions!";
            goal_handle->abort(result);
            return;
        }

        target_joints_ = goal->joint_positions;

        move_group_->clearPoseTargets();
        move_group_->clearPathConstraints();
        move_group_->setJointValueTarget(target_joints_);

        // ── Plan với retry ───────────────────────────────────────────────
        MoveGroupInterface::Plan plan;
        bool plan_ok = false;

        for (int i = 0; i < 5 && !plan_ok; ++i) {
            bool ok = static_cast<bool>(move_group_->plan(plan));
            if (ok && !plan.trajectory_.joint_trajectory.points.empty()) {
                plan_ok = true;
                RCLCPP_INFO(this->get_logger(), "Plan found on attempt %d/5", i + 1);
            } else {
                RCLCPP_WARN(this->get_logger(), "Plan attempt %d/5 failed", i + 1);
            }
        }

        if (!plan_ok) {
            RCLCPP_ERROR(this->get_logger(), "Planning failed after 5 attempts");
            result->success = false;
            result->message = "Motion planning failed!";
            goal_handle->abort(result);
            return;
        }

        // ── Feedback giả lập trong khi execute ──────────────────────────
        auto feedback = std::make_shared<MoveToHome::Feedback>();
        for (int i = 0; i <= 10; ++i) {
            feedback->progress = i * 0.1f;
            goal_handle->publish_feedback(feedback);
            rclcpp::sleep_for(std::chrono::milliseconds(50));
        }

        // ── Execute ──────────────────────────────────────────────────────
        if (executeWithTimeout(plan, std::chrono::seconds(60))) {
            result->success = true;
            result->message = "Robot successfully moved!";
            goal_handle->succeed(result);
        } else {
            result->success = false;
            result->message = "Execution failed!";
            goal_handle->abort(result);
        }
    }
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<MoveToHomeServer>();
    node->initialize();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}