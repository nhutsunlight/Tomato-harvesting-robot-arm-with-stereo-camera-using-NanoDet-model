#include "rclcpp/rclcpp.hpp"
#include "rclcpp_action/rclcpp_action.hpp"
#include "std_msgs/msg/float64.hpp"
#include "gripper_action/action/gripper_control.hpp"
#include "connect_msgs/msg/connect_msg.hpp"

#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <moveit_msgs/msg/move_it_error_codes.hpp>

class GripperActionServer : public rclcpp::Node {
public:
    using GripperControl = gripper_action::action::GripperControl;
    using GoalHandle     = rclcpp_action::ServerGoalHandle<GripperControl>;
    using MoveGroupInterface = moveit::planning_interface::MoveGroupInterface;

    GripperActionServer() : Node("gripper_action_server") {
        gripper_pub_ = create_publisher<std_msgs::msg::Float64>(
            "/gripper_controller/command", 10);

        server_ = rclcpp_action::create_server<GripperControl>(
            this, "gripper_action",
            std::bind(&GripperActionServer::handle_goal,     this, std::placeholders::_1, std::placeholders::_2),
            std::bind(&GripperActionServer::handle_cancel,   this, std::placeholders::_1),
            std::bind(&GripperActionServer::handle_accepted, this, std::placeholders::_1));

        connection_sub_ = create_subscription<connect_msgs::msg::ConnectMsg>(
            "/connect_msg", 10,
            std::bind(&GripperActionServer::connection_callback, this, std::placeholders::_1));

        RCLCPP_INFO(get_logger(), "Gripper Action Server started.");
    }

    void initialize() {
        gripper_group_ = std::make_unique<MoveGroupInterface>(shared_from_this(), "gripper");
        RCLCPP_INFO(get_logger(), "Gripper MoveGroup initialized.");
    }

    ~GripperActionServer() = default;

private:
    rclcpp_action::Server<GripperControl>::SharedPtr        server_;
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr    gripper_pub_;
    rclcpp::Subscription<connect_msgs::msg::ConnectMsg>::SharedPtr connection_sub_;
    std::unique_ptr<MoveGroupInterface> gripper_group_;

    std::atomic<bool> latest_connection_status_{false};
    std::atomic<bool> reset_status_{false};
    bool is_reset_ = false;

    // =========================================================
    rclcpp_action::GoalResponse handle_goal(
        const rclcpp_action::GoalUUID&,
        std::shared_ptr<const GripperControl::Goal> goal)
    {
        RCLCPP_INFO(get_logger(), "Received goal: position=%.2f id=%zu",
                    goal->position, goal->id);

        if (is_reset_ && goal->pass_permit == 0) {
            RCLCPP_WARN(get_logger(), "Robot resetting — reject goal.");
            return rclcpp_action::GoalResponse::REJECT;
        }
        return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
    }

    rclcpp_action::CancelResponse handle_cancel(const std::shared_ptr<GoalHandle>) {
        RCLCPP_INFO(get_logger(), "Goal canceled.");
        return rclcpp_action::CancelResponse::ACCEPT;
    }

    // ✅ Chạy đồng bộ trong executor thread — không dùng std::thread/promise
    void handle_accepted(const std::shared_ptr<GoalHandle> goal_handle) {
        execute(goal_handle);
    }

    void execute(const std::shared_ptr<GoalHandle> goal_handle) {
        auto result   = std::make_shared<GripperControl::Result>();
        const auto& g = goal_handle->get_goal();

        // Chọn target
        if (g->position > 0.5) {
            RCLCPP_INFO(get_logger(), "Opening gripper...");
            gripper_group_->setNamedTarget("open");
        } else {
            RCLCPP_INFO(get_logger(), "Closing gripper...");
            gripper_group_->setNamedTarget("closed");
        }

        // Plan
        MoveGroupInterface::Plan plan;
        if (gripper_group_->plan(plan) != moveit::core::MoveItErrorCode::SUCCESS ||
            plan.trajectory_.joint_trajectory.points.empty())
        {
            RCLCPP_WARN(get_logger(), "Planning failed or empty trajectory");
            result->success = false;
            goal_handle->abort(result);
            return;
        }

        // Execute — đồng bộ, không cần thread/promise/future
        if (gripper_group_->execute(plan) != moveit::core::MoveItErrorCode::SUCCESS) {
            RCLCPP_ERROR(get_logger(), "Gripper execute failed!");
            result->success = false;
            goal_handle->abort(result);
            return;
        }

        result->success = true;
        goal_handle->succeed(result);
        RCLCPP_INFO(get_logger(), "Gripper action succeeded.");
    }

    void connection_callback(const connect_msgs::msg::ConnectMsg& msg) {
        if (!msg.connect_msg.empty()) {
            latest_connection_status_ = msg.connect_msg.front().connection;
            reset_status_             = msg.connect_msg.front().wait_key;
            is_reset_                 = reset_status_.load();
        }
    }
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<GripperActionServer>();

    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(node->get_node_base_interface());

    std::thread spin_thread([&executor]() { executor.spin(); });
    node->initialize();
    spin_thread.join();

    rclcpp::shutdown();
    return 0;
}
