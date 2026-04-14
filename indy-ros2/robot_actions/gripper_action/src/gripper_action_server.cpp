#include "rclcpp/rclcpp.hpp"
#include "rclcpp_action/rclcpp_action.hpp"
#include "std_msgs/msg/float64.hpp"  // <--- dòng bị thiếu
#include "gripper_action/action/gripper_control.hpp"
#include "connect_msgs/msg/connect_msg.hpp"

// MoveIt
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <moveit_msgs/msg/move_it_error_codes.hpp>



class GripperActionServer : public rclcpp::Node {
public:
    using GripperControl = gripper_action::action::GripperControl;
    using GoalHandle = rclcpp_action::ServerGoalHandle<GripperControl>;
    using MoveGroupInterface = moveit::planning_interface::MoveGroupInterface;

    GripperActionServer() : Node("gripper_action_server") {
        // Tạo publisher điều khiển gripper
        gripper_pub_ = this->create_publisher<std_msgs::msg::Float64>("/gripper_controller/command", 10);

        // Tạo action server
        server_ = rclcpp_action::create_server<GripperControl>(
            this,
            "gripper_action",
            std::bind(&GripperActionServer::handle_goal, this, std::placeholders::_1, std::placeholders::_2),
            std::bind(&GripperActionServer::handle_cancel, this, std::placeholders::_1),
            std::bind(&GripperActionServer::handle_accepted, this, std::placeholders::_1)
        );

        connection_sub_ = this->create_subscription<connect_msgs::msg::ConnectMsg>(
            "/connect_msg", 10, std::bind(&GripperActionServer::connection_callback, this, std::placeholders::_1));
        
        //startConnectionMonitorThread();

        RCLCPP_INFO(this->get_logger(), "Gripper Action Server started.");
    }

    // Thêm hàm initialize
    void initialize() {
        gripper_group_ = std::make_unique<MoveGroupInterface>(
            shared_from_this(), "gripper");
        RCLCPP_INFO(this->get_logger(), "Gripper MoveGroup initialized.");
    }

    void startConnectionMonitorThread() {
        connection_monitor_thread_ = std::thread([this]() {
            while (rclcpp::ok() && !stop_connection_monitor_) {
                is_server_ready_ = latest_connection_status_;
                is_reset_ = reset_status_;
                std::this_thread::sleep_for(std::chrono::milliseconds(10)); // thêm dòng này
            }
        });
    }

    ~GripperActionServer() {
        stop_connection_monitor_ = true;
        if (connection_monitor_thread_.joinable()) {
            connection_monitor_thread_.join();
        }
    }

private:
    rclcpp_action::Server<GripperControl>::SharedPtr server_;
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr gripper_pub_;
    rclcpp::Subscription<connect_msgs::msg::ConnectMsg>::SharedPtr connection_sub_;
    std::unique_ptr<MoveGroupInterface> gripper_group_;
    std::atomic<bool> latest_connection_status_{false};
    std::atomic<bool> reset_status_{false};
    std::thread connection_monitor_thread_;
    bool stop_connection_monitor_ = false;
    bool is_server_ready_ = false;
    bool is_reset_ = false;
    size_t res_id = 0;
    size_t id = 0;

    rclcpp_action::GoalResponse handle_goal(const rclcpp_action::GoalUUID &, std::shared_ptr<const GripperControl::Goal> goal) {
        RCLCPP_INFO(this->get_logger(), "Received goal to move gripper to position: %.2f", goal->position);
        if (is_reset_ && goal->pass_permit == 0) {
            RCLCPP_WARN(this->get_logger(), "Robot is resetting. Rejecting goal.");
            return rclcpp_action::GoalResponse::REJECT;
        }else {
            RCLCPP_INFO(this->get_logger(), "Action goal accepted.");
            return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
        }
    }

    rclcpp_action::CancelResponse handle_cancel(const std::shared_ptr<GoalHandle>) {
        RCLCPP_INFO(this->get_logger(), "Goal canceled.");
        return rclcpp_action::CancelResponse::ACCEPT;
    }

    void waitForReconnect() {
        while (rclcpp::ok()) {
            RCLCPP_WARN(this->get_logger(), "lost connection. Trying to reconnect...");
            if(is_server_ready_){
                RCLCPP_INFO(this->get_logger(), "Reconnected successfully.");
                break;
            }
        }
    } 

    void handle_accepted(const std::shared_ptr<GoalHandle> goal_handle) {
        std::thread([this, goal_handle]() {
            auto goal = goal_handle->get_goal()->position;
            id = goal_handle->get_goal()->id;
            if (id >= 1000) {
                id = id / 1000;
            }
            auto result = std::make_shared<GripperControl::Result>();

            if (goal > 0.5) {
                RCLCPP_INFO(this->get_logger(), "Opening gripper...");
                gripper_group_->setNamedTarget("open");
            } else {
                RCLCPP_INFO(this->get_logger(), "Closing gripper...");
                gripper_group_->setNamedTarget("closed");
            }

            moveit::planning_interface::MoveGroupInterface::Plan plan;
            auto result_code = gripper_group_->plan(plan);
            bool success = (result_code == moveit::core::MoveItErrorCode::SUCCESS);

            if (!success || plan.trajectory_.joint_trajectory.points.empty())
            {
                RCLCPP_WARN(this->get_logger(), "Planning failed or empty trajectory");
                result->success = false;
                goal_handle->abort(result);
                return;
            }

            // Execute với timeout
            std::promise<moveit::core::MoveItErrorCode> exec_promise;
            auto exec_future = exec_promise.get_future();

            std::thread exec_thread([&]() {
                try {
                    exec_promise.set_value(gripper_group_->execute(plan));
                } catch (const std::exception& e) {
                    RCLCPP_ERROR(this->get_logger(), "Execute crash: %s", e.what());
                    exec_promise.set_value(moveit::core::MoveItErrorCode::FAILURE);
                }
            });

            if (exec_future.wait_for(std::chrono::seconds(10)) != std::future_status::ready)
            {
                RCLCPP_ERROR(this->get_logger(), "Gripper execute timeout!");
                exec_thread.detach();
                result->success = false;
                goal_handle->abort(result);
                return;
            }

            exec_thread.join();

            if (exec_future.get() != moveit::core::MoveItErrorCode::SUCCESS)
            {
                RCLCPP_ERROR(this->get_logger(), "Gripper execute failed!");
                result->success = false;
                goal_handle->abort(result);
                return;
            }

            result->success = true;
            goal_handle->succeed(result);

        }).detach();
    }

    void connection_callback(const connect_msgs::msg::ConnectMsg& msg) {
        if (!msg.connect_msg.empty()) {
            const auto& result = msg.connect_msg.front();
            latest_connection_status_ = result.connection;
            reset_status_ = result.wait_key;
            res_id = result.id;
        }
    }
};

// Sửa main
int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    
    auto node = std::make_shared<GripperActionServer>();
    
    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(node->get_node_base_interface());
    
    // Spin trên thread riêng trước khi initialize
    std::thread spin_thread([&executor]() {
        executor.spin();
    });
    
    node->initialize();
    
    spin_thread.join();
    rclcpp::shutdown();
    return 0;
}
