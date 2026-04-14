#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include "robot_home_action/action/move_to_home.hpp"
#include "connect_msgs/msg/connect_msg.hpp"
#include <optional>
#include <future>

using moveit::planning_interface::MoveGroupInterface;

class MoveToHomeServer : public rclcpp::Node
{
public:
    using MoveToHome = robot_home_action::action::MoveToHome;
    using GoalHandleMoveToHome = rclcpp_action::ServerGoalHandle<MoveToHome>;

    MoveToHomeServer()
        : Node("move_to_home_server")
        //moveit_node_(std::make_shared<rclcpp::Node>("robot_home_node"))
        //move_group_interface_(moveit_node_, "indy_manipulator")
    {
        action_server_ = rclcpp_action::create_server<MoveToHome>(
            this,
            "move_to_home",
            std::bind(&MoveToHomeServer::handle_goal, this, std::placeholders::_1, std::placeholders::_2),
            std::bind(&MoveToHomeServer::handle_cancel, this, std::placeholders::_1),
            std::bind(&MoveToHomeServer::handle_accepted, this, std::placeholders::_1));

        connection_sub_ = this->create_subscription<connect_msgs::msg::ConnectMsg>(
            "/connect_msg", 10,
            [this](const connect_msgs::msg::ConnectMsg::SharedPtr msg) {
              if (!msg->connect_msg.empty()) {
                latest_connection_status_ = msg->connect_msg.front().connection;
                reset_status_ = msg->connect_msg.front().wait_key;
                res_id = msg->connect_msg.front().id;
              }
            });

        //startConnectionMonitorThread();

        RCLCPP_INFO(this->get_logger(), "MoveToHome action server started.");
    }

    void initialize() {
        move_group_interface_ = std::make_unique<MoveGroupInterface>(shared_from_this(), "indy_manipulator");
    }

    void startUpdatePoseThread(const std::string& frame_id) {
        if (update_pose_thread_.joinable()) return;  // Nếu thread đang chạy thì không tạo mới
        pose_promise_ = std::promise<bool>();        // Reset promise
        auto future_stop = pose_promise_.get_future();
        update_pose_thread_ = std::thread([this, future_stop = std::move(future_stop), frame_id]() mutable {
            RCLCPP_INFO(this->get_logger(), "Getting current pose infformation...");
            while (future_stop.wait_for(std::chrono::milliseconds(10)) == std::future_status::timeout && rclcpp::ok()) {
                auto state = move_group_interface_->getCurrentState(1.0);
                if (state) {
                    move_group_interface_->setStartState(*state);
                    break;
                }
            }
            RCLCPP_INFO(this->get_logger(), "Current pose information retrieved.");
        });
    }

    void stopUpdatePoseThread() {
        pose_promise_.set_value(true);  // Gửi tín hiệu dừng
        if (update_pose_thread_.joinable()) {
            update_pose_thread_.join();  // Đợi thread kết thúc
        }
    }

    void startConnectionMonitorThread() {
        connection_monitor_thread_ = std::thread([this]() {
            while (rclcpp::ok() && !stop_connection_monitor_) {
                is_server_ready_ = latest_connection_status_;
                is_reset_ = reset_status_;
            }
        });
    }

    ~MoveToHomeServer() {
        stopUpdatePoseThread();
        stop_connection_monitor_ = true;
        if (connection_monitor_thread_.joinable()) {
            connection_monitor_thread_.join();
        }
    }

private:
    //std::shared_ptr<rclcpp::Node> moveit_node_;
    std::unique_ptr<MoveGroupInterface> move_group_interface_;
    std::thread update_pose_thread_;
    std::promise<bool> pose_promise_;
    rclcpp_action::Server<MoveToHome>::SharedPtr action_server_;
    rclcpp::Subscription<connect_msgs::msg::ConnectMsg>::SharedPtr connection_sub_;
    std::atomic<bool> latest_connection_status_{false};
    std::atomic<bool> reset_status_{false};
    std::thread connection_monitor_thread_;
    std::vector<double> target_joint_positions;
    bool stop_connection_monitor_ = false;
    bool is_server_ready_ = false;
    bool is_reset_ = false;
    size_t res_id = 0;
    size_t id = 0;

    rclcpp_action::GoalResponse handle_goal(const rclcpp_action::GoalUUID &, std::shared_ptr<const MoveToHome::Goal> goal)
    {
        if ((goal->joint_positions.empty() || is_reset_) && goal->pass_permit == 0) {
            RCLCPP_WARN(this->get_logger(), "Received empty joint_positions.");
            return rclcpp_action::GoalResponse::REJECT;
        } else {
            RCLCPP_INFO(this->get_logger(), "Received goal to move to home position.");
            return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
        }
    }

    rclcpp_action::CancelResponse handle_cancel(const std::shared_ptr<GoalHandleMoveToHome> goal_handle)
    {
        (void)goal_handle;
        RCLCPP_WARN(this->get_logger(), "Goal canceled.");
        return rclcpp_action::CancelResponse::ACCEPT;
    }

    void handle_accepted(const std::shared_ptr<GoalHandleMoveToHome> goal_handle)
    {
        std::thread{std::bind(&MoveToHomeServer::execute, this, goal_handle)}.detach();
    }

    void updateStartState(const std::string& frame_id) {
        move_group_interface_->clearPoseTargets();
        startUpdatePoseThread(frame_id);
        stopUpdatePoseThread();
    }

    void waitForReconnect() {
        while (rclcpp::ok()) {
            RCLCPP_WARN(this->get_logger(), "Lost connection! Waiting for reconnect...");
            if(is_server_ready_){
                RCLCPP_INFO(this->get_logger(), "Reconnected successfully!");
                break;
            }
        }
    }

    void execute(const std::shared_ptr<GoalHandleMoveToHome> goal_handle)
    {
        auto feedback = std::make_shared<MoveToHome::Feedback>();
        auto result = std::make_shared<MoveToHome::Result>();

        if (!goal_handle->get_goal()->joint_positions.empty()){
            target_joint_positions = goal_handle->get_goal()->joint_positions;
            id = goal_handle->get_goal()->id;
        }
        else
        {
            result->success = false;
            goal_handle->abort(result);
            return;
        }

        move_group_interface_->clearPoseTargets();
        move_group_interface_->clearPathConstraints();
        move_group_interface_->setJointValueTarget(target_joint_positions);
        move_group_interface_->setPlanningTime(10.0);
        move_group_interface_->setPlannerId("RRTConnectkConfigDefault");
        move_group_interface_->setMaxVelocityScalingFactor(1.0);
        move_group_interface_->setMaxAccelerationScalingFactor(1.0);

        moveit::planning_interface::MoveGroupInterface::Plan plan;
        bool success = false;
        int max_attempts = 5;

        for (int attempt = 0; attempt < max_attempts; attempt++) {
            auto result_code = move_group_interface_->plan(plan);
            success = (result_code == moveit::core::MoveItErrorCode::SUCCESS);

            if (!success || plan.trajectory_.joint_trajectory.points.empty()) {
                RCLCPP_WARN(this->get_logger(), 
                    "Plan attempt %d/%d failed, retrying...", attempt+1, max_attempts);
                continue;
            }

            RCLCPP_INFO(this->get_logger(), 
                "Valid plan found on attempt %d/%d", attempt+1, max_attempts);
            break;
        }

        if (!success || plan.trajectory_.joint_trajectory.points.empty())
        {
            RCLCPP_WARN(this->get_logger(), "Motion planning failed after %d attempts!", max_attempts);
            result->success = false;
            result->message = "Motion planning failed!";
            goal_handle->abort(result);
            return;
        }

        for (int i = 0; i <= 10; ++i)
        {
            feedback->progress = i * 0.1;
            goal_handle->publish_feedback(feedback);
            rclcpp::sleep_for(std::chrono::milliseconds(50));
        }

        try
        {
            auto exec_result = move_group_interface_->execute(plan);
            
            if (exec_result != moveit::core::MoveItErrorCode::SUCCESS)
            {
                RCLCPP_ERROR(this->get_logger(), "Execute failed with code: %d", 
                    exec_result.val);
                result->success = false;
                result->message = "Execution failed!";
                goal_handle->abort(result);
                return;
            }
        }
        catch (const std::exception &e)
        {
            RCLCPP_ERROR(this->get_logger(), "Execute exception: %s", e.what());
            result->success = false;
            result->message = "Execution crashed!";
            goal_handle->abort(result);
            return;
        }

        result->success = true;
        result->message = "Robot successfully moved!";
        goal_handle->succeed(result);
    }
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<MoveToHomeServer>();
    node->initialize();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}