#include "rclcpp/rclcpp.hpp"
#include "rclcpp_action/rclcpp_action.hpp"
#include "control_action/action/move_robot.hpp"

class MoveItServiceClient : public rclcpp::Node {
public:
    using MoveRobot = control_action::action::MoveRobot;
    using GoalHandleMoveRobot = rclcpp_action::ClientGoalHandle<MoveRobot>;

    MoveItServiceClient() : Node("moveit_service_client") {
        action_client_ = rclcpp_action::create_client<MoveRobot>(this, "move_robot");

        // Chờ action server sẵn sàng
        while (!action_client_->wait_for_action_server(std::chrono::seconds(2)) && rclcpp::ok()) {
            RCLCPP_WARN(this->get_logger(), "Waiting for MoveIt Action Server...");
        }

        RCLCPP_INFO(this->get_logger(), "MoveIt Action Server is ready!");
        
        send_goal();  // Gửi yêu cầu di chuyển ngay khi khởi động
    }

private:
    rclcpp_action::Client<MoveRobot>::SharedPtr action_client_;

    void send_goal()
    {
        auto goal_msg = MoveRobot::Goal();
        goal_msg.request_move = true;

        auto send_goal_options = rclcpp_action::Client<MoveRobot>::SendGoalOptions();
        
        send_goal_options.goal_response_callback =
            [this](GoalHandleMoveRobot::SharedPtr goal_handle) {
                this->goal_response_callback(goal_handle);
            };

        send_goal_options.feedback_callback =
            std::bind(&MoveItServiceClient::feedback_callback, this, std::placeholders::_1, std::placeholders::_2);

        send_goal_options.result_callback =
            std::bind(&MoveItServiceClient::result_callback, this, std::placeholders::_1);

        action_client_->async_send_goal(goal_msg, send_goal_options);
    }

    void goal_response_callback(GoalHandleMoveRobot::SharedPtr goal_handle)
    {
        if (!goal_handle)
        {
            RCLCPP_ERROR(this->get_logger(), "Goal was rejected by server.");
        }
        else
        {
            RCLCPP_INFO(this->get_logger(), "Goal accepted by server, waiting for result.");
        }
    }

    void feedback_callback(
        GoalHandleMoveRobot::SharedPtr,
        const std::shared_ptr<const MoveRobot::Feedback> feedback)
    {
        RCLCPP_INFO(this->get_logger(), "Progress: %.2f%%", feedback->progress * 100);
    }

    void result_callback(const GoalHandleMoveRobot::WrappedResult & result)
    {
        if (result.code == rclcpp_action::ResultCode::SUCCEEDED)
        {
            RCLCPP_INFO(this->get_logger(), "Result received: %s", result.result->message.c_str());
        }
        else
        {
            RCLCPP_ERROR(this->get_logger(), "Goal failed!");
        }
    }
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<MoveItServiceClient>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
