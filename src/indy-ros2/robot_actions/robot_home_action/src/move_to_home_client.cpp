#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include "robot_home_action/action/move_to_home.hpp"

class MoveToHomeClient : public rclcpp::Node
{
public:
    using MoveToHome = robot_home_action::action::MoveToHome;
    using GoalHandleMoveToHome = rclcpp_action::ClientGoalHandle<MoveToHome>;

    MoveToHomeClient() : Node("move_to_home_client")
    {
        action_client_ = rclcpp_action::create_client<MoveToHome>(this, "move_to_home");

        if (!action_client_->wait_for_action_server(std::chrono::seconds(2)) && rclcpp::ok())
        {
            RCLCPP_ERROR(this->get_logger(), "Action server not available!");
            return;
        }

        send_goal();
    }

    void send_goal()
    {
        this->declare_parameter<std::vector<double>>("joint_positions", {0.0, 0.0, -1.57, 0.0, -1.57, 0.0});
        auto joint_positions = this->get_parameter("joint_positions").as_double_array();
        this->declare_parameter<int>("id", 1);
        auto id = this->get_parameter("id").as_int();
        this->declare_parameter<int>("pass_permit", 0);
        auto pass_permit = this->get_parameter("pass_permit").as_int();

        auto goal_msg = MoveToHome::Goal();
        goal_msg.joint_positions = std::vector<double>(joint_positions.begin(), joint_positions.end());
        goal_msg.id = id;
        goal_msg.pass_permit = pass_permit;

        RCLCPP_INFO(this->get_logger(), "Sending goal to move robot to target joint position...");
        auto send_goal_options = rclcpp_action::Client<MoveToHome>::SendGoalOptions();

        send_goal_options.result_callback = [](const GoalHandleMoveToHome::WrappedResult &result)
        {
            if (result.code == rclcpp_action::ResultCode::SUCCEEDED)
            {
                RCLCPP_INFO(rclcpp::get_logger("move_to_home_client"), "Result: %s", result.result->message.c_str());
            }
            else
            {
                RCLCPP_ERROR(rclcpp::get_logger("move_to_home_client"), "Action failed!");
            }
        };

        action_client_->async_send_goal(goal_msg, send_goal_options);
    }

private:
    rclcpp_action::Client<MoveToHome>::SharedPtr action_client_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<MoveToHomeClient>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
