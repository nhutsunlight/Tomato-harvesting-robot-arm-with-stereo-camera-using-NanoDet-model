#include "rclcpp/rclcpp.hpp"
#include "rclcpp_action/rclcpp_action.hpp"
#include "gripper_action/action/gripper_control.hpp"

using namespace std::chrono_literals;
using GripperControl = gripper_action::action::GripperControl;

class GripperClient : public rclcpp::Node {
public:
    GripperClient() : Node("gripper_action_client") {
        client_ = rclcpp_action::create_client<GripperControl>(this, "gripper_action");

        while (!client_->wait_for_action_server(2s) && rclcpp::ok()) {
            RCLCPP_INFO(this->get_logger(), "Waiting for gripper action server...");
        }

        this->declare_parameter<int>("id", 1);
        auto id = this->get_parameter("id").as_int();

        auto goal_msg = GripperControl::Goal();
        goal_msg.position = 0.05;  // Đóng/mở tuỳ giá trị
        goal_msg.id = id; 
        goal_msg.pass_permit = 0; // 0: close, 1: open

        RCLCPP_INFO(this->get_logger(), "Sending goal...");
        client_->async_send_goal(goal_msg);
    }

private:
    rclcpp_action::Client<GripperControl>::SharedPtr client_;
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<GripperClient>());
    rclcpp::shutdown();
    return 0;
}
