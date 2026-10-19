#include "rclcpp/rclcpp.hpp"
#include "start_request_service/srv/start_request.hpp"
#include "connect_msgs/msg/connect_msg.hpp"
#include "res_msgs/msg/pose_res.hpp"

using std::placeholders::_1;
using std::placeholders::_2;

class StartRequestServer : public rclcpp::Node
{
public:
  StartRequestServer() : Node("start_request_server")
  {
    service_ = this->create_service<start_request_service::srv::StartRequest>(
      "start_request", std::bind(&StartRequestServer::handle_service, this, _1, _2));

    connect_pub_ = this->create_publisher<connect_msgs::msg::ConnectMsg>("/connect_msg", 10);
    pose_pub_ = this->create_publisher<res_msgs::msg::PoseRes>("/pose_res", 10);
    connection_ = this->create_subscription<connect_msgs::msg::ConnectMsg>(
    "/connect_msg", 10, std::bind(&StartRequestServer::connection_callback, this, std::placeholders::_1));

    RCLCPP_INFO(this->get_logger(), "Start Request Service Ready.");
  }

private:
  rclcpp::Service<start_request_service::srv::StartRequest>::SharedPtr service_;
  rclcpp::Publisher<connect_msgs::msg::ConnectMsg>::SharedPtr connect_pub_;
  rclcpp::Publisher<res_msgs::msg::PoseRes>::SharedPtr pose_pub_;
  rclcpp::Subscription<connect_msgs::msg::ConnectMsg>::SharedPtr connection_;
  std::atomic<bool> reset_status_{false};

  void handle_service(
    const std::shared_ptr<start_request_service::srv::StartRequest::Request> request,
    std::shared_ptr<start_request_service::srv::StartRequest::Response> response)
  {
    (void)request;
    if (reset_status_) {
      auto connect_msg = connect_msgs::msg::ConnectMsg();
      auto conn = connect_msgs::msg::ConnectStatus();
      conn.connection = true;
      conn.wait_key = false;
      conn.id = 0;
      connect_msg.connect_msg.push_back(conn);
      connect_pub_->publish(connect_msg);

      auto pose_msg = res_msgs::msg::PoseRes();
      auto pose = res_msgs::msg::ResFlag();
      pose.x = 0.0;
      pose.flag = 0;
      pose_msg.pose_res.push_back(pose);
      pose_pub_->publish(pose_msg);

      response->success = true;

      RCLCPP_INFO(this->get_logger(), "Published connect_msg and pose_res.");
    } else {
      response->success = false;
      RCLCPP_WARN(this->get_logger(), "Service not available.");
    }
  }

  void connection_callback(const connect_msgs::msg::ConnectMsg& msg) {
    if (!msg.connect_msg.empty()) {
        const auto& result = msg.connect_msg.front();
        reset_status_ = result.wait_key;
    }
  }
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<StartRequestServer>());
  rclcpp::shutdown();
  return 0;
}
