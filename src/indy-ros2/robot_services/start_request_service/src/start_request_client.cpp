#include "rclcpp/rclcpp.hpp"
#include "start_request_service/srv/start_request.hpp"

using StartRequest = start_request_service::srv::StartRequest;

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  auto node = rclcpp::Node::make_shared("start_request_client");

  auto client = node->create_client<StartRequest>("start_request");

  while (!client->wait_for_service(std::chrono::seconds(1))) {
    RCLCPP_INFO(node->get_logger(), "Waiting for service...");
  }

  auto request = std::make_shared<StartRequest::Request>();

  auto future = client->async_send_request(request);
  if (rclcpp::spin_until_future_complete(node, future) ==
      rclcpp::FutureReturnCode::SUCCESS)
  {
    if (future.get()->success)
      RCLCPP_INFO(node->get_logger(), "Service call succeeded.");
    else
      RCLCPP_WARN(node->get_logger(), "Service call failed.");
  } else {
    RCLCPP_ERROR(node->get_logger(), "Failed to call service.");
  }

  rclcpp::shutdown();
  return 0;
}
