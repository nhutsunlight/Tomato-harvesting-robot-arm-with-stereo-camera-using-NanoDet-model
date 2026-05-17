#include <memory>

#include "connect_msgs/msg/connect_msg.hpp"
#include "connect_msgs/msg/connect_status.hpp"
#include "rclcpp/rclcpp.hpp"
#include "yolov8_msgs/msg/yolov8_inference.hpp"

using std::placeholders::_1;

class HarvestFlagBridge : public rclcpp::Node
{
public:
  HarvestFlagBridge()
  : Node("harvest_flag_bridge")
  {
    connect_pub_ = create_publisher<connect_msgs::msg::ConnectMsg>("/connect1_msg", 10);

    connect_sub_ = create_subscription<connect_msgs::msg::ConnectMsg>(
      "/connect_msg", 10, std::bind(&HarvestFlagBridge::connectCallback, this, _1));

    yolo_sub_ = create_subscription<yolov8_msgs::msg::Yolov8Inference>(
      "/Yolov8_Inference", 10, std::bind(&HarvestFlagBridge::yoloCallback, this, _1));

    RCLCPP_INFO(get_logger(), "Harvest flag bridge started.");
  }

private:
  void connectCallback(const connect_msgs::msg::ConnectMsg::SharedPtr msg)
  {
    if (msg->connect_msg.empty()) {
      RCLCPP_WARN(get_logger(), "Received empty /connect_msg.");
      return;
    }

    latest_status_ = msg->connect_msg.front();

    publishConnect1(latest_status_.harvest_flag);
  }

  void yoloCallback(const yolov8_msgs::msg::Yolov8Inference::SharedPtr)
  {
    if (harvest_flag_) {
      return;
    }

    publishConnect1(true);
  }

  void publishConnect1(const bool harvest_flag)
  {
    connect_msgs::msg::ConnectMsg out_msg;
    auto out_status = latest_status_;
    out_status.harvest_flag = harvest_flag;
    out_msg.connect_msg.push_back(out_status);

    connect_pub_->publish(out_msg);
    harvest_flag_ = harvest_flag;

    RCLCPP_INFO(
      get_logger(), "Published /connect1_msg: connection=%s wait_key=%s harvest_flag=%s id=%ld",
      out_status.connection ? "true" : "false",
      out_status.wait_key ? "true" : "false",
      out_status.harvest_flag ? "true" : "false",
      out_status.id);
  }

  rclcpp::Publisher<connect_msgs::msg::ConnectMsg>::SharedPtr connect_pub_;
  rclcpp::Subscription<connect_msgs::msg::ConnectMsg>::SharedPtr connect_sub_;
  rclcpp::Subscription<yolov8_msgs::msg::Yolov8Inference>::SharedPtr yolo_sub_;

  connect_msgs::msg::ConnectStatus latest_status_;
  bool harvest_flag_{false};
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<HarvestFlagBridge>());
  rclcpp::shutdown();
  return 0;
}
