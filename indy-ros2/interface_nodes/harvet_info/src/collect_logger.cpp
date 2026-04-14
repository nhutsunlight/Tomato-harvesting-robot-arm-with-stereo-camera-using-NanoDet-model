#include <rclcpp/rclcpp.hpp>
#include <collect_msgs/msg/collect_msg.hpp>  // Đảm bảo đúng tên package & msg
#include <fstream>

class CollectLogger : public rclcpp::Node {
public:
    CollectLogger() : Node("collect_logger_node"), count_(0) {
        subscription_ = this->create_subscription<collect_msgs::msg::CollectMsg>(
            "/collect3_msg", 10,
            std::bind(&CollectLogger::topic_callback, this, std::placeholders::_1)
        );

        // Mở file ở chế độ append
        outfile_.open("/home/nhut/indy-ros2/collect_log.txt", std::ios::app);
        if (!outfile_.is_open()) {
            RCLCPP_ERROR(this->get_logger(), "Failed to open log file.");
        } else {
            RCLCPP_INFO(this->get_logger(), "Logging to /home/nhut/collect_log.txt");
        }
    }

    ~CollectLogger() {
        if (outfile_.is_open()) {
            outfile_.close();
        }
    }

private:
    void topic_callback(const collect_msgs::msg::CollectMsg::SharedPtr msg) {
        if (msg->collect_msg[0].total_time > 0.0f) {
            count_++;

            float total_time_sec = msg->collect_msg[0].total_time;
            float detection_time_sec = msg->collect_msg[0].detection_time;
            float positioning_time_sec = msg->collect_msg[0].positioning_time;

            std::ostringstream total_time_str, detection_time_str, positioning_time_str;
            total_time_str << std::fixed << std::setprecision(2) << total_time_sec;
            detection_time_str << std::fixed << std::setprecision(2) << detection_time_sec;
            positioning_time_str << std::fixed << std::setprecision(2) << positioning_time_sec;

            if (outfile_.is_open()) {
                outfile_ << count_ << "\t" << total_time_str.str()
                        << "\t" << detection_time_str.str() << "\t" << positioning_time_str.str() << std::endl;
                RCLCPP_INFO(this->get_logger(), "Count: %d\t, Move time: %s\t, Detect time: %s\t, Positioning time: %s", count_,
                            total_time_str.str().c_str(), detection_time_str.str().c_str(), positioning_time_str.str().c_str());
            }
        }
    }


    rclcpp::Subscription<collect_msgs::msg::CollectMsg>::SharedPtr subscription_;
    std::ofstream outfile_;
    int count_;
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<CollectLogger>());
    rclcpp::shutdown();
    return 0;
}
