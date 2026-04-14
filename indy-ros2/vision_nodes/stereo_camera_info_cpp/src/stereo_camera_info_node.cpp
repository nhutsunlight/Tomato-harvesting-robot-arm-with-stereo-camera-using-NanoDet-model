#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
//#include <sensor_msgs/msg/image.hpp>
#include <camera_info_manager/camera_info_manager.hpp>
//#include <message_filters/subscriber.h>
//#include <message_filters/synchronizer.h>
//#include <message_filters/sync_policies/approximate_time.h>
//#include "res_msgs/msg/pose_res.hpp"
#include <yaml-cpp/yaml.h>
#include <filesystem>

class StereoCameraInfoNode : public rclcpp::Node
{
public:
    StereoCameraInfoNode() : Node("stereo_camera_info_node")
    {
        //left_path_ = "package://stereo_camera_info_cpp/config/left.yaml";
        //right_path_ = "package://stereo_camera_info_cpp/config/right.yaml";
        // Load camera info
        camera_info_manager::CameraInfoManager left_camera_info_manager(this, "stereo/left", "package://stereo_camera_info_cpp/config/left.yaml");
        camera_info_manager::CameraInfoManager right_camera_info_manager(this, "stereo/right", "package://stereo_camera_info_cpp/config/right.yaml");

        if (left_camera_info_manager.loadCameraInfo("package://stereo_camera_info_cpp/config/left.yaml")) {
            left_camera_info_ = left_camera_info_manager.getCameraInfo();
        }
        if (right_camera_info_manager.loadCameraInfo("package://stereo_camera_info_cpp/config/right.yaml")) {
            right_camera_info_ = right_camera_info_manager.getCameraInfo();
        }

        // Publishers
        left_camera_info_pub_ = this->create_publisher<sensor_msgs::msg::CameraInfo>("/stereo/left/camera_info_calib", rclcpp::QoS(1).transient_local().reliable());
        right_camera_info_pub_ = this->create_publisher<sensor_msgs::msg::CameraInfo>("/stereo/right/camera_info_calib", rclcpp::QoS(1).transient_local().reliable());
        //left_image_pub_ = this->create_publisher<sensor_msgs::msg::Image>("/stereo/left/image_raw_calib", 10);
        //right_image_pub_ = this->create_publisher<sensor_msgs::msg::Image>("/stereo/right/image_raw_calib", 10);

        Camera_info_publish();
/*
        timer_ = this->create_wall_timer(
            std::chrono::seconds(2),
            std::bind(&StereoCameraInfoNode::checkFiles, this)
        );
*/
        // Subscribers với message_filters
        //left_image_sub_.subscribe(this, "/stereo/left/image_rect");
        //right_image_sub_.subscribe(this, "/stereo/right/image_rect");
//        subscription_ = this->create_subscription<res_msgs::msg::PoseRes>(
//            "/pose_res", 10, std::bind(&StereoCameraInfoNode::timestamp_callback, this, std::placeholders::_1));

        // Đồng bộ hóa với ApproximateTime
        //sync_ = std::make_shared<message_filters::Synchronizer<ApproximateSyncPolicy>>(
        //    ApproximateSyncPolicy(10), left_image_sub_, right_image_sub_);

        //sync_->registerCallback(std::bind(&StereoCameraInfoNode::imageCallback, this, std::placeholders::_1, std::placeholders::_2));

    }

private:
//    using ApproximateSyncPolicy = message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::Image, sensor_msgs::msg::Image>;

//    message_filters::Subscriber<sensor_msgs::msg::Image> left_image_sub_;
//    message_filters::Subscriber<sensor_msgs::msg::Image> right_image_sub_;
//    //rclcpp::Subscription<res_msgs::msg::PoseRes>::SharedPtr subscription_;

//    std::shared_ptr<message_filters::Synchronizer<ApproximateSyncPolicy>> sync_;

    rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr left_camera_info_pub_;
    rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr right_camera_info_pub_;
//    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr left_image_pub_;
//    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr right_image_pub_;

    sensor_msgs::msg::CameraInfo left_camera_info_;
    sensor_msgs::msg::CameraInfo right_camera_info_;

    //std::string left_path_;
    //std::string right_path_;
    //std::filesystem::file_time_type last_left_time_;
    //std::filesystem::file_time_type last_right_time_;
    //rclcpp::TimerBase::SharedPtr timer_;

//    double last_move_;
//    int flag_;
/*
    void timestamp_callback(const res_msgs::msg::PoseRes::SharedPtr msg) {    
        if (!msg->pose_res.empty()) {
            last_move_ = msg->pose_res[0].x;
            flag_ = msg->pose_res[0].flag;
        }
    }

    void imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr &left_msg,
                       const sensor_msgs::msg::Image::ConstSharedPtr &right_msg)
    {

        if (rclcpp::Time(left_msg->header.stamp).seconds() < last_move_ || flag_ == 1) {
            return;
        }
        else {
            last_move_ = 0.0;
        }

        left_camera_info_.header = left_msg->header;
        right_camera_info_.header = right_msg->header;
        left_camera_info_pub_->publish(left_camera_info_);
        right_camera_info_pub_->publish(right_camera_info_);
        left_image_pub_->publish(*left_msg);
        right_image_pub_->publish(*right_msg);
    }
*/
/*
    void checkFiles()
    {
        auto new_left_time = std::filesystem::last_write_time(left_path_);
        auto new_right_time = std::filesystem::last_write_time(right_path_);

        if (new_left_time != last_left_time_ ||
            new_right_time != last_right_time_)
        {
            RCLCPP_INFO(this->get_logger(), "YAML changed → reloading...");
            last_left_time_ = new_left_time;
            last_right_time_ = new_right_time;

            Camera_info_publish();
        }
    }
*/
    void Camera_info_publish() {
        left_camera_info_.header.stamp = this->now();
        right_camera_info_.header.stamp = this->now();
        left_camera_info_pub_->publish(left_camera_info_);
        right_camera_info_pub_->publish(right_camera_info_);
        RCLCPP_INFO(this->get_logger(), "Camera Info Published! ========================================================================");
    }
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<StereoCameraInfoNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
