#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>
#include "res_msgs/msg/pose_res.hpp"
#include "yolov8_msgs/msg/yolov8_inference.hpp"

class StereoRectifyNode : public rclcpp::Node
{
public:
    StereoRectifyNode()
    : Node("camera_rectify_node")
    {
        load_calibration_files();

        left_sub_.subscribe(this, "/stereo/left/image_raw");
        right_sub_.subscribe(this, "/stereo/right/image_raw");

        sync_ = std::make_shared<Sync>(SyncPolicy(10), left_sub_, right_sub_);
        sync_->registerCallback(
            std::bind(&StereoRectifyNode::image_callback, this, std::placeholders::_1, std::placeholders::_2));

        subscription_ = this->create_subscription<res_msgs::msg::PoseRes>(
            "/pose_res", 10, std::bind(&StereoRectifyNode::timestamp_callback, this, std::placeholders::_1));

        sub_yolo_ = this->create_subscription<yolov8_msgs::msg::Yolov8Inference>(
            "/detect_msg", 10, std::bind(&StereoRectifyNode::yolo_callback, this, std::placeholders::_1));

        left_pub_ = this->create_publisher<sensor_msgs::msg::Image>("/stereo/left/image_rect", 10);
        right_pub_ = this->create_publisher<sensor_msgs::msg::Image>("/stereo/right/image_rect", 10);

        RCLCPP_INFO(this->get_logger(), "Camera Rectify Node Started!");
    }

private:
    using SyncPolicy = message_filters::sync_policies::ApproximateTime<
        sensor_msgs::msg::Image, sensor_msgs::msg::Image>;
    using Sync = message_filters::Synchronizer<SyncPolicy>;

    message_filters::Subscriber<sensor_msgs::msg::Image> left_sub_, right_sub_;
    std::shared_ptr<Sync> sync_;

    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr left_pub_, right_pub_;
    cv::Mat left_map1_, left_map2_, right_map1_, right_map2_;
    rclcpp::Subscription<res_msgs::msg::PoseRes>::SharedPtr subscription_;
    rclcpp::Subscription<yolov8_msgs::msg::Yolov8Inference>::SharedPtr sub_yolo_;
    bool flag_ = false;
    bool allow_rectify_ = true;

    void timestamp_callback(const res_msgs::msg::PoseRes::SharedPtr msg) {    
        if (!msg->pose_res.empty()) {
            flag_ = msg->pose_res[0].pause;
            if (!flag_) {
                allow_rectify_ = true;  // Allow processing when pause is false
            }
        }
    }

    void yolo_callback(const yolov8_msgs::msg::Yolov8Inference::SharedPtr msg) {
        if (!msg->yolov8_inference.empty()) {
            allow_rectify_ = false;
        }
    }

    void load_calibration_files()
    {
        std::string package_share_dir = ament_index_cpp::get_package_share_directory("camera_rectify_cpp");
        std::string left_yaml_path = package_share_dir + "/config/left_camera.yaml";
        std::string right_yaml_path = package_share_dir + "/config/right_camera.yaml";

        YAML::Node left_data = YAML::LoadFile(left_yaml_path);
        YAML::Node right_data = YAML::LoadFile(right_yaml_path);

        cv::Size image_size(left_data["image_width"].as<int>(), left_data["image_height"].as<int>());

        auto K1 = cv::Mat(3, 3, CV_64F, left_data["camera_matrix"]["data"].as<std::vector<double>>().data()).clone();
        auto D1 = cv::Mat(1, 5, CV_64F, left_data["distortion_coefficients"]["data"].as<std::vector<double>>().data()).clone();
        auto R1 = cv::Mat(3, 3, CV_64F, left_data["rectification_matrix"]["data"].as<std::vector<double>>().data()).clone();
        auto P1 = cv::Mat(3, 4, CV_64F, right_data["projection_matrix"]["data"].as<std::vector<double>>().data()).clone();

        auto K2 = cv::Mat(3, 3, CV_64F, right_data["camera_matrix"]["data"].as<std::vector<double>>().data()).clone();
        auto D2 = cv::Mat(1, 5, CV_64F, right_data["distortion_coefficients"]["data"].as<std::vector<double>>().data()).clone();
        auto R2 = cv::Mat(3, 3, CV_64F, right_data["rectification_matrix"]["data"].as<std::vector<double>>().data()).clone();
        auto P2 = cv::Mat(3, 4, CV_64F, right_data["projection_matrix"]["data"].as<std::vector<double>>().data()).clone();

        cv::initUndistortRectifyMap(K1, D1, R1, P1(cv::Rect(0, 0, 3, 3)), image_size, CV_16SC2, left_map1_, left_map2_);
        cv::initUndistortRectifyMap(K2, D2, R2, P2(cv::Rect(0, 0, 3, 3)), image_size, CV_16SC2, right_map1_, right_map2_);
    }

    void image_callback(
        const sensor_msgs::msg::Image::ConstSharedPtr &left_msg,
        const sensor_msgs::msg::Image::ConstSharedPtr &right_msg)
    {
        if (!allow_rectify_) {return;}  // Skip processing if flag is set

        cv::Mat left_image = cv_bridge::toCvCopy(left_msg, "bgr8")->image;
        cv::Mat right_image = cv_bridge::toCvCopy(right_msg, "bgr8")->image;

        cv::Mat left_rect, right_rect;
        cv::remap(left_image, left_rect, left_map1_, left_map2_, cv::INTER_LINEAR);
        cv::remap(right_image, right_rect, right_map1_, right_map2_, cv::INTER_LINEAR);

        auto left_msg_rect = cv_bridge::CvImage(left_msg->header, "bgr8", left_rect).toImageMsg();
        auto right_msg_rect = cv_bridge::CvImage(right_msg->header, "bgr8", right_rect).toImageMsg();

        left_pub_->publish(*left_msg_rect);
        right_pub_->publish(*right_msg_rect);
    }
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<StereoRectifyNode>());
    rclcpp::shutdown();
    return 0;
}
