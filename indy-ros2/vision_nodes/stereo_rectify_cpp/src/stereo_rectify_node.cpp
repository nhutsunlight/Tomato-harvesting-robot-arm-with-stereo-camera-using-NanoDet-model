#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <thread>
#include <future>
#include <mutex>
#include "res_msgs/msg/pose_res.hpp"
#include <vector>
#include <cmath>
#include <numeric>
#include <unordered_map>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>

#include <set>
#include <array>



// message_filters
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/synchronizer.h>
#include <iostream>

using namespace std::placeholders;
using namespace cv;
using namespace std;
using SyncPolicy = message_filters::sync_policies::ApproximateTime<
    sensor_msgs::msg::Image, sensor_msgs::msg::Image>;

class StereoRectifyNode : public rclcpp::Node
{
public:
    StereoRectifyNode()
        : Node("stereo_rectify_node"),
          max_time_diff_(rclcpp::Duration::from_seconds(0.05))
    {
        RCLCPP_INFO(this->get_logger(), "✅ StereoRectifyNode started (ApproximateTime Sync)");

        // --- Camera Info ---
        left_camera_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
            "/stereo/left/camera_info_calib", rclcpp::QoS(1).transient_local().reliable(),
            std::bind(&StereoRectifyNode::left_camera_info_callback, this, std::placeholders::_1));

        right_camera_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
            "/stereo/right/camera_info_calib", rclcpp::QoS(1).transient_local().reliable(),
            std::bind(&StereoRectifyNode::right_camera_info_callback, this, std::placeholders::_1));

        // --- Pose subscription ---
        subscription_ = this->create_subscription<res_msgs::msg::PoseRes>(
               "/pose_res", 10,
            std::bind(&StereoRectifyNode::timestamp_callback, this, std::placeholders::_1));

        // --- Publishers ---
        //left_pub_ = this->create_publisher<sensor_msgs::msg::Image>("/stereo/left/stereo_img", 10);
        //right_pub_ = this->create_publisher<sensor_msgs::msg::Image>("/stereo/right/stereo_img", 10);
        img_for_yolo_pub_ = this->create_publisher<sensor_msgs::msg::Image>("/stereo/left/img_for_yolo", 10);
        img_for_yolo_pub_r_ = this->create_publisher<sensor_msgs::msg::Image>("/stereo/right/img_for_yolo", 10);


        // --- message_filters Subscribers ---
        left_mf_sub_ = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::Image>>(
            //this, "/stereo/left/image_raw_calib", rmw_qos_profile_default);
            this, "/stereo/left/image_rect", rmw_qos_profile_default);
        right_mf_sub_ = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::Image>>(
            //this, "/stereo/right/image_raw_calib", rmw_qos_profile_default);
            this, "/stereo/right/image_rect", rmw_qos_profile_default);
        //clahe_ = cv::createCLAHE(2.0, cv::Size(8, 8));
        
        sync_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(
            SyncPolicy(10), *left_mf_sub_, *right_mf_sub_);

        sync_->registerCallback(std::bind(&StereoRectifyNode::stereo_callback, this, _1, _2));
    }

private:
    // === ROS Components ===
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr left_camera_info_sub_;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr right_camera_info_sub_;
    rclcpp::Subscription<res_msgs::msg::PoseRes>::SharedPtr subscription_;
    //rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr left_pub_, right_pub_, img_for_yolo_pub_, img_for_yolo_pub_r_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr img_for_yolo_pub_, img_for_yolo_pub_r_;

    // === message_filters ===
    std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::Image>> left_mf_sub_, right_mf_sub_;
    std::shared_ptr<message_filters::Synchronizer<
        message_filters::sync_policies::ApproximateTime<
            sensor_msgs::msg::Image,
            sensor_msgs::msg::Image>>> sync_;

    // === Camera info & maps ===
    sensor_msgs::msg::CameraInfo::SharedPtr left_camera_info_, right_camera_info_;
    cv::Mat left_map1_, left_map2_, right_map1_, right_map2_;

    // === Misc ===
    rclcpp::Duration max_time_diff_;
    //cv::Ptr<cv::CLAHE> clahe_;
    double last_move_{0.0};
    bool flag_{false};

    // ============= CALLBACKS =============
    void timestamp_callback(const res_msgs::msg::PoseRes::SharedPtr msg)
    {
        if (!msg->pose_res.empty()) {
            last_move_ = msg->pose_res[0].x;
            flag_ = msg->pose_res[0].flag;
        }
    }

    void left_camera_info_callback(const sensor_msgs::msg::CameraInfo::SharedPtr msg)
    {
        left_camera_info_ = msg;
        update_rectification();
    }

    void right_camera_info_callback(const sensor_msgs::msg::CameraInfo::SharedPtr msg)
    {
        right_camera_info_ = msg;
        update_rectification();
    }

    void update_rectification()
    {
        if (!left_camera_info_ || !right_camera_info_)
            return;
            
        cv::Mat K1(3, 3, CV_64F, (void*)left_camera_info_->k.data());
        cv::Mat D1(1, 5, CV_64F, (void*)left_camera_info_->d.data());
        cv::Mat R1(3, 3, CV_64F, (void*)left_camera_info_->r.data());
        cv::Mat P1(3, 4, CV_64F, (void*)left_camera_info_->p.data());

        cv::Mat K2(3, 3, CV_64F, (void*)right_camera_info_->k.data());
        cv::Mat D2(1, 5, CV_64F, (void*)right_camera_info_->d.data());
        cv::Mat R2(3, 3, CV_64F, (void*)right_camera_info_->r.data());
        cv::Mat P2(3, 4, CV_64F, (void*)right_camera_info_->p.data());

        cv::Size image_size(left_camera_info_->width, left_camera_info_->height);

        try {
            cv::initUndistortRectifyMap(K1, D1, R1, P1, image_size, CV_16SC2, left_map1_, left_map2_);
            cv::initUndistortRectifyMap(K2, D2, R2, P2, image_size, CV_16SC2, right_map1_, right_map2_);
            RCLCPP_INFO(this->get_logger(), "Rectification maps updated.");
        } catch (const cv::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "OpenCV error in update_rectification: %s", e.what());
        }
    }

    void stereo_callback(const sensor_msgs::msg::Image::ConstSharedPtr &left_msg,
                        const sensor_msgs::msg::Image::ConstSharedPtr &right_msg)
    {
        if (left_map1_.empty() || left_map2_.empty() || right_map1_.empty() || right_map2_.empty())
        {
            RCLCPP_WARN(this->get_logger(), "Rectification maps not ready yet, skipping frame");
            return;
        }
        if (rclcpp::Time(left_msg->header.stamp).seconds() < last_move_ || rclcpp::Time(right_msg->header.stamp).seconds() < last_move_ || flag_)
        {
            return;
        }
        // ===== 1. Lấy ảnh gốc =====
        cv::Mat left = cv_bridge::toCvShare(left_msg, "bgr8")->image;
        cv::Mat right = cv_bridge::toCvShare(right_msg, "bgr8")->image;

        // ===== 2. RECTIFY — dùng INTER_LINEAR thay INTER_LANCZOS4 (nhanh hơn nhiều) =====
        cv::Mat rectL, rectR;
        cv::remap(left, rectL, left_map1_, left_map2_, cv::INTER_LINEAR, cv::BORDER_REPLICATE);
        cv::remap(right, rectR, right_map1_, right_map2_, cv::INTER_LINEAR, cv::BORDER_REPLICATE);

        // ===== Publish color =====
        img_for_yolo_pub_->publish(*cv_bridge::CvImage(left_msg->header, "bgr8", rectL).toImageMsg());
        img_for_yolo_pub_r_->publish(*cv_bridge::CvImage(right_msg->header, "bgr8", rectR).toImageMsg());
/*
        // ===== 3. Convert sang grayscale =====
        cv::Mat grayL, grayR;
        cv::cvtColor(rectL, grayL, cv::COLOR_BGR2GRAY);
        cv::cvtColor(rectR, grayR, cv::COLOR_BGR2GRAY);

        // 4. CLAHE — dùng clahe_ member thay vì tạo mới mỗi lần
        clahe_->apply(grayL, grayL);
        clahe_->apply(grayR, grayR);

        // 5. Gaussian
        cv::GaussianBlur(grayL, grayL, cv::Size(5,5), 0);
        cv::GaussianBlur(grayR, grayR, cv::Size(5,5), 0);

        // 6. Match brightness
        double meanL = cv::mean(grayL)[0];
        double meanR = cv::mean(grayR)[0];
        grayR.convertTo(grayR, -1, meanL / (meanR + 1e-6), 0);

        // 7. Publish
        left_pub_->publish(*cv_bridge::CvImage(left_msg->header, "mono8", grayL).toImageMsg());
        right_pub_->publish(*cv_bridge::CvImage(right_msg->header, "mono8", grayR).toImageMsg());
*/
        // imwrite chỉ dùng khi debug — comment lại khi chạy thật
        // cv::imwrite("left.png", grayL);
        // cv::imwrite("right.png", grayR);
    }
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<StereoRectifyNode>();
    rclcpp::executors::MultiThreadedExecutor exec;
    exec.add_node(node);
    exec.spin();
    rclcpp::shutdown();
    return 0;
}
