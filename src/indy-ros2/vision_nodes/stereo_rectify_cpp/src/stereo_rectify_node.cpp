#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <thread>
#include <mutex>
#include <atomic>
#include <condition_variable>
#include <optional>

#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>

#include "res_msgs/msg/pose_res.hpp"
#include "yolov8_msgs/msg/yolov8_inference.hpp"
#include "depth_signal_msgs/msg/depth_signal.hpp"
#include "position_signal_msgs/msg/position_signal.hpp"

using std::placeholders::_1;
using std::placeholders::_2;

using SyncPolicy = message_filters::sync_policies::ApproximateTime<
    sensor_msgs::msg::Image,
    sensor_msgs::msg::Image>;

class StereoRectifyFusionNode : public rclcpp::Node
{
public:
    StereoRectifyFusionNode() : Node("stereo_rectify_fusion_node")
    {
        cv::setUseOptimized(true);

        load_yaml_calibration();

        left_info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
            "/stereo/left/camera_info_calib",
            rclcpp::QoS(1).transient_local().reliable(),
            std::bind(&StereoRectifyFusionNode::left_info_cb, this, _1));

        right_info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
            "/stereo/right/camera_info_calib",
            rclcpp::QoS(1).transient_local().reliable(),
            std::bind(&StereoRectifyFusionNode::right_info_cb, this, _1));

        // ===== ApproximateTime sync =====
        left_sub_.subscribe(this, "/stereo/left/image_raw");
        right_sub_.subscribe(this, "/stereo/right/image_raw");

        sync_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(
            SyncPolicy(3), left_sub_, right_sub_);
        sync_->registerCallback(
            std::bind(&StereoRectifyFusionNode::sync_callback, this, _1, _2));

        pose_sub_ = create_subscription<res_msgs::msg::PoseRes>(
            "/pose_res", 10,
            std::bind(&StereoRectifyFusionNode::pose_cb, this, _1));

        yolo_sub_ = create_subscription<yolov8_msgs::msg::Yolov8Inference>(
            "/detect_msg", 10,
            std::bind(&StereoRectifyFusionNode::yolo_cb, this, _1));

        depth_signal_sub_ = create_subscription<depth_signal_msgs::msg::DepthSignal>(
            "/depth_signal", 10,
            std::bind(&StereoRectifyFusionNode::depth_signal_cb, this, _1));

        position_signal_sub_ = create_subscription<position_signal_msgs::msg::PositionSignal>(
            "/position_signal", 10,
            std::bind(&StereoRectifyFusionNode::position_signal_cb, this, _1));

        yolo_pub_l_ = create_publisher<sensor_msgs::msg::Image>("/stereo/left/img_for_yolo",  10);
        yolo_pub_r_ = create_publisher<sensor_msgs::msg::Image>("/stereo/right/img_for_yolo", 10);

        processing_thread_ = std::thread(&StereoRectifyFusionNode::process_stereo, this);

        RCLCPP_INFO(get_logger(), "StereoRectifyFusionNode started");
    }

    ~StereoRectifyFusionNode()
    {
        stop_thread_.store(true);
        condition_.notify_all();
        if (processing_thread_.joinable())
            processing_thread_.join();
    }

private:
    // ===== Subscribers =====
    message_filters::Subscriber<sensor_msgs::msg::Image> left_sub_, right_sub_;
    std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;

    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr     left_info_sub_, right_info_sub_;
    rclcpp::Subscription<res_msgs::msg::PoseRes>::SharedPtr            pose_sub_;
    rclcpp::Subscription<yolov8_msgs::msg::Yolov8Inference>::SharedPtr yolo_sub_;
    rclcpp::Subscription<depth_signal_msgs::msg::DepthSignal>::SharedPtr depth_signal_sub_;
    rclcpp::Subscription<position_signal_msgs::msg::PositionSignal>::SharedPtr position_signal_sub_;

    // ===== Publishers =====
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr yolo_pub_l_, yolo_pub_r_;

    // ===== Threading =====
    std::thread             processing_thread_;
    std::mutex              pair_mutex_;
    std::condition_variable condition_;
    std::atomic<bool>       stop_thread_{false};

    // ===== Atomic flags =====
    std::atomic<bool>   allow_rectify_{true};
    std::atomic<bool>   computing_depth_{false};
    std::atomic<bool>   computing_position_{false};
    std::atomic<bool>   flag_{false};
    std::atomic<double> last_move_{0.0};

    // ===== Pair buffering =====
    std::optional<std::pair<sensor_msgs::msg::Image::ConstSharedPtr,
                            sensor_msgs::msg::Image::ConstSharedPtr>> latest_stereo_pair_;

    // ===== Remap maps =====
    cv::Mat yaml_map1_left_,  yaml_map2_left_;
    cv::Mat yaml_map1_right_, yaml_map2_right_;
    cv::Mat info_map1_left_,  info_map2_left_;
    cv::Mat info_map1_right_, info_map2_right_;

    // ===== Reusable image buffers =====
    cv::Mat yaml_left_,  yaml_right_;
    cv::Mat final_left_, final_right_;
    sensor_msgs::msg::Image pub_msg_l_, pub_msg_r_;
    bool msg_init_ = false;

    // ===== CameraInfo =====
    sensor_msgs::msg::CameraInfo::SharedPtr left_info_msg_, right_info_msg_;

    // =========================================================
    void load_yaml_calibration()
    {
        std::string pkg = ament_index_cpp::get_package_share_directory("stereo_rectify_cpp");

        YAML::Node left_yaml  = YAML::LoadFile(pkg + "/config/left_camera.yaml");
        YAML::Node right_yaml = YAML::LoadFile(pkg + "/config/right_camera.yaml");

        cv::Size img_size(
            left_yaml["image_width"].as<int>(),
            left_yaml["image_height"].as<int>());

        auto load_mat = [](YAML::Node& node, const std::string& key,
                           int rows, int cols) -> cv::Mat {
            auto v = node[key]["data"].as<std::vector<double>>();
            return cv::Mat(rows, cols, CV_64F, v.data()).clone();
        };

        cv::Mat K1 = load_mat(left_yaml,  "camera_matrix",           3, 3);
        cv::Mat D1 = load_mat(left_yaml,  "distortion_coefficients", 1, 5);
        cv::Mat R1 = load_mat(left_yaml,  "rectification_matrix",    3, 3);
        cv::Mat P1 = load_mat(left_yaml,  "projection_matrix",       3, 4);

        cv::Mat K2 = load_mat(right_yaml, "camera_matrix",           3, 3);
        cv::Mat D2 = load_mat(right_yaml, "distortion_coefficients", 1, 5);
        cv::Mat R2 = load_mat(right_yaml, "rectification_matrix",    3, 3);
        cv::Mat P2 = load_mat(right_yaml, "projection_matrix",       3, 4);

        cv::initUndistortRectifyMap(K1, D1, R1, P1(cv::Rect(0,0,3,3)),
            img_size, CV_16SC2, yaml_map1_left_,  yaml_map2_left_);
        cv::initUndistortRectifyMap(K2, D2, R2, P2(cv::Rect(0,0,3,3)),
            img_size, CV_16SC2, yaml_map1_right_, yaml_map2_right_);

        RCLCPP_INFO(get_logger(), "YAML calibration loaded OK");
    }

    // =========================================================
    void update_info_maps()
    {
        if (!left_info_msg_ || !right_info_msg_) return;

        cv::Size img_size(
            static_cast<int>(left_info_msg_->width),
            static_cast<int>(left_info_msg_->height));

        auto make_mat = [](const auto& arr, int rows, int cols) -> cv::Mat {
            std::vector<double> v(arr.begin(), arr.end());
            return cv::Mat(rows, cols, CV_64F, v.data()).clone();
        };

        cv::Mat iK1 = make_mat(left_info_msg_->k, 3, 3);
        cv::Mat iD1 = make_mat(left_info_msg_->d, 1, static_cast<int>(left_info_msg_->d.size()));
        cv::Mat iR1 = make_mat(left_info_msg_->r, 3, 3);
        cv::Mat iP1 = make_mat(left_info_msg_->p, 3, 4);

        cv::Mat iK2 = make_mat(right_info_msg_->k, 3, 3);
        cv::Mat iD2 = make_mat(right_info_msg_->d, 1, static_cast<int>(right_info_msg_->d.size()));
        cv::Mat iR2 = make_mat(right_info_msg_->r, 3, 3);
        cv::Mat iP2 = make_mat(right_info_msg_->p, 3, 4);

        cv::initUndistortRectifyMap(iK1, iD1, iR1, iP1,
            img_size, CV_16SC2, info_map1_left_,  info_map2_left_);
        cv::initUndistortRectifyMap(iK2, iD2, iR2, iP2,
            img_size, CV_16SC2, info_map1_right_, info_map2_right_);

        RCLCPP_INFO(get_logger(), "CameraInfo maps updated OK");
    }

    // =========================================================
    void left_info_cb(const sensor_msgs::msg::CameraInfo::SharedPtr msg)
    {
        left_info_msg_ = msg;
        update_info_maps();
    }

    void right_info_cb(const sensor_msgs::msg::CameraInfo::SharedPtr msg)
    {
        right_info_msg_ = msg;
        update_info_maps();
    }

    void pose_cb(const res_msgs::msg::PoseRes::SharedPtr msg)
    {
        if (msg->pose_res.empty()) return;
        last_move_.store(msg->pose_res[0].x);
        flag_.store(msg->pose_res[0].flag);
        if (!msg->pose_res[0].flag)
            allow_rectify_.store(true);
    }

    void yolo_cb(const yolov8_msgs::msg::Yolov8Inference::SharedPtr msg)
    {
        if (!msg->yolov8_inference.empty())
            allow_rectify_.store(true);
    }

    void depth_signal_cb(const depth_signal_msgs::msg::DepthSignal::SharedPtr msg)
    {
        computing_depth_.store(msg->computing_depth);
        allow_rectify_.store(!computing_depth_.load() && !computing_position_.load());
    }

    void position_signal_cb(const position_signal_msgs::msg::PositionSignal::SharedPtr msg)
    {
        computing_position_.store(msg->computing_position);
        allow_rectify_.store(!computing_depth_.load() && !computing_position_.load());
    }

    // =========================================================
    // ✅ ApproximateTime sync callback — thay thế left_cb + right_cb
    void sync_callback(
        const sensor_msgs::msg::Image::ConstSharedPtr& left_msg,
        const sensor_msgs::msg::Image::ConstSharedPtr& right_msg)
    {
        if (!allow_rectify_.load(std::memory_order_relaxed)) return;
        if (computing_depth_.load() || computing_position_.load()) return;
        if (rclcpp::Time(left_msg->header.stamp).seconds() < last_move_.load()) return;

        {
            std::lock_guard<std::mutex> lock(pair_mutex_);
            latest_stereo_pair_ = std::make_pair(left_msg, right_msg);
        }
        condition_.notify_one();
    }

    // =========================================================
    void process_stereo()
    {
        while (!stop_thread_.load() && rclcpp::ok()) {
            decltype(latest_stereo_pair_) pair;
            {
                std::unique_lock<std::mutex> lock(pair_mutex_);
                condition_.wait(lock, [this] {
                    return latest_stereo_pair_.has_value() || stop_thread_.load();
                });
                if (stop_thread_.load()) break;
                pair = std::move(latest_stereo_pair_);
                latest_stereo_pair_.reset();
            }

            stereo_callback(pair->first, pair->second);
        }
    }

    // =========================================================
    void stereo_callback(
        const sensor_msgs::msg::Image::ConstSharedPtr& left_msg,
        const sensor_msgs::msg::Image::ConstSharedPtr& right_msg)
    {
        if (!allow_rectify_.load() || flag_.load())
        {     
            RCLCPP_WARN(get_logger(), "Rectify skipped: flag=%d, last_move=%.2f", flag_.load(), last_move_.load());  
            return;
        }

        if (computing_depth_.load() || computing_position_.load()) 
        {
            RCLCPP_WARN(get_logger(), "Rectify skipped: computing_depth=%d, computing_position=%d",
                computing_depth_.load(), computing_position_.load());
            return;
        }
        
        if (rclcpp::Time(left_msg->header.stamp).seconds() < last_move_.load()) 
        {
            RCLCPP_WARN(get_logger(), "Rectify skipped: timestamp mismatch");
            return;
        }

        if (yaml_map1_left_.empty()  || yaml_map1_right_.empty() ||
            info_map1_left_.empty()  || info_map1_right_.empty()) {
            RCLCPP_WARN_ONCE(get_logger(), "Maps not ready, skipping");
            return;
        }

        cv_bridge::CvImageConstPtr left_ptr, right_ptr;
        try {
            left_ptr  = cv_bridge::toCvShare(left_msg,  "bgr8");
            right_ptr = cv_bridge::toCvShare(right_msg, "bgr8");
        } catch (const cv_bridge::Exception& e) {
            RCLCPP_ERROR(get_logger(), "cv_bridge: %s", e.what());
            return;
        }
        if (!left_ptr  || left_ptr->image.empty() ||
            !right_ptr || right_ptr->image.empty()) return;

        const cv::Mat& raw_left  = left_ptr->image;
        const cv::Mat& raw_right = right_ptr->image;

        // ---- remap left async, right trên main thread ----
        cv::remap(raw_left,   yaml_left_,  yaml_map1_left_,  yaml_map2_left_,  cv::INTER_LINEAR);
        cv::remap(yaml_left_, final_left_, info_map1_left_,  info_map2_left_,  cv::INTER_LINEAR);

        cv::remap(raw_right,    yaml_right_,  yaml_map1_right_, yaml_map2_right_, cv::INTER_LINEAR);
        cv::remap(yaml_right_,  final_right_, info_map1_right_, info_map2_right_, cv::INTER_LINEAR);

        if (!msg_init_) {
            pub_msg_l_.encoding = "bgr8";
            pub_msg_r_.encoding = "bgr8";

            pub_msg_l_.height = final_left_.rows;
            pub_msg_l_.width  = final_left_.cols;
            pub_msg_l_.step   = final_left_.cols * 3;

            pub_msg_r_ = pub_msg_l_;

            pub_msg_l_.data.resize(pub_msg_l_.step * pub_msg_l_.height);
            pub_msg_r_.data.resize(pub_msg_r_.step * pub_msg_r_.height);

            msg_init_ = true;
        }

        pub_msg_l_.header = left_msg->header;
        pub_msg_r_.header = right_msg->header;

        // ⚠️ copy nhanh hơn cv_bridge rất nhiều
        memcpy(pub_msg_l_.data.data(), final_left_.data, pub_msg_l_.data.size());
        memcpy(pub_msg_r_.data.data(), final_right_.data, pub_msg_r_.data.size());

        yolo_pub_l_->publish(pub_msg_l_);
        yolo_pub_r_->publish(pub_msg_r_);
    }
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<StereoRectifyFusionNode>();
    rclcpp::executors::MultiThreadedExecutor exec;
    exec.add_node(node);
    exec.spin();
    rclcpp::shutdown();
    return 0;
}
