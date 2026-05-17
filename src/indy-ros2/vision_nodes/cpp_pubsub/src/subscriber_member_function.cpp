#include <pcl/ModelCoefficients.h>
#include <pcl/common/pca.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/surface/mls.h>
#include <pcl/features/normal_3d.h>

#include <cmath>
#include <memory>
#include <opencv2/opencv.hpp>
#include <random>
#include <string>
#include <filesystem>
#include <yaml-cpp/yaml.h>
#include <future>

#include "cv_bridge/cv_bridge.h"
#include "control_action/action/move_robot.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_action/rclcpp_action.hpp"
#include "res_msgs/msg/pose_res.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "stereo_msgs/msg/disparity_image.hpp"
#include <image_geometry/stereo_camera_model.h>
#include "test_msgs/msg/ros_yolo.hpp"
#include "test_msgs/msg/yolo_pose.hpp"
#include "yolov8_msgs/msg/yolov8_inference.hpp"
#include "collect_msgs/msg/collect_msg.hpp"
#include "connect_msgs/msg/connect_msg.hpp"
#include "tomato_octomap_msgs/msg/tomato_octomaps.hpp"
#include "tomato_octomap_msgs/msg/tomato_octomap.hpp"
#include "config_manager/msg/system_config.hpp"
#include "position_signal_msgs/msg/position_signal.hpp"
#include "depth_signal_msgs/msg/depth_signal.hpp"
#include "skip_signal_msgs/msg/skip_signal.hpp"
#include "move_signal_msgs/msg/move_signal.hpp"
    
#include <octomap_msgs/msg/octomap.hpp>
#include <octomap_msgs/conversions.h>

using std::placeholders::_1;
using std::placeholders::_2;

struct PoseInfo {
    float distance;
    float X_final, Y_final, Z_final;
    float roll, pitch, yaw;
    int idx;
};

struct Candidate {
    Eigen::Vector3f pt;
    float z;
    bool valid = false;
};

struct TrackedBbox {
    cv::Rect            bbox;
    cv::Point2f         center;
    float               z_depth     = -1.0f;
    float               roll        = 0.0f;
    float               pitch       = 0.0f;
    float               yaw         = 0.0f;
    //float               median_depth = 0.0f;
    float               z_min       = 0.0f;
    float               center_fix  = 0.0f;
};

struct NewDet {
    int x1, y1, x2, y2;
    int center_x, center_y;
};

class Tomato3DDetector : public rclcpp::Node {
public:
    using MoveRobot           = control_action::action::MoveRobot;
    using GoalHandleMoveRobot = rclcpp_action::ClientGoalHandle<MoveRobot>;

    Tomato3DDetector() : Node("tomato_3d_detector") {
        rclcpp::QoS qos_profile(rclcpp::KeepLast(1));
        qos_profile.transient_local();
        qos_profile.reliable();

        client_ = rclcpp_action::create_client<MoveRobot>(this, "move_robot");
        while (rclcpp::ok() && !client_->wait_for_action_server(std::chrono::seconds(1)))
            RCLCPP_INFO(get_logger(), "Waiting for MoveIt Action Server...");
        RCLCPP_INFO(get_logger(), "MoveIt Action Server is ready!");

        sub_yolo_ = create_subscription<yolov8_msgs::msg::Yolov8Inference>(
            "/detect_msg", 10,
            std::bind(&Tomato3DDetector::yolo_callback, this, _1));

        sub_disparity_ = create_subscription<stereo_msgs::msg::DisparityImage>(
            "/stereo/disparity", 10,
            std::bind(&Tomato3DDetector::disparity_callback, this, _1));

        sub_left_cam_ = create_subscription<sensor_msgs::msg::CameraInfo>(
            "/stereo/left/camera_info_calib",
            rclcpp::QoS(1).transient_local().reliable(),
            std::bind(&Tomato3DDetector::left_camera_info_callback, this, _1));

        sub_right_cam_ = create_subscription<sensor_msgs::msg::CameraInfo>(
            "/stereo/right/camera_info_calib",
            rclcpp::QoS(1).transient_local().reliable(),
            std::bind(&Tomato3DDetector::right_camera_info_callback, this, _1));

        sub_pose_res_ = create_subscription<res_msgs::msg::PoseRes>(
            "/pose_res", 10,
            std::bind(&Tomato3DDetector::pose_res_callback, this, _1));

        time_sub_ = create_subscription<collect_msgs::msg::CollectMsg>(
            "/collect_msg", 10,
            std::bind(&Tomato3DDetector::collectmsg_callback, this, _1));

        left_img_sub_ = create_subscription<sensor_msgs::msg::Image>(
            "/ref_img", 10,
            std::bind(&Tomato3DDetector::left_camera_callback, this, _1));

//        obs_img_sub_ = create_subscription<sensor_msgs::msg::Image>(
//            "/obs_img", 10,
//            std::bind(&Tomato3DDetector::obs_camera_callback, this, _1));

        config_sub_ = create_subscription<config_manager::msg::SystemConfig>(
            "/system_config",
            rclcpp::QoS(1).transient_local().reliable(),
            std::bind(&Tomato3DDetector::config_callback, this, _1));

        publisher_          = create_publisher<test_msgs::msg::RosYolo>("/ros_yolo", 10);
        time_pub            = create_publisher<collect_msgs::msg::CollectMsg>("/collect2_msg", 10);
        connect_msg_pub     = create_publisher<connect_msgs::msg::ConnectMsg>("/connect_msg", qos_profile);
        res_msg_pub         = create_publisher<res_msgs::msg::PoseRes>("/pose_res", qos_profile);
        octomap_pub_        = create_publisher<octomap_msgs::msg::Octomap>("/obstacle_octomap", 10);
        tomato_octomap_pub_ = create_publisher<tomato_octomap_msgs::msg::TomatoOctomaps>("/tomato_octomaps", 10);
        position_signal_pub = create_publisher<position_signal_msgs::msg::PositionSignal>("/position_signal", 10);
        depth_signal_pub    = create_publisher<depth_signal_msgs::msg::DepthSignal>("/depth_signal", 10);
        skip_signal_pub     = create_publisher<skip_signal_msgs::msg::SkipSignal>("/skip_signal", 10);
        move_signal_pub     = create_publisher<move_signal_msgs::msg::MoveSignal>("/move_signal", 10);

        config_path = std::filesystem::current_path().string() + "/config/setup.yaml";

        // Pre-reserve member vectors để tránh realloc trong hot path
        valid_points_.reserve(2000);
        valid_points_obs_.reserve(2000);
        valid_points_obs_all_.reserve(5000);
        valid_obs_point_only_.reserve(2000);

        RCLCPP_INFO(get_logger(), "Tomato 3D Detector Node Started!");
    }

private:
    // ── Camera parameters ─────────────────────────────────────────────────────
    float fx_ = 0.f, fy_ = 0.f, cx_ = 0.f, cy_ = 0.f;
    float baseline_ = 0.f, fx_offset = 0.f, object_offset = 0.f, baseline_offset = 0.f;

    // ── Flags & scalars ───────────────────────────────────────────────────────
    std::atomic<bool> processing_request_{false};
    std::atomic<bool> data_ready_{false};
    double last_move_              = 0.0;
    double disparity_time_         = 0.0;
    double detection_time_         = 0.0;
    double start_detection_time_   = 0.0;
    double start_positioning_time_ = 0.0;
    bool flag_             = false;
    bool skip_first_frame_ = false;
    //bool orientation_fix   = false;
    bool config_received_  = false;
    bool skip_disparity_   = false;
    bool first_run_        = true;
    bool multi_collect_mode = false;
    bool resent_           = false;
    bool time_recieved_    = false;
    bool skip_target_      = false;

    size_t tomato_index_      = 0;
    size_t last_object_count_ = 0;
    float obstacle_radius     = 0.03f;
    float min_d = 0.f, max_d = 0.f;
    int x1_off = 0, y1_off = 0, w = 0, h = 0;
    int limit = 0, detection_idx = 0;

    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::TimerBase::SharedPtr retry_timer_;

    std::vector<tomato_octomap_msgs::msg::TomatoOctomap> tomato_octomaps;
    std::map<int, octomap_msgs::msg::Octomap> octomap_map_temp;
    std::vector<PoseInfo> poses;
    std::vector<NewDet>   new_dets;

    // ── Member vectors — reuse across calls, không alloc mỗi frame ────────────
    std::vector<Eigen::Vector3f> valid_points_;
    std::vector<Eigen::Vector3f> valid_points_obs_;
    std::vector<Eigen::Vector3f> valid_points_obs_all_;
    std::vector<Eigen::Vector3f> valid_obs_point_only_;

    yolov8_msgs::msg::Yolov8Inference::SharedPtr last_yolo_msg_ = nullptr;

    std::vector<TrackedBbox> tracked_store_;
    static constexpr float HUNGARIAN_DIST_THRESHOLD = 80.0f;

    // ── Shared disparity data ─────────────────────────────────────────────────
    std::mutex data_mtx_;
    cv::Mat_<cv::Vec3f> points_mat_;       // disparity thread writes
    cv::Mat_<cv::Vec3f> points_snapshot_;  // processing thread reads

    cv::Mat left_img_, obs_img_;
    std::mutex image_mtx_;
    std::string config_path;
    std::vector<PoseInfo> tomato_list_;
    std::mutex mtx_;
    std::condition_variable cv_;
    image_geometry::StereoCameraModel model_;
    std::optional<std::vector<PoseInfo>> nearest_tomato;

    // ── Subscribers / Publishers ──────────────────────────────────────────────
    rclcpp::Subscription<yolov8_msgs::msg::Yolov8Inference>::SharedPtr  sub_yolo_;
    rclcpp::Subscription<stereo_msgs::msg::DisparityImage>::SharedPtr   sub_disparity_;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr       sub_left_cam_, sub_right_cam_;
    rclcpp::Subscription<res_msgs::msg::PoseRes>::SharedPtr             sub_pose_res_;
    rclcpp::Subscription<collect_msgs::msg::CollectMsg>::SharedPtr      time_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr            left_img_sub_;
    rclcpp::Subscription<config_manager::msg::SystemConfig>::SharedPtr  config_sub_;
    sensor_msgs::msg::CameraInfo::SharedPtr left_camera_info_, right_camera_info_;

    rclcpp::Publisher<test_msgs::msg::RosYolo>::SharedPtr                   publisher_;
    rclcpp::Publisher<collect_msgs::msg::CollectMsg>::SharedPtr             time_pub;
    rclcpp::Publisher<connect_msgs::msg::ConnectMsg>::SharedPtr             connect_msg_pub;
    rclcpp::Publisher<res_msgs::msg::PoseRes>::SharedPtr                    res_msg_pub;
    rclcpp::Publisher<octomap_msgs::msg::Octomap>::SharedPtr                octomap_pub_;
    rclcpp::Publisher<tomato_octomap_msgs::msg::TomatoOctomaps>::SharedPtr  tomato_octomap_pub_;
    rclcpp::Publisher<position_signal_msgs::msg::PositionSignal>::SharedPtr position_signal_pub;
    rclcpp::Publisher<skip_signal_msgs::msg::SkipSignal>::SharedPtr         skip_signal_pub;
    rclcpp::Publisher<depth_signal_msgs::msg::DepthSignal>::SharedPtr depth_signal_pub;
    rclcpp::Publisher<move_signal_msgs::msg::MoveSignal>::SharedPtr         move_signal_pub;

    rclcpp_action::Client<MoveRobot>::SharedPtr client_;

    // =========================================================================
    void pose_res_callback(const res_msgs::msg::PoseRes::SharedPtr msg) {
        if (msg->pose_res.empty()) return;
        last_move_      = msg->pose_res[0].x;
        flag_           = msg->pose_res[0].flag;
        skip_disparity_ = msg->pose_res[0].skip;
        if (!skip_disparity_) first_run_ = true;
        if (!flag_)            processing_request_ = false;
    }

    void config_callback(const config_manager::msg::SystemConfig::SharedPtr msg) {
        config_received_   = false;
        RCLCPP_INFO(get_logger(), "Load setup");
        object_offset      = msg->object_offset;
        fx_offset          = msg->fx_offset;
        multi_collect_mode = msg->multi_collect_mode;
        config_received_   = true;
    }

    void left_camera_info_callback(const sensor_msgs::msg::CameraInfo::SharedPtr msg) {
        left_camera_info_ = std::make_shared<sensor_msgs::msg::CameraInfo>(*msg);
        update_camera_model();
        fx_ = static_cast<float>(msg->k[0]);
        fy_ = static_cast<float>(msg->k[4]);
        cx_ = static_cast<float>(msg->k[2]);
        cy_ = static_cast<float>(msg->k[5]);
    }

    void right_camera_info_callback(const sensor_msgs::msg::CameraInfo::SharedPtr msg) {
        right_camera_info_ = std::make_shared<sensor_msgs::msg::CameraInfo>(*msg);
        update_camera_model();
        baseline_ = std::abs(static_cast<float>(msg->p[3] / msg->p[0]));
    }

    void update_camera_model() {
        if (left_camera_info_ && right_camera_info_)
            model_.fromCameraInfo(*left_camera_info_, *right_camera_info_);
    }

    void left_camera_callback(const sensor_msgs::msg::Image::ConstSharedPtr& msg) {
        try {
            auto img = cv_bridge::toCvCopy(msg, "mono8")->image;
            std::lock_guard<std::mutex> lock(image_mtx_);
            left_img_ = img;
        } catch (const cv_bridge::Exception& e) {
            RCLCPP_ERROR(get_logger(), "cv_bridge: %s", e.what());
        }
    }

//    void obs_camera_callback(const sensor_msgs::msg::Image::ConstSharedPtr& msg) {
//        try {
//            obs_img_ = cv_bridge::toCvCopy(msg, "mono8")->image;
//        } catch (const cv_bridge::Exception& e) {
//            RCLCPP_ERROR(get_logger(), "cv_bridge: %s", e.what());
//        }
//    }

    // =========================================================================
    void disparity_callback(const stereo_msgs::msg::DisparityImage::SharedPtr msg) {
        if (skip_first_frame_) { skip_first_frame_ = false; return; }

        // Heavy projection work OUTSIDE lock
        const sensor_msgs::msg::Image& dimage = msg->image;
        const cv::Mat_<float> dmat(
            dimage.height, dimage.width,
            const_cast<float*>(reinterpret_cast<const float*>(dimage.data.data())),
            dimage.step);

        cv::Mat_<cv::Vec3f> tmp;
        model_.projectDisparityImageTo3d(dmat, tmp, true);

        {
            std::lock_guard<std::mutex> lock(data_mtx_);
            points_mat_     = std::move(tmp);   // O(1) move, không copy pixels
            disparity_time_ = rclcpp::Time(msg->header.stamp).seconds();
            w      = msg->valid_window.width;
            h      = msg->valid_window.height;
            x1_off = msg->valid_window.x_offset;
            y1_off = msg->valid_window.y_offset;
            min_d  = msg->min_disparity;
            max_d  = msg->max_disparity;
        }
    }

    // =========================================================================
    void yolo_callback(const yolov8_msgs::msg::Yolov8Inference::SharedPtr msg) {
        RCLCPP_INFO(get_logger(), "Received YOLO: %zu detections", msg->yolov8_inference.size());
        if (!processing_request_) last_yolo_msg_ = msg;
        try_process_yolo();
    }

    void try_process_yolo() {
        if (!last_yolo_msg_) {
            RCLCPP_WARN(get_logger(), "No YOLO message to process.");
            return;
        }

        if (skip_disparity_ && !first_run_)
            disparity_time_ = last_move_;

        if (disparity_time_ < last_move_ || flag_ || !config_received_) {
            processing_request_ = false;
            RCLCPP_WARN(get_logger(),
                "Disparity old, retry... flag:%d disp:%.3f move:%.3f",
                flag_, disparity_time_, last_move_);
            schedule_retry();
            return;
        }

        if (processing_request_) return;

        // Check points_mat_ emptiness with minimal lock time
//        bool pts_empty;
//        {
//            std::lock_guard<std::mutex> lk(data_mtx_);
//            pts_empty = points_mat_.empty();
//        }

        bool points_empty = true;
        {
            std::lock_guard<std::mutex> lk(data_mtx_);
            points_empty = points_mat_.empty();
        }

        bool left_img_empty = true;
        {
            std::lock_guard<std::mutex> lk(image_mtx_);
            left_img_empty = left_img_.empty();
        }

        if (points_empty || left_img_empty || fx_ == 0.f || baseline_ == 0.f || !time_recieved_) {
            RCLCPP_WARN(get_logger(), "Waiting for calibration/disparity...");
            schedule_retry();
            return;
        }

        auto msg       = last_yolo_msg_;
        last_yolo_msg_ = nullptr;
        publish_position_signal(true);

        if (skip_disparity_) {
            nearest_tomato = first_run_
                ? compute3DCoordinates(*msg)
                : computeFromTracked(*msg);
        } else {
            nearest_tomato = compute3DCoordinates(*msg);
        }

        if (multi_collect_mode) first_run_ = false;

        RCLCPP_WARN(get_logger(), "Check11111");

        if (nearest_tomato.has_value()) {
            tomato_list_ = *nearest_tomato;
            process_next_tomato();
            //publish_position_signal(false);
        } else {
            RCLCPP_INFO(get_logger(), "No tomato detected.");
            publish_position_signal(false);
            publish_depth_signal(false);
            handle_no_tomato_detected();
            return;
        }
    }

    // Cancel & recreate retry timer only when needed
    void schedule_retry() {
        if (retry_timer_ && !retry_timer_->is_canceled())
            retry_timer_->cancel();
        retry_timer_ = create_wall_timer(
            std::chrono::milliseconds(200),
            [this]() { retry_timer_->cancel(); try_process_yolo(); });
    }

    // =========================================================================
    void process_next_tomato() {
        if (tomato_list_.empty()) {
            RCLCPP_WARN(get_logger(), "Tomato list is empty.");
            return;
        }
        RCLCPP_WARN(get_logger(), "Poses computed: %zu", tomato_list_.size());

        test_msgs::msg::RosYolo ros_yolo_msg;
        ros_yolo_msg.ros_yolo.reserve(tomato_list_.size());
        for (const auto& t : tomato_list_) {
            test_msgs::msg::YoloPose yp;
            yp.x     = t.X_final;  yp.y     = t.Y_final;  yp.z    = t.Z_final;
            yp.roll  = t.roll;     yp.pitch  = t.pitch;    yp.yall = t.yaw;
            ros_yolo_msg.ros_yolo.push_back(yp);
        }
        publisher_->publish(ros_yolo_msg);
        RCLCPP_INFO(get_logger(), "Published %zu poses", tomato_list_.size());

        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        timer_ = create_wall_timer(std::chrono::milliseconds(0), [this]() {
            time_publisher(now().seconds());
            timer_->cancel();
        });
        send_move_request();
    }

    // =========================================================================
    std::vector<Eigen::Vector3f> filterOutliersSOR(
        const std::vector<Eigen::Vector3f>& points,
        int k = 15, float std_mul = 1.5f)
    {
        // ❗ 1. Nếu quá ít điểm → bỏ qua SOR (tránh filter sai)
        if (points.size() < static_cast<size_t>(k * 2)) {
            RCLCPP_WARN(get_logger(), "SOR skipped: too few points (%zu)", points.size());
            return points;
        }

        // ❗ 2. Filter theo Z trước (rất quan trọng)
        std::vector<Eigen::Vector3f> valid_points;
        valid_points.reserve(points.size());

        for (const auto& p : points)
        {
            // chỉnh lại range theo setup của bạn
            if (std::isfinite(p.z()) && p.z() > 0.1f && p.z() < 2.0f)
                valid_points.push_back(p);
        }

        if (valid_points.size() < static_cast<size_t>(k * 2)) {
            RCLCPP_WARN(get_logger(), "SOR skipped after Z filter (%zu)", valid_points.size());
            return valid_points;
        }

        // ❗ 3. Convert sang PCL
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
        cloud->reserve(valid_points.size());

        for (const auto& p : valid_points)
            cloud->emplace_back(p.x(), p.y(), p.z());

        // ❗ 4. SOR filter (đã tune lại param)
        pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
        sor.setInputCloud(cloud);

        sor.setMeanK(k);                 // ⚠️ giảm từ 50 → 15
        sor.setStddevMulThresh(std_mul); // ⚠️ tăng từ 1.0 → 1.5

        pcl::PointCloud<pcl::PointXYZ>::Ptr filtered(new pcl::PointCloud<pcl::PointXYZ>);
        sor.filter(*filtered);

        // ❗ 5. Fallback nếu SOR “lọc ngu”
        if (filtered->size() < valid_points.size() * 0.3)
        {
            RCLCPP_WARN(get_logger(),
                "SOR removed too many points (%zu → %zu), fallback to valid_points",
                valid_points.size(), filtered->size());

            return valid_points;
        }

        // ❗ 6. Convert lại Eigen
        std::vector<Eigen::Vector3f> result;
        result.reserve(filtered->size());

        for (const auto& pt : filtered->points)
            result.emplace_back(pt.x, pt.y, pt.z);

        RCLCPP_DEBUG(get_logger(),
            "SOR: %zu → %zu (after Z: %zu)",
            points.size(), result.size(), valid_points.size());

        return result;
    }

    // =========================================================================
    std::tuple<float, float, float> computeSurfaceOrientation(
        const std::vector<Eigen::Vector3f>& points, float threshold, bool yaw_flag_)
    {
        if (points.size() < 10) return {0.f, 0.f, 0.f};

        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
        cloud->reserve(points.size());
        for (const auto& pt : points)
            cloud->emplace_back(pt(0), pt(1), pt(2));

        pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
        pcl::PointIndices::Ptr      inliers(new pcl::PointIndices);
        pcl::SACSegmentation<pcl::PointXYZ> seg;
        seg.setOptimizeCoefficients(true);
        seg.setModelType(pcl::SACMODEL_PLANE);
        seg.setMethodType(pcl::SAC_RANSAC);
        seg.setDistanceThreshold(threshold);
        seg.setMaxIterations(1000);
        seg.setInputCloud(cloud);
        seg.segment(*inliers, *coefficients);

        if (inliers->indices.size() < 10) {
            RCLCPP_WARN(rclcpp::get_logger("tomato"), "RANSAC: not enough inliers");
            return {0.f, 0.f, 0.f};
        }

        Eigen::Vector3f normal(
            coefficients->values[0],
            coefficients->values[1],
            coefficients->values[2]);
        normal.normalize();
        if (normal(2) < 0) normal = -normal;

        RCLCPP_INFO(get_logger(), "Normal: %f %f %f", normal(0), normal(1), normal(2));

        float nxy = std::sqrt(normal(0)*normal(0) + normal(1)*normal(1));
        float nz  = std::sqrt(normal(2)*normal(2));
        if (nxy > nz)
            normal(2) = nxy * (normal(2) / std::abs(normal(2)));

        float roll  = -std::atan2(normal(1), normal(2));
        float pitch =  std::atan2(normal(0), normal(2));
        float yaw   = 0.f;

        if (yaw_flag_) {
            // ── PCA 2D trên X-Y của point cloud (giống computeYawFromContour2D) ──

            // Tính centroid
            double mean_x = 0.0, mean_y = 0.0, mean_z = 0.0;
            for (const auto& pt : points) {
                mean_x += pt(0);
                mean_y += pt(1);
                mean_z += pt(2);
            }
            mean_x /= points.size();
            mean_y /= points.size();
            mean_z /= points.size();

            // Tính covariance matrix 2D
            double cxx = 0.0, cxy = 0.0, cyy = 0.0, czz = 0.0;
            for (const auto& pt : points) {
                const double dx = pt(0) - mean_x;
                const double dy = pt(1) - mean_y;
                const double dz = pt(2) - mean_z;
                cxx += dx * dx;
                cxy += dx * dy;
                cyy += dy * dy;
                czz += dz * dz;  // ✅ thêm Z variance
            }
            cxx /= points.size();
            cxy /= points.size();
            cyy /= points.size();
            czz /= points.size();

            // ✅ Check phân bố dọc theo Z
            // Nếu var_z >> var_xy → điểm phân bố dọc theo Z → không phải bề mặt
            const double xy_var    = (cxx + cyy) * 0.5;
            const double z_xy_ratio = (xy_var > 1e-6) ? (czz / xy_var) : 999.0;

            if (z_xy_ratio > 5.0) {
                RCLCPP_WARN(get_logger(),
                    "Points distributed along Z (ratio=%.2f) → invalid object",
                    z_xy_ratio);
                skip_target_ = true;  // ✅ flag về false
                return {0.f, 0.f, 0.f};
            }

            // Eigenvalue để check elongation
            const double trace        = cxx + cyy;
            const double diff         = cxx - cyy;
            const double disc         = std::sqrt(diff * diff + 4.0 * cxy * cxy);
            const double lambda_major = 0.5 * (trace + disc);
            const double lambda_minor = 0.5 * (trace - disc);

            const double elongation = (lambda_major > 1e-6)
                                    ? 1.0 - (lambda_minor / lambda_major)
                                    : 0.0;

            if (elongation >= 0.20) {
                // Eigenvector của lambda_major → trục dài nhất
                yaw = static_cast<float>(0.5 * std::atan2(2.0 * cxy, diff));
                if (yaw >  M_PI_2) yaw -= static_cast<float>(M_PI);
                if (yaw < -M_PI_2) yaw += static_cast<float>(M_PI);
            } else {
                // Point cloud gần tròn → không xác định được yaw
                RCLCPP_WARN(get_logger(), "Yaw: elongation=%.2f too low, skip", elongation);
                yaw = 0.f;
            }
        }
        //orientation_fix = false;
        return {roll, pitch, yaw};
    }

    std::tuple<float, float> computeRollPitchFromXYZ(float x, float y, float z)
    {
        // Tạo vector
        Eigen::Vector3f normal(x, y, z);

        // Tránh chia 0
        if (normal.norm() < 1e-6f) {
            return {0.0f, 0.0f};
        }

        // Normalize
        normal.normalize();

        // Tính roll, pitch theo công thức của bạn
        float roll  = -std::atan2(normal(1), normal(2));
        float pitch =  std::atan2(normal(0), normal(2));

        return {roll, pitch};
    }

    std::optional<float> computeYawFromContour2D(const std::vector<cv::Point>& contour)
    {
        if (contour.size() < 5) return std::nullopt;

        double mean_x = 0.0, mean_y = 0.0;
        for (const auto& p : contour) {
            mean_x += p.x;
            mean_y += p.y;
        }
        mean_x /= static_cast<double>(contour.size());
        mean_y /= static_cast<double>(contour.size());

        double cxx = 0.0, cxy = 0.0, cyy = 0.0;
        for (const auto& p : contour) {
            const double dx = p.x - mean_x;
            const double dy = p.y - mean_y;
            cxx += dx * dx;
            cxy += dx * dy;
            cyy += dy * dy;
        }
        cxx /= static_cast<double>(contour.size());
        cxy /= static_cast<double>(contour.size());
        cyy /= static_cast<double>(contour.size());

        const double trace = cxx + cyy;
        const double diff  = cxx - cyy;
        const double disc  = std::sqrt(diff * diff + 4.0 * cxy * cxy);
        const double lambda_major = 0.5 * (trace + disc);
        const double lambda_minor = 0.5 * (trace - disc);
        if (lambda_major <= 1e-6) return std::nullopt;

        const double elongation = 1.0 - (lambda_minor / lambda_major);
        if (elongation < 0.20) return std::nullopt;

        float yaw = static_cast<float>(- 0.5 * std::atan2(2.0 * cxy, diff));
        if (yaw > M_PI_2) yaw -= static_cast<float>(M_PI);
        if (yaw < -M_PI_2) yaw += static_cast<float>(M_PI);
        return yaw;
    }

    // =========================================================================
    // buildOctomap: insert points directly, skip intermediate pcl cloud
    octomap_msgs::msg::Octomap buildOctomap(const std::vector<Eigen::Vector3f>& points) {
        octomap::OcTree tree(0.01);
        tree.setProbHit(0.7);
        tree.setProbMiss(0.4);
        for (const auto& p : points)
            tree.updateNode(octomap::point3d(p.x(), p.y(), p.z()), true);
        tree.updateInnerOccupancy();

        octomap_msgs::msg::Octomap msg;
        octomap_msgs::binaryMapToMsg(tree, msg);
        msg.header.frame_id = "left_camera_link";
        msg.header.stamp    = now();
        return msg;
    }

    // =========================================================================
    std::vector<cv::Point> findContourMono8(
        const cv::Mat& mono8_img, int x1, int y1, int x2, int y2)
    {
        if (x1 >= x2 || y1 >= y2 || x1 < 0 || y1 < 0 ||
            x2 > mono8_img.cols || y2 > mono8_img.rows) {
            RCLCPP_INFO(get_logger(),
                "Invalid parameters! Image: %dx%d, x1:%d y1:%d x2:%d y2:%d",
                mono8_img.cols, mono8_img.rows, x1, y1, x2, y2);
            return {};
        }

        cv::Mat roi = mono8_img(cv::Range(y1, y2), cv::Range(x1, x2)).clone();
        if (roi.channels() > 1) cv::cvtColor(roi, roi, cv::COLOR_BGR2GRAY);

        // ── Làm mịn trước để giảm noise ─────────────────────────────────────────
        cv::Mat blurred;
        cv::GaussianBlur(roi, blurred, cv::Size(5, 5), 1.5);

        // ── Canny edge detection ─────────────────────────────────────────────────
        // Otsu để tự động chọn threshold phù hợp với từng ảnh
        cv::Mat otsu_bin;
        double otsu_thresh = cv::threshold(blurred, otsu_bin, 0, 255,
                                        cv::THRESH_BINARY | cv::THRESH_OTSU);
        cv::Mat edges;
        cv::Canny(blurred, edges,
                otsu_thresh * 0.3,   // low  = 30% of Otsu
                otsu_thresh * 0.9);  // high = 90% of Otsu

        // ── Đóng kín các đường biên bị hở ───────────────────────────────────────
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7));
        cv::Mat closed;
        cv::morphologyEx(edges, closed, cv::MORPH_CLOSE, kernel, cv::Point(-1,-1), 2);

        // ── Fill để lấy vùng đặc, rồi lấy contour ngoài ─────────────────────────
        // floodFill từ 4 góc để fill background → lấy phần còn lại là object
        cv::Mat filled = closed.clone();
        cv::floodFill(filled, cv::Point(0, 0), 255);
        cv::floodFill(filled, cv::Point(filled.cols-1, 0), 255);
        cv::floodFill(filled, cv::Point(0, filled.rows-1), 255);
        cv::floodFill(filled, cv::Point(filled.cols-1, filled.rows-1), 255);

        // Invert: object = trắng, background = đen
        cv::Mat object_mask;
        cv::bitwise_not(filled, object_mask);

        // Nếu fill từ góc không hiệu quả, fallback về Otsu binary
        int white_px = cv::countNonZero(object_mask);
        int roi_area  = (x2-x1) * (y2-y1);
        if (white_px < 0.05 * roi_area || white_px > 0.95 * roi_area) {
            // Fallback: dùng Otsu binary trực tiếp
            object_mask = otsu_bin.clone();
            // Morphological closing để lấp lỗ hổng
            cv::morphologyEx(object_mask, object_mask, cv::MORPH_CLOSE, kernel,
                            cv::Point(-1,-1), 3);
        }

        // ── Tìm contour lớn nhất ─────────────────────────────────────────────────
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(object_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        if (contours.empty()) return {};

        double min_area = 0.001 * roi_area;
        int    best_idx = -1;
        double max_area = 0.0;
        for (size_t i = 0; i < contours.size(); ++i) {
            double area = cv::contourArea(contours[i]);
            if (area >= min_area && area > max_area) {
                max_area = area;
                best_idx = (int)i;
            }
        }
        if (best_idx < 0) return {};

        // ── Smooth contour (convex hull nếu object hình cầu) ────────────────────
        // Quả cà chua gần hình tròn → convex hull cho biên sạch hơn
        std::vector<cv::Point> hull;
        cv::convexHull(contours[best_idx], hull);

        // Translate về tọa độ ảnh gốc
        std::vector<cv::Point> result;
        result.reserve(hull.size());
        for (const auto& pt : hull)
            result.emplace_back(pt.x + x1, pt.y + y1);
        return result;
    }

    // =========================================================================
    std::vector<int> hungarianMatch(const std::vector<std::vector<float>>& cost_matrix) {
        int n_new = (int)cost_matrix.size();
        if (n_new == 0) return {};
        int n_old = (int)cost_matrix[0].size();
        if (n_old == 0) return std::vector<int>(n_new, -1);

        int n = std::max(n_new, n_old);
        std::vector<std::vector<float>> C(n, std::vector<float>(n, 1e9f));
        for (int i = 0; i < n_new; ++i)
            for (int j = 0; j < n_old; ++j)
                C[i][j] = cost_matrix[i][j];

        std::vector<int>   u(n+1,0), v(n+1,0), p(n+1,0), way(n+1,0);
        for (int i = 1; i <= n; ++i) {
            p[0] = i; int j0 = 0;
            std::vector<float> minV(n+1, 1e18f);
            std::vector<bool>  used(n+1, false);
            do {
                used[j0] = true;
                int i0 = p[j0], j1 = -1;
                float delta = 1e18f;
                for (int j = 1; j <= n; ++j) {
                    if (!used[j]) {
                        float cur = C[i0-1][j-1] - u[i0] - v[j];
                        if (cur < minV[j]) { minV[j] = cur; way[j] = j0; }
                        if (minV[j] < delta) { delta = minV[j]; j1 = j; }
                    }
                }
                for (int j = 0; j <= n; ++j)
                    used[j] ? (u[p[j]] += delta, v[j] -= delta) : (minV[j] -= delta);
                j0 = j1;
            } while (p[j0] != 0);
            do { p[j0] = p[way[j0]]; j0 = way[j0]; } while (j0);
        }

        std::vector<int> assignment(n_new, -1);
        for (int j = 1; j <= n_old; ++j) {
            int row = p[j] - 1;
            if (row >= 0 && row < n_new && cost_matrix[row][j-1] < HUNGARIAN_DIST_THRESHOLD)
                assignment[row] = j - 1;
        }
        return assignment;
    }

    // =========================================================================
    std::optional<std::vector<PoseInfo>> computeFromTracked(
        const yolov8_msgs::msg::Yolov8Inference& msg)
    {
        if (tracked_store_.empty()) return std::nullopt;

        // Shallow copy under lock — O(1), no pixel copy
        {
            std::lock_guard<std::mutex> lock(data_mtx_);
            points_snapshot_ = points_mat_;
        }
        if (points_snapshot_.empty()) {
            RCLCPP_WARN(get_logger(), "[Track] Empty points_snapshot");
            return std::nullopt;
        }

        new_dets.clear();
        for (const auto& det : msg.yolov8_inference) {
            int x1 = std::clamp((int)det.top,         0, w-1);
            int y1 = std::clamp((int)det.left,        0, h-1);
            int x2 = std::clamp(x1+(int)det.bottom,   0, w-1);
            int y2 = std::clamp(y1+(int)det.right,    0, h-1);
            if (x1 >= x2 || y1 >= y2) continue;
            new_dets.push_back({x1, y1, x2, y2, (x1+x2)/2, (y1+y2)/2});
        }
        if (new_dets.empty()) return std::nullopt;

        int n_new = (int)new_dets.size();
        int n_old = (int)tracked_store_.size();

        std::vector<std::vector<float>> cost(n_new, std::vector<float>(n_old));
        for (int i = 0; i < n_new; ++i)
            for (int j = 0; j < n_old; ++j) {
                float dx = new_dets[i].center_x - tracked_store_[j].center.x;
                float dy = new_dets[i].center_y - tracked_store_[j].center.y;
                cost[i][j] = std::sqrt(dx*dx + dy*dy);
            }

        auto assignment = hungarianMatch(cost);

        octomap_map_temp.clear();
        poses.clear();
        int det_idx    = 0;
        //const bool has_obs = !obs_img_.empty();

        for (int i = 0; i < n_new; ++i) {
            if (assignment[i] < 0) {
                RCLCPP_WARN(get_logger(), "[Track] det %d → no match, skip", i);
                ++det_idx; continue;
            }
            const TrackedBbox& matched = tracked_store_[assignment[i]];
            if (matched.z_depth <= 0.f) { ++det_idx; continue; }

            RCLCPP_INFO(get_logger(), "[Track] det %d → store[%d]  Z=%.3f  dist=%.1fpx",
                i, assignment[i], matched.z_depth, cost[i][assignment[i]]);

            const int x1 = new_dets[i].x1, y1 = new_dets[i].y1;
            const int x2 = new_dets[i].x2, y2 = new_dets[i].y2;
            //const float median     = matched.median_depth;
            const float z_min      = matched.z_min;
            const float center_fix = matched.center_fix;

            valid_points_obs_all_.clear();
            valid_obs_point_only_.clear();

            // Precompute expanded bbox bounds once per detection
            const int exp_y1 = y1 - 1.5*(y1 - y2), exp_y2 = y2 + 1.5*(y2 - y1);
            const int exp_x1 = x1 - 1.5*(x1 - x2), exp_x2 = x2 + 1.5*(x2 - x1);

            // Use row pointers for faster pixel access
            for (int pi = 0; pi < h; pi += 5) {
                const cv::Vec3f* row_ptr = points_snapshot_.ptr<cv::Vec3f>(pi);
                //const bool obs_row_valid = pi % 10 == 0;
                //const uchar* obs_row = (obs_row_valid) ? obs_img_.ptr<uchar>(pi) : nullptr;

                for (int pj = 0; pj < w; pj += 5) {
                    const cv::Vec3f& pt_obs = row_ptr[pj];
                    if (pt_obs[2] <= 0.f || pt_obs[2] >= 2.f) continue;

                    const bool in_expanded = (pi >= exp_y1 && pi <= exp_y2 &&
                                              pj >= exp_x1 && pj <= exp_x2);

                    if (!in_expanded && (pi % 10 == 0) && (pj % 10 == 0)) {
                        valid_points_obs_all_.emplace_back(pt_obs[2], -pt_obs[0], -pt_obs[1]);
//                        if (obs_row && obs_row[pj] > 0)
//                            valid_obs_point_only_.emplace_back(pt_obs[2], -pt_obs[0], -pt_obs[1]);
                    }
                }
            }

            //std::vector<Eigen::Vector3f> filtered_obs_all;
            //filtered_obs_all.reserve(valid_obs_point_only_.size() + valid_points_obs_all_.size());
            //for (const auto& p : valid_obs_point_only_)
            //    filtered_obs_all.push_back(p);
            //for (const auto& p : valid_points_obs_all_)
            //    if (p.x() > median) filtered_obs_all.push_back(p);

            if (!valid_points_obs_all_.empty())
                octomap_map_temp[det_idx] = buildOctomap(valid_points_obs_all_);

            const int   cx_det = new_dets[i].center_x;
            const int   cy_det = new_dets[i].center_y;
            float Z = z_min + std::abs(x2-x1) * z_min / (fx_ * 2) + center_fix / 2;
            float Y = -(cx_det - cx_) * Z / fx_;
            float X =  (cy_det - cy_) * Z / fy_;

            RCLCPP_INFO(get_logger(), "[Track] X=%.3f Y=%.3f Z=%.3f", X, Y, Z);

            float X_final, Y_final, Z_final;
            compute_offset_position(X, Y, Z, X_final, Y_final, Z_final);

            PoseInfo pi_info;
            pi_info.distance = std::sqrt(X*X + Y*Y + Z*Z);
            pi_info.X_final  = X_final;  pi_info.Y_final = Y_final;  pi_info.Z_final = Z_final;
            pi_info.roll     = matched.roll;
            pi_info.pitch    = matched.pitch;
            pi_info.yaw      = matched.yaw;
            pi_info.idx      = det_idx;
            poses.push_back(pi_info);
            ++det_idx;
        }

        if (poses.empty()) return std::nullopt;

        std::sort(poses.begin(), poses.end(),
            [](const PoseInfo& a, const PoseInfo& b){ return a.distance < b.distance; });

        tomato_octomap_msgs::msg::TomatoOctomaps msg_out;
        for (size_t k = 0; k < poses.size(); ++k) {
            auto it = octomap_map_temp.find(poses[k].idx);
            if (it != octomap_map_temp.end()) {
                tomato_octomap_msgs::msg::TomatoOctomap to;
                to.idx = (int)k;  to.octomap = it->second;
                msg_out.octomaps.push_back(to);
            }
        }
        tomato_octomap_pub_->publish(msg_out);

        time_recieved_ = false;
        if (processing_request_) return std::nullopt;
        return poses;
    }

    // =========================================================================
    std::optional<std::vector<PoseInfo>> compute3DCoordinates(
        const yolov8_msgs::msg::Yolov8Inference& msg)
    {
        // Shallow copy under lock — O(1)
        {
            std::lock_guard<std::mutex> lock(data_mtx_);
            points_snapshot_ = points_mat_;
        }
        if (points_snapshot_.empty() || fx_ == 0.f || fy_ == 0.f || baseline_ == 0.f) {
            RCLCPP_WARN(get_logger(), "Invalid calibration or empty disparity");
            time_recieved_ = false;
            return std::nullopt;
        }

        cv::Mat left_img_snapshot;
        {
            std::lock_guard<std::mutex> lock(image_mtx_);
            left_img_snapshot = left_img_;
        }
        if (left_img_snapshot.empty()) {
            RCLCPP_WARN(get_logger(), "Empty left image");
            time_recieved_ = false;
            return std::nullopt;
        }

        tracked_store_.clear();
        octomap_map_temp.clear();
        poses.clear();
        limit        = 0;
        detection_idx = 0;

        //const bool has_obs = !obs_img_.empty();

        for (const auto& detection : msg.yolov8_inference) {
            // Reuse member vectors — clear is O(n) size reset, no free
            valid_points_.clear();
            valid_points_obs_.clear();
            valid_points_obs_all_.clear();
            valid_obs_point_only_.clear();
            skip_target_ = false;

            if (detection.id != 1 && limit > 10) continue;

            int x1 = std::clamp((int)detection.top,          0, w-1);
            int y1 = std::clamp((int)detection.left,         0, h-1);
            int x2 = std::clamp(x1 + (int)detection.bottom,  0, w-1);
            int y2 = std::clamp(y1 + (int)detection.right,   0, h-1);
            RCLCPP_INFO(get_logger(), "ROI raw: x1=%d y1=%d x2=%d y2=%d w=%d h=%d",
                x1, y1, x2, y2, w, h);
            if (x1 >= x2 || y1 >= y2) continue;

            const int center_x = (x1+x2)/2, center_y = (y1+y2)/2;
            baseline_offset = baseline_ / 2;

            auto contour = findContourMono8(left_img_snapshot, x1, y1, x2, y2);
            if (contour.size() < 3) {
                RCLCPP_WARN(get_logger(), "Contour invalid (%zu pts) → fallback bbox", contour.size());
                contour = { {x1,y1},{x2,y1},{x2,y2},{x1,y2} };
            }

            //cv::Mat mask = cv::Mat::zeros(h, w, CV_8UC1);
            //std::vector<std::vector<cv::Point>> contours_wrap{contour};
            //cv::fillPoly(mask, contours_wrap, cv::Scalar(255));
            cv::Mat mask;
            cv::threshold(left_img_snapshot, mask, 0, 255, cv::THRESH_BINARY);
            ++limit;
/*
            {
                static std::atomic<int> dbg_idx{0};
                int idx = dbg_idx.fetch_add(1);
                std::string base = "hmi_dbg_" + std::to_string(idx);

                // 1. ROI crop từ ảnh gốc (grayscale → BGR để dễ xem)
                cv::Mat roi = left_img_(cv::Rect(x1, y1, x2 - x1, y2 - y1));
                cv::Mat roi_bgr;
                cv::cvtColor(roi, roi_bgr, cv::COLOR_GRAY2BGR);
                cv::imwrite(base + "_roi.png", roi_bgr);

                // 2. Mask contour (toàn ảnh, không crop)
                cv::Mat mask_dbg = cv::Mat::zeros(h, w, CV_8UC1);
                std::vector<std::vector<cv::Point>> wrap{contour};
                cv::fillPoly(mask_dbg, wrap, cv::Scalar(255));
                cv::imwrite(base + "_mask.png", mask_dbg);

                // 3. Overlay: ảnh gốc + contour vẽ lên
                cv::Mat overlay;
                cv::cvtColor(left_img_, overlay, cv::COLOR_GRAY2BGR);
                cv::polylines(overlay, wrap, true, cv::Scalar(0, 255, 0), 2);
                cv::rectangle(overlay, {x1, y1}, {x2, y2}, cv::Scalar(0, 0, 255), 1);
                cv::circle(overlay, {center_x, center_y}, 4, cv::Scalar(255, 0, 0), -1);
                cv::imwrite(base + "_overlay.png", overlay);

                RCLCPP_INFO(get_logger(), "[DBG] saved %s_{roi,mask,overlay}.png", base.c_str());
            }
*/
            int count = 0;

            // Precompute expanded bbox bounds once
            const int exp_y1 = 2*y1 - y2, exp_y2 = 2*y2 - y1;
            const int exp_x1 = 2*x1 - x2, exp_x2 = 2*x2 - x1;

            for (int i = 0; i < h; i += 5) {
                // Row pointers — avoid repeated at<> overhead
                const cv::Vec3f* row_pts  = points_snapshot_.ptr<cv::Vec3f>(i);
                const uchar*     row_mask = mask.ptr<uchar>(i);
                //const bool obs_row_valid  = i % 10 == 0;
                //const uchar* row_obs      = obs_row_valid ? obs_img_.ptr<uchar>(i) : nullptr;

                for (int j = 0; j < w; j += 5) {
                    const cv::Vec3f& pt_obs = row_pts[j];
                    if (pt_obs[2] <= 0.f || pt_obs[2] >= 2.f) continue;

                    const bool in_expanded = (i >= exp_y1 && i <= exp_y2 &&
                                              j >= exp_x1 && j <= exp_x2);
                    const bool in_bbox     = (i >= y1 && i < y2 && j >= x1 && j < x2);
                    const bool sparse      = (i % 10 == 0 && j % 10 == 0);

                    // Obstacle in 2x expanded bbox
                    if (in_expanded && sparse) {
                        valid_points_obs_.emplace_back(pt_obs[1], -pt_obs[0], pt_obs[2]);
                    }

                    // Tomato face points inside bbox + mask
                    if (in_bbox && row_mask[j] > 0) {
                    //if (in_bbox) {
                        //if (y2 - y1 < x2 - x1) orientation_fix = true;
                        if (pt_obs[2] > 0.1f && pt_obs[2] < 2.f)
                            valid_points_.emplace_back(pt_obs[1], -pt_obs[0], pt_obs[2]);
                    }

                    // Background obstacles outside expanded bbox
                    if (!in_expanded && sparse) {
                        ++count;
                        valid_points_obs_all_.emplace_back(pt_obs[2], -pt_obs[0], -pt_obs[1]);
                        //if (row_obs && row_obs[j] > 0)
                        //    valid_obs_point_only_.emplace_back(pt_obs[2], -pt_obs[0], -pt_obs[1]);
                    }
                }
            }

            RCLCPP_INFO(get_logger(), "valid_points size: %zu", valid_points_.size());

            if (valid_points_.size() < 15) {
                RCLCPP_WARN(get_logger(), "Not enough valid points: %zu", valid_points_.size());
                continue;
            }

            RCLCPP_INFO(get_logger(), "Count: %d  valid_points_obs: %zu", count, valid_points_obs_.size());

            std::vector<Eigen::Vector3f> filtered = filterOutliersSOR(valid_points_, 20, 2.0f);
            RCLCPP_INFO(get_logger(), "After SOR: %zu points", filtered.size());
            if (filtered.size() < 15) continue;

            size_t n   = filtered.size();
            size_t mid = n / 2;
            float z_min = std::numeric_limits<float>::max();
            float y_min = std::numeric_limits<float>::max();
            float y_max = std::numeric_limits<float>::lowest();

            std::vector<float> depths(n);
            for (size_t k = 0; k < n; ++k) {
                float z = filtered[k].z(), y = filtered[k].y();
                depths[k] = z;
                if (z < z_min) z_min = z;
                if (y < y_min) y_min = y;
                if (y > y_max) y_max = y;
            }

            // Median Z
            float median;
            if (n % 2 == 0) {
                std::nth_element(depths.begin(), depths.begin()+mid-1, depths.end());
                float m1 = depths[mid-1];
                std::nth_element(depths.begin(), depths.begin()+mid, depths.end());
                median = (m1 + depths[mid]) * 0.5f;
            } else {
                std::nth_element(depths.begin(), depths.begin()+mid, depths.end());
                median = depths[mid];
            }

            float center_fix = std::abs(
                std::abs(y_max + (x2-cx_)*z_min/fx_) -
                std::abs(-(x1-cx_)*z_min/fx_ - y_min));
            float Z = z_min + std::abs(x2-x1)*z_min/(fx_*2) + center_fix/2;
            float Y = -(center_x - cx_) * Z / fx_;
            float X =  (center_y - cy_) * Z / fy_;

            RCLCPP_INFO(get_logger(), "Raw coords - X:%.3f Y:%.3f Z:%.3f", X, Y, Z);

            // Filter obs: copy_if is idiomatic and clear to read
            std::vector<Eigen::Vector3f> filtered_obs;
            filtered_obs.reserve(valid_points_obs_.size());
            std::copy_if(valid_points_obs_.begin(), valid_points_obs_.end(),
                std::back_inserter(filtered_obs),
                [median](const Eigen::Vector3f& p){ return p.z() <= median; });

            //std::vector<Eigen::Vector3f> filtered_obs_all;
            //filtered_obs_all.reserve(valid_obs_point_only_.size() + valid_points_obs_all_.size());
            //for (const auto& p : valid_obs_point_only_)
            //    filtered_obs_all.push_back(p);
            //for (const auto& p : valid_points_obs_all_)
            //    if (p.x() > median) filtered_obs_all.push_back(p);

            if (!valid_points_obs_all_.empty())
                octomap_map_temp[detection_idx] = buildOctomap(valid_points_obs_all_);

            //const float yaw_contour = computeYawFromContour2D(contour).value_or(0.f);

            auto [roll_obj, pitch_obj, yaw_obj] = computeSurfaceOrientation(filtered,     0.01f, true);
            auto [roll_obs, pitch_obs, yaw_obs] = computeSurfaceOrientation(filtered_obs, 0.01f, false);

            float roll  = (std::abs(roll_obs)  > M_PI/90.f) ? roll_obs  : roll_obj;
            float pitch = (std::abs(pitch_obs) > M_PI/90.f) ? pitch_obs : pitch_obj;
            float yaw   = yaw_obj;

            RCLCPP_INFO(get_logger(),
                "Obs orient  roll=%.2f° pitch=%.2f° yaw=%.2f°",
                roll_obs*180/M_PI, pitch_obs*180/M_PI, yaw_obs*180/M_PI);
            RCLCPP_INFO(get_logger(),
                "Surf orient roll=%.2f° pitch=%.2f° yaw=%.2f°",
                roll_obj*180/M_PI, pitch_obj*180/M_PI, yaw_obj*180/M_PI);
            RCLCPP_INFO(get_logger(),
                "Contour yaw=%.2f°",
                yaw_obj*180/M_PI);
            RCLCPP_INFO(get_logger(),
                "Final orient roll=%.2f° pitch=%.2f° yaw=%.2f°",
                roll*180/M_PI, pitch*180/M_PI, yaw*180/M_PI);

            if (std::isnan(roll) || std::isnan(pitch) || std::isnan(yaw))
                roll = pitch = yaw = 0.f;

            float X_final, Y_final, Z_final;
            //compute_offset_position(X, Y, Z, roll, pitch, yaw,
            //                        object_offset, X_final, Y_final, Z_final);

            compute_offset_position(X, Y, Z, X_final, Y_final, Z_final);

            if (!skip_target_) {
                TrackedBbox tb;
                tb.bbox         = cv::Rect(x1, y1, x2-x1, y2-y1);
                tb.center       = cv::Point2f((float)center_x, (float)center_y);
                tb.z_depth      = Z;
                tb.roll         = roll;  tb.pitch = pitch;  tb.yaw = yaw;
                tb.center_fix   = center_fix;
                tb.z_min        = z_min;
                //tb.median_depth = median;
                tracked_store_.push_back(tb);

                PoseInfo pi_info;
                pi_info.distance = std::sqrt(X*X + Y*Y + Z*Z);
                pi_info.X_final  = X_final;  pi_info.Y_final = Y_final;  pi_info.Z_final = Z_final;
                pi_info.roll     = roll;      pi_info.pitch   = pitch;     pi_info.yaw    = yaw;
                pi_info.idx      = detection_idx;
                poses.push_back(pi_info);
            }
            ++detection_idx;
        }

        std::sort(poses.begin(), poses.end(),
            [](const PoseInfo& a, const PoseInfo& b){ return a.distance < b.distance; });

        tomato_octomap_msgs::msg::TomatoOctomaps msg_out;
        for (size_t i = 0; i < poses.size(); ++i) {
            auto it = octomap_map_temp.find(poses[i].idx);
            if (it != octomap_map_temp.end()) {
                tomato_octomap_msgs::msg::TomatoOctomap to;
                to.idx = (int)i;  to.octomap = it->second;
                msg_out.octomaps.push_back(to);
            }
        }
        tomato_octomap_pub_->publish(msg_out);

        RCLCPP_WARN(get_logger(), "Check222222");
        time_recieved_ = false;
        if (poses.empty() || processing_request_) return std::nullopt;
        return poses;
    }

    // =========================================================================
    // compute_offset_position: only column-2 of rotation matrix is needed for
    // the offset direction, so skip computing the other 6 elements entirely.
    void compute_offset_position(
        float x, float y, float z,
        float& x_out, float& y_out, float& z_out)
    {
        x_out = x - fx_offset;
        y_out = y + baseline_offset;
        z_out = z;
    }

    // =========================================================================
    void send_move_request() {
        if (processing_request_) return;
        processing_request_ = true;

        auto goal_msg = MoveRobot::Goal();
        goal_msg.request_move = true;

        RCLCPP_INFO(get_logger(), "Sending request to action server...");

        auto opts = rclcpp_action::Client<MoveRobot>::SendGoalOptions();
        opts.feedback_callback      = std::bind(&Tomato3DDetector::feedback_callback,    this, _1, _2);
        opts.goal_response_callback = std::bind(&Tomato3DDetector::handle_goal_response, this, _1);
        opts.result_callback        = std::bind(&Tomato3DDetector::handle_result,        this, _1);
        client_->async_send_goal(goal_msg, opts);
    }

    void feedback_callback(
        GoalHandleMoveRobot::SharedPtr,
        const std::shared_ptr<const MoveRobot::Feedback> feedback)
    {
        RCLCPP_INFO(get_logger(), "Feedback: %.2f%%", feedback->progress * 100.f);
        if (feedback->progress >= 1.f)
            RCLCPP_INFO(get_logger(), "***************************************************");
    }

    void handle_result(const GoalHandleMoveRobot::WrappedResult& result) {
        if (result.code == rclcpp_action::ResultCode::SUCCEEDED && result.result->success) {
            RCLCPP_INFO(get_logger(), "Goal succeeded");
            resent_ = false;
        } else {
            RCLCPP_WARN(get_logger(), "Goal failed: code=%d msg=%s",
                (int)result.code, result.result->message.c_str());
            if (result.result->message == "RE") resent_ = true;
        }
        processing_request_ = false;
    }

    void handle_goal_response(GoalHandleMoveRobot::SharedPtr goal_handle) {
        if (!goal_handle) {
            RCLCPP_WARN(get_logger(), "Action goal rejected.");
            processing_request_ = false;
            time_publisher(now().seconds(), true);
        }
    }

    // =========================================================================
    void collectmsg_callback(const collect_msgs::msg::CollectMsg& msg) {
        if (!msg.collect_msg.empty() && !time_recieved_) {
            const auto& t           = msg.collect_msg.front();
            start_detection_time_   = t.start_detection;
            detection_time_         = t.detection_time;
            start_positioning_time_ = t.start_positioning_time;
            time_recieved_          = true;
        }
    }

    void time_publisher(double start_time, bool check = false) {
        collect_msgs::msg::CollectMsg msg;
        collect_msgs::msg::CollectTime t;
        t.start_detection        = start_detection_time_;
        t.detection_time         = detection_time_;
        t.positioning_time       = start_time - start_positioning_time_;
        t.check                  = check;
        msg.collect_msg.push_back(t);
        time_pub->publish(msg);
    }

    void publish_position_signal(bool computing_position) {
        position_signal_msgs::msg::PositionSignal msg;
        msg.computing_position = computing_position;
        position_signal_pub->publish(msg);
    }

    void publish_skip_signal(bool skip) {
        skip_signal_msgs::msg::SkipSignal msg;
        msg.skip = skip;
        skip_signal_pub->publish(msg);
    }

    void publish_depth_signal(bool computing_depth) {
        depth_signal_msgs::msg::DepthSignal msg;
        msg.computing_depth = computing_depth;
        depth_signal_pub->publish(msg);
    }

    void publish_move_signal(bool move) {
        move_signal_msgs::msg::MoveSignal msg;
        msg.move = move;
        move_signal_pub->publish(msg);
    }

    void handle_no_tomato_detected() {
        processing_request_ = false;
        last_yolo_msg_ = nullptr;
        tomato_list_.clear();
        nearest_tomato.reset();

        publish_skip_signal(true);
        publish_move_signal(true);
        publish_no_tomato_status();
    }

    void publish_no_tomato_status() {
        connect_msgs::msg::ConnectMsg msg;
        connect_msgs::msg::ConnectStatus state;
        res_msgs::msg::PoseRes res_msg;
        res_msgs::msg::ResFlag flag_msg;

        state.connection = false;
        state.wait_key = false;
        flag_msg.flag = false;
        flag_msg.x = now().seconds();

        msg.connect_msg.push_back(state);
        res_msg.pose_res.push_back(flag_msg);
        connect_msg_pub->publish(msg);
        res_msg_pub->publish(res_msg);
    }

    void publish_status() {
        connect_msgs::msg::ConnectMsg msg;
        connect_msgs::msg::ConnectStatus state;
        res_msgs::msg::PoseRes res_msg;
        res_msgs::msg::ResFlag flag_msg;
        state.connection = false;  state.wait_key = true;
        flag_msg.flag    = true;
        msg.connect_msg.push_back(state);
        res_msg.pose_res.push_back(flag_msg);
        connect_msg_pub->publish(msg);
        res_msg_pub->publish(res_msg);
    }
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<Tomato3DDetector>());
    rclcpp::shutdown();
    return 0;
}
