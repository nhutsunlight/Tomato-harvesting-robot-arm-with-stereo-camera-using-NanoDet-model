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

struct Candidate
{
    Eigen::Vector3f pt;
    float z;
    bool valid=false;
};

class Tomato3DDetector : public rclcpp::Node {
   public:
    using MoveRobot = control_action::action::MoveRobot;
    using GoalHandleMoveRobot = rclcpp_action::ClientGoalHandle<MoveRobot>;

    Tomato3DDetector() : Node("tomato_3d_detector") {
        rclcpp::QoS qos_profile(rclcpp::KeepLast(1));
        qos_profile.transient_local();
        qos_profile.reliable(); 
        // Action Client
        client_ = rclcpp_action::create_client<MoveRobot>(this, "move_robot");
        while (rclcpp::ok() && !client_->wait_for_action_server(std::chrono::seconds(1))) {
            RCLCPP_INFO(this->get_logger(), "Waiting for MoveIt Action Server...");
        }
        RCLCPP_INFO(this->get_logger(), "MoveIt Action Server is ready!");

        // Subscribers
        sub_yolo_ = this->create_subscription<yolov8_msgs::msg::Yolov8Inference>(
            "/detect_msg", 10, std::bind(&Tomato3DDetector::yolo_callback, this, _1));

        sub_disparity_ = this->create_subscription<stereo_msgs::msg::DisparityImage>(
            "/stereo/disparity", 10, std::bind(&Tomato3DDetector::disparity_callback, this, _1));

        sub_left_cam_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
            "/stereo/left/camera_info_calib", rclcpp::QoS(1).transient_local().reliable()
            , std::bind(&Tomato3DDetector::left_camera_info_callback, this, _1));

        sub_right_cam_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
            "/stereo/right/camera_info_calib", rclcpp::QoS(1).transient_local().reliable()
            , std::bind(&Tomato3DDetector::right_camera_info_callback, this, _1));

        sub_pose_res_ = this->create_subscription<res_msgs::msg::PoseRes>(
            "/pose_res", 10, std::bind(&Tomato3DDetector::pose_res_callback, this, _1));

        time_sub_ = this->create_subscription<collect_msgs::msg::CollectMsg>(
            "/collect_msg", 10, std::bind(&Tomato3DDetector::collectmsg_callback, this, std::placeholders::_1));

        left_img_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/ref_img", 10, std::bind(&Tomato3DDetector::left_camera_callback, this, _1));
        obs_img_sub_ = this->create_subscription<sensor_msgs::msg::Image>(

            "/obs_img", 10, std::bind(&Tomato3DDetector::obs_camera_callback, this, _1));
                
        config_sub_ = this->create_subscription<config_manager::msg::SystemConfig>(
            "/system_config", 
            rclcpp::QoS(1).transient_local().reliable(),
            std::bind(&Tomato3DDetector::config_callback, this, std::placeholders::_1)
        );

        // Publisher
        publisher_ = this->create_publisher<test_msgs::msg::RosYolo>("/ros_yolo", 10);

        time_pub = create_publisher<collect_msgs::msg::CollectMsg>("/collect2_msg", 10);

        connect_msg_pub = this->create_publisher<connect_msgs::msg::ConnectMsg>("/connect_msg", qos_profile);

        res_msg_pub = this->create_publisher<res_msgs::msg::PoseRes>("/pose_res", qos_profile);

        octomap_pub_ = this->create_publisher<octomap_msgs::msg::Octomap>(
            "/obstacle_octomap", 10);
        tomato_octomap_pub_ = this->create_publisher<tomato_octomap_msgs::msg::TomatoOctomaps>(
            "/tomato_octomaps", 10);

        std::filesystem::path base_path = std::filesystem::current_path(); // sẽ là đường dẫn từ nơi bạn chạy `ros2 run`
        config_path = base_path.string() + "/config/setup.yaml";

        RCLCPP_INFO(this->get_logger(), "Tomato 3D Detector Node Started!");
    }

   private:
    // Camera parameters
    float fx_, fy_, cx_, cy_, baseline_, fx_offset, object_offset, baseline_offset;

    // Flags
    std::atomic<bool> processing_request_ = false;
    std::atomic<bool> data_ready_{false};
    double last_move_ = 0.0;
    double disparity_time_ = 0.0;
    double detection_time_ = 0.0;
    double start_detection_time_ = 0.0;
    double start_positioning_time_ = 0.0;
    bool flag_ = false;
    bool skip_first_frame_ = false;
    bool orientation_fix = false;
    bool config_received_ = false;
//    bool multi_collect_mode = false;
    //std::atomic<bool> rotate_check_ = true;
    //bool ws_check_ = true;
    bool resent_ = false;
    bool time_recieved_ = false;
    //std::atomic<bool> start_time_check_ = false;
    size_t tomato_index_;
    rclcpp::TimerBase::SharedPtr timer_;
    size_t last_object_count_ = 0;
    float obstacle_radius = 0.03f; // chỉnh bán kính nếu cần
    float min_d;  // tiêu cự
    float max_d;  // baseline
    int x1_off;
    int y1_off;
    int w;
    int h;
    // member variables
    yolov8_msgs::msg::Yolov8Inference::SharedPtr last_yolo_msg_ = nullptr;
    rclcpp::TimerBase::SharedPtr retry_timer_;

    // Data
    std::mutex data_mtx_; 
    cv::Mat_<cv::Vec3f> points_mat;  // scratch buffer
    cv::Mat left_img_, obs_img_;
    std::string config_path;
    std::vector<PoseInfo> tomato_list_;
    std::mutex mtx_;
    std::condition_variable cv_;
    image_geometry::StereoCameraModel model_;

    // Subscribers
    rclcpp::Subscription<yolov8_msgs::msg::Yolov8Inference>::SharedPtr sub_yolo_;
    rclcpp::Subscription<stereo_msgs::msg::DisparityImage>::SharedPtr sub_disparity_;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr sub_left_cam_;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr sub_right_cam_;
    rclcpp::Subscription<res_msgs::msg::PoseRes>::SharedPtr sub_pose_res_;
    rclcpp::Subscription<collect_msgs::msg::CollectMsg>::SharedPtr time_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr left_img_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr obs_img_sub_;
    sensor_msgs::msg::CameraInfo::SharedPtr left_camera_info_;
    sensor_msgs::msg::CameraInfo::SharedPtr right_camera_info_;
    rclcpp::Subscription<config_manager::msg::SystemConfig>::SharedPtr config_sub_;

    // Publisher
    rclcpp::Publisher<test_msgs::msg::RosYolo>::SharedPtr publisher_;
    rclcpp::Publisher<collect_msgs::msg::CollectMsg>::SharedPtr time_pub;
    rclcpp::Publisher<connect_msgs::msg::ConnectMsg>::SharedPtr connect_msg_pub;
    rclcpp::Publisher<res_msgs::msg::PoseRes>::SharedPtr res_msg_pub;
    rclcpp::Publisher<octomap_msgs::msg::Octomap>::SharedPtr octomap_pub_;
    rclcpp::Publisher<tomato_octomap_msgs::msg::TomatoOctomaps>::SharedPtr tomato_octomap_pub_;
    //std::map<int, octomap_msgs::msg::Octomap> octomap_map_temp;


    // Action Client
    rclcpp_action::Client<MoveRobot>::SharedPtr client_;

    void pose_res_callback(const res_msgs::msg::PoseRes::SharedPtr msg) {
        if (!msg->pose_res.empty()) {
            last_move_ = msg->pose_res[0].x;
            flag_ = msg->pose_res[0].flag;
            //RCLCPP_INFO(this->get_logger(), "Check point at : %.2f", last_move_);
        }
    }

    void load_setup_params(const std::string &filename) {
        RCLCPP_INFO(this->get_logger(), "loading setup params");
        YAML::Node config = YAML::LoadFile(filename);
        auto setup = config["setup"];
        object_offset = setup["ObjectOffset"].as<double>();
        fx_offset = setup["FxOffset"].as<double>();
//        multi_collect_mode = setup["Multi_collect_mode"].as<bool>();
    }

    void config_callback(const config_manager::msg::SystemConfig::SharedPtr msg)
    {
        config_received_ = false;  // reset flag để chờ config mới cho lần sau
        RCLCPP_INFO(this->get_logger(), "Load setup");

        object_offset = msg->object_offset;
        fx_offset = msg->fx_offset;
        config_received_ = true;
    }

    void left_camera_info_callback(const sensor_msgs::msg::CameraInfo::SharedPtr msg) {
        left_camera_info_ = std::make_shared<sensor_msgs::msg::CameraInfo>(*msg);
        update_camera_model();
        const auto &K = msg->k;  // là std::array<float, 9>
        fx_ = K[0];
        fy_ = K[4];
        cx_ = K[2];
        cy_ = K[5];
    }

    void right_camera_info_callback(const sensor_msgs::msg::CameraInfo::SharedPtr msg) {
        right_camera_info_ = std::make_shared<sensor_msgs::msg::CameraInfo>(*msg);
        update_camera_model();
        const auto &P = msg->p;  // là std::array<float, 12>
        baseline_ = std::abs(static_cast<float>(P[3] / P[0]));
    }

    void update_camera_model() {
        if (left_camera_info_ && right_camera_info_) {
            model_.fromCameraInfo(*left_camera_info_, *right_camera_info_);
            //RCLCPP_INFO(this->get_logger(), "Stereo camera model updated!");
        }
    }

    void left_camera_callback(const sensor_msgs::msg::Image::ConstSharedPtr &msg)
    {
        try {
            left_img_ = cv::Mat();  // Clear old map
            left_img_ = cv_bridge::toCvCopy(msg, "mono8")->image;
        } catch (cv_bridge::Exception &e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
            return;
        }
    }

    void obs_camera_callback(const sensor_msgs::msg::Image::ConstSharedPtr &msg)
    {
        try {
            obs_img_ = cv::Mat();  // Clear old map
            obs_img_ = cv_bridge::toCvCopy(msg, "mono8")->image;
        } catch (cv_bridge::Exception &e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
            return;
        }
    }

    void disparity_callback(const stereo_msgs::msg::DisparityImage::SharedPtr msg) {

        // ✅ 1. Kiểm tra skip TRƯỚC — không làm gì cả nếu skip
        if (skip_first_frame_) {
            skip_first_frame_ = false;
            return;
        }

        // ✅ 2. Chuẩn bị data NGOÀI lock — tránh giữ lock lâu
        const sensor_msgs::msg::Image& dimage = msg->image;
        const float* data = reinterpret_cast<const float*>(dimage.data.data());
        const cv::Mat_<float> dmat(dimage.height, dimage.width,
                                const_cast<float*>(data),   // OpenCV yêu cầu non-const
                                dimage.step);

        cv::Mat_<cv::Vec3f> tmp;
        model_.projectDisparityImageTo3d(dmat, tmp, true);  // ✅ heavy work ngoài lock

        // ✅ 3. Chỉ lock khi ghi shared data
        {
            std::lock_guard<std::mutex> lock(data_mtx_);
            points_mat    = std::move(tmp);
            disparity_time_ = rclcpp::Time(msg->header.stamp).seconds();
            w     = msg->valid_window.width;
            h     = msg->valid_window.height;
            x1_off = msg->valid_window.x_offset;
            y1_off = msg->valid_window.y_offset;
            min_d  = msg->min_disparity;
            max_d  = msg->max_disparity;
        }
    }

    void yolo_callback(const yolov8_msgs::msg::Yolov8Inference::SharedPtr msg) {
        RCLCPP_INFO(this->get_logger(), "DEBUG: Received YOLO message with %zu detections",
                    msg->yolov8_inference.size());

        // Cache lại message để retry
        if (!processing_request_) {
            last_yolo_msg_ = msg;
        }

        try_process_yolo();
    }

    void try_process_yolo() {
        if (!last_yolo_msg_) return;

        if (disparity_time_ < last_move_ || flag_ || !config_received_) {
            processing_request_ = false;
            RCLCPP_WARN(this->get_logger(),
                "Disparity is old, will retry... flag: %d, disparity_time: %f, last_move: %f",
                flag_, disparity_time_, last_move_);
            retry_timer_ = this->create_wall_timer(
                std::chrono::milliseconds(200),
                [this]() {
                    retry_timer_->cancel();
                    try_process_yolo();
                });
            return;
        }

        last_move_ = 0.0;
        //load_setup_params(config_path);

        if (processing_request_) return;

        // ✅ Thay vì return thẳng, retry chờ đủ điều kiện
        if (points_mat.empty() || fx_ == 0.0 || baseline_ == 0.0) {
            RCLCPP_WARN(this->get_logger(), "Waiting for calibration/disparity map...");
            retry_timer_ = this->create_wall_timer(
                std::chrono::milliseconds(200),
                [this]() {
                    retry_timer_->cancel();
                    try_process_yolo();
                });
            return;
        }

        auto msg = last_yolo_msg_;
        last_yolo_msg_ = nullptr;

        auto nearest_tomato = compute3DCoordinates(*msg);
        RCLCPP_WARN(this->get_logger(), "Check11111");

        if (nearest_tomato.has_value()) {
            tomato_list_ = *nearest_tomato;
            process_next_tomato();
        } else {
            RCLCPP_INFO(this->get_logger(), "No tomato detected.");
        }
    }

    void process_next_tomato()
    {
        if (tomato_list_.empty()) {
            RCLCPP_WARN(this->get_logger(), "Tomato list is empty.");
            return;
        }

        RCLCPP_WARN(this->get_logger(), "Poses computed: %zu", tomato_list_.size());
        
        test_msgs::msg::RosYolo ros_yolo_msg;
        for (const auto& t : tomato_list_) {
            test_msgs::msg::YoloPose yp;
            yp.x = t.X_final;
            yp.y = t.Y_final;
            yp.z = t.Z_final;
            yp.roll = t.roll;
            yp.pitch = t.pitch;
            yp.yall = t.yaw;
            ros_yolo_msg.ros_yolo.push_back(yp);
        }

        // 1. Publish data TRƯỚC
        publisher_->publish(ros_yolo_msg);
        RCLCPP_INFO(this->get_logger(), "Published %zu poses", tomato_list_.size());

        // 2. Đợi 1 chút để action server nhận được data
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        // 3. Gửi request
        send_move_request();

        // 4. time_publisher KHÔNG block ở đây — chạy async
        // An toàn hơn: one-shot timer
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(0),
            [this]() {
                time_publisher(this->now().seconds());
                timer_->cancel();
            });
    }

    // ===== HÀM LỌC OUTLIERS OPTIMIZED =====
    std::vector<Eigen::Vector3f> filterOutliersSOR(
        const std::vector<Eigen::Vector3f>& points,
        int k = 50,  // Tăng k để ổn định hơn
        float std_mul = 1.0f  // Giảm ngưỡng để giữ nhiều điểm hơn
    ) {
        if (points.size() < static_cast<size_t>(k * 2)) return points;

        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
        cloud->width = points.size();
        cloud->height = 1;
        cloud->is_dense = true;
        cloud->points.reserve(points.size());
        
        for (const auto& p : points) {
            cloud->points.emplace_back(p(0), p(1), p(2));
        }

        pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
        sor.setInputCloud(cloud);
        sor.setMeanK(k);
        sor.setStddevMulThresh(std_mul);

        pcl::PointCloud<pcl::PointXYZ>::Ptr filtered(new pcl::PointCloud<pcl::PointXYZ>);
        sor.filter(*filtered);

        std::vector<Eigen::Vector3f> result;
        result.reserve(filtered->points.size());
        for (const auto& pt : filtered->points) {
            result.emplace_back(pt.x, pt.y, pt.z);
        }
        
        RCLCPP_DEBUG(rclcpp::get_logger("tomato"), 
            "SOR filtering: %lu -> %lu points", points.size(), result.size());
        
        return result;
    }

    std::tuple<float, float, float> computeSurfaceOrientation(
        const std::vector<Eigen::Vector3f>& points, float threshold, bool yaw_flag_
    ) {
        if (points.size() < 10) {
            return {0.0f, 0.0f, 0.0f};
        }

        // 1️⃣ Chuyển sang PCL cloud
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
        for (const auto& pt : points) {
            cloud->points.emplace_back(pt(0), pt(1), pt(2));
        }

        pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
        pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
        pcl::SACSegmentation<pcl::PointXYZ> seg;

        seg.setOptimizeCoefficients(true);
        seg.setModelType(pcl::SACMODEL_PLANE);
        seg.setMethodType(pcl::SAC_RANSAC);
        seg.setDistanceThreshold(threshold);   // 1 cm
        seg.setMaxIterations(1000);
        seg.setInputCloud(cloud);
        seg.segment(*inliers, *coefficients);

        if (inliers->indices.size() < 10) {
            RCLCPP_WARN(rclcpp::get_logger("tomato"), "RANSAC failed, not enough inliers");
            return {0.0f, 0.0f, 0.0f};
        }

        // 4️⃣ Vector pháp tuyến của mặt phẳng
        Eigen::Vector3f normal(
            coefficients->values[0],
            coefficients->values[1],
            coefficients->values[2]
        );
        normal.normalize();

        if (normal(2) < 0) {
            normal = -normal;  // Đảm bảo normal hướng về phía camera
        }
        RCLCPP_INFO(this->get_logger(), "Normal principal: %f, %f, %f", normal(0), normal(1), normal(2));

        if (std::sqrt(normal(0)*normal(0) + normal(1)*normal(1)) > std::sqrt(normal(2)*normal(2))) {
            normal(2) = (std::sqrt(normal(0)*normal(0) + normal(1)*normal(1)))*(normal(2)/std::abs(normal(2)));
        }

        // 5️⃣ Tính Roll và Pitch từ normal (radian)
        float roll  = -std::atan2(normal(1), normal(2));
        float pitch = std::atan2(normal(0), normal(2));
        float yaw = 0.0f;  // mặc định

        if (yaw_flag_) {
            if (!orientation_fix) {
                // 7️⃣ PCA để tìm hướng chính của mặt phẳng
                pcl::PCA<pcl::PointXYZ> pca;
                pca.setInputCloud(cloud);
                Eigen::Vector3f main_dir(pca.getEigenVectors().col(0).data());
                main_dir.normalize();

                // 🔹 Tính yaw trong mặt phẳng
                if (main_dir(0) >= 0) {
                    yaw = std::atan2(main_dir(1), main_dir(0));
                } else {
                    yaw = std::atan2(-main_dir(1), -main_dir(0));
                }
            }
        }
        orientation_fix = false;  // reset flag sau khi tính toán
        return {roll, pitch, yaw};
    }

    void Obstacles_Octomap(const std::vector<Eigen::Vector3f>& points)
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

        for (const auto &p : points)
        {
            cloud->points.emplace_back(p.x(), p.y(), p.z());
        }

        octomap::OcTree tree(0.01);

        for (const auto &pt : cloud->points)
        {
            tree.updateNode(octomap::point3d(pt.x, pt.y, pt.z), true);
        }

        tree.updateInnerOccupancy();

        octomap_msgs::msg::Octomap octomap_msg;

        octomap_msgs::binaryMapToMsg(tree, octomap_msg);

        octomap_msg.header.frame_id = "left_camera_link";
        octomap_msg.header.stamp = this->now();

        octomap_pub_->publish(octomap_msg);
        RCLCPP_WARN(this->get_logger(), "==========DEBUG OBSTACLES MAP PUBLISHED==========");
    }

    octomap_msgs::msg::Octomap buildOctomap(const std::vector<Eigen::Vector3f>& points)
    {
        octomap::OcTree tree(0.01);
        for (const auto& p : points) {
            tree.updateNode(octomap::point3d(p.x(), p.y(), p.z()), true);
        }
        tree.updateInnerOccupancy();

        octomap_msgs::msg::Octomap octomap_msg;
        octomap_msgs::binaryMapToMsg(tree, octomap_msg);
        octomap_msg.header.frame_id = "left_camera_link";
        octomap_msg.header.stamp = this->now();
        return octomap_msg;
    }

    std::vector<cv::Point> findContourMono8(const cv::Mat& mono8_img,
                                            int x1, int y1, int x2, int y2)
    {
        // --- Kiểm tra tọa độ ---
        if (x1 >= x2 || y1 >= y2 || x1 < 0 || y1 < 0 ||
            x2 > mono8_img.cols || y2 > mono8_img.rows)
        {
            std::cout << "Invalid parameters!" << std::endl;
            RCLCPP_INFO(this->get_logger(), "Invalid parameters! Image size: %d x %d, x1: %d, y1: %d, x2: %d, y2: %d",
                mono8_img.cols, mono8_img.rows, x1, y1, x2, y2);
            return {};
        }

        // --- Cắt ROI ---
        cv::Mat roi = mono8_img(cv::Range(y1, y2), cv::Range(x1, x2));

        if (roi.channels() > 1)
            cv::cvtColor(roi, roi, cv::COLOR_BGR2GRAY);

        // --- Threshold đơn giản (giả sử mask trắng - nền đen) ---
        cv::Mat binary;
        cv::threshold(roi, binary, 127, 255, cv::THRESH_BINARY);

        // --- Tìm contours ---
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        // Không có contour
        if (contours.empty())
            return {};

        // --- Tìm contour lớn nhất ---
        double roi_area = static_cast<double>((x2 - x1) * (y2 - y1));
        double min_area = 0.001 * roi_area;  // ngưỡng noise

        int best_idx = -1;
        double max_area = 0.0;

        for (size_t i = 0; i < contours.size(); ++i)
        {
            double area = cv::contourArea(contours[i]);

            if (area >= min_area && area > max_area)
            {
                max_area = area;
                best_idx = static_cast<int>(i);
            }
        }

        // Không có contour đủ lớn
        if (best_idx < 0)
            return {};

        // --- Offset về tọa độ ảnh gốc ---
        std::vector<cv::Point> result;
        result.reserve(contours[best_idx].size());

        for (const auto& pt : contours[best_idx])
            result.emplace_back(pt.x + x1, pt.y + y1);

        return result;
    }

    std::optional<std::vector<PoseInfo>> compute3DCoordinates(
        const yolov8_msgs::msg::Yolov8Inference &msg) {
        if (points_mat.empty() || fx_ == 0.0 || fy_ == 0.0 || baseline_ == 0.0) {
            RCLCPP_WARN(this->get_logger(), "Invalid calibration or empty disparity");
            time_recieved_ = false;
            return std::nullopt;
        }
        std::vector<tomato_octomap_msgs::msg::TomatoOctomap> tomato_octomaps;
        std::map<int, octomap_msgs::msg::Octomap> octomap_map_temp; 
        octomap_map_temp.clear();  // ← thêm dòng này
        int limit = 0;
        int detection_idx = 0;
        std::tuple<float, float, float, float, float, float> nearest_tomato;
        std::vector<PoseInfo> poses; 
                    std::vector<Eigen::Vector3f> valid_points_obs_all, valid_obs_point_only, valid_points_obs, valid_points;

        // khúc xử lý này đang bị nhiễu, chưa cắt đúng obj (khusc này lấy contour trong bbox từ cái ảnh mask)
        for (const auto &detection : msg.yolov8_inference) {
            valid_points_obs_all.clear();
            valid_obs_point_only.clear();
            valid_points.clear();
            valid_points_obs.clear();
            if (detection.id != 1 && limit > 10) continue;

            int x1 = std::clamp(static_cast<int>(detection.top), 0, w - 1);
            int y1 = std::clamp(static_cast<int>(detection.left), 0, h - 1);
            int x2 = std::clamp(x1 + static_cast<int>(detection.bottom), 0, w - 1);
            int y2 = std::clamp(y1 + static_cast<int>(detection.right), 0, h - 1);
            RCLCPP_INFO(this->get_logger(), "ROI raw: x1=%d, y1=%d, x2=%d, y2=%d, w=%d, h=%d", x1, y1, x2, y2, w, h);
            if (x1 >= x2 || y1 >= y2) continue;

            int center_x = (x1 + x2) / 2;
            int center_y = (y1 + y2) / 2;
            baseline_offset = baseline_ / 2;

            auto contour = findContourMono8(left_img_, x1, y1, x2, y2);

            cv::Mat mask = cv::Mat::zeros(h, w, CV_8UC1);

            // 🔥 fallback nếu contour lỗi
            if (contour.size() < 3)
            {
                RCLCPP_WARN(this->get_logger(), "Contour invalid (%zu pts) → fallback bbox", contour.size());

                contour.clear();
                contour.emplace_back(x1, y1);
                contour.emplace_back(x2, y1);
                contour.emplace_back(x2, y2);
                contour.emplace_back(x1, y2);
            }

            // đảm bảo luôn hợp lệ
            std::vector<std::vector<cv::Point>> contours_wrap{contour};
            cv::fillPoly(mask, contours_wrap, cv::Scalar(255));
            //if (contour.size() < 3)
            //{
            //   // Không đủ điểm để tạo polygon
            //    continue;
            //}

            limit++;
      
            int count = 0;

            for (int i = 0; i < h; i += 5) {
                for (int j = 0; j < w; j += 5) {
                    cv::Vec3f pt_obs = points_mat.at<cv::Vec3f>(i, j);
                    if (pt_obs[2] > 0.0f && pt_obs[2] < 2.0f) {
                        if (i >= 2*y1-y2 && i <= 2*y2-y1 && j >= 2*x1-x2 && j <= 2*x2-x1 && i % 10 == 0 && j % 10 == 0) 
                        {
                            float X_obs, Y_obs, Z_obs;
//                            if (pt_obs[2] > median || (i>= y1 && i <= y2 && j >= x1 && j <= x2) || 
//                            mask.at<uchar>(i, j) > 0) {
//                                Z_obs = median;
//                                Y_obs = -(j - cx_) * Z_obs / fx_;
//                                X_obs = (i - cy_) * Z_obs / fy_;
//                            } else {
                                //count++;
                                Z_obs = pt_obs[2]; // gán giá trị lớn hơn ngưỡng để loại bỏ
                                Y_obs = -pt_obs[0];
                                X_obs = pt_obs[1];
//                            }
                            valid_points_obs.emplace_back(X_obs, Y_obs, Z_obs);
                        }
                        if (i >= y1 && i < y2 && j >= x1 && j < x2)
                        {
                            if (mask.at<uchar>(i, j) > 0) {
                                if (y2 - y1 < x2 - x1) {
                                    orientation_fix = true;
                                }
                                cv::Vec3f pt = points_mat.at<cv::Vec3f>(i, j);
                                if (pt[2] > 0.1f && pt[2] < 2.0f) {
                                    float Z_face = pt[2];
                                    float Y_face = -pt[0];
                                    float X_face = pt[1];
                                    valid_points.emplace_back(X_face, Y_face, Z_face);
                                }
                            }
                        }
                        if (!(i >= 2*y1-y2 && i <= 2*y2-y1 && j >= 2*x1-x2 && j <= 2*x2-x1) && i % 10 == 0 && j % 10 == 0) 
                        {
                            count++;
                            float X_obs_all = pt_obs[2];
                            float Y_obs_all = -pt_obs[0];
                            float Z_obs_all = -pt_obs[1];
                            valid_points_obs_all.emplace_back(X_obs_all, Y_obs_all, Z_obs_all);
                            if (obs_img_.at<uchar>(i, j) > 0) 
                            {
                                float X_obs_only = pt_obs[2];
                                float Y_obs_only = -pt_obs[0];
                                float Z_obs_only = -pt_obs[1];
                                valid_obs_point_only.emplace_back(X_obs_only, Y_obs_only, Z_obs_only);
                            }
                        }
                    }
                }
            }

            RCLCPP_INFO(this->get_logger(), "points_vec size: %zu", valid_points.size());   

            if (valid_points.size() < 15) {
                RCLCPP_WARN(this->get_logger(), "Not enough valid points: %zu", valid_points.size());
                continue;
            }

            RCLCPP_INFO(this->get_logger(), "Count: %u points", count);
            RCLCPP_INFO(this->get_logger(), "valid_points_obs: %zu points", valid_points_obs.size());
            // Lọc outliers
            std::vector<Eigen::Vector3f> filtered = filterOutliersSOR(valid_points, 20, 2.0f);
            RCLCPP_INFO(this->get_logger(), "After filtering: %zu points", filtered.size());

            if (filtered.size() < 15) continue;

            size_t n = filtered.size();
            size_t mid = n / 2;

            // Init
            float z_min = std::numeric_limits<float>::max();
            float y_min = std::numeric_limits<float>::max();
            float y_max = std::numeric_limits<float>::lowest();

            std::vector<float> depths(n);

            // 1 PASS: lấy hết thông tin cần thiết
            for (size_t i = 0; i < n; ++i) {
                const auto& p = filtered[i];

                float z = p.z();
                float y = p.y();

                depths[i] = z;

                if (z < z_min) z_min = z;
                if (y < y_min) y_min = y;
                if (y > y_max) y_max = y;
            }

            // Median Z (optional nhưng nên giữ)
            std::nth_element(depths.begin(), depths.begin() + mid, depths.end());

            float median;
            if (n % 2 == 0) {
                std::nth_element(depths.begin(), depths.begin() + mid - 1, depths.end());
                float m1 = depths[mid - 1];
                std::nth_element(depths.begin(), depths.begin() + mid, depths.end());
                float m2 = depths[mid];
                median = (m1 + m2) * 0.5f;
            } else {
                std::nth_element(depths.begin(), depths.begin() + mid, depths.end());
                median = depths[mid];
            }

            float center_fix = std::abs(std::abs((y_max + (x2 - cx_) * z_min / fx_)) - std::abs((-(x1 - cx_) * z_min / fx_ - y_min)));
            //float Z = z_min + std::abs(x2 - x1) * z_min / (fx_ * 2);
            float Z = z_min + std::abs(x2 - x1) * z_min / (fx_ * 2) + center_fix / 2;
            float Y = -(center_x - cx_) * Z / fx_;
            float X = (center_y - cy_) * Z / fy_;

            RCLCPP_INFO(this->get_logger(), "Raw coordinates - X: %f, Y: %f, Z: %f", X, Y, Z);

            // Filter obs points dùng median vừa tính, partition thay vì copy
            std::vector<Eigen::Vector3f> filtered_obs;
            filtered_obs.reserve(valid_points_obs.size());
            std::copy_if(valid_points_obs.begin(), valid_points_obs.end(),
                std::back_inserter(filtered_obs),
                [median](const Eigen::Vector3f& p) { return p.z() <= median; });

            // Gộp 2 nguồn: valid_obs_point_only + points > median từ valid_points_obs_all
            std::vector<Eigen::Vector3f> filtered_obs_all;
            filtered_obs_all.reserve(valid_points_obs_all.size());

            // 1. Lấy tất cả từ valid_obs_point_only
            for (const auto& p : valid_obs_point_only) {
                filtered_obs_all.push_back(p);
            }

            // 2. Từ valid_points_obs_all, chỉ lấy z > median
            for (const auto& p : valid_points_obs_all) {
                if (p.x() > median) {
                    filtered_obs_all.push_back(p);
                }
            }

            if (!filtered_obs_all.empty()) {
                octomap_map_temp[detection_idx] = buildOctomap(filtered_obs_all);
            }

            // Tính orientation từ surface
            auto [roll_obj, pitch_obj, yaw_obj] = computeSurfaceOrientation(filtered, 0.01f, true);
            auto [roll_obs, pitch_obs, yaw_obs] = computeSurfaceOrientation(filtered_obs, 0.01f, false);
            //auto [roll_obs, pitch_obs, yaw_obs] = computeRollPitchFromObstacleCloud(valid_points_obs, X, Y);

            float roll, pitch, yaw;
            if (std::abs(roll_obs) > M_PI/90.0f) {
                roll = roll_obs;
            } else {
                roll = roll_obj;
            } 
            if (std::abs(pitch_obs) > M_PI/90.0f) {
                pitch = pitch_obs;
            } else {
                pitch = pitch_obj;
            }
            yaw = yaw_obj;  // luôn ưu tiên yaw từ object

            RCLCPP_INFO(this->get_logger(), 
                "Obstacles orientation - Roll: %.2f rad (%.1f°), Pitch: %.2f rad (%.1f°), Yaw: %.2f rad (%.1f°)",
                roll_obs, roll_obs * 180.0/M_PI, pitch_obs, pitch_obs * 180.0/M_PI, yaw_obs, yaw_obs * 180.0/M_PI);
            RCLCPP_INFO(this->get_logger(), 
                "Surface orientation - Roll: %.2f rad (%.1f°), Pitch: %.2f rad (%.1f°), Yaw: %.2f rad (%.1f°)",
                roll_obj, roll_obj * 180.0/M_PI, pitch_obj, pitch_obj * 180.0/M_PI, yaw_obj, yaw_obj * 180.0/M_PI);
            RCLCPP_INFO(this->get_logger(), 
                "Final orientation - Roll: %.2f rad (%.1f°), Pitch: %.2f rad (%.1f°), Yaw: %.2f rad (%.1f°)",
                roll, roll * 180.0/M_PI, pitch, pitch * 180.0/M_PI, yaw, yaw * 180.0/M_PI);

            // Kiểm tra giá trị hợp lệ
            if (std::isnan(roll) || std::isnan(pitch) || std::isnan(yaw)) {
                RCLCPP_WARN(this->get_logger(), "Invalid orientation, using default");
                roll = 0.0f;
                pitch = 0.0f;
                yaw = 0.0f;
            }

            float distance = sqrt(pow(X,2)+pow(Y,2)+pow(Z,2));

            float X_final, Y_final, Z_final;
            compute_offset_position(X, Y, Z, roll, pitch, yaw, object_offset, X_final, Y_final, Z_final);
            PoseInfo p;
            p.distance = distance;
            p.X_final = X_final;
            p.Y_final = Y_final;
            p.Z_final = Z_final;
            p.roll = roll;
            p.pitch = pitch;
            p.yaw = yaw;
            p.idx = detection_idx;   // ← gán idx trước khi push
            poses.push_back(p);
            detection_idx++;
        }
        //Obstacles_Octomap(valid_points_obs_all);
        std::sort(poses.begin(), poses.end(), [](const PoseInfo& a, const PoseInfo& b) {
            return a.distance < b.distance;
        });

        // Publish octomap theo đúng thứ tự pose sau sort
        tomato_octomap_msgs::msg::TomatoOctomaps msg_out;
        for (size_t i = 0; i < poses.size(); i++) {
            int original_idx = poses[i].idx;  // idx gốc trước sort
            
            // Tìm octomap tương ứng với original_idx
            auto it = octomap_map_temp.find(original_idx);
            if (it != octomap_map_temp.end()) {
                tomato_octomap_msgs::msg::TomatoOctomap to;
                to.idx = static_cast<int>(i);  // idx mới sau sort
                to.octomap = it->second;
                msg_out.octomaps.push_back(to);
            }
        }
        tomato_octomap_pub_->publish(msg_out);
        
        RCLCPP_WARN(this->get_logger(), "Check222222");
        time_recieved_ = false;
        if (poses.empty() || processing_request_) return std::nullopt;
        return poses;
    }

    void compute_offset_position(float x, float y, float z,
                             float roll, float pitch, float yaw,
                             float offset_distance,
                             float& x_out, float& y_out, float& z_out) {
        // Tính ma trận quay từ RPY (Z-Y-X)
        float cr = cos(roll),   sr = sin(roll);
        float cp = cos(pitch),  sp = sin(pitch);
        float cy = cos(yaw),    sy = sin(yaw);

        // Ma trận quay 3x3
        float R[3][3] = {
            {cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr},
            {sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr},
            {-sp,     cp * sr,                cp * cr}
        };

        // Offset theo trục -Z local
        float dx = -offset_distance * R[0][2];
        float dy = -offset_distance * R[1][2];
        float dz = -offset_distance * R[2][2];

        //Vị trí mới
        x_out = x - fx_offset + dx;
        y_out = y + baseline_offset  + dy;
        z_out = z + dz;
    }

    void send_move_request() {
        if (processing_request_) {
        //RCLCPP_WARN(this->get_logger(), "Đang xử lý request trước đó, không gửi thêm.");
        return;
        }
        processing_request_ = true;

        auto goal_msg = MoveRobot::Goal();
        goal_msg.request_move = true;

        RCLCPP_INFO(this->get_logger(), "Sending request to action server...");

        auto send_goal_options = rclcpp_action::Client<MoveRobot>::SendGoalOptions();
        send_goal_options.feedback_callback =
            std::bind(&Tomato3DDetector::feedback_callback, this, std::placeholders::_1, std::placeholders::_2);
        send_goal_options.goal_response_callback =
            std::bind(&Tomato3DDetector::handle_goal_response, this, std::placeholders::_1);
        send_goal_options.result_callback = std::bind(&Tomato3DDetector::handle_result, this, std::placeholders::_1);

        client_->async_send_goal(goal_msg, send_goal_options);
    }

    void feedback_callback(GoalHandleMoveRobot::SharedPtr, const std::shared_ptr<const MoveRobot::Feedback> feedback) {
        RCLCPP_INFO(this->get_logger(), "Feedback: Progress = %.2f%%", feedback->progress * 100.0f);
        if (feedback->progress >= 1.0f) {
            RCLCPP_INFO(this->get_logger(), "*************************************************************************");
        }
    }

    void handle_result(const GoalHandleMoveRobot::WrappedResult &result) {
        // Luôn reset ở cuối bất kể kết quả
        auto reset = [this]() {
            processing_request_ = false;
        };

        if (result.code == rclcpp_action::ResultCode::SUCCEEDED && result.result->success) {
            RCLCPP_INFO(this->get_logger(), "Goal succeeded");
            resent_ = false;
            reset();
            return;
        }

        // Xử lý các trường hợp fail
        RCLCPP_WARN(this->get_logger(), 
            "Goal failed: code=%d message=%s",
            static_cast<int>(result.code),
            result.result->message.c_str());

        if (result.result->message == "ER") {
            RCLCPP_WARN(this->get_logger(), "CHECK: %s", result.result->message.c_str());
        } else if (result.result->message == "RE") {
            resent_ = true;
        }

        // Luôn reset dù fail
        reset();
    }

    void handle_goal_response(GoalHandleMoveRobot::SharedPtr goal_handle) {
        if (!goal_handle) {
            RCLCPP_WARN(this->get_logger(), "Action goal was rejected by server.");
            processing_request_ = false;
            time_publisher(this->now().seconds(), true);
            return;
        }
        //RCLCPP_INFO(this->get_logger(), "Action goal accepted by server, waiting for result...");
    }

    void collectmsg_callback(const collect_msgs::msg::CollectMsg& msg) {
        // chạy xử lý callback trong một luồng riêng
        //std::thread([this, msg]() {
            if (!msg.collect_msg.empty() && !time_recieved_) {
                const auto& time = msg.collect_msg.front();
                //std::lock_guard<std::mutex> lock(mtx_);
                start_detection_time_ = 0.0;
                detection_time_ = 0.0;
                start_positioning_time_ = 0.0;
                start_detection_time_ = time.start_detection;
                detection_time_ = time.detection_time;
                start_positioning_time_ = time.start_positioning_time;
                //start_time_check_ = true;
                //rotate_check_ = time.check;
                time_recieved_ = true; 
                //data_ready_.store(true, std::memory_order_release); 
            }
        //}).detach();  // detach để không block ROS executor
    }

    void time_publisher(double start_time, bool check = false) {
//        while (!data_ready_.load(std::memory_order_acquire) && rclcpp::ok()) {
//            std::this_thread::sleep_for(std::chrono::milliseconds(10));
//        }

        collect_msgs::msg::CollectMsg msg;
        collect_msgs::msg::CollectTime time;
        //time.start_time = start_time;
        
        time.start_detection = start_detection_time_;
        time.detection_time = detection_time_;
        time.positioning_time = start_time - start_positioning_time_;
        time.check = check;
        msg.collect_msg.push_back(time);

        time_pub->publish(msg);

        //rotate_check_ = false;
        //start_time_check_ = false;
    }

    void publish_status() {
        connect_msgs::msg::ConnectMsg msg;
        connect_msgs::msg::ConnectStatus state;
        res_msgs::msg::PoseRes res_msg;
        res_msgs::msg::ResFlag flag;
        state.connection = false;
        state.wait_key = true;
        flag.flag = true;
        msg.connect_msg.push_back(state);
        res_msg.pose_res.push_back(flag);
        connect_msg_pub->publish(msg);
        res_msg_pub->publish(res_msg);
    }

};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<Tomato3DDetector>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
