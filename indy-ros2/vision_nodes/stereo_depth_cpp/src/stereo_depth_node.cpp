#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <stereo_msgs/msg/disparity_image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc.hpp>
#include <message_filters/subscriber.h>
#include <yaml-cpp/yaml.h>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <cassert>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <image_geometry/stereo_camera_model.h>
#include <thread>
#include <mutex>
#include <future>  
#include <filesystem>
#include <condition_variable>
#include <boost/asio.hpp>
#include "res_msgs/msg/pose_res.hpp"
#include "collect_msgs/msg/collect_msg.hpp"
#include "yolov8_msgs/msg/yolov8_inference.hpp"
#include "config_manager/msg/system_config.hpp"
#include <opencv2/core.hpp>
#include <queue>

#ifndef RCUTILS_ASSERT
#define RCUTILS_ASSERT assert
#endif

using namespace std::chrono_literals;
using std::placeholders::_1;
using std::placeholders::_2;
//using ImgConstPtr = sensor_msgs::msg::Image::ConstSharedPtr;
//using StereoPair = std::pair<ImgConstPtr, ImgConstPtr>;
using SyncPolicy = message_filters::sync_policies::ApproximateTime<
    sensor_msgs::msg::Image,
    sensor_msgs::msg::Image>;

struct HSVRange {
    int h_low, h_high;
    int s_low, s_high;
    int v_low, v_high;
};

struct FilterResult {
    cv::Mat gray_filtered;
    cv::Mat color_filtered;
    cv::Mat mask;
};

struct ColorMasks {
    cv::Mat red_mask;
    cv::Mat green_mask;
    cv::Mat combined_mask;
};


class StereoDepthNode : public rclcpp::Node {
public:
    StereoDepthNode() : Node("stereo_depth_node"), pool_(2) {

        cv::setUseOptimized(true);  
        cv::setNumThreads(std::thread::hardware_concurrency()); 

        left_sub.subscribe(this, "/stereo/left/img_for_yolo");
        right_sub.subscribe(this, "/stereo/right/img_for_yolo");

        //using SyncPolicy = message_filters::sync_policies::ExactTime<sensor_msgs::msg::Image, sensor_msgs::msg::Image>;
        //sync = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(SyncPolicy(10), left_sub, right_sub);
        //sync->registerCallback(std::bind(&StereoDepthNode::stereo_callback, this, _1, _2));
        // Trong constructor
        sync = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(
            SyncPolicy(10), left_sub, right_sub);
        sync->registerCallback(std::bind(&StereoDepthNode::stereo_callback, this, _1, _2));

        left_camera_info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
            "/stereo/left/camera_info_calib", rclcpp::QoS(1).transient_local().reliable(),
            std::bind(&StereoDepthNode::left_camera_info_callback, this, _1));

        right_camera_info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
            "/stereo/right/camera_info_calib", rclcpp::QoS(1).transient_local().reliable(),
            std::bind(&StereoDepthNode::right_camera_info_callback, this, _1));

//        left_color_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
//            "/stereo/left/img_for_yolo", 10, std::bind(&StereoDepthNode::left_camera_callback, this, _1));

//        right_color_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
//            "/stereo/right/img_for_yolo", 10, std::bind(&StereoDepthNode::right_camera_callback, this, _1));

        subscription_ = this->create_subscription<res_msgs::msg::PoseRes>(
            "/pose_res", 10, std::bind(&StereoDepthNode::timestamp_callback, this, std::placeholders::_1));

                // Subscribers
        sub_yolo_ = this->create_subscription<yolov8_msgs::msg::Yolov8Inference>(
            "/Yolov8_Inference", 10, std::bind(&StereoDepthNode::yolo_callback, this, _1));

        config_sub_ = this->create_subscription<config_manager::msg::SystemConfig>(
            "/system_config", 
            rclcpp::QoS(1).transient_local().reliable(),
            std::bind(&StereoDepthNode::config_callback, this, std::placeholders::_1)
        );

//        time_sub_ = this->create_subscription<collect_msgs::msg::CollectMsg>(
//            "/collect_msg", 10, std::bind(&StereoDepthNode::collectmsg_callback, this, std::placeholders::_1));

        disparity_pub = create_publisher<stereo_msgs::msg::DisparityImage>("/stereo/disparity", 10);
        //disparity_obs_pub = create_publisher<stereo_msgs::msg::DisparityImage>("/stereo/disparity_obs", 10);

        seg_pub = this->create_publisher<sensor_msgs::msg::Image>("/ref_img", 10);

        obs_seg_pub = this->create_publisher<sensor_msgs::msg::Image>("/obs_img", 10);

        time_pub = this->create_publisher<collect_msgs::msg::CollectMsg>("/collect_msg", 10);

        detect_pub = this->create_publisher<yolov8_msgs::msg::Yolov8Inference>("/detect_msg", 10);

        std::filesystem::path base_path = std::filesystem::current_path(); // sẽ là đường dẫn từ nơi bạn chạy `ros2 run`
        config_path = base_path.string() + "/config/stereo_config.yaml";
        // Chạy ROS 2 trong một thread riêng
        stop_thread_ = false;
        processing_thread_ = std::thread(&StereoDepthNode::process_stereo, this);

        RCLCPP_INFO(this->get_logger(), "✅ OpenCV build version: %s", CV_VERSION);
        RCLCPP_INFO(this->get_logger(), "✅ OpenCV runtime version: %s", cv::getVersionString().c_str());
    }

    ~StereoDepthNode() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_thread_ = true;
        }
        condition_.notify_all();
        if (processing_thread_.joinable()) {
            processing_thread_.join();
        }
    }

private:
    //std::queue<StereoPair> stereo_queue_;
    //const size_t MAX_QUEUE_SIZE = 3;
    std::vector<HSVRange> hsv_color_ranges_;
    std::vector<HSVRange> full_hsv_range_;
    boost::asio::thread_pool pool_;  // Tạo thread pool với 2 thread
    float s = 0.5; // scale factor (0.5 hoặc 0.66 để tăng tốc)
    float strength = 30.0f; // độ mạnh của bias ánh sáng
    int DPP = 16;  // disparities per pixel
    //cv::Mat l_bb_mask_, r_bb_mask_;
    double inv_dpp = 1.0 / DPP;
    double last_move_= 0.0;
    double detect_time_ = 0.0;
    double start_detection_time_ = 0.0;
    //double start_positioning_ = 0.0;
    //int image_width_, image_height_;
    int minDisparity, numDisparities, blockSize, P1, P2, disp12MaxDiff, uniquenessRatio, speckleWindowSize, speckleRange, 
        preFilterCap, preFilterType, preFilterSize, textureThreshold, mode, scaledNumDisp, scaledMinDisp;
    bool stop_thread_ = false;
    bool timer_ = true;
    bool yolo_check_ = false;
    bool allow_image_ = true;
    bool first_run_ = false;
    bool flag_ = false;
    bool config_ready_ = false;
    std::string config_path;
    std::string stereo_method;
    std::string temp_method;
    std::thread processing_thread_;
    std::mutex mutex_;
    std::condition_variable condition_;
    cv::Ptr<cv::StereoMatcher> left_matcher;
    cv::Ptr<cv::StereoMatcher> right_matcher;
    cv::Ptr<cv::ximgproc::DisparityWLSFilter> wls_filter;
    cv::Mat left_img_;
    cv::Mat right_img_;
    cv::Mat kernel_nrg_ = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7));

    std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync;
    message_filters::Subscriber<sensor_msgs::msg::Image> left_sub, right_sub;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr left_camera_info_sub_;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr right_camera_info_sub_;
    rclcpp::Subscription<res_msgs::msg::PoseRes>::SharedPtr subscription_;
    rclcpp::Subscription<yolov8_msgs::msg::Yolov8Inference>::SharedPtr sub_yolo_;
    sensor_msgs::msg::CameraInfo::SharedPtr left_camera_info_;
    sensor_msgs::msg::CameraInfo::SharedPtr right_camera_info_;
    rclcpp::Subscription<config_manager::msg::SystemConfig>::SharedPtr config_sub_;
//    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr left_color_sub_;
//    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr right_color_sub_;
//    rclcpp::Subscription<collect_msgs::msg::CollectMsg>::SharedPtr time_sub_;
    rclcpp::Publisher<stereo_msgs::msg::DisparityImage>::SharedPtr disparity_pub;
    //rclcpp::Publisher<stereo_msgs::msg::DisparityImage>::SharedPtr disparity_obs_pub;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr seg_pub;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr obs_seg_pub;
    rclcpp::Publisher<collect_msgs::msg::CollectMsg>::SharedPtr time_pub;
    rclcpp::Publisher<yolov8_msgs::msg::Yolov8Inference>::SharedPtr detect_pub;
    image_geometry::StereoCameraModel model_;
    sensor_msgs::msg::Image::ConstSharedPtr left_msg_, right_msg_;
    std::optional<std::pair<sensor_msgs::msg::Image::ConstSharedPtr,
                        sensor_msgs::msg::Image::ConstSharedPtr>> latest_stereo_pair_;
    std::optional<yolov8_msgs::msg::Yolov8Inference> latest_yolo_msg_;

    void timestamp_callback(const res_msgs::msg::PoseRes::SharedPtr msg) {    
        if (!msg->pose_res.empty()) {
            RCLCPP_WARN(this->get_logger(), "DEBUG3");
            last_move_ = msg->pose_res[0].x;
            flag_ = msg->pose_res[0].flag;
            if (flag_) {
                allow_image_ = true;  // Re-enable image processing when flag is 1
            }
            RCLCPP_WARN(this->get_logger(), "DEBUG allow_image_ flag: %d", allow_image_);
        }
    }
    
    void yolo_callback(const yolov8_msgs::msg::Yolov8Inference::SharedPtr msg)
    {
        if (msg->yolov8_inference.empty() || !allow_image_) {
            RCLCPP_WARN(this->get_logger(), "No detections in the message.");
            return;
        }

        RCLCPP_INFO(this->get_logger(), "About to publish detect_pub, subscribers: %zu",
            detect_pub->get_subscription_count());
        latest_yolo_msg_ = *msg; 
        //detect_pub->publish(*latest_yolo_msg_);

        //RCLCPP_INFO(this->get_logger(), "detect_pub published OK");
        detect_time_ = rclcpp::Time(msg->header.stamp).seconds();
        yolo_check_ = true;

        // ----- Reset ranges -----
        hsv_color_ranges_.clear();
        full_hsv_range_.clear();

        // ----- RED ranges -----
        HSVRange red1 = {0, 10, 60, 255, 40, 255};     // đỏ đầu
        HSVRange red2 = {160, 180, 60, 255, 40, 255};  // đỏ cuối (wrap)

        // hsv_color_ranges_ → chỉ đỏ
        hsv_color_ranges_.push_back(red1);
        hsv_color_ranges_.push_back(red2);

        // full_hsv_range_ → đỏ + xanh + vàng
        full_hsv_range_.push_back(red1);
        full_hsv_range_.push_back(red2);

        // ----- GREEN -----
        HSVRange green = {35, 85, 50, 255, 50, 255};
        full_hsv_range_.push_back(green);

        // ----- YELLOW / ORANGE -----
        HSVRange yellow = {11, 25, 80, 255, 80, 255};
        full_hsv_range_.push_back(yellow);
    }
/*
    void left_camera_callback(const sensor_msgs::msg::Image::ConstSharedPtr &msg)
    {
        if (allow_image_)
        {
            if (last_move_ == 0.0) {last_move_ = rclcpp::Time(msg->header.stamp).seconds();} 
            try {
                left_img_ = cv::Mat();  // Clear old map
                left_img_ = cv_bridge::toCvCopy(msg, "bgr8")->image;
                image_width_ = left_img_.cols;
                image_height_ = left_img_.rows;
            } catch (cv_bridge::Exception &e) {
                RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
                return;
            }
        }
    }

    void right_camera_callback(const sensor_msgs::msg::Image::ConstSharedPtr &msg)
    {
        if (allow_image_)
        {
            try {
                right_img_ = cv::Mat();  // Clear old map
                right_img_ = cv_bridge::toCvCopy(msg, "bgr8")->image;
            } catch (cv_bridge::Exception &e) {
                RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
                return;
            }
        }
    }
*/    
    void setup() {
        //load_method_params(config_path);
        scaledNumDisp = std::max(16, static_cast<int>(std::round(numDisparities * s)));
        if (scaledNumDisp % 16 != 0)
            scaledNumDisp = ((scaledNumDisp / 16) + 1) * 16;
        scaledMinDisp = std::round(minDisparity * s);

        if (stereo_method == "sgbm") {
            left_matcher = cv::StereoSGBM::create(scaledMinDisp, scaledNumDisp, blockSize, 8*blockSize*blockSize, 32*blockSize*blockSize, disp12MaxDiff, preFilterCap, uniquenessRatio, 
                            speckleWindowSize, speckleRange, mode); // cv::StereoSGBM::MODE_SGBM
        } else if (stereo_method == "bm") {
            // For StereoBM
            left_matcher = cv::StereoBM::create(scaledNumDisp, blockSize);
            auto bm = left_matcher.dynamicCast<cv::StereoBM>();
            // Set parameters using the correct method names
            bm->setPreFilterType(preFilterType);        // This might need to be removed or replaced
            bm->setPreFilterSize(preFilterSize);        // Use setPreFilterSize if available, otherwise remove
            bm->setPreFilterCap(preFilterCap);          // Use setPreFilterCap if available, otherwise remove
            bm->setTextureThreshold(textureThreshold);  // This might not be available in StereoBM
            bm->setUniquenessRatio(uniquenessRatio);
            bm->setSpeckleWindowSize(speckleWindowSize);
            bm->setSpeckleRange(speckleRange);
            bm->setDisp12MaxDiff(disp12MaxDiff);
        }
        wls_filter = cv::ximgproc::createDisparityWLSFilter(left_matcher);
        wls_filter->setLambda(1000);
        wls_filter->setSigmaColor(0.5);
        temp_method = stereo_method;
    }

    void load_method_params(const std::string &filename) {
        YAML::Node config = YAML::LoadFile(filename);
        stereo_method = config["stereo_method"].as<std::string>();
        RCLCPP_INFO(this->get_logger(), "Using stereo method: %s", stereo_method.c_str());
        if (stereo_method == "sgbm") {
            auto sgbm = config["stereo_sgbm"];
            minDisparity = sgbm["minDisparity"].as<int>();
            numDisparities = sgbm["numDisparities"].as<int>();
            blockSize = sgbm["blockSize"].as<int>();
            P1 = sgbm["P1"].as<int>();
            P2 = sgbm["P2"].as<int>();
            disp12MaxDiff = sgbm["disp12MaxDiff"].as<int>();
            uniquenessRatio = sgbm["uniquenessRatio"].as<int>();
            speckleWindowSize = sgbm["speckleWindowSize"].as<int>();
            speckleRange = sgbm["speckleRange"].as<int>();
            preFilterCap = sgbm["preFilterCap"].as<int>();
            mode = sgbm["mode"].as<int>();
            //RCLCPP_INFO(this->get_logger(), "Loaded minDisparity from YAML: %d", minDisparity);
            //RCLCPP_INFO(this->get_logger(), "Loaded numDisparities from YAML: %d", numDisparities);
            //RCLCPP_INFO(this->get_logger(), "Loaded blockSize from YAML: %d", blockSize);
            //RCLCPP_INFO(this->get_logger(), "Loaded P1 from YAML: %d", P1);
            //RCLCPP_INFO(this->get_logger(), "Loaded P2 from YAML: %d", P2);
            //RCLCPP_INFO(this->get_logger(), "Loaded disp12MaxDiff from YAML: %d", disp12MaxDiff);
            //RCLCPP_INFO(this->get_logger(), "Loaded uniquenessRatio from YAML: %d", uniquenessRatio);
            //RCLCPP_INFO(this->get_logger(), "Loaded speckleWindowSize from YAML: %d", speckleWindowSize);
            //RCLCPP_INFO(this->get_logger(), "Loaded speckleRange from YAML: %d", speckleRange);
            //RCLCPP_INFO(this->get_logger(), "Loaded preFilterCap from YAML: %d", preFilterCap);
        } else if (stereo_method == "bm") {
            auto bm = config["stereo_bm"];
            minDisparity = 0;
            numDisparities = bm["numDisparities"].as<int>();
            blockSize = bm["blockSize"].as<int>();
            preFilterType = bm["preFilterType"].as<int>();
            preFilterSize = bm["preFilterSize"].as<int>();
            preFilterCap = bm["preFilterCap"].as<int>();
            textureThreshold = bm["textureThreshold"].as<int>();
            uniquenessRatio = bm["uniquenessRatio"].as<int>();
            speckleWindowSize = bm["speckleWindowSize"].as<int>();
            speckleRange = bm["speckleRange"].as<int>();
            disp12MaxDiff = bm["disp12MaxDiff"].as<int>();
            //RCLCPP_INFO(this->get_logger(), "Loaded numDisparities from YAML: %d", numDisparities);
            //RCLCPP_INFO(this->get_logger(), "Loaded blockSize from YAML: %d", blockSize);
            //RCLCPP_INFO(this->get_logger(), "Loaded preFilterType from YAML: %d", preFilterType);
            //RCLCPP_INFO(this->get_logger(), "Loaded preFilterSize from YAML: %d", preFilterSize);
            //RCLCPP_INFO(this->get_logger(), "Loaded preFilterCap from YAML: %d", preFilterCap);
            //RCLCPP_INFO(this->get_logger(), "Loaded textureThreshold from YAML: %d", textureThreshold);
            //RCLCPP_INFO(this->get_logger(), "Loaded uniquenessRatio from YAML: %d", uniquenessRatio);
            //RCLCPP_INFO(this->get_logger(), "Loaded speckleWindowSize from YAML: %d", speckleWindowSize);
            //RCLCPP_INFO(this->get_logger(), "Loaded speckleRange from YAML: %d", speckleRange);
           // RCLCPP_INFO(this->get_logger(), "Loaded disp12MaxDiff from YAML: %d", disp12MaxDiff);
        } else {
            RCLCPP_ERROR(this->get_logger(), "Unknown stereo method: %s", stereo_method.c_str());
            throw std::runtime_error("Invalid stereo method in YAML");
        }
    }

    void config_callback(const config_manager::msg::SystemConfig::SharedPtr msg)
    {
        config_ready_ = false; // Đặt lại trạng thái config_ready_ trước khi cập nhật
        stereo_method = msg->stereo_method;

        RCLCPP_INFO(this->get_logger(), "Using stereo method: %s", stereo_method.c_str());

        if (stereo_method == "sgbm")
        {
            minDisparity     = msg->sgbm_min_disparity;
            numDisparities   = msg->sgbm_num_disparities;
            blockSize        = msg->sgbm_block_size;
            P1               = msg->sgbm_p1;
            P2               = msg->sgbm_p2;
            disp12MaxDiff    = msg->sgbm_disp12maxdiff;
            uniquenessRatio  = msg->sgbm_uniquenessratio;
            speckleWindowSize= msg->sgbm_specklewindowsize;
            speckleRange     = msg->sgbm_specklerange;
            preFilterCap     = msg->sgbm_prefiltercap;
            mode             = msg->sgbm_mode;
        }
        else if (stereo_method == "bm")
        {
            minDisparity     = 0;
            numDisparities   = msg->bm_num_disparities;
            blockSize        = msg->bm_block_size;
            preFilterType    = msg->bm_prefiltertype;
            preFilterSize    = msg->bm_prefiltersize;
            preFilterCap     = msg->bm_prefiltercap;
            textureThreshold = msg->bm_texturethreshold;
            uniquenessRatio  = msg->bm_uniquenessratio;
            speckleWindowSize= msg->bm_specklewindowsize;
            speckleRange     = msg->bm_specklerange;
            disp12MaxDiff    = msg->bm_disp12maxdiff;
        }
        else
        {
            RCLCPP_ERROR(this->get_logger(), "Unknown stereo method: %s", stereo_method.c_str());
        }
        setup();
        config_ready_ = true;
    }

    void left_camera_info_callback(const sensor_msgs::msg::CameraInfo::SharedPtr msg) {
        left_camera_info_ = std::make_shared<sensor_msgs::msg::CameraInfo>(*msg);
        update_camera_model();
    }

    void right_camera_info_callback(const sensor_msgs::msg::CameraInfo::SharedPtr msg) {
        right_camera_info_ = std::make_shared<sensor_msgs::msg::CameraInfo>(*msg);
        update_camera_model();
    }

    void update_camera_model() {
        if (left_camera_info_ && right_camera_info_) {
            model_.fromCameraInfo(*left_camera_info_, *right_camera_info_);
           //RCLCPP_INFO(this->get_logger(), "Stereo camera model updated!");
        }
    }

    FilterResult filter_gray_by_colors(
        const cv::Mat &color_img,
        const cv::Mat &gray_img,
        const std::vector<HSVRange> &ranges)
    {
        CV_Assert(color_img.type() == CV_8UC3);
        CV_Assert(gray_img.type() == CV_8UC1);

        cv::Mat color_resized = color_img;
        if (color_img.size() != gray_img.size()) {
            cv::resize(color_resized, color_resized, gray_img.size());
        }

        cv::Mat hsv;
        cv::cvtColor(color_resized, hsv, cv::COLOR_BGR2HSV);

        cv::Mat mask = cv::Mat::zeros(gray_img.size(), CV_8UC1);

        for (const auto &r : ranges)
        {
            cv::Mat temp;
            cv::inRange(hsv,
                        cv::Scalar(r.h_low, r.s_low, r.v_low),
                        cv::Scalar(r.h_high, r.s_high, r.v_high),
                        temp);
            mask |= temp;
        }

        cv::Mat gray_filtered = cv::Mat::zeros(gray_img.size(), CV_8UC1);
        gray_img.copyTo(gray_filtered, mask);

        cv::Mat color_filtered = cv::Mat::zeros(color_resized.size(), CV_8UC3);
        color_img.copyTo(color_filtered, mask);

        return {gray_filtered, color_filtered, mask};
    }

    ColorMasks extractRedGreenNRG(const cv::Mat& bgr)
    {
        CV_Assert(bgr.type() == CV_8UC3);

        // ── 1. Tách channel CV_8U — không cần convert toàn ảnh sang float ──
        cv::Mat ch[3];
        cv::split(bgr, ch);  // ch[0]=B, ch[1]=G, ch[2]=R

        // Convert từng channel đơn (3x nhỏ hơn convert cả ảnh 3ch)
        cv::Mat Bf, Gf, Rf;
        ch[0].convertTo(Bf, CV_32F);
        ch[1].convertTo(Gf, CV_32F);
        ch[2].convertTo(Rf, CV_32F);

        // ── 2. Tính denom + NRG một lần ──
        cv::Mat denom, nrg;
        cv::add(Rf, Gf, denom);
        denom += 1e-6f;
        cv::subtract(Rf, Gf, nrg);
        cv::divide(nrg, denom, nrg);  // nrg = (R-G)/(R+G+eps), range [-1, 1]

        // ── 3. Scale về [0,255] — dùng convertTo thay vì biểu thức Mat ──
        // nrg_norm = (nrg + 1) / 2 * 255  →  convertTo với alpha=127.5, beta=127.5
        cv::Mat nrg_norm, ngr_norm;
        nrg.convertTo(nrg_norm, CV_8U, 127.5, 127.5);   // red: nrg > 0 → R > G
        nrg.convertTo(ngr_norm, CV_8U, -127.5, 127.5);  // green: flip dấu, không tính lại

        // ── 4. Threshold + condition song song ──
        cv::Mat red_mask, green_mask;

        // Precompute compare masks — dùng chung cho cả red và green
        cv::Mat R_gt_G, R_gt_B, G_gt_R, G_gt_B;
        cv::compare(ch[2], ch[1], R_gt_G, cv::CMP_GT);  // R > G
        cv::compare(ch[2], ch[0], R_gt_B, cv::CMP_GT);  // R > B
        cv::compare(ch[1], ch[2], G_gt_R, cv::CMP_GT);  // G > R
        cv::compare(ch[1], ch[0], G_gt_B, cv::CMP_GT);  // G > B

        auto process_mask = [&](
            const cv::Mat& norm,
            const cv::Mat& cond1,
            const cv::Mat& cond2,
            cv::Mat& out_mask)
        {
            cv::threshold(norm, out_mask, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

            // Apply color condition
            cv::Mat condition;
            cv::bitwise_and(cond1, cond2, condition);
            cv::bitwise_and(out_mask, condition, out_mask);

            // Morphology + fill contours
            cv::morphologyEx(out_mask, out_mask, cv::MORPH_CLOSE, kernel_nrg_);
            std::vector<std::vector<cv::Point>> contours;
            cv::findContours(out_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
            if (!contours.empty())
                cv::drawContours(out_mask, contours, -1, 255, cv::FILLED);
        };

        // ── 5. Red + Green song song ──
        auto fut_red = std::async(std::launch::async, [&]{
            process_mask(nrg_norm, R_gt_G, R_gt_B, red_mask);
        });
        auto fut_green = std::async(std::launch::async, [&]{
            process_mask(ngr_norm, G_gt_R, G_gt_B, green_mask);
        });
        fut_red.get();
        fut_green.get();

        // ── 6. Combine ──
        cv::Mat combined;
        cv::bitwise_or(red_mask, green_mask, combined);

        return {red_mask, green_mask, combined};
    }

    void addGradientChannel(const cv::Mat& gray, cv::Mat& out)
    {
        //cv::Mat gray;

        // Gradient theo X (hướng disparity)
        cv::Mat gradx;
        cv::Sobel(gray, gradx, CV_32F, 1, 0, 3);

        cv::Mat gradx_abs;
        cv::convertScaleAbs(gradx, gradx_abs);

        // Convert sang float
        cv::Mat f_gray, f_grad;
        gray.convertTo(f_gray, CV_32F);
        gradx_abs.convertTo(f_grad, CV_32F);

        // Blend
        cv::Mat result = 0.7f * f_gray + 0.3f * f_grad;

        cv::normalize(result, result, 0, 255, cv::NORM_MINMAX);
        result.convertTo(out, CV_8U);
    }

    cv::Mat cleanMaskMorphology(const cv::Mat& mask)
    {
        cv::Mat clean;

        // kernel ellipse
        cv::Mat kernel_close = cv::getStructuringElement(cv::MORPH_ELLIPSE, {9,9});
        cv::Mat kernel_open  = cv::getStructuringElement(cv::MORPH_ELLIPSE, {7,7});

        // đóng lỗ
        cv::morphologyEx(mask, clean, cv::MORPH_CLOSE, kernel_close);

        // xoá nhiễu nhỏ
        cv::morphologyEx(clean, clean, cv::MORPH_OPEN, kernel_open);

        return clean;
    }

    cv::Mat cleanTomatoMask(const cv::Mat& mask)
    {
        cv::Mat clean = cleanMaskMorphology(mask);
        clean = filterMaskByArea(clean, 300);
        return clean;
    }

    cv::Mat filterMaskByArea(const cv::Mat& mask, int min_area = 600)
    {
        cv::Mat labels, stats, centroids;

        int n = cv::connectedComponentsWithStats(mask, labels, stats, centroids);

        cv::Mat filtered = cv::Mat::zeros(mask.size(), CV_8U);

        for (int i = 1; i < n; i++)
        {
            int area = stats.at<int>(i, cv::CC_STAT_AREA);

            if (area > min_area)
            {
                filtered.setTo(255, labels == i);
            }
        }

        return filtered;
    }

    void stereo_callback(
        const sensor_msgs::msg::Image::ConstSharedPtr& left_msg, 
        const sensor_msgs::msg::Image::ConstSharedPtr& right_msg) 
    {
        if (last_move_ == 0.0) {last_move_ = this->now().seconds();} 
        if (!allow_image_) return;

        std::lock_guard<std::mutex> lock(mutex_);
        latest_stereo_pair_ = std::make_pair(left_msg, right_msg);
        condition_.notify_one();
    }

    void process_stereo() {
        while (!stop_thread_ && rclcpp::ok()) {
            std::unique_lock<std::mutex> lock(mutex_);
            condition_.wait(lock, [this] {
                return latest_stereo_pair_.has_value() || stop_thread_;
            });

            if (stop_thread_) break;

            auto [left_msg, right_msg] = *latest_stereo_pair_;
            latest_stereo_pair_.reset();
            lock.unlock();

            RCLCPP_INFO(this->get_logger(), "DEBUG222222222222222222222222");
            RCLCPP_WARN(this->get_logger(), "DEBUG detect_time_: %f, last_move_: %f, flag_: %d, yolo_check_: %d, left_msg: %f, config_received_: %d", detect_time_, last_move_, flag_, yolo_check_, rclcpp::Time(left_msg->header.stamp).seconds(), config_ready_);
            if (detect_time_ < last_move_ || flag_ || !yolo_check_ || !config_ready_ || rclcpp::Time(left_msg->header.stamp).seconds() < last_move_) {
                continue;
            }

//            if (first_run_) {
//                first_run_ = false;
 //               continue;
 //           }

            // Khởi tạo last_move_ nếu chưa có giá trị
            start_detection_time_ = last_move_;
            //start_positioning_ = rclcpp::Time(left_msg->header.stamp).seconds();
            last_move_ = 0.0;
            allow_image_ = false;  // Disable further image processing until re-enabled
            yolo_check_ = false;  // Reset yolo_check after processing
            //setup();

            RCLCPP_INFO(this->get_logger(), "DEBUG0000000000000000000000000000000000000000");

            // 1. Decode ảnh
            auto left_cv_ptr  = cv_bridge::toCvShare(left_msg, "bgr8");
            auto right_cv_ptr = cv_bridge::toCvShare(right_msg, "bgr8");
            const cv::Mat& left_img_  = left_cv_ptr->image;
            const cv::Mat& right_img_ = right_cv_ptr->image;

            // 2. Song song: gray + NRG mask
            cv::Mat left_full, right_full;
            auto fut_gray_l = std::async(std::launch::async,
                [&]{ cv::Mat g; cv::cvtColor(left_img_,  g, cv::COLOR_BGR2GRAY); return g; });
            auto fut_gray_r = std::async(std::launch::async,
                [&]{ cv::Mat g; cv::cvtColor(right_img_, g, cv::COLOR_BGR2GRAY); return g; });
            auto fut_mask_l = std::async(std::launch::async,
                [&]{ return extractRedGreenNRG(left_img_); });
            auto fut_mask_r = std::async(std::launch::async,
                [&]{ return extractRedGreenNRG(right_img_); });

            left_full        = fut_gray_l.get();
            right_full       = fut_gray_r.get();
            ColorMasks left_masks  = fut_mask_l.get();
            ColorMasks right_masks = fut_mask_r.get();

            // 3. Apply mask — dùng bitwise_and
            cv::Mat left_clean, right_clean;
            cv::bitwise_and(left_full,  left_masks.combined_mask,  left_clean);
            cv::bitwise_and(right_full, right_masks.combined_mask, right_clean);
            cv::Mat left_guide_clean = cv::Mat::zeros(left_img_.size(), CV_8UC3);
            left_img_.copyTo(left_guide_clean, left_masks.combined_mask);

            RCLCPP_INFO(this->get_logger(), "DEBUG999999999999999999999999999999999999");

            cv::Mat left_gray, right_gray;
            addGradientChannel(left_clean, left_gray);
            addGradientChannel(right_clean, right_gray);
            cv::Mat obj_img = cv::Mat::zeros(left_gray.size(), CV_8UC1);
            left_gray.copyTo(obj_img, left_masks.red_mask);
            seg_pub->publish(*cv_bridge::CvImage(left_msg->header, sensor_msgs::image_encodings::MONO8, obj_img).toImageMsg());
            obs_seg_pub->publish(*cv_bridge::CvImage(left_msg->header, sensor_msgs::image_encodings::MONO8, left_masks.green_mask).toImageMsg());


            //cv::imwrite("left_gray.png", left_gray);
            //cv::imwrite("right_gray.png", right_gray);

            // Downsample
            cv::Mat left_small, right_small;
            cv::resize(left_gray, left_small, cv::Size(), s, s, cv::INTER_LINEAR_EXACT);
            cv::resize(right_gray, right_small, cv::Size(), s, s, cv::INTER_LINEAR_EXACT);

            // Compute disparity
            cv::Mat left_disp_small, right_disp_small, right_disp_flip;
/*
            cv::setNumThreads(1);

            right_matcher = cv::ximgproc::createRightMatcher(left_matcher);

            std::thread t1([&](){
                left_matcher->compute(left_small, right_small, left_disp_small);
                RCLCPP_INFO(this->get_logger(), "DEBUGLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL");
                
            });

            std::thread t2([&](){
                right_matcher->compute(right_small, left_small, right_disp_small);
                RCLCPP_INFO(this->get_logger(), "DEBUGRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRR");
            });

            t1.join();
            t2.join();
*/
            left_matcher->compute(left_small, right_small, left_disp_small);
            RCLCPP_INFO(this->get_logger(), "DEBUGLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL");
            right_matcher = cv::ximgproc::createRightMatcher(left_matcher);
            right_matcher->compute(right_small, left_small, right_disp_small);
            RCLCPP_INFO(this->get_logger(), "DEBUGRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRR");
            left_disp_small.setTo(0, left_disp_small < 0);

            RCLCPP_WARN(this->get_logger(), "6666666666666666666666666666666666666666666666666666666666");

            // ===== DEBUG 1: Kiểm tra raw disparity =====
            double minVal, maxVal;
            cv::minMaxLoc(left_disp_small, &minVal, &maxVal);
            //RCLCPP_INFO(this->get_logger(), "Raw disparity: min=%f, max=%f", minVal, maxVal);
            
            if (maxVal == 0) {
                RCLCPP_ERROR(this->get_logger(), "❌ Raw disparity toàn 0! Kiểm tra stereo matcher!");
                allow_image_ = true; 
                return;
            }

            // WLS filter
            cv::Mat disparity;
            wls_filter->filter(left_disp_small, left_guide_clean, disparity, right_disp_small);
            RCLCPP_INFO(this->get_logger(), "Disparity image size: %d x %d", disparity.cols, disparity.rows);
            cv::Mat disp_masked = cv::Mat::zeros(disparity.size(), CV_16S);
            disparity.copyTo(disp_masked, left_masks.combined_mask);
            //cv::Mat disp_obs = cv::Mat::zeros(disparity.size(), CV_16S);
            //disparity.copyTo(disp_obs, result_mask);
            RCLCPP_INFO(this->get_logger(), "Disparity image size after masking: %d x %d", disp_masked.cols, disp_masked.rows);

            cv::Mat confidence_map = wls_filter->getConfidenceMap(); // Lấy confidence map nếu cần
            //imwrite("confidence.png", confidence_map);

            // left_disp_small: kiểu CV_16S hoặc CV_32F, disparity SGBM scale = 16

            cv::Mat raw_dispL_vis, raw_dispR_vis;
            // Tạo ảnh hiển thị từ raw disparity
            cv::ximgproc::getDisparityVis(left_disp_small, raw_dispL_vis, 1.0);
            cv::ximgproc::getDisparityVis(right_disp_small, raw_dispR_vis, -1.0);

            // Lưu ảnh raw disparity visualization
            //cv::imwrite("raw-dispL-test.png", raw_dispL_vis);
            //cv::imwrite("raw-dispR-test.png", raw_dispR_vis);

           // cv::minMaxLoc(disparity, &minVal, &maxVal);
            //RCLCPP_INFO(this->get_logger(), "Final disparity: min=%f, max=%f, type: %d", minVal, maxVal, disparity.type());


            // -> disp_visual sẵn sàng hiển thị hoặc pub ra ROS topic
            RCLCPP_WARN(this->get_logger(), "DEBUGDEBUG777777777777777777777777");

            auto disp_msg = stereo_msgs::msg::DisparityImage();
            disp_msg.header.stamp = rclcpp::Time(static_cast<uint64_t>(this->now().seconds() * 1e9));;
            disp_msg.header.frame_id = left_camera_info_->header.frame_id;
            sensor_msgs::msg::Image & dimage = disp_msg.image;
            dimage.height = disp_masked.rows;
            dimage.width = disp_masked.cols;
            dimage.encoding = sensor_msgs::image_encodings::TYPE_32FC1;
            dimage.step = dimage.width * sizeof(float);
            dimage.data.resize(dimage.step * dimage.height);
            cv::Mat_<float> dmat(
                dimage.height, dimage.width, reinterpret_cast<float *>(&dimage.data[0]), dimage.step);
            disp_masked.convertTo(dmat, dmat.type(), inv_dpp, -(model_.left().cx() - model_.right().cx()));
            RCUTILS_ASSERT(dmat.data == &dimage.data[0]);

            //double minVal, maxVal;
            cv::minMaxLoc(dmat, &minVal, &maxVal);
            RCLCPP_INFO(this->get_logger(), "Disparity: min = %f, max = %f", minVal, maxVal);

            int border = blockSize / 2;
            int left = numDisparities + minDisparity + border - 1;
            int wtf;
            if (minDisparity >= 0) {
                wtf = border + minDisparity;
            } else {
                wtf = std::max(border, minDisparity);
            }

            int right = dimage.width - 1 - wtf;
            int top = border;
            int bottom = dimage.height - 1 - border;
            disp_msg.valid_window.x_offset = left;
            disp_msg.valid_window.y_offset = top;
            disp_msg.valid_window.width = right - left;
            disp_msg.valid_window.height = bottom - top;

            disp_msg.f = model_.right().fx();
            disp_msg.t = model_.baseline();
            disp_msg.min_disparity = minDisparity;
            disp_msg.max_disparity = minDisparity + numDisparities - 1;
            disp_msg.delta_d = inv_dpp;

            time_publisher();
            disparity_pub->publish(disp_msg);
            detect_pub->publish(*latest_yolo_msg_);

            RCLCPP_INFO(this->get_logger(), "detect_pub published OK");
        }
    }
/*
    void collectmsg_callback(const collect_msgs::msg::CollectMsg& msg) {
        if (!msg.collect_msg.empty()) {
            const auto& time = msg.collect_msg.front();
            timer_ = time.check;
        }
    }
*/
    void time_publisher() {
//        if (timer_) {
        collect_msgs::msg::CollectMsg msg;
        collect_msgs::msg::CollectTime time;
        time.start_detection = start_detection_time_;
        time.detection_time = detect_time_ - start_detection_time_;
        time.start_positioning_time = detect_time_;
        time.check = false;
        msg.collect_msg.push_back(time);
        time_pub->publish(msg);
        timer_ = false;  // Reset timer after publishing
//        }

    }
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<StereoDepthNode>();

    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(node);
    executor.spin();

    rclcpp::shutdown();
    return 0;
}
