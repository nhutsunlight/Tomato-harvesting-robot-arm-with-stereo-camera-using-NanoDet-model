#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <stereo_msgs/msg/disparity_image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc.hpp>
#include <yaml-cpp/yaml.h>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <cassert>
#include <image_geometry/stereo_camera_model.h>
#include <thread>
#include <mutex>
#include <atomic>
#include <filesystem>
#include <condition_variable>
#include <array>
#include "res_msgs/msg/pose_res.hpp"
#include "collect_msgs/msg/collect_msg.hpp"
#include "yolov8_msgs/msg/yolov8_inference.hpp"
#include "config_manager/msg/system_config.hpp"
#include "connect_msgs/msg/connect_msg.hpp"
#include "depth_signal_msgs/msg/depth_signal.hpp"
#include "position_signal_msgs/msg/position_signal.hpp"
#include "skip_signal_msgs/msg/skip_signal.hpp"
#include "stereo_depth_cpp/dual_stereo_matcher.hpp"

#ifndef RCUTILS_ASSERT
#define RCUTILS_ASSERT assert
#endif

using std::placeholders::_1;

struct ColorDef {
    cv::Vec3f lab;
    cv::Vec3b bgr;
    int id;
};

class StereoDepthNode : public rclcpp::Node {
public:
    StereoDepthNode() : Node("stereo_depth_node") {
        cv::setUseOptimized(true);

        left_sub_ = create_subscription<sensor_msgs::msg::Image>(
            "/stereo/left/img_for_yolo", 10,
            std::bind(&StereoDepthNode::left_callback, this, _1));

        right_sub_ = create_subscription<sensor_msgs::msg::Image>(
            "/stereo/right/img_for_yolo", 10,
            std::bind(&StereoDepthNode::right_callback, this, _1));

        left_camera_info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
            "/stereo/left/camera_info_calib",
            rclcpp::QoS(1).transient_local().reliable(),
            std::bind(&StereoDepthNode::left_camera_info_callback, this, _1));

        right_camera_info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
            "/stereo/right/camera_info_calib",
            rclcpp::QoS(1).transient_local().reliable(),
            std::bind(&StereoDepthNode::right_camera_info_callback, this, _1));

        subscription_ = create_subscription<res_msgs::msg::PoseRes>(
            "/pose_res", 10,
            std::bind(&StereoDepthNode::timestamp_callback, this, _1));

        sub_yolo_ = create_subscription<yolov8_msgs::msg::Yolov8Inference>(
            "/Yolov8_Inference", 10,
            std::bind(&StereoDepthNode::yolo_callback, this, _1));

        connect_sub_ = create_subscription<connect_msgs::msg::ConnectMsg>(
            "/connect1_msg", 10,
            std::bind(&StereoDepthNode::connect_callback, this, _1));

        config_sub_ = create_subscription<config_manager::msg::SystemConfig>(
            "/system_config",
            rclcpp::QoS(1).transient_local().reliable(),
            std::bind(&StereoDepthNode::config_callback, this, _1));

        skip_signal_sub_ = create_subscription<skip_signal_msgs::msg::SkipSignal>(
            "/skip_signal", 10,
            std::bind(&StereoDepthNode::skip_signal_callback, this, _1));

        disparity_pub = create_publisher<stereo_msgs::msg::DisparityImage>("/stereo/disparity", 10);
        seg_pub       = create_publisher<sensor_msgs::msg::Image>("/ref_img", 10);
        time_pub      = create_publisher<collect_msgs::msg::CollectMsg>("/collect_msg", 10);
        detect_pub    = create_publisher<yolov8_msgs::msg::Yolov8Inference>("/detect_msg", 10);
        depth_signal_pub = create_publisher<depth_signal_msgs::msg::DepthSignal>("/depth_signal", 10);
        position_signal_pub = create_publisher<position_signal_msgs::msg::PositionSignal>("/position_signal", 10);

        config_path        = std::filesystem::current_path().string() + "/config/stereo_config.yaml";
        processing_thread_ = std::thread(&StereoDepthNode::process_stereo, this);

        RCLCPP_INFO(get_logger(), "✅ OpenCV %s", CV_VERSION);
    }

    ~StereoDepthNode() {
        stop_thread_.store(true);
        condition_.notify_all();
        if (processing_thread_.joinable()) processing_thread_.join();
    }

private:
    float  s          = 0.5f;
    int    DPP        = 16;
    double inv_dpp    = 1.0 / 16;
    double wls_sigma  = 0.5;
    int    wls_lambda = 1000;
    int    minDisparity = 0, numDisparities = 64, blockSize = 5;
    int    P1 = 0, P2 = 0, disp12MaxDiff = 0, uniquenessRatio = 0;
    int    speckleWindowSize = 0, speckleRange = 0;
    int    preFilterCap = 0, preFilterType = 0, preFilterSize = 0;
    int    textureThreshold = 0, mode = 0;
    int    scaledNumDisp = 0, scaledMinDisp = 0;
    double minVal, maxVal;

    // ===== Atomic flags =====
    std::atomic<bool>   stop_thread_{false};
    std::atomic<bool>   allow_image_{true};
    std::atomic<bool>   flag_{false};
    std::atomic<bool>   yolo_check_{false};
    std::atomic<bool>   config_ready_{false};
    std::atomic<bool>   skip_disparity_{false};
    std::atomic<bool>   first_run_{false};
    std::atomic<bool>   multi_collect_mode{false};
    std::atomic<bool>   harvest_flag_{false};
    std::atomic<double> last_move_{0.0};
    std::atomic<double> detect_time_{0.0};

    bool        timer_ = true;
    double      start_detection_time_ = 0.0;
    std::string config_path, stereo_method, temp_method;

    // ===== Threading =====
    std::thread             processing_thread_;
    std::mutex              proc_mutex_;
    std::condition_variable condition_;

    ///cv::Ptr<cv::StereoMatcher>                left_matcher, right_matcher;
    ///cv::Ptr<cv::ximgproc::DisparityWLSFilter> wls_filter;
    DualStereoMatcher dual_matcher_;
    //cv::Ptr<cv::ximgproc::DisparityWLSFilter> wls_filter_obj_, wls_filter_obs_;
    cv::Ptr<cv::CLAHE>                        clahe_;

    // ===== Kernels =====
    cv::Mat kernel_close_ = cv::getStructuringElement(cv::MORPH_ELLIPSE, {7, 7});
    cv::Mat kernel_open_  = cv::getStructuringElement(cv::MORPH_ELLIPSE, {5, 5});

    // ===== Reusable image buffers =====
    cv::Mat left_full_,       right_full_;
    cv::Mat left_guide_clean_;
    cv::Mat left_mask_combined;
    cv::Mat left_small_,      right_small_;
    cv::Mat left_disp_small_, right_disp_small_;
    cv::Mat left_mask_green, left_mask_roy_only;
    cv::Mat right_mask_green, right_mask_roy_only;
    cv::Mat left_clean_roy,  left_clean_green;
    cv::Mat right_clean_roy, right_clean_green;
    cv::Mat left_gray_roy;  
    //left_gray_green;
    //cv::Mat right_gray_roy, right_gray_green;
    cv::Mat disparity_,       disp_masked_;
    cv::Mat obs_disp_left, obs_disp_right;
    cv::Mat obj_disp_left, obj_disp_right;
    cv::Mat obs_gray_left_small, obs_gray_right_small;
    cv::Mat obj_gray_left_small, obj_gray_right_small;
    cv::Mat obj_disp_masked, obs_disp_masked;
    cv::Mat obj_disp, obs_disp;

    std::vector<uint8_t> disparity_buffer_;
    cv::Mat              disparity_float_;
    uint32_t             buffer_width_  = 0;
    uint32_t             buffer_height_ = 0;

    // ===== Color LUT =====
    cv::Mat lab_buf_, label_buf_;
    // [OPT-1] Flatten 3D LUT → 1D với key = R*65536 + G*256 + B
    // Tránh pointer chasing 3 lần, cache-friendly hơn
    std::vector<uint8_t> color_id_lut_;  // size 256*256*256
    bool    lut_built_ = false;

    // ── Constants tiện dùng ──────────────────────────────────────────────────────
    static constexpr uint16_t COLOR_RED    = 1 << 0;
    static constexpr uint16_t COLOR_ORANGE = 1 << 1;
    static constexpr uint16_t COLOR_YELLOW = 1 << 2;
    static constexpr uint16_t COLOR_GREEN  = 1 << 3;
    static constexpr uint16_t COLOR_BLUE   = 1 << 4;
    static constexpr uint16_t COLOR_PURPLE = 1 << 5;
    static constexpr uint16_t COLOR_PINK   = 1 << 6;
    static constexpr uint16_t COLOR_WHITE  = 1 << 7;
    static constexpr uint16_t COLOR_BLACK  = 1 << 8;
    static constexpr uint16_t COLOR_ROY    = COLOR_RED | COLOR_ORANGE | COLOR_YELLOW;
    static constexpr uint16_t COLOR_ALL_WARM = COLOR_ROY | COLOR_GREEN;

    // ===== ROS =====
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr           left_sub_, right_sub_;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr      left_camera_info_sub_, right_camera_info_sub_;
    rclcpp::Subscription<res_msgs::msg::PoseRes>::SharedPtr            subscription_;
    rclcpp::Subscription<yolov8_msgs::msg::Yolov8Inference>::SharedPtr sub_yolo_;
    rclcpp::Subscription<config_manager::msg::SystemConfig>::SharedPtr config_sub_;
    rclcpp::Subscription<connect_msgs::msg::ConnectMsg>::SharedPtr connect_sub_;
    rclcpp::Subscription<skip_signal_msgs::msg::SkipSignal>::SharedPtr skip_signal_sub_;

    sensor_msgs::msg::CameraInfo::SharedPtr left_camera_info_, right_camera_info_;

    rclcpp::Publisher<stereo_msgs::msg::DisparityImage>::SharedPtr  disparity_pub;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr           seg_pub;
    rclcpp::Publisher<collect_msgs::msg::CollectMsg>::SharedPtr     time_pub;
    rclcpp::Publisher<yolov8_msgs::msg::Yolov8Inference>::SharedPtr detect_pub;
    rclcpp::Publisher<depth_signal_msgs::msg::DepthSignal>::SharedPtr depth_signal_pub;
    rclcpp::Publisher<position_signal_msgs::msg::PositionSignal>::SharedPtr position_signal_pub;

    image_geometry::StereoCameraModel model_;

    sensor_msgs::msg::Image::ConstSharedPtr last_left_msg_, last_right_msg_;
    yolov8_msgs::msg::Yolov8Inference       latest_yolo_msg_;
    bool                                    yolo_msg_valid_ = false;

    std::optional<std::pair<sensor_msgs::msg::Image::ConstSharedPtr,
                            sensor_msgs::msg::Image::ConstSharedPtr>> latest_stereo_pair_;

    // =========================================================
    void left_callback(const sensor_msgs::msg::Image::ConstSharedPtr msg) {
        if (last_move_.load() == 0.0) last_move_.store(now().seconds());
        RCLCPP_INFO(get_logger(), "Left image received");
        if (!allow_image_.load(std::memory_order_relaxed) || !harvest_flag_.load()) return;

        last_left_msg_ = msg;
        if (!last_right_msg_) return;

        {
            std::lock_guard<std::mutex> lock(proc_mutex_);
            latest_stereo_pair_ = std::make_pair(last_left_msg_, last_right_msg_);
            last_left_msg_  = nullptr;
            last_right_msg_ = nullptr;
        }
        RCLCPP_INFO(get_logger(), "LEFT LEFT LEFT LEFT LEFT LEFT LEFT LEFT LEFT LEFT LEFT");
        condition_.notify_one();
    }

    void right_callback(const sensor_msgs::msg::Image::ConstSharedPtr msg) {
        RCLCPP_INFO(get_logger(), "Right image received");
        if (!allow_image_.load(std::memory_order_relaxed) || !harvest_flag_.load()) return;

        last_right_msg_ = msg;
        if (!last_left_msg_) return;

        {
            std::lock_guard<std::mutex> lock(proc_mutex_);
            latest_stereo_pair_ = std::make_pair(last_left_msg_, last_right_msg_);
            last_left_msg_  = nullptr;
            last_right_msg_ = nullptr;
        }
        RCLCPP_INFO(get_logger(), "RIGHT RIGH RIGHT RIGH RIGHT RIGH RIGHT RIGH RIGHT RIGH RIGHT RIGH RIGHT");
        condition_.notify_one();
    }

    void skip_signal_callback(const skip_signal_msgs::msg::SkipSignal::SharedPtr msg) {
        if (!msg->skip) return;
        allow_image_.store(true);
        RCLCPP_INFO(get_logger(), "Skip signal received. Reset allow_image_.");
    }

    void timestamp_callback(const res_msgs::msg::PoseRes::SharedPtr msg) {
        if (msg->pose_res.empty()) return;
        last_move_.store(msg->pose_res[0].x);
        flag_.store(msg->pose_res[0].flag);
        skip_disparity_.store(msg->pose_res[0].skip);
        if (!msg->pose_res[0].skip) first_run_.store(true);
        if (msg->pose_res[0].flag)  allow_image_.store(true);
    }

    void connect_callback(const connect_msgs::msg::ConnectMsg::SharedPtr msg) {
        if (msg->connect_msg.empty()) return;
        harvest_flag_.store(msg->connect_msg[0].harvest_flag);
    }

    void yolo_callback(const yolov8_msgs::msg::Yolov8Inference::SharedPtr msg) {
        if (msg->yolov8_inference.empty() || !allow_image_.load() || !harvest_flag_.load()) return;
        latest_yolo_msg_ = *msg;
        yolo_msg_valid_  = true;
        detect_time_.store(rclcpp::Time(msg->header.stamp).seconds());
        yolo_check_.store(true);
        RCLCPP_INFO(get_logger(), "DETECT DETECT DETECT DETECT DETECT DETECT DETECT DETECT DETECT");
    }

    // =========================================================
    void setup() {
        scaledNumDisp = std::max(16, (int)std::round(numDisparities * s));
        if (scaledNumDisp % 16 != 0)
            scaledNumDisp = (scaledNumDisp / 16 + 1) * 16;
        scaledMinDisp = (int)std::round(minDisparity * s);
/*
        if (stereo_method == "sgbm") {
            left_matcher = cv::StereoSGBM::create(
                scaledMinDisp, scaledNumDisp, blockSize,
                8*blockSize*blockSize, 32*blockSize*blockSize,
                disp12MaxDiff, preFilterCap, uniquenessRatio,
                speckleWindowSize, speckleRange, mode);
        } else if (stereo_method == "bm") {
            left_matcher = cv::StereoBM::create(scaledNumDisp, blockSize);
            auto bm = left_matcher.dynamicCast<cv::StereoBM>();
            bm->setPreFilterType(preFilterType);
            bm->setPreFilterSize(preFilterSize);
            bm->setPreFilterCap(preFilterCap);
            bm->setTextureThreshold(textureThreshold);
            bm->setUniquenessRatio(uniquenessRatio);
            bm->setSpeckleWindowSize(speckleWindowSize);
            bm->setSpeckleRange(speckleRange);
            bm->setDisp12MaxDiff(disp12MaxDiff);
        }
        right_matcher = cv::ximgproc::createRightMatcher(left_matcher);
        wls_filter    = cv::ximgproc::createDisparityWLSFilter(left_matcher);
        wls_filter->setLambda(wls_lambda);
        wls_filter->setSigmaColor(wls_sigma);
*/
        if (stereo_method == "bm") {
            dual_matcher_.init_bm(
                /*numDisp*/        scaledNumDisp,
                /*blkSize*/        blockSize,
                /*minDisp*/        scaledMinDisp,
                /*preFilterType*/  preFilterType,
                /*preFilterSize*/  preFilterSize,
                /*preFilterCap*/   preFilterCap,
                /*textureThreshold*/textureThreshold,
                /*uniquenessRatio*/uniquenessRatio,
                /*speckleWindow*/  speckleWindowSize,
                /*speckleRange*/   speckleRange,
                /*disp12MaxDiff*/  disp12MaxDiff
            );
        } else if (stereo_method == "sgbm") {
            const int sgbm_p1 = (P1 > 0) ? P1 : 8 * blockSize * blockSize;
            const int sgbm_p2 = (P2 > 0) ? P2 : 32 * blockSize * blockSize;
            dual_matcher_.init_sgbm(
                scaledMinDisp, scaledNumDisp, blockSize,
                sgbm_p1, sgbm_p2,
                disp12MaxDiff, preFilterCap, uniquenessRatio,
                speckleWindowSize, speckleRange, mode
            );
        }
        //wls_filter_obj_ = cv::ximgproc::createDisparityWLSFilter(dual_matcher_.left_matcher_a());
        //wls_filter_obs_ = cv::ximgproc::createDisparityWLSFilter(dual_matcher_.left_matcher_b());
        //wls_filter_obj_->setLambda(wls_lambda);
        //wls_filter_obj_->setSigmaColor(wls_sigma);
        //wls_filter_obs_->setLambda(wls_lambda);
        //wls_filter_obs_->setSigmaColor(wls_sigma);
        dual_matcher_.set_wls_params(wls_lambda, wls_sigma);
        temp_method = stereo_method;
    }

    void config_callback(const config_manager::msg::SystemConfig::SharedPtr msg) {
        config_ready_.store(false);
        stereo_method = msg->stereo_method;
        multi_collect_mode.store(msg->multi_collect_mode);

        if (stereo_method == "sgbm") {
            minDisparity      = msg->sgbm_min_disparity;
            numDisparities    = msg->sgbm_num_disparities;
            blockSize         = msg->sgbm_block_size;
            P1                = msg->sgbm_p1;
            P2                = msg->sgbm_p2;
            disp12MaxDiff     = msg->sgbm_disp12maxdiff;
            uniquenessRatio   = msg->sgbm_uniquenessratio;
            speckleWindowSize = msg->sgbm_specklewindowsize;
            speckleRange      = msg->sgbm_specklerange;
            preFilterCap      = msg->sgbm_prefiltercap;
            mode              = msg->sgbm_mode;
            wls_lambda        = msg->sgbm_wls_lambda;
            wls_sigma         = msg->sgbm_wls_sigma;
        } else if (stereo_method == "bm") {
            minDisparity      = 0;
            numDisparities    = msg->bm_num_disparities;
            blockSize         = msg->bm_block_size;
            preFilterType     = msg->bm_prefiltertype;
            preFilterSize     = msg->bm_prefiltersize;
            preFilterCap      = msg->bm_prefiltercap;
            textureThreshold  = msg->bm_texturethreshold;
            uniquenessRatio   = msg->bm_uniquenessratio;
            speckleWindowSize = msg->bm_specklewindowsize;
            speckleRange      = msg->bm_specklerange;
            disp12MaxDiff     = msg->bm_disp12maxdiff;
            wls_lambda        = msg->bm_wls_lambda;
            wls_sigma         = msg->bm_wls_sigma;
        } else {
            RCLCPP_ERROR(get_logger(), "Unknown stereo method: %s", stereo_method.c_str());
        }
        setup();
        config_ready_.store(true);
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
        if (left_camera_info_ && right_camera_info_)
            model_.fromCameraInfo(*left_camera_info_, *right_camera_info_);
    }

    // =========================================================
    // [OPT-1] build_color_lut: flatten LUT + parallel_for
    // - Dùng vector<uint8_t> thay uint8_t[256][256][256] → liên tục trong RAM
    // - cv::parallel_for_ chia R-loop ra các thread → tận dụng đa lõi
    // - Không còn 3 lần deref mảng (cache miss) → truy cập tuyến tính
    void build_color_lut()
    {
        color_id_lut_.resize(256 * 256 * 256);

        // Build BGR → HSV toàn bộ 16M pixel 1 lần
        cv::Mat bgr_all(1, 256 * 256 * 256, CV_8UC3);
        int idx = 0;
        for (int R = 0; R < 256; ++R)
        for (int G = 0; G < 256; ++G)
        for (int B = 0; B < 256; ++B)
            bgr_all.at<cv::Vec3b>(0, idx++) = cv::Vec3b(B, G, R);

        cv::Mat hsv_all;
        cv::cvtColor(bgr_all, hsv_all, cv::COLOR_BGR2HSV);

        // [OPT] parallel_for_ chia theo R (256 slices)
        uint8_t* lut_ptr = color_id_lut_.data();
        const cv::Vec3b* hsv_ptr = hsv_all.ptr<cv::Vec3b>(0);

        cv::parallel_for_(cv::Range(0, 256), [&](const cv::Range& range) {
            for (int R = range.start; R < range.end; ++R) {
                int base = R * 256 * 256;
                for (int G = 0; G < 256; ++G) {
                    int base2 = base + G * 256;
                    for (int B = 0; B < 256; ++B) {
                        const cv::Vec3b& p = hsv_ptr[base2 + B];
                        float h = p[0] * 2.0f;
                        float s = p[1] / 255.f;
                        float v = p[2] / 255.f;

                        uint8_t id;
                        if      (v < 0.2f)            id = 8;
                        else if (s < 0.2f)            id = 7;
                        else if (h < 15 || h >= 345)  id = 0;
                        else if (h < 30)              id = 1;
                        else if (h < 50)              id = 2;
                        else if (h < 150)             id = 3;
                        else if (h < 255)             id = 4;
                        else if (h < 290)             id = 5;
                        else                          id = 6;

                        lut_ptr[base2 + B] = id;
                    }
                }
            }
        });

        lut_built_ = true;
        RCLCPP_INFO(get_logger(), "Color ID LUT built (16MB, flat, parallel)");
    }

    // ── Helper: tạo mask từ tập id ────────────────────────────────────────────────
    // [OPT-2] extractMaskByIds: parallel_for_ theo row + flat LUT access
    // - Tránh 3 lần deref: color_id_lut_[R][G][B] → lut_ptr[R*65536 + G*256 + B]
    // - parallel_for_ chia hàng ra các thread
    cv::Mat extractMaskByIds(const cv::Mat& bgr, uint16_t ids_mask,
                             bool do_morphology = true)
    {
        if (!lut_built_) build_color_lut();
        CV_Assert(bgr.type() == CV_8UC3);

        const int rows = bgr.rows, cols = bgr.cols;
        cv::Mat mask(bgr.size(), CV_8U);
        const uint8_t* lut_ptr = color_id_lut_.data();

        // [OPT] Tính sẵn lookup byte: id → 0/255, tránh shift mỗi pixel
        uint8_t id2val[9];
        for (int id = 0; id < 9; ++id)
            id2val[id] = ((ids_mask >> id) & 1) ? 255 : 0;

        cv::parallel_for_(cv::Range(0, rows), [&](const cv::Range& range) {
            for (int y = range.start; y < range.end; ++y) {
                const uchar* src = bgr.ptr<uchar>(y);
                uchar*       dst = mask.ptr<uchar>(y);
                for (int x = 0; x < cols; ++x) {
                    // [OPT] Flat index: R*65536 + G*256 + B
                    uint8_t id = lut_ptr[src[x*3+2] * 65536 + src[x*3+1] * 256 + src[x*3]];
                    dst[x] = id2val[id];
                }
            }
        });

        if (do_morphology) {
            cv::morphologyEx(mask, mask, cv::MORPH_OPEN,  kernel_open_);
            cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel_close_);

            const double min_area = 0.1 * rows * cols;
            std::vector<std::vector<cv::Point>> contours;
            cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

            mask.setTo(0);
            for (const auto& c : contours) {
                if (cv::contourArea(c) >= min_area)
                    cv::drawContours(mask, std::vector<std::vector<cv::Point>>{c},
                                     -1, 255, cv::FILLED);
            }
        }

        return mask;
    }

    std::vector<StereoMaskPair> extractMaskPairsByIdMasks(
        const cv::Mat& left_bgr,
        const cv::Mat& right_bgr,
        const std::vector<uint16_t>& id_masks,
        bool do_morphology = false)
    {
        if (!lut_built_) build_color_lut();
        CV_Assert(left_bgr.type() == CV_8UC3 && right_bgr.type() == CV_8UC3);
        CV_Assert(left_bgr.size() == right_bgr.size());

        std::vector<StereoMaskPair> mask_pairs(id_masks.size());
        std::vector<std::array<uint8_t, 9>> id2vals(id_masks.size());
        for (size_t i = 0; i < id_masks.size(); ++i) {
            mask_pairs[i].left.create(left_bgr.size(), CV_8U);
            mask_pairs[i].right.create(right_bgr.size(), CV_8U);
            for (int id = 0; id < 9; ++id)
                id2vals[i][id] = ((id_masks[i] >> id) & 1) ? 255 : 0;
        }

        auto fill_masks = [this, &id2vals](const cv::Mat& bgr,
                                           std::vector<cv::Mat>& masks) {
            const int rows = bgr.rows;
            const int cols = bgr.cols;
            const size_t mask_count = masks.size();
            const uint8_t* lut_ptr = color_id_lut_.data();

            cv::parallel_for_(cv::Range(0, rows), [&](const cv::Range& range) {
                std::vector<uchar*> dst_rows(mask_count);
                for (int y = range.start; y < range.end; ++y) {
                    const uchar* src = bgr.ptr<uchar>(y);
                    for (size_t i = 0; i < mask_count; ++i)
                        dst_rows[i] = masks[i].ptr<uchar>(y);

                    for (int x = 0; x < cols; ++x) {
                        const uint8_t id = lut_ptr[
                            src[x*3+2] * 65536 + src[x*3+1] * 256 + src[x*3]];
                        for (size_t i = 0; i < mask_count; ++i)
                            dst_rows[i][x] = id2vals[i][id];
                    }
                }
            });
        };

        std::vector<cv::Mat> left_masks(id_masks.size()), right_masks(id_masks.size());
        for (size_t i = 0; i < id_masks.size(); ++i) {
            left_masks[i] = mask_pairs[i].left;
            right_masks[i] = mask_pairs[i].right;
        }
        fill_masks(left_bgr, left_masks);
        fill_masks(right_bgr, right_masks);

        const double total_pixels = static_cast<double>(left_bgr.total());
        for (auto& pair : mask_pairs) {
            if (do_morphology) {
                cv::morphologyEx(pair.left, pair.left, cv::MORPH_OPEN, kernel_open_);
                cv::morphologyEx(pair.left, pair.left, cv::MORPH_CLOSE, kernel_close_);
                cv::morphologyEx(pair.right, pair.right, cv::MORPH_OPEN, kernel_open_);
                cv::morphologyEx(pair.right, pair.right, cv::MORPH_CLOSE, kernel_close_);
            }
            pair.left_coverage = cv::countNonZero(pair.left) / total_pixels;
            pair.right_coverage = cv::countNonZero(pair.right) / total_pixels;
        }

        return mask_pairs;
    }

    // [OPT-3] extractMaskLUT: single-pass dual-mask + parallel_for_
    // Trước: 2 lần loop riêng. Giờ: 1 pass duy nhất tính cả 2 mask
    std::pair<cv::Mat, cv::Mat> extractMaskLUT(
        const cv::Mat& bgr, const std::vector<int>& keep_ids)
    {
        if (!lut_built_) build_color_lut();

        uint16_t full_mask = 0;
        for (int id : keep_ids) full_mask |= (1 << id);
        constexpr uint16_t roy_mask = COLOR_ROY;

        CV_Assert(bgr.type() == CV_8UC3);
        cv::Mat mask_full(bgr.size(), CV_8U);
        cv::Mat mask_roy (bgr.size(), CV_8U);

        const int rows = bgr.rows, cols = bgr.cols;
        const uint8_t* lut_ptr = color_id_lut_.data();

        // [OPT] Precompute id→val tables
        uint8_t full_id2val[9], roy_id2val[9];
        for (int id = 0; id < 9; ++id) {
            full_id2val[id] = ((full_mask >> id) & 1) ? 255 : 0;
            roy_id2val [id] = ((roy_mask  >> id) & 1) ? 255 : 0;
        }

        // [OPT] Single-pass parallel
        cv::parallel_for_(cv::Range(0, rows), [&](const cv::Range& range) {
            for (int y = range.start; y < range.end; ++y) {
                const uchar* src = bgr.ptr<uchar>(y);
                uchar*       df  = mask_full.ptr<uchar>(y);
                uchar*       dr  = mask_roy.ptr<uchar>(y);
                for (int x = 0; x < cols; ++x) {
                    uint8_t id = lut_ptr[src[x*3+2] * 65536 + src[x*3+1] * 256 + src[x*3]];
                    df[x] = full_id2val[id];
                    dr[x] = roy_id2val[id];
                }
            }
        });

        cv::morphologyEx(mask_full, mask_full, cv::MORPH_OPEN,  kernel_open_);
        cv::morphologyEx(mask_full, mask_full, cv::MORPH_CLOSE, kernel_close_);
        cv::morphologyEx(mask_roy,  mask_roy,  cv::MORPH_OPEN,  kernel_open_);
        cv::morphologyEx(mask_roy,  mask_roy,  cv::MORPH_CLOSE, kernel_close_);

        return {mask_full, mask_roy};
    }

    // [OPT-4] addGradientChannel: loại bỏ convertTo trung gian thừa
    // - Dùng trực tiếp CV_32F từ Sobel, tránh convertScaleAbs + convertTo thêm
    // - normalize + convertTo gộp 1 lần duy nhất
    void addGradientChannel(const cv::Mat& gray, cv::Mat& out,
                            float gray_weight = 0.7f,
                            float grad_weight = 0.3f,
                            bool  both_axes   = false)
    {
        // [OPT] Dùng member buffers thay vì local Mat
        static thread_local cv::Mat f_gray, f_grad, gx, gy, result;

        gray.convertTo(f_gray, CV_32F);

        if (both_axes) {
            cv::Sobel(gray, gx, CV_32F, 1, 0, 3);
            cv::Sobel(gray, gy, CV_32F, 0, 1, 3);
            // [OPT] cv::magnitude thay cho abs(gx)*0.5 + abs(gy)*0.5 → ít alloc hơn
            // Dùng addWeighted trực tiếp trên CV_32F, tránh convertScaleAbs
            cv::Mat gx_abs, gy_abs;
            gx_abs = cv::abs(gx);
            gy_abs = cv::abs(gy);
            cv::addWeighted(gx_abs, 0.5f, gy_abs, 0.5f, 0.0, f_grad);
        } else {
            cv::Sobel(gray, gx, CV_32F, 1, 0, 3);
            // [OPT] Tránh convertScaleAbs (alloc thêm CV_8U rồi convert lại CV_32F)
            // Làm thẳng trên CV_32F
            f_grad = cv::abs(gx);
        }

        // [OPT] addWeighted trực tiếp → kết quả CV_32F, normalize, convertTo 1 lần
        cv::addWeighted(f_gray, static_cast<double>(gray_weight),
                        f_grad, static_cast<double>(grad_weight),
                        0.0, result);
        cv::normalize(result, result, 0, 255, cv::NORM_MINMAX);
        result.convertTo(out, CV_8U);
    }


    // =========================================================
    void process_stereo() {
        while (!stop_thread_.load() && rclcpp::ok()) {
            decltype(latest_stereo_pair_) pair;
            yolov8_msgs::msg::Yolov8Inference yolo_copy;
            bool has_yolo = false;
            bool depth_signal_active = false;
            auto finish_depth_signal = [this, &depth_signal_active]() {
                if (!depth_signal_active) return;
                publish_depth_signal(false);
                publish_position_signal(false);
                depth_signal_active = false;
            };

            {
                std::unique_lock<std::mutex> lock(proc_mutex_);
                condition_.wait(lock, [this] {
                    return latest_stereo_pair_.has_value() || stop_thread_.load();
                });
                if (stop_thread_.load()) break;
                pair = std::move(latest_stereo_pair_);
                latest_stereo_pair_.reset();
                if (yolo_msg_valid_) {
                    yolo_copy = latest_yolo_msg_;
                    has_yolo  = true;
                }
            }

            const double d_time  = detect_time_.load();
            const double l_move  = last_move_.load();
            const bool   f_flag  = flag_.load();
            const double l_stamp = rclcpp::Time(pair->first->header.stamp).seconds();

            RCLCPP_DEBUG(get_logger(),
                "detect:%.3f move:%.3f flag:%d yolo:%d stamp:%.3f cfg:%d",
                d_time, l_move, f_flag, yolo_check_.load(), l_stamp, config_ready_.load());

            if (d_time < l_move || f_flag || !yolo_check_.load() ||
                !config_ready_.load() || l_stamp < l_move)
                continue;

            start_detection_time_ = l_move;
            allow_image_.store(false);
            yolo_check_.store(false);
            publish_depth_signal(true);
            depth_signal_active = true;

            if (skip_disparity_.load() && !first_run_.load()) {
                time_publisher();
                if (has_yolo) detect_pub->publish(yolo_copy);
                finish_depth_signal();
                continue;
            }
            if (multi_collect_mode.load()) first_run_.store(false);

            RCLCPP_DEBUG(get_logger(), "Start stereo disparity processing");

            // 1. Decode
            cv_bridge::CvImageConstPtr left_cv_ptr, right_cv_ptr;
            try {
                left_cv_ptr  = cv_bridge::toCvShare(pair->first,  "bgr8");
                right_cv_ptr = cv_bridge::toCvShare(pair->second, "bgr8");
            } catch (const cv_bridge::Exception& e) {
                RCLCPP_ERROR(get_logger(), "cv_bridge: %s", e.what());
                allow_image_.store(true); finish_depth_signal(); continue;
            }
            if (!left_cv_ptr  || left_cv_ptr->image.empty() ||
                !right_cv_ptr || right_cv_ptr->image.empty()) {
                allow_image_.store(true); finish_depth_signal(); continue;
            }

            const cv::Mat& left_bgr  = left_cv_ptr->image;
            const cv::Mat& right_bgr = right_cv_ptr->image;

            if (!lut_built_) build_color_lut();

            // 2. Gray
            cv::cvtColor(left_bgr,  left_full_,  cv::COLOR_BGR2GRAY);
            cv::cvtColor(right_bgr, right_full_, cv::COLOR_BGR2GRAY);

            // 3. Extract only the masks requested by the disparity pipeline.
            static const std::vector<uint16_t> disparity_id_masks = {
                COLOR_ROY,
                COLOR_GREEN
            };
            const auto disparity_mask_pairs = extractMaskPairsByIdMasks(
                left_bgr, right_bgr, disparity_id_masks, false);

            const cv::Mat& left_mask_roy_debug = disparity_mask_pairs.front().left;

            RCLCPP_DEBUG(get_logger(), "Masks extracted");

            if (seg_pub->get_subscription_count() > 0) {
                cv::bitwise_and(left_full_, left_mask_roy_debug, left_clean_roy);
                addGradientChannel(left_clean_roy, left_gray_roy, 0.5f, 0.5f, false);

                // 4. Publish seg/debug mask
                seg_pub->publish(*cv_bridge::CvImage(
                    pair->first->header, sensor_msgs::image_encodings::MONO8,
                    left_gray_roy).toImageMsg());
            }

            RCLCPP_DEBUG(get_logger(), "Debug segmentation published");

            // 5. Masked disparity: apply masks, resize, compute, WLS-filter, remask, merge.
            dual_matcher_.compute_masked_disparity(
                left_full_, right_full_,
                disparity_mask_pairs,
                s,
                disp_masked_);

            RCLCPP_DEBUG(get_logger(), "Masked disparity computed");

            cv::minMaxLoc(disp_masked_, &minVal, &maxVal);
            RCLCPP_DEBUG(get_logger(), "Disparity after WLS (full): min=%d max=%d", (int)minVal, (int)maxVal);
            if (maxVal == 0) {
                RCLCPP_ERROR(get_logger(), "❌ disparity toàn 0!");
                allow_image_.store(true); finish_depth_signal(); continue;
            }

            RCLCPP_DEBUG(get_logger(), "Building disparity message");

            // 10. Build disparity message
            stereo_msgs::msg::DisparityImage disp_msg;
            disp_msg.header.stamp    = now();
            disp_msg.header.frame_id = left_camera_info_->header.frame_id;

            auto& dimage    = disp_msg.image;
            dimage.height   = disp_masked_.rows;
            dimage.width    = disp_masked_.cols;
            dimage.encoding = sensor_msgs::image_encodings::TYPE_32FC1;
            dimage.step     = dimage.width * sizeof(float);

            if (buffer_width_ != dimage.width || buffer_height_ != dimage.height) {
                buffer_width_  = dimage.width;
                buffer_height_ = dimage.height;
                disparity_buffer_.resize(dimage.step * dimage.height);
                disparity_float_ = cv::Mat(dimage.height, dimage.width,
                    CV_32FC1, disparity_buffer_.data(), dimage.step);
            }

            disp_masked_.convertTo(disparity_float_, CV_32F, inv_dpp,
                -(model_.left().cx() - model_.right().cx()));
            dimage.data = disparity_buffer_;

            cv::Mat_<float> dmat(dimage.height, dimage.width,
                reinterpret_cast<float*>(&dimage.data[0]), dimage.step);
            disp_masked_.convertTo(dmat, dmat.type(), inv_dpp,
                -(model_.left().cx() - model_.right().cx()));
            RCUTILS_ASSERT(dmat.data == &dimage.data[0]);

            int border  = blockSize / 2;
            int left_w  = numDisparities + minDisparity + border - 1;
            int wtf     = (minDisparity >= 0) ? (border + minDisparity) : std::max(border, minDisparity);
            int right_w = dimage.width - 1 - wtf;

            disp_msg.valid_window.x_offset = left_w;
            disp_msg.valid_window.y_offset = border;
            disp_msg.valid_window.width    = right_w - left_w;
            disp_msg.valid_window.height   = dimage.height - 1 - border - border;
            disp_msg.f             = model_.right().fx();
            disp_msg.t             = model_.baseline();
            disp_msg.min_disparity = minDisparity;
            disp_msg.max_disparity = minDisparity + numDisparities - 1;
            disp_msg.delta_d       = inv_dpp;

            if (has_yolo) detect_pub->publish(yolo_copy);
            time_publisher();
            disparity_pub->publish(std::move(disp_msg));
            RCLCPP_INFO(get_logger(), "Published OK");
        }
    }

    void publish_depth_signal(bool computing_depth) {
        depth_signal_msgs::msg::DepthSignal msg;
        msg.computing_depth = computing_depth;
        depth_signal_pub->publish(msg);
    }

    void publish_position_signal(bool computing_position) {
        position_signal_msgs::msg::PositionSignal msg;
        msg.computing_position = computing_position;
        position_signal_pub->publish(msg);
    }

    void time_publisher() {
        collect_msgs::msg::CollectMsg msg;
        collect_msgs::msg::CollectTime t;
        t.start_detection        = start_detection_time_;
        t.detection_time         = detect_time_.load() - start_detection_time_;
        t.start_positioning_time = detect_time_.load();
        t.check                  = false;
        msg.collect_msg.push_back(t);
        time_pub->publish(msg);
        timer_ = false;
    }
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<StereoDepthNode>();
    rclcpp::executors::SingleThreadedExecutor executor;
    executor.add_node(node);
    executor.spin();
    rclcpp::shutdown();
    return 0;
}
