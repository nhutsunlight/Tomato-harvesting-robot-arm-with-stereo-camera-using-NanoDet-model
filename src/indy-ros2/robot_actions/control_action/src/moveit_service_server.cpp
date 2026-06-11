#include <geometric_shapes/solid_primitive_dims.h>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <moveit_msgs/msg/planning_scene.hpp>
#include <moveit_msgs/msg/planning_scene_world.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include <moveit/planning_scene/planning_scene.h>
#include <moveit/robot_state/robot_state.h>
#include <moveit/collision_detection/collision_matrix.h>
#include <moveit_msgs/srv/get_planning_scene.hpp>
#include <moveit_msgs/msg/planning_scene_components.hpp>

#include <chrono>
#include <functional>
#include <future>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <memory>
#include <moveit_msgs/msg/constraints.hpp>
#include <moveit_msgs/msg/orientation_constraint.hpp>
#include <rclcpp/rclcpp.hpp>
#include <shape_msgs/msg/solid_primitive.hpp>
#include <string>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <vector>
#include <filesystem>
#include <yaml-cpp/yaml.h>
#include <Eigen/Geometry>
#include <tf2_eigen/tf2_eigen.hpp>

#include "control_action/action/move_robot.hpp"
#include "gripper_action/action/gripper_control.hpp"
#include "robot_move_action/action/move_robot.hpp"
#include "rclcpp_action/rclcpp_action.hpp"
#include "res_msgs/msg/pose_res.hpp"
#include "robot_home_action/action/move_to_home.hpp"
#include "test_msgs/msg/ros_yolo.hpp"
#include "connect_msgs/msg/connect_msg.hpp"
#include "collect_msgs/msg/collect_msg.hpp"
#include "config_manager/msg/system_config.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include <image_geometry/stereo_camera_model.h>

#include <octomap_msgs/msg/octomap.hpp>
#include <octomap_msgs/conversions.h>
#include "tomato_octomap_msgs/msg/tomato_octomaps.hpp"
#include "tomato_octomap_msgs/msg/tomato_octomap.hpp"
#include <moveit_msgs/srv/apply_planning_scene.hpp>
#include "depth_signal_msgs/msg/depth_signal.hpp"
#include "position_signal_msgs/msg/position_signal.hpp"
#include "skip_signal_msgs/msg/skip_signal.hpp"
#include "move_signal_msgs/msg/move_signal.hpp"

using moveit::planning_interface::MoveGroupInterface;

// ═══════════════════════════════════════════════════════════════════════
//  CollisionInfo
// ═══════════════════════════════════════════════════════════════════════
struct CollisionInfo
{
    bool collision = false;

    Eigen::Vector3d position = Eigen::Vector3d::Zero();        // centroid TCP
    Eigen::Vector3d position_world = Eigen::Vector3d::Zero();  // centroid world
    Eigen::Vector3d normal = Eigen::Vector3d::Zero();

    double depth = 0.0;        // max depth
    double avg_depth = 0.0;
    double total_depth = 0.0;

    size_t contact_count = 0;

    std::string body1;
    std::string body2;

    struct ContactSample
    {
        Eigen::Vector3d position_world = Eigen::Vector3d::Zero();
        Eigen::Vector3d normal = Eigen::Vector3d::Zero();

        double depth = 0.0;

        std::string body1;
        std::string body2;
    };

    std::vector<ContactSample> contact_points;
};


// ═══════════════════════════════════════════════════════════════════════
//  OrientationRange
// ═══════════════════════════════════════════════════════════════════════

struct OrientationRange
{
    double roll_min  = 0.0;
    double roll_max  = 0.0;

    double pitch_min = 0.0;
    double pitch_max = 0.0;
};

// ═══════════════════════════════════════════════════════════════════════
//  MoveItController
// ═══════════════════════════════════════════════════════════════════════
class MoveItController : public rclcpp::Node {
public:
    // ── Constructor / Destructor ────────────────────────────────────────
    MoveItController()
        : Node("moveit_controller"),
          tf_buffer_(std::make_shared<tf2_ros::Buffer>(this->get_clock())),
          tf_listener_(*tf_buffer_)
    {
        subscription_ = this->create_subscription<test_msgs::msg::RosYolo>(
            "/ros_yolo", 10, std::bind(&MoveItController::topic_callback, this, std::placeholders::_1));
        connection_ = this->create_subscription<connect_msgs::msg::ConnectMsg>(
            "/connect_msg", 10, std::bind(&MoveItController::connection_callback, this, std::placeholders::_1));
        time_sub_ = this->create_subscription<collect_msgs::msg::CollectMsg>(
            "/collect2_msg", 10, std::bind(&MoveItController::collectmsg_callback, this, std::placeholders::_1));
        publisher_ = this->create_publisher<res_msgs::msg::PoseRes>("/pose_res", 10);
        time_publisher_ = this->create_publisher<collect_msgs::msg::CollectMsg>("/collect3_msg", 10);
        skip_signal_pub = create_publisher<skip_signal_msgs::msg::SkipSignal>("/skip_signal", 10);
        tomato_octomap_sub_ = this->create_subscription<tomato_octomap_msgs::msg::TomatoOctomaps>(
            "/tomato_octomaps", 10,
            std::bind(&MoveItController::tomatoOctomapCallback, this, std::placeholders::_1));
        config_sub_ = this->create_subscription<config_manager::msg::SystemConfig>(
            "/system_config",
            rclcpp::QoS(1).transient_local().reliable(),
            std::bind(&MoveItController::config_callback, this, std::placeholders::_1));
        sub_left_cam_ = create_subscription<sensor_msgs::msg::CameraInfo>(
            "/stereo/left/camera_info_calib",
            rclcpp::QoS(1).transient_local().reliable(),
            std::bind(&MoveItController::left_camera_info_callback, this, std::placeholders::_1));
        sub_right_cam_ = create_subscription<sensor_msgs::msg::CameraInfo>(
            "/stereo/right/camera_info_calib",
            rclcpp::QoS(1).transient_local().reliable(),
            std::bind(&MoveItController::right_camera_info_callback, this, std::placeholders::_1));
        depth_signal_pub    = create_publisher<depth_signal_msgs::msg::DepthSignal>("/depth_signal", 10);
        position_signal_pub = create_publisher<position_signal_msgs::msg::PositionSignal>("/position_signal", 10);
        move_signal_pub     = create_publisher<move_signal_msgs::msg::MoveSignal>("/move_signal", 10);

        action_server_ = rclcpp_action::create_server<ControlRobot>(
            this, "move_robot",
            std::bind(&MoveItController::handle_goal,     this, std::placeholders::_1, std::placeholders::_2),
            std::bind(&MoveItController::handle_cancel,   this, std::placeholders::_1),
            std::bind(&MoveItController::handle_accepted, this, std::placeholders::_1));
        move_client_          = rclcpp_action::create_client<MoveRobot>(this, "robot_move_action");
        move_to_home_client_  = rclcpp_action::create_client<MoveToHome>(this, "move_to_home");
        gripper_client_       = rclcpp_action::create_client<GripperControl>(this, "gripper_action");
        planning_scene_client_ = this->create_client<moveit_msgs::srv::GetPlanningScene>("/get_planning_scene");
        apply_client_ = this->create_client<moveit_msgs::srv::ApplyPlanningScene>("/apply_planning_scene");

        std::filesystem::path base_path = std::filesystem::current_path();
        config_path = base_path.string() + "/config/setup.yaml";
        RCLCPP_INFO(this->get_logger(), "MoveIt Action Server started.");
    }

    ~MoveItController() {
        stop_connection_monitor_ = true;
        if (connection_monitor_thread_.joinable()) {
            connection_monitor_thread_.join();
        }
    }

    // ── Initialization ──────────────────────────────────────────────────
    void initialize() {
        move_group_interface_ = std::make_unique<MoveGroupInterface>(shared_from_this(), "indy_manipulator");
        saveOriginalACM();
        setGripperIgnoreCollision(true);
        near_tcp_range_ = estimateLink5MaxReach("link5");
        tcp_range_ = estimateLink5MaxReach("tcp0");
    }

private:
    // ── Type aliases ────────────────────────────────────────────────────
    using ControlRobot            = control_action::action::MoveRobot;
    using GoalHandleControlRobot  = rclcpp_action::ServerGoalHandle<ControlRobot>;
    using MoveRobot               = robot_move_action::action::MoveRobot;
    using GoalHandleMoveRobot     = rclcpp_action::ClientGoalHandle<MoveRobot>;
    using MoveToHome              = robot_home_action::action::MoveToHome;
    using GoalHandleMoveToHome    = rclcpp_action::ClientGoalHandle<MoveToHome>;
    using GripperControl          = gripper_action::action::GripperControl;
    using GoalHandleGripperControl = rclcpp_action::ClientGoalHandle<GripperControl>;

    // ── Inner structs ───────────────────────────────────────────────────
    struct ClusterEntry
    {
        std::array<double, 10> pose;
        size_t original_idx;
    };

    struct ValidTarget
    {
        geometry_msgs::msg::Pose pose;
        size_t original_idx;
    };

    // ── Member variables ────────────────────────────────────────────────

    // MoveIt / planning
    moveit::planning_interface::PlanningSceneInterface planning_scene_interface_;
    std::unique_ptr<MoveGroupInterface>                move_group_interface_;
    std::shared_ptr<planning_scene::PlanningScene>     cached_scene_;
    collision_detection::AllowedCollisionMatrix        original_acm_;
    std::atomic<bool>                                  acm_saved_{false};
    std::atomic<bool>                                  scene_valid_{false};

    // TF
    std::shared_ptr<tf2_ros::Buffer>  tf_buffer_;
    tf2_ros::TransformListener        tf_listener_;
    geometry_msgs::msg::TransformStamped target_base_transform_;
    bool target_base_transform_ready_ = false;

    // Robot state / targets
    std::vector<std::array<double, 10>> target_position_;
    std::array<double, 6>               test_position_ref;
    std::array<double, 6>               test_position_ref_offset;
    std::array<double, 6>               target_idx_position_;
    std::array<double, 6>               sub_test_position_ref_offset;
    std::array<double, 6>               previous_position_;
    std::vector<double>                 home_position_;
    std::vector<double>                 drop_position_;
    std::vector<geometry_msgs::msg::Pose> target_pose_list_;
    geometry_msgs::msg::Pose target_pose;
    geometry_msgs::msg::Pose pose;
    geometry_msgs::msg::Pose test_pose;
    geometry_msgs::msg::Pose next_pose;
    geometry_msgs::msg::Pose input_test_pose;
    geometry_msgs::msg::Pose sub_input_test_pose;
    geometry_msgs::msg::Pose temp_pose;

    // Octomap
    octomap_msgs::msg::Octomap          octomap_single_;
    octomap_msgs::msg::Octomap          octomap_cache_;
    octomap_msgs::msg::Octomap          octomap_temp_;
    octomap_msgs::msg::Octomap          octomap_combine_;
    geometry_msgs::msg::TransformStamped octomap_to_link0_tf_;
    std::mutex                           octomap_map_mutex_;
    bool                                 octomap_cache_valid_ = false;

    // Camera
    float fx_ = 0.f, fy_ = 0.f, cx_ = 0.f, cy_ = 0.f;
    sensor_msgs::msg::CameraInfo::SharedPtr left_camera_info_, right_camera_info_;
    image_geometry::StereoCameraModel       model_;

    // Config / params
    double offset_distance_   = 0.0;
    double object_offset_     = 0.0;
    double y_offset_distance_ = 0.0;
    double offset_angle_      = 0.0;
    double eef_scale_         = 0.0;
    double tcp_range_         = 0.0;
    double near_tcp_range_    = 0.0;
    std::string config_path;

    // Timing
    double detection_time       = 0.0;
    double total_time           = 0.0;
    double start_detection_time = 0.0;
    double positioning_time     = 0.0;
    double temp_total_time      = 0.0;

    // Flags
    bool is_robot_moving_     = false;
    bool is_server_ready_     = false;
    bool stop_connection_monitor_ = false;
    bool is_reset_            = false;
    bool bypass               = false;
    bool ws_check             = true;
    bool pose_check           = false;
    ///bool recompute            = false;
    bool found_test_ik        = false;
    bool sub_found_test_ik    = false;
    bool obs_check            = true;
    bool obs_check_1          = false;
    bool obs_check_2          = false;
    bool obs_check_3          = false;
    bool allow_request_       = false;
    bool go_home_             = false;
    bool time_recieved_       = false;
    bool pass_all_            = false;
    bool target_ready_        = false;
    bool obs_ready            = false;
    bool config_received_     = false;
    bool move_action_ready_   = false;
    bool home_action_ready_   = false;
    bool gripper_action_ready_ = false;
    std::atomic<bool> mul_mode_{false};
    std::atomic<bool> latest_connection_status_{false};
    std::atomic<bool> reset_status_{false};
    std::atomic<bool> move_success_{false};

    // Counters
    std::size_t success_count = 0;

    // Mutexes
    std::mutex pub_mutex;
    std::mutex acm_mutex_;
    std::mutex scene_mutex_;

    // Thread
    std::thread connection_monitor_thread_;

    // ROS interfaces
    rclcpp_action::Server<ControlRobot>::SharedPtr     action_server_;
    rclcpp_action::Client<GripperControl>::SharedPtr   gripper_client_;
    rclcpp_action::Client<MoveToHome>::SharedPtr       move_to_home_client_;
    rclcpp_action::Client<MoveRobot>::SharedPtr        move_client_;
    rclcpp::Subscription<test_msgs::msg::RosYolo>::SharedPtr          subscription_;
    rclcpp::Subscription<connect_msgs::msg::ConnectMsg>::SharedPtr    connection_;
    rclcpp::Subscription<collect_msgs::msg::CollectMsg>::SharedPtr    time_sub_;
    rclcpp::Subscription<config_manager::msg::SystemConfig>::SharedPtr config_sub_;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr     sub_left_cam_, sub_right_cam_;
    rclcpp::Subscription<tomato_octomap_msgs::msg::TomatoOctomaps>::SharedPtr tomato_octomap_sub_;
    rclcpp::Publisher<depth_signal_msgs::msg::DepthSignal>::SharedPtr    depth_signal_pub;
    rclcpp::Publisher<position_signal_msgs::msg::PositionSignal>::SharedPtr position_signal_pub;
    rclcpp::Publisher<res_msgs::msg::PoseRes>::SharedPtr                 publisher_;
    rclcpp::Publisher<collect_msgs::msg::CollectMsg>::SharedPtr          time_publisher_;
    rclcpp::Publisher<skip_signal_msgs::msg::SkipSignal>::SharedPtr      skip_signal_pub;
    rclcpp::Publisher<move_signal_msgs::msg::MoveSignal>::SharedPtr      move_signal_pub;
    rclcpp::Client<moveit_msgs::srv::GetPlanningScene>::SharedPtr        planning_scene_client_;
    rclcpp::Client<moveit_msgs::srv::ApplyPlanningScene>::SharedPtr      apply_client_;

    // ════════════════════════════════════════════════════════════════════
    //  ACTION SERVER CALLBACKS
    // ════════════════════════════════════════════════════════════════════
    rclcpp_action::GoalResponse handle_goal(
        const rclcpp_action::GoalUUID& uuid,
        std::shared_ptr<const ControlRobot::Goal> goal)
    {
        RCLCPP_INFO(this->get_logger(), "Received action request!");
        (void)goal; (void)uuid;
        if (!goal->request_move && !allow_request_) {
            RCLCPP_WARN(this->get_logger(), "Action server not ready. Rejecting goal.");
            return rclcpp_action::GoalResponse::REJECT;
        }
        allow_request_ = false;
        RCLCPP_INFO(this->get_logger(), "Action goal accepted.");
        return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
    }

    rclcpp_action::CancelResponse handle_cancel(
        const std::shared_ptr<GoalHandleControlRobot> goal_handle)
    {
        (void)goal_handle;
        RCLCPP_INFO(this->get_logger(), "Goal canceled!");
        return rclcpp_action::CancelResponse::ACCEPT;
    }

    void handle_accepted(const std::shared_ptr<GoalHandleControlRobot> goal_handle) {
        std::thread{std::bind(&MoveItController::execute, this, goal_handle)}.detach();
    }

    // ════════════════════════════════════════════════════════════════════
    //  TOPIC / CONFIG CALLBACKS
    // ════════════════════════════════════════════════════════════════════
    void topic_callback(const test_msgs::msg::RosYolo& msg)
    {
        if (!target_ready_ && !msg.ros_yolo.empty()) {
            allow_request_ = true;
            target_position_.clear();
            target_pose_list_.clear();
            target_base_transform_ready_ = false;
            for (const auto& result : msg.ros_yolo) {
                target_position_.push_back({
                    result.x, result.y, result.z,
                    result.roll, result.pitch, result.yall,
                    static_cast<double>(result.x1), static_cast<double>(result.y1),
                    static_cast<double>(result.x2), static_cast<double>(result.y2)
                });
            }
            is_robot_moving_ = true;
            target_ready_ = true;
            RCLCPP_INFO(this->get_logger(), "Received %zu tomato poses:", target_position_.size());
        }
    }

    void connection_callback(const connect_msgs::msg::ConnectMsg& msg) {
        if (!msg.connect_msg.empty()) {
            const auto& result = msg.connect_msg.front();
            latest_connection_status_ = result.connection;
            reset_status_ = result.wait_key;
        }
    }

    void collectmsg_callback(const collect_msgs::msg::CollectMsg& msg) {
        if (!msg.collect_msg.empty() && !time_recieved_) {
            RCLCPP_INFO(this->get_logger(), "DEBUG collectmsg_callback: Received collect message with %zu entries", msg.collect_msg.size());
            start_detection_time = 0.0;
            detection_time = 0.0;
            positioning_time = 0.0;
            const auto& time = msg.collect_msg.front();
            start_detection_time = time.start_detection;
            detection_time = time.detection_time;
            positioning_time = time.positioning_time;
            time_recieved_ = true;
            RCLCPP_WARN(this->get_logger(), "DEBUG time_sub: start_detection_time=%.3f, positioning_time=%.3f, detection_time=%.3f",
                        start_detection_time, positioning_time, detection_time);
        }
    }

    void config_callback(const config_manager::msg::SystemConfig::SharedPtr msg)
    {
        RCLCPP_INFO(this->get_logger(), "Load setup");
        config_received_ = false;
        home_position_      = msg->home_pose;
        drop_position_      = msg->drop_pose;
        object_offset_      = msg->object_offset;
        offset_distance_    = msg->offset_distance;
        y_offset_distance_  = msg->y_offset_distance;
        offset_angle_       = msg->offset_angle;
        mul_mode_           = msg->multi_collect_mode;
        config_received_ = true;
    }

    void tomatoOctomapCallback(
        const tomato_octomap_msgs::msg::TomatoOctomaps::SharedPtr msg)
    {
        if (!obs_ready) {
            if (msg->octomaps.empty()) return;
            octomap_single_ = msg->octomaps[0].octomap;
            try {
                octomap_to_link0_tf_ = tf_buffer_->lookupTransform(
                    "link0", octomap_single_.header.frame_id,
                    tf2::TimePointZero, tf2::durationFromSec(1.0));
            } catch (const tf2::TransformException& ex) {
                RCLCPP_ERROR(this->get_logger(),
                             "Failed to cache octomap->link0 transform: %s", ex.what());
                obs_ready = false;
                return;
            }
            obs_ready = true;
            RCLCPP_INFO(this->get_logger(), "Stored single octomap in frame '%s'",
                        octomap_single_.header.frame_id.c_str());
        }
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
    }

    void update_camera_model() {
        if (left_camera_info_ && right_camera_info_)
            model_.fromCameraInfo(*left_camera_info_, *right_camera_info_);
    }

    // ════════════════════════════════════════════════════════════════════
    //  PLANNING SCENE — refresh / snapshot / ACM
    // ════════════════════════════════════════════════════════════════════
    void saveOriginalACM()
    {
        using GetPlanningScene = moveit_msgs::srv::GetPlanningScene;
        if (!planning_scene_client_) {
            RCLCPP_ERROR(this->get_logger(), "planning_scene_client_ is null!"); return;
        }
        if (!planning_scene_client_->wait_for_service(std::chrono::seconds(5))) {
            RCLCPP_ERROR(this->get_logger(), "Service /get_planning_scene not available"); return;
        }
        auto request = std::make_shared<GetPlanningScene::Request>();
        request->components.components = moveit_msgs::msg::PlanningSceneComponents::ALLOWED_COLLISION_MATRIX;

        std::promise<GetPlanningScene::Response::SharedPtr> promise;
        auto future_result = promise.get_future();
        planning_scene_client_->async_send_request(request,
            [&promise](rclcpp::Client<GetPlanningScene>::SharedFuture future) {
                promise.set_value(future.get());
            });
        if (future_result.wait_for(std::chrono::seconds(5)) != std::future_status::ready) {
            RCLCPP_ERROR(this->get_logger(), "Timeout saving original ACM"); return;
        }
        auto response = future_result.get();
        if (!response) { RCLCPP_ERROR(this->get_logger(), "Got null response!"); return; }
        {
            std::lock_guard<std::mutex> lock(acm_mutex_);
            original_acm_ = collision_detection::AllowedCollisionMatrix(response->scene.allowed_collision_matrix);
            acm_saved_ = true;
        }
        RCLCPP_INFO(this->get_logger(), "Original ACM saved successfully.");
    }

    bool refreshPlanningScene()
    {
        using GetPlanningScene = moveit_msgs::srv::GetPlanningScene;
 
        // Reset flag trước — bắt buộc waitForSceneReady() phải chờ build xong
        scene_valid_.store(false, std::memory_order_release);
 
        if (!planning_scene_client_->wait_for_service(std::chrono::seconds(3))) {
            RCLCPP_ERROR(get_logger(), "Service /get_planning_scene not available");
            return false;
        }
 
        auto request = std::make_shared<GetPlanningScene::Request>();
        request->components.components =
            moveit_msgs::msg::PlanningSceneComponents::SCENE_SETTINGS        |
            moveit_msgs::msg::PlanningSceneComponents::ROBOT_STATE           |
            moveit_msgs::msg::PlanningSceneComponents::WORLD_OBJECT_GEOMETRY |
            moveit_msgs::msg::PlanningSceneComponents::ALLOWED_COLLISION_MATRIX |
            moveit_msgs::msg::PlanningSceneComponents::OCTOMAP;
 
        std::promise<GetPlanningScene::Response::SharedPtr> promise;
        auto fut = promise.get_future();
        planning_scene_client_->async_send_request(
            request,
            [&promise](rclcpp::Client<GetPlanningScene>::SharedFuture f) {
                promise.set_value(f.get());
            });
 
        if (fut.wait_for(std::chrono::seconds(3)) != std::future_status::ready) {
            RCLCPP_ERROR(get_logger(), "Timeout getting planning scene");
            return false;
        }
 
        auto response    = fut.get();
        auto robot_model = move_group_interface_->getRobotModel();
 
        auto new_scene = std::make_shared<planning_scene::PlanningScene>(robot_model);
        new_scene->setPlanningSceneDiffMsg(response->scene);
 
        setScene(new_scene);  // sets scene_valid_ = true
        return true;
    }

    std::shared_ptr<planning_scene::PlanningScene> getSceneSnapshot()
    {
        return std::atomic_load(&cached_scene_);
    }

    void setScene(std::shared_ptr<planning_scene::PlanningScene> new_scene)
    {
        std::atomic_store(&cached_scene_, new_scene);
        scene_valid_.store(true, std::memory_order_release);
    }

    bool waitForSceneReady(std::chrono::milliseconds timeout = std::chrono::milliseconds(15000))
    {
        const auto deadline = std::chrono::steady_clock::now() + timeout;
        while (rclcpp::ok() &&
               !scene_valid_.load(std::memory_order_acquire) &&
               std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        return scene_valid_.load(std::memory_order_acquire);
    }

    void setOctomapCollision(bool allow_collision)
    {
        std::lock_guard<std::mutex> lock(acm_mutex_);
        if (!acm_saved_) { RCLCPP_ERROR(this->get_logger(), "Original ACM not saved yet!"); return; }
        moveit_msgs::msg::PlanningScene diff_scene;
        diff_scene.is_diff = true;
        collision_detection::AllowedCollisionMatrix acm = original_acm_;
        std::vector<std::string> gripper_links = {
            "gripper_left1", "gripper_left2", "gripper_left3",
            "gripper_right1", "gripper_right2", "gripper_right3"
        };
        const auto& all_links = move_group_interface_->getRobotModel()->getLinkModelNames();
        for (const auto& link : gripper_links) {
            for (const auto& other : all_links) acm.setEntry(link, other, true);
            acm.setDefaultEntry(link, true);
        }
        if (allow_collision) {
            for (const auto& link : all_links) acm.setDefaultEntry(link, true);
            RCLCPP_WARN(this->get_logger(), "Octomap collision: DISABLED");
        } else {
            RCLCPP_INFO(this->get_logger(), "Octomap collision: ENABLED");
        }
        acm.getMessage(diff_scene.allowed_collision_matrix);
        planning_scene_interface_.applyPlanningScene(diff_scene);
    }

    void setGripperIgnoreCollision(bool allow)
    {
        std::lock_guard<std::mutex> lock(acm_mutex_);
        if (!acm_saved_) { RCLCPP_ERROR(this->get_logger(), "Original ACM not saved yet!"); return; }
        moveit_msgs::msg::PlanningScene diff_scene;
        diff_scene.is_diff = true;
        if (allow) {
            collision_detection::AllowedCollisionMatrix acm = original_acm_;
            std::vector<std::string> gripper_links = {
                "gripper_left1", "gripper_left2", "gripper_left3",
                "gripper_right1", "gripper_right2", "gripper_right3"
            };
            const auto& all_links = move_group_interface_->getRobotModel()->getLinkModelNames();
            for (const auto& link : gripper_links) {
                for (const auto& other : all_links) acm.setEntry(link, other, true);
                acm.setDefaultEntry(link, true);
            }
            acm.getMessage(diff_scene.allowed_collision_matrix);
            planning_scene_interface_.applyPlanningScene(diff_scene);
            RCLCPP_INFO(this->get_logger(), "Collision DISABLED for all gripper links");
        } else {
            original_acm_.getMessage(diff_scene.allowed_collision_matrix);
            planning_scene_interface_.applyPlanningScene(diff_scene);
            RCLCPP_INFO(this->get_logger(), "Collision ENABLED for all gripper links");
        }
    }

    void setObstacleCollision(bool allow)
    {
        std::lock_guard<std::mutex> lock(acm_mutex_);
        if (!acm_saved_) { RCLCPP_ERROR(this->get_logger(), "Original ACM not saved yet!"); return; }
        moveit_msgs::msg::PlanningScene diff_scene;
        diff_scene.is_diff = true;
        if (allow) {
            collision_detection::AllowedCollisionMatrix acm = original_acm_;
            const auto& all_links = move_group_interface_->getRobotModel()->getLinkModelNames();
            for (const auto& link : all_links) acm.setDefaultEntry(link, true);
            acm.getMessage(diff_scene.allowed_collision_matrix);
            planning_scene_interface_.applyPlanningScene(diff_scene);
            RCLCPP_INFO(this->get_logger(), "Obstacle collision check: DISABLED");
        } else {
            original_acm_.getMessage(diff_scene.allowed_collision_matrix);
            planning_scene_interface_.applyPlanningScene(diff_scene);
            RCLCPP_INFO(this->get_logger(), "Obstacle collision check: ENABLED");
        }
    }

    // ════════════════════════════════════════════════════════════════════
    //  OCTOMAP — apply / crop / mask / combine
    // ════════════════════════════════════════════════════════════════════
    bool applyOctomapMessage(
        const octomap_msgs::msg::Octomap& octomap_to_apply,
        const char* success_log)
    {
        moveit_msgs::msg::PlanningScene planning_scene_msg;
        planning_scene_msg.is_diff = true;
        planning_scene_msg.world.octomap.octomap = octomap_to_apply;
        planning_scene_msg.world.octomap.header  = octomap_to_apply.header;
        planning_scene_msg.world.octomap.origin.orientation.w = 1.0;

        if (!apply_client_->wait_for_service(std::chrono::seconds(3))) {
            RCLCPP_ERROR(get_logger(), "Service /apply_planning_scene not available");
            return false;
        }

        auto request = std::make_shared<moveit_msgs::srv::ApplyPlanningScene::Request>();
        request->scene = planning_scene_msg;

        // ✅ Dùng shared_ptr để tránh dangling reference khi lambda capture
        auto promise_ptr = std::make_shared<std::promise<moveit_msgs::srv::ApplyPlanningScene::Response::SharedPtr>>();
        auto future = promise_ptr->get_future();

        apply_client_->async_send_request(request,
            [promise_ptr](rclcpp::Client<moveit_msgs::srv::ApplyPlanningScene>::SharedFuture f) {
                // ✅ Chỉ set_value nếu chưa set — tránh double-set
                try {
                    promise_ptr->set_value(f.get());
                } catch (...) {
                    // promise đã bị set hoặc future invalid — bỏ qua
                }
            });

        if (future.wait_for(std::chrono::seconds(10)) != std::future_status::ready) {
            RCLCPP_ERROR(get_logger(), "Timeout applying octomap");
            return false;
        }

        auto response = future.get();
        if (!response || !response->success) {
            RCLCPP_ERROR(get_logger(), "ApplyPlanningScene failed");
            return false;
        }

        RCLCPP_INFO(get_logger(), "%s", success_log);
        return true;
    }

    void applyOctomap(
        int x1, int y1, int x2, int y2,
        float fx, float fy, float cx, float cy)
    {
        octomap_msgs::msg::Octomap octomap_to_apply;
        {
            std::lock_guard<std::mutex> lock(octomap_map_mutex_);
            if (!obs_ready) {
                RCLCPP_WARN(this->get_logger(), "No octomap ready");
                return;
            }
            octomap_to_apply = octomap_single_;
        }
 
        octomap_to_apply = cropOctomapByBbox(
            octomap_to_apply, x1, y1, x2, y2, fx, fy, cx, cy);
        {
            std::lock_guard<std::mutex> lock(octomap_map_mutex_);
            octomap_to_apply = transformOctomapWithTransform(
                octomap_to_apply, octomap_to_link0_tf_);
        }
 
        RCLCPP_INFO(this->get_logger(),
            "Cropped octomap (bbox=[%d,%d,%d,%d])", x1, y1, x2, y2);
 
        if (octomap_to_apply.header.frame_id.empty()) {
            octomap_to_apply.header.frame_id = "link0";
        }
 
        octomap_cache_ = octomap_to_apply;
        octomap_cache_valid_ = true;
 
        while(rclcpp::ok() && !applyOctomapMessage(octomap_to_apply, "Octomap applied from msg"))
        {
            RCLCPP_WARN(this->get_logger(), "Retrying to apply octomap from msg...");
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
 
        // Wait for move_group to fully process the applied octomap before
        // refreshPlanningScene() reads it back — prevents stale scene cache.
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }

    void applyOctomapTemp()   { octomap_temp_ = octomap_cache_; }

    void applyOctocmapFromTemp()
    {
        while (rclcpp::ok() && !applyOctomapMessage(octomap_temp_, "Octomap applied from temp")) {
            RCLCPP_WARN(this->get_logger(), "Retrying to apply octomap from temp...");
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
    }

    void ocotmapCombine()
    {
        if (!octomap_temp_.data.empty()) {
            octomap_combine_ = intersectOctomaps(octomap_cache_, octomap_temp_);
        } else {
            RCLCPP_WARN(this->get_logger(), "Temp octomap is empty, return to cache octomap");
            octomap_combine_ = octomap_cache_;
        }
        while (rclcpp::ok() && !applyOctomapMessage(octomap_combine_, "Octomap applied from combine")) {
            RCLCPP_WARN(this->get_logger(), "Retrying to apply octomap from combine...");
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
    }

    void clearOctomapCache()
    {
        octomap_cache_   = octomap_msgs::msg::Octomap();
        octomap_temp_    = octomap_msgs::msg::Octomap();
        octomap_combine_ = octomap_msgs::msg::Octomap();
        octomap_cache_valid_ = false;
    }

    bool applyMaskedOctomapFromCache(
        const std::vector<CollisionInfo::ContactSample>& contact_samples,
        double clear_radius = 0.02)
    {
        if (!octomap_cache_valid_)
        {
            RCLCPP_WARN(
                get_logger(),
                "No cached octomap available");
            return false;
        }

        if (contact_samples.empty())
            return false;

        auto octomap_to_apply = octomap_cache_;

        // ===== Cluster đơn giản bằng khoảng cách =====

        constexpr double merge_distance = 0.01; // 1cm

        std::vector<CollisionInfo::ContactSample> representative_contacts;

        for (const auto& sample : contact_samples)
        {
            bool duplicate = false;

            for (const auto& rep : representative_contacts)
            {
                if ((sample.position_world -
                    rep.position_world).norm() < merge_distance)
                {
                    duplicate = true;
                    break;
                }
            }

            if (!duplicate)
            {
                representative_contacts.push_back(sample);
            }
        }

        // Giới hạn số điểm tối đa
        constexpr size_t max_contact_points = 10;

        if (representative_contacts.size() > max_contact_points)
        {
            representative_contacts.resize(max_contact_points);
        }

        RCLCPP_INFO(
            get_logger(),
            "Contacts: %zu -> Representative contacts: %zu",
            contact_samples.size(),
            representative_contacts.size());

        // ===== Clear Octomap =====

        for (const auto& sample : representative_contacts)
        {
            Eigen::Vector3d normal_dir =
                sample.normal.norm() > 1e-9
                    ? sample.normal.normalized()
                    : Eigen::Vector3d::UnitZ();

            for (int i = -1; i <= 1; ++i)
            {
                double offset =
                    static_cast<double>(i) *
                    clear_radius * 0.4;

                Eigen::Vector3d clear_center =
                    sample.position_world +
                    normal_dir * offset;

                octomap_to_apply =
                    maskOctomapAroundPoint(
                        octomap_to_apply,
                        clear_center,
                        clear_radius);
            }
        }

        octomap_cache_ = octomap_to_apply;
        octomap_cache_valid_ = true;

        while (rclcpp::ok())
        {
            if (applyOctomapMessage(
                    octomap_to_apply,
                    "Octomap applied from cache"))
            {
                break;
            }

            RCLCPP_WARN(
                get_logger(),
                "Retrying to apply octomap from cache...");

            std::this_thread::sleep_for(
                std::chrono::milliseconds(500));
        }

        return true;
    }

    // ── Octomap geometry helpers ─────────────────────────────────────────
    octomap_msgs::msg::Octomap cropOctomapByBbox(
        const octomap_msgs::msg::Octomap& input_octomap,
        int x1, int y1, int x2, int y2,
        float fx, float fy, float cx, float cy,
        float z_min = 0.05f, float z_max = 5.0f)
    {
        std::unique_ptr<octomap::AbstractOcTree> abstract_tree(octomap_msgs::msgToMap(input_octomap));
        auto* tree = dynamic_cast<octomap::OcTree*>(abstract_tree.get());
        if (!tree) return input_octomap;
        octomap::OcTree out_tree(tree->getResolution());
        for (auto it = tree->begin_leafs(); it != tree->end_leafs(); ++it) {
            if (!tree->isNodeOccupied(*it)) continue;
            const float X = it.getX(), Y = it.getY(), Z = it.getZ();
            bool inside_bbox = false;
            if (X >= z_min && X <= z_max) {
                const float u = fx * (-Y) / X + cx;
                const float v = fy * (-Z) / X + cy;
                inside_bbox = (u >= x1 && u <= x2 && v >= y1 && v <= y2);
            }
            if (inside_bbox) continue;
            out_tree.updateNode(octomap::point3d(X, Y, Z), true);
        }
        out_tree.updateInnerOccupancy();
        octomap_msgs::msg::Octomap out_msg;
        octomap_msgs::binaryMapToMsg(out_tree, out_msg);
        out_msg.header = input_octomap.header;
        return out_msg;
    }

    octomap_msgs::msg::Octomap maskOctomapAroundPoint(
        const octomap_msgs::msg::Octomap& input_octomap,
        const Eigen::Vector3d& center_world, double clear_radius)
    {
        std::unique_ptr<octomap::AbstractOcTree> abstract_tree(octomap_msgs::msgToMap(input_octomap));
        auto* tree = dynamic_cast<octomap::OcTree*>(abstract_tree.get());
        if (!tree) return input_octomap;
        octomap::OcTree out_tree(tree->getResolution());
        const double radius_sq = clear_radius * clear_radius;
        for (auto it = tree->begin_leafs(); it != tree->end_leafs(); ++it) {
            if (!tree->isNodeOccupied(*it)) continue;
            const double dx = it.getX() - center_world.x();
            const double dy = it.getY() - center_world.y();
            const double dz = it.getZ() - center_world.z();
            if ((dx*dx + dy*dy + dz*dz) <= radius_sq) continue;
            out_tree.updateNode(octomap::point3d(it.getX(), it.getY(), it.getZ()), true);
        }
        out_tree.updateInnerOccupancy();
        octomap_msgs::msg::Octomap out_msg;
        octomap_msgs::binaryMapToMsg(out_tree, out_msg);
        out_msg.header = input_octomap.header;
        return out_msg;
    }

    octomap_msgs::msg::Octomap intersectOctomaps(
        const octomap_msgs::msg::Octomap& cache_msg,
        const octomap_msgs::msg::Octomap& temp_msg)
    {
        std::unique_ptr<octomap::AbstractOcTree> cache_abs(octomap_msgs::msgToMap(cache_msg));
        std::unique_ptr<octomap::AbstractOcTree> temp_abs(octomap_msgs::msgToMap(temp_msg));
        auto* cache_tree = dynamic_cast<octomap::OcTree*>(cache_abs.get());
        auto* temp_tree  = dynamic_cast<octomap::OcTree*>(temp_abs.get());
        if (!cache_tree || !temp_tree) return cache_msg;
        octomap::OcTree out_tree(cache_tree->getResolution());
        for (auto it = cache_tree->begin_leafs(); it != cache_tree->end_leafs(); ++it) {
            if (!cache_tree->isNodeOccupied(*it)) continue;
            auto* temp_node = temp_tree->search(it.getX(), it.getY(), it.getZ());
            if (temp_node && temp_tree->isNodeOccupied(temp_node))
                out_tree.updateNode(octomap::point3d(it.getX(), it.getY(), it.getZ()), true);
        }
        out_tree.updateInnerOccupancy();
        octomap_msgs::msg::Octomap out_msg;
        octomap_msgs::binaryMapToMsg(out_tree, out_msg);
        out_msg.header = cache_msg.header;
        return out_msg;
    }

    octomap_msgs::msg::Octomap transformOctomapWithTransform(
        const octomap_msgs::msg::Octomap& input_octomap,
        const geometry_msgs::msg::TransformStamped& tf_msg)
    {
        if (input_octomap.header.frame_id == "link0") return input_octomap;
        std::unique_ptr<octomap::AbstractOcTree> abstract_tree(octomap_msgs::msgToMap(input_octomap));
        auto* input_tree = dynamic_cast<octomap::OcTree*>(abstract_tree.get());
        if (!input_tree) { RCLCPP_ERROR(this->get_logger(), "Failed to convert octomap msg to OcTree"); return input_octomap; }
        tf2::Transform tf;
        tf2::fromMsg(tf_msg.transform, tf);
        octomap::OcTree output_tree(input_tree->getResolution());
        for (auto it = input_tree->begin_leafs(); it != input_tree->end_leafs(); ++it) {
            if (!input_tree->isNodeOccupied(*it)) continue;
            const tf2::Vector3 pt_in(it.getX(), it.getY(), it.getZ());
            const tf2::Vector3 pt_out = tf * pt_in;
            output_tree.updateNode(octomap::point3d(pt_out.x(), pt_out.y(), pt_out.z()), true);
        }
        output_tree.updateInnerOccupancy();
        octomap_msgs::msg::Octomap output_octomap;
        octomap_msgs::binaryMapToMsg(output_tree, output_octomap);
        output_octomap.header.frame_id = "link0";
        output_octomap.header.stamp    = input_octomap.header.stamp;
        return output_octomap;
    }

    octomap_msgs::msg::Octomap transformOctomapToLink0(const octomap_msgs::msg::Octomap& input_octomap)
    {
        if (input_octomap.header.frame_id == "link0") return input_octomap;
        geometry_msgs::msg::TransformStamped tf_msg;
        try {
            tf_msg = tf_buffer_->lookupTransform("link0", input_octomap.header.frame_id,
                                                  tf2::TimePointZero, tf2::durationFromSec(1.0));
        } catch (const tf2::TransformException& ex) {
            RCLCPP_ERROR(this->get_logger(), "Failed to transform octomap: %s", ex.what());
            return input_octomap;
        }
        return transformOctomapWithTransform(input_octomap, tf_msg);
    }

    // ════════════════════════════════════════════════════════════════════
    //  COLLISION CHECKING
    // ════════════════════════════════════════════════════════════════════
    bool solveIKWithSeed(
        const geometry_msgs::msg::Pose& target_pose,
        const std::vector<double>& joint_values,
        std::unique_ptr<moveit::core::RobotState>& robot_state,
        double timeout_seconds = 0.1)
    {
        const auto robot_model = move_group_interface_->getRobotModel();
        if (!robot_model) { RCLCPP_ERROR(get_logger(), "Robot model is null."); return false; }
        const auto* joint_model_group = robot_model->getJointModelGroup(move_group_interface_->getName());
        if (!joint_model_group) { RCLCPP_ERROR(get_logger(), "Joint model group not found."); return false; }
        if (!joint_values.empty() && joint_values.size() != joint_model_group->getVariableCount()) {
            RCLCPP_ERROR(get_logger(), "Mismatch joint count: expected %zu, got %zu.",
                         static_cast<size_t>(joint_model_group->getVariableCount()), joint_values.size());
            return false;
        }
        robot_state = std::make_unique<moveit::core::RobotState>(robot_model);
        robot_state->setToDefaultValues();
        if (!joint_values.empty()) robot_state->setJointGroupPositions(joint_model_group, joint_values);
        const bool found_ik = robot_state->setFromIK(joint_model_group, target_pose, timeout_seconds);
        if (found_ik) robot_state->update();
        return found_ik;
    }

    CollisionInfo checkCollisionWithState(
        const moveit::core::RobotState& robot_state)
    {
        CollisionInfo info;

        if (!waitForSceneReady())
        {
            RCLCPP_WARN(get_logger(), "Scene not ready");
            return info;
        }

        auto scene_snapshot = getSceneSnapshot();

        if (!scene_snapshot)
        {
            RCLCPP_WARN(get_logger(), "Scene snapshot null");
            return info;
        }

        collision_detection::CollisionRequest req;
        collision_detection::CollisionResult res;

        req.contacts = true;

        // Octomap nên lấy nhiều contact hơn
        req.max_contacts = 1000;
        req.max_contacts_per_pair = 1000;

        scene_snapshot->checkCollision(
            req,
            res,
            robot_state);

        if (!res.collision)
        {
            return info;
        }

        info.collision = true;

        const Eigen::Isometry3d& tcp_tf =
            robot_state.getGlobalLinkTransform("tcp0");

        const Eigen::Isometry3d world_to_tcp =
            tcp_tf.inverse();

        Eigen::Vector3d centroid_world =
            Eigen::Vector3d::Zero();

        Eigen::Vector3d avg_normal_tcp =
            Eigen::Vector3d::Zero();

        double max_depth = 0.0;
        double total_depth = 0.0;
        size_t count = 0;

        for (const auto& pair : res.contacts)
        {
            for (const auto& contact : pair.second)
            {
                CollisionInfo::ContactSample sample;

                sample.body1 = pair.first.first;
                sample.body2 = pair.first.second;

                sample.position_world = contact.pos;
                sample.depth = contact.depth;

                Eigen::Vector3d tcp_normal =
                    world_to_tcp.linear() * contact.normal;

                if (tcp_normal.norm() > 1e-9)
                    tcp_normal.normalize();

                sample.normal = tcp_normal;

                info.contact_points.emplace_back(sample);

                centroid_world += contact.pos;
                avg_normal_tcp += tcp_normal;

                total_depth += contact.depth;
                max_depth = std::max(max_depth, contact.depth);

                ++count;
            }
        }

        if (count == 0)
            return info;

        centroid_world /= static_cast<double>(count);

        if (avg_normal_tcp.norm() > 1e-9)
            avg_normal_tcp.normalize();

        info.contact_count = count;

        info.position_world = centroid_world;
        info.position = world_to_tcp * centroid_world;

        info.normal = avg_normal_tcp;

        info.depth = max_depth;
        info.total_depth = total_depth;
        info.avg_depth = total_depth / count;

        info.body1 = info.contact_points.front().body1;
        info.body2 = info.contact_points.front().body2;

        RCLCPP_INFO(
            get_logger(),
            "Collision: contacts=%zu max_depth=%.4f avg_depth=%.4f total_depth=%.4f",
            info.contact_count,
            info.depth,
            info.avg_depth,
            info.total_depth);

        return info;
    }

    CollisionInfo checkCollisionAtTarget(
        const geometry_msgs::msg::Pose& target_pose, double ik_timeout_seconds = 0.1)
    {
        CollisionInfo info;
        std::unique_ptr<moveit::core::RobotState> robot_state;
        if (!solveIKWithSeed(target_pose, std::vector<double>{}, robot_state, ik_timeout_seconds)) {
            RCLCPP_WARN(get_logger(), "IK not found for target pose"); return info;
        }
        info = checkCollisionWithState(*robot_state);
        if (info.collision) RCLCPP_WARN(get_logger(), "Collision detected at target pose!");
        return info;
    }

    // Tìm range roll/pitch hợp lệ tại một vị trí xyz cho trước.
    // Sample theo grid với step_rad, dùng solveIKWithSeed để check IK.
    // base_yaw: góc yaw cố định (thường lấy từ target_position_[i][5]).
    // search_range: phạm vi tìm kiếm ±search_range (radian) cho cả roll và pitch.
    // step_rad: bước nhảy khi sample (radian).
    OrientationRange getOrientationRange(
        double x, double y, double z,
        double base_yaw,
        const std::vector<double>& current_joint_values,
        double search_range = M_PI / 2.0,   // ±90 deg
        double step_rad     = M_PI / 18.0)  // 10 deg step
    {
        OrientationRange range;
        bool found_any = false;

        // Precompute xyz offset position dùng roll/pitch gốc để lấy base offset,
        // nhưng ta sẽ override roll/pitch trong loop nên dùng trực tiếp xyz.
        for (double roll = -search_range; roll <= search_range + 1e-9; roll += step_rad)
        {
            for (double pitch = -search_range; pitch <= search_range + 1e-9; pitch += step_rad)
            {
                // Build pose với xyz + roll/pitch/yaw hiện tại
                //std::array<double, 6> test_pos = {x, y, z, roll, pitch, base_yaw};

                double ox, oy, oz;
                compute_offset_position(x, y, z, roll, pitch, base_yaw,
                                        object_offset_, ox, oy, oz);

                std::array<double, 6> offset_pos = {ox, oy, oz, roll, pitch, base_yaw};
                geometry_msgs::msg::Pose test_pose = transformToBaseFrame(offset_pos);

                std::unique_ptr<moveit::core::RobotState> robot_state;
                const bool ik_ok = solveIKWithSeed(
                    test_pose, current_joint_values, robot_state, 0.05);

                if (!ik_ok) continue;

                // Update range
                if (!found_any) {
                    range.roll_min  = roll;
                    range.roll_max  = roll;
                    range.pitch_min = pitch;
                    range.pitch_max = pitch;
                    found_any = true;
                } else {
                    range.roll_min  = std::min(range.roll_min,  roll);
                    range.roll_max  = std::max(range.roll_max,  roll);
                    range.pitch_min = std::min(range.pitch_min, pitch);
                    range.pitch_max = std::max(range.pitch_max, pitch);
                }
            }
        }

        if (found_any) {
            RCLCPP_INFO(get_logger(),
                "OrientationRange at (%.3f, %.3f, %.3f): "
                "roll=[%.3f, %.3f] pitch=[%.3f, %.3f] (rad)",
                x, y, z,
                range.roll_min,  range.roll_max,
                range.pitch_min, range.pitch_max);
        } else {
            RCLCPP_WARN(get_logger(),
                "OrientationRange: no valid IK found at (%.3f, %.3f, %.3f)", x, y, z);
        }

        return range;
    }

    // Check collision tại trạng thái hiện tại của robot (không cần IK).
    // Dùng getCurrentState() từ move_group để lấy joint state thực tế.
    // Returns CollisionInfo với collision=false nếu không có va chạm.
    CollisionInfo checkCurrentStateCollision()
    {
        // Lấy current state từ move_group (reflect joint state thực của robot)
        const auto current_state = move_group_interface_->getCurrentState(2.0);
        if (!current_state) {
            RCLCPP_WARN(get_logger(), "checkCurrentStateCollision: failed to get current state");
            return CollisionInfo{};
        }
        return checkCollisionWithState(*current_state);
    }

    bool clearCurrentStateContacts(int max_iterations = 50)
    {
        for (int i = 0; i < max_iterations; ++i) {
            // Lấy state thực của robot
            const auto current_state = move_group_interface_->getCurrentState(2.0);
            if (!current_state) {
                RCLCPP_WARN(get_logger(), "clearCurrentStateContacts: failed to get current state");
                return false;
            }

            current_state->update();

            const auto col = checkCollisionWithState(*current_state);

            if (!col.collision) {
                RCLCPP_INFO(get_logger(),
                    "clearCurrentStateContacts: no collision at iteration %d", i);
                return true;
            }

            RCLCPP_WARN(get_logger(),
                "clearCurrentStateContacts: collision detected (iter=%d, contacts=%zu, depth=%.4f), masking...",
                i, col.contact_count, col.depth);

            if (!applyMaskedOctomapFromCache(col.contact_points, 0.02)) {
                RCLCPP_WARN(get_logger(),
                    "clearCurrentStateContacts: applyMaskedOctomapFromCache failed");
                return false;
            }

            // Refresh để cached_scene_ phản ánh octomap mới
            while (rclcpp::ok() && !refreshPlanningScene()) {
                RCLCPP_WARN(get_logger(), "clearCurrentStateContacts: retrying scene refresh...");
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        }

        RCLCPP_ERROR(get_logger(),
            "clearCurrentStateContacts: still in collision after %d iterations", max_iterations);
        return false;
    }

    // ════════════════════════════════════════════════════════════════════
    //  POSE COMPUTATION HELPERS
    // ════════════════════════════════════════════════════════════════════
    void compute_offset_position(
        double x, double y, double z,
        double roll, double pitch, double yaw,
        double offset_distance,
        double& x_out, double& y_out, double& z_out)
    {
        double cr = std::cos(roll),  sr = std::sin(roll);
        double cp = std::cos(pitch), sp = std::sin(pitch);
        double cy = std::cos(yaw),   sy = std::sin(yaw);
        double R02 = cy*sp*cr + sy*sr;
        double R12 = sy*sp*cr - cy*sr;
        double R22 = cp*cr;
        x_out = x - offset_distance * R02;
        y_out = y - offset_distance * R12;
        z_out = z - offset_distance * R22;
    }

    geometry_msgs::msg::Pose offsetPose(
        const geometry_msgs::msg::Pose& input_pose,
        double y_offset, double z_offset, double yaw_offset)
    {
        pose = input_pose;
        tf2::Quaternion q;
        tf2::fromMsg(pose.orientation, q);
        tf2::Vector3 offset_tcp(y_offset, 0, z_offset);
        tf2::Vector3 offset_world = tf2::quatRotate(q, offset_tcp);
        tf2::Quaternion q_yaw;
        q_yaw.setRPY(0.0, 0.0, yaw_offset);
        tf2::Quaternion q_new = q * q_yaw;
        q_new.normalize();
        pose.position.x += offset_world.x();
        pose.position.y += offset_world.y();
        pose.position.z += offset_world.z();
        pose.orientation = tf2::toMsg(q_new);
        return pose;
    }

    bool captureTargetBaseTransform()
    {
        if (!tf_buffer_ ||
            !tf_buffer_->canTransform("link0", "tcp0", tf2::TimePointZero, tf2::durationFromSec(1.0))) {
            target_base_transform_ready_ = false;
            RCLCPP_WARN(this->get_logger(), "TF buffer not available for target pose conversion.");
            return false;
        }
        try {
            target_base_transform_ = tf_buffer_->lookupTransform("link0", "tcp0", tf2::TimePointZero);
            target_base_transform_ready_ = true;
            return true;
        } catch (const tf2::TransformException& ex) {
            target_base_transform_ready_ = false;
            RCLCPP_ERROR(this->get_logger(), "Failed to capture target transform: %s", ex.what());
            return false;
        }
    }

    geometry_msgs::msg::Pose transformToBaseFrame(const std::array<double, 6>& position)
    {
        geometry_msgs::msg::Pose msg;
        msg.position.x = position[0]; msg.position.y = position[1]; msg.position.z = position[2];
        tf2::Quaternion q_new;
        q_new.setRPY(position[3], position[4], position[5]);
        q_new.normalize();
        tf2::convert(q_new, msg.orientation);
        geometry_msgs::msg::Pose transformed_pose;
        if (target_base_transform_ready_) {
            tf2::doTransform(msg, transformed_pose, target_base_transform_);
            return transformed_pose;
        }
        if (!tf_buffer_ ||
            !tf_buffer_->canTransform("link0", "tcp0", tf2::TimePointZero, tf2::durationFromSec(1.0))) {
            RCLCPP_WARN(this->get_logger(), "TF buffer not available. Returning original pose.");
            return msg;
        }
        try {
            auto transform = tf_buffer_->lookupTransform("link0", "tcp0", tf2::TimePointZero);
            tf2::doTransform(msg, transformed_pose, transform);
        } catch (const tf2::TransformException& ex) {
            RCLCPP_ERROR(this->get_logger(), "Transform failed: %s", ex.what());
            return msg;
        }
        return transformed_pose;
    }

    geometry_msgs::msg::Pose targetPositionToBasePose(std::size_t idx)
    {
        if (idx >= target_position_.size()) {
            RCLCPP_ERROR(this->get_logger(), "Target index %zu out of range", idx);
            return geometry_msgs::msg::Pose();
        }
        const auto& target = target_position_[idx];
        compute_offset_position(
            target[0], target[1], target[2], target[3], target[4], target[5],
            object_offset_,
            target_idx_position_[0], target_idx_position_[1], target_idx_position_[2]);
        target_idx_position_[3] = target[3];
        target_idx_position_[4] = target[4];
        target_idx_position_[5] = target[5];
        auto converted_pose = transformToBaseFrame(target_idx_position_);
        if (idx >= target_pose_list_.size()) target_pose_list_.resize(idx + 1);
        target_pose_list_[idx] = converted_pose;
        return converted_pose;
    }

    void rebuildTargetPoseList()
    {
        target_pose_list_.resize(target_position_.size());
        for (std::size_t i = 0; i < target_position_.size(); ++i)
            target_pose_list_[i] = targetPositionToBasePose(i);
    }

    geometry_msgs::msg::Pose jointStatesToPose(const std::vector<double>& joint_values)
    {
        auto robot_model = move_group_interface_->getRobotModel();
        const moveit::core::JointModelGroup* jmg = robot_model->getJointModelGroup("indy_manipulator");
        moveit::core::RobotState robot_state(robot_model);
        robot_state.setJointGroupPositions(jmg, joint_values);
        robot_state.update();
        const std::string& eef_link = move_group_interface_->getEndEffectorLink();
        const Eigen::Isometry3d& transform = robot_state.getGlobalLinkTransform(eef_link);
        geometry_msgs::msg::Pose pose;
        pose.position.x = transform.translation().x();
        pose.position.y = transform.translation().y();
        pose.position.z = transform.translation().z();
        Eigen::Quaterniond q(transform.rotation());
        pose.orientation.x = q.x(); pose.orientation.y = q.y();
        pose.orientation.z = q.z(); pose.orientation.w = q.w();
        return pose;
    }

    double estimateLink5MaxReach(
        const std::string& link_name = "link5",
        int samples = 50000)
    {
        auto robot_model = move_group_interface_->getRobotModel();

        if (!robot_model)
        {
            RCLCPP_ERROR(get_logger(), "Robot model null");
            return 0.0;
        }

        const auto* jmg =
            robot_model->getJointModelGroup(
                move_group_interface_->getName());

        if (!jmg)
        {
            RCLCPP_ERROR(get_logger(), "JointModelGroup null");
            return 0.0;
        }

        moveit::core::RobotState state(robot_model);

        double max_reach = 0.0;
        Eigen::Vector3d best_pos = Eigen::Vector3d::Zero();

        for (int i = 0; i < samples; ++i)
        {
            state.setToRandomPositions(jmg);
            state.update();

            const Eigen::Vector3d p =
                state.getGlobalLinkTransform(link_name)
                    .translation();

            const double d = p.norm();

            if (d > max_reach)
            {
                max_reach = d;
                best_pos = p;
            }
        }

        RCLCPP_INFO(
            get_logger(),
            "%s max reach = %.4f m  at (%.3f %.3f %.3f)",
            link_name.c_str(),
            max_reach,
            best_pos.x(),
            best_pos.y(),
            best_pos.z());

        return max_reach;
    }

    // ════════════════════════════════════════════════════════════════════
    //  RECOMPUTE HELPERS (roll/pitch adjustment)
    // ════════════════════════════════════════════════════════════════════
    std::tuple<double,double> computeRollPitchFromCollision(
        const Eigen::Vector3d& target_position_tcp,
        const Eigen::Vector3d& collision_position_tcp,
        double roll_prev,
        double pitch_prev,
        double reach5,
        double reach_tcp,
        double penalty)
    {
        if(target_position_tcp.norm() < 1e-6 ||
        collision_position_tcp.norm() < 1e-6)
        {
            RCLCPP_WARN(
                get_logger(),
                "Invalid TCP positions for collision recompute. Stop recompute.");
            return {roll_prev, pitch_prev};
        }

        //----------------------------------------
        // contact direction
        //----------------------------------------

        Eigen::Vector3d contact_dir =
            (target_position_tcp - collision_position_tcp).normalized();

        double pitch_contact =
            std::atan2(
                contact_dir.x(),
                contact_dir.z());

        double roll_contact =
            -std::atan2(
                contact_dir.y(),
                contact_dir.z());

        double roll_max, pitch_max, roll_min, pitch_min;
        roll_max = roll_prev;
        pitch_max = pitch_prev;
        roll_min = roll_contact;
        pitch_min = pitch_contact;
        if (roll_contact > roll_prev) 
        {
            roll_max = roll_contact;
            roll_min = roll_prev;
        }
        if (pitch_contact > pitch_prev) 
        {
            pitch_max = pitch_contact;
            pitch_min = pitch_prev;
        }

        //----------------------------------------
        // search theta
        //----------------------------------------

        double roll_new, pitch_new;

        for(double theta = 0.0;
            theta < 2.0 * M_PI;
            theta += M_PI / 72.0)   // 5 deg
        {
            Eigen::Vector3d wrist =
                computeWristCenter(
                    target_position_tcp,
                    theta,
                    penalty,
                    reach5,
                    reach_tcp);

            //------------------------------------
            // tcp z direction generated by theta
            //------------------------------------

            Eigen::Vector3d tcp_z =
                (target_position_tcp - wrist).normalized();

            if(tcp_z.z() < 0.0)
                tcp_z = -tcp_z;

            double pitch =
                std::atan2(
                    tcp_z.x(),
                    tcp_z.z());

            double roll =
                -std::atan2(
                    tcp_z.y(),
                    tcp_z.z());

            if ((roll > roll_max && pitch > pitch_max)
                || (roll < roll_min && pitch < pitch_min)
                || (roll > roll_max && pitch < pitch_min)
                || (roll < roll_min && pitch > pitch_max)
            )
            {
                roll_new = roll;
                pitch_new = pitch;
                break;
            }

        }

        RCLCPP_INFO(
            get_logger(),
            "Collision avoidance: roll=%.2f deg pitch=%.2f deg",
            roll_new * 180.0 / M_PI,
            pitch_new * 180.0 / M_PI);

        return {roll_new, pitch_new};
    }

    // Tính basis (u, v) ổn định cho vector n bất kỳ.
    // Dùng Gram-Schmidt với danh sách fallback reference vectors
    // để đảm bảo u luôn nhất quán với cùng một n.
    static std::pair<Eigen::Vector3d, Eigen::Vector3d>
    stableBasis(const Eigen::Vector3d& n)
    {
        // Thử lần lượt các reference vector cho đến khi tìm được
        // cái không song song với n (|cross| > 1e-3)
        const std::array<Eigen::Vector3d, 3> refs = {
            Eigen::Vector3d::UnitX(),
            Eigen::Vector3d::UnitY(),
            Eigen::Vector3d::UnitZ()
        };

        Eigen::Vector3d u;
        for (const auto& ref : refs) {
            u = n.cross(ref);
            if (u.norm() > 1e-3) { u.normalize(); break; }
        }

        Eigen::Vector3d v = n.cross(u).normalized();
        return {u, v};
    }

    std::pair<double, double> estimateRollPitchFromTheta(
        const Eigen::Vector3d& target,
        double theta,
        double penalty,
        double max_reach_link5,
        double max_reach_tcp)
    {
        double R1 = max_reach_link5;
        double R2 = max_reach_tcp - max_reach_link5;
        double d  = target.norm();

        if (d < 1e-6) return {0.0, 0.0};
        if (d + R2 < R1) R1 = 0.9 * d;

        const double a = (R1*R1 - R2*R2 + d*d) / (2.0 * d);
        const double r = penalty * std::sqrt(std::max(0.0, R1*R1 - a*a));

        RCLCPP_INFO(
            get_logger(),
            "estimateRollPitchFromTheta: d=%.4f a=%.4f r=%.4f, R1=%.4f, R2=%.4f",
            d, a, r, R1, R2);

        const Eigen::Vector3d n = target.normalized();
        const auto [u, v] = stableBasis(n);          // ← stable, deterministic

        const Eigen::Vector3d wrist =
            a * n + r * (std::cos(theta) * u + std::sin(theta) * v);

        Eigen::Vector3d tcp_z = (target - wrist).normalized();
        if (tcp_z.z() < 0.0) tcp_z = -tcp_z;

        const double pitch = std::atan2(tcp_z.x(), tcp_z.z());
        const double roll  = - std::atan2(tcp_z.y(), tcp_z.z());

        return {roll, pitch};
    }

    Eigen::Vector3d computeWristCenter(
        const Eigen::Vector3d& target,
        double theta,
        double penalty,
        double max_reach_link5,
        double max_reach_tcp)
    {
        double R1 = max_reach_link5;
        double R2 = max_reach_tcp - max_reach_link5;
        double d  = target.norm();

        if (d < 1e-6) return Eigen::Vector3d::Zero();
        if (d + R2 < R1) R1 = 0.9 * d;

        const double a = (R1*R1 - R2*R2 + d*d) / (2.0 * d);
        const double r = penalty * std::sqrt(std::max(0.0, R1*R1 - a*a));

        RCLCPP_INFO(
            get_logger(),
            "computeWristCenter: d=%.4f a=%.4f r=%.4f, R1=%.4f, R2=%.4f",
            d, a, r, R1, R2);

        const Eigen::Vector3d n = target.normalized();
        const auto [u, v] = stableBasis(n);          // ← stable, deterministic

        return a * n + r * (std::cos(theta) * u + std::sin(theta) * v);
    }

    // ════════════════════════════════════════════════════════════════════
    //  POSE CHECK & RECOMPUTE
    // ════════════════════════════════════════════════════════════════════

    // Helper: scan forward along the approach axis from the confirmed IK pose,
    // masking any octomap voxels that block the path segment.
    // Returns true if all iterations finished without error.
    bool clearApproachPath(
        const std::array<double, 6>& base_offset_pos,
        const std::vector<double>& joint_values,
        int max_crop_iters = 100)
    {
        std::unique_ptr<moveit::core::RobotState> sub_state;
        for (int k = 1; k < max_crop_iters; ++k) {
            compute_offset_position(
                base_offset_pos[0], base_offset_pos[1], base_offset_pos[2],
                base_offset_pos[3], base_offset_pos[4], base_offset_pos[5],
                k * 0.01,
                sub_test_position_ref_offset[0],
                sub_test_position_ref_offset[1],
                sub_test_position_ref_offset[2]);
            sub_test_position_ref_offset[3] = base_offset_pos[3];
            sub_test_position_ref_offset[4] = base_offset_pos[4];
            sub_test_position_ref_offset[5] = base_offset_pos[5];

            sub_input_test_pose = transformToBaseFrame(sub_test_position_ref_offset);
            sub_found_test_ik   = solveIKWithSeed(sub_input_test_pose, joint_values, sub_state, 0.1);

            if (!sub_found_test_ik || !sub_state) {
                RCLCPP_WARN(get_logger(),
                    "clearApproachPath: no IK at step %d, stopping early", k);
                break;
            }

            const auto col = checkCollisionWithState(*sub_state);
            if (!col.collision && k >= 30) {
                RCLCPP_INFO(get_logger(),
                    "clearApproachPath: path clear at step %d", k);
                break;
            }
            RCLCPP_WARN(get_logger(),
                "clearApproachPath: collision at step %d (contacts=%zu, depth=%.4f), masking...",
                k, col.contact_count, col.depth);
            if (col.collision)
            {
                if (!applyMaskedOctomapFromCache(col.contact_points, 0.02)) {
                    RCLCPP_WARN(get_logger(), "clearApproachPath: masked octomap failed");
                    return false;
                }
                while (rclcpp::ok() && !refreshPlanningScene()) {
                    RCLCPP_WARN(get_logger(), "Retrying refresh planning scene");
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                }
            }
        }
        return true;
    }

    // Helper: iteratively mask small-depth contacts from the octomap until
    // the given robot state is collision-free or max iterations are exhausted.
    // Returns true if contacts were cleared, false otherwise.
    bool clearSmallContacts(
        moveit::core::RobotState& ik_state,
        CollisionInfo& col,
        int max_crop_iters = 100)
    {
        for (int k = 0; k < max_crop_iters; ++k) {
            if (!col.collision) return true;
            if (!applyMaskedOctomapFromCache(col.contact_points, 0.02)) {
                RCLCPP_WARN(get_logger(), "clearSmallContacts: masked octomap failed");
                return false;
            }
            while (rclcpp::ok() && !refreshPlanningScene()) {
                RCLCPP_WARN(get_logger(), "Retrying refresh planning scene");
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
            col = checkCollisionWithState(ik_state);
        }
        return false;  // still in collision after max iterations
    }

    void posecheck_and_recompute(
        const std::array<double, 10>& test_position,
        const std::vector<double>& test_joint_values,
        std::size_t idx)
    {
        test_position_ref = {
            test_position[0], test_position[1], test_position[2],
            test_position[3], test_position[4], test_position[5]
        };
        test_position_ref_offset = test_position_ref;
        found_test_ik = false;
        obs_check     = true;

        const int max_iterations = 100;
        int iteration = 0;

        while (rclcpp::ok() && iteration < max_iterations) {
            ++iteration;

            // ── Compute offset pose ───────────────────────────────────────
            compute_offset_position(
                test_position_ref[0], test_position_ref[1], test_position_ref[2],
                test_position_ref[3], test_position_ref[4], test_position_ref[5],
                object_offset_,
                test_position_ref_offset[0],
                test_position_ref_offset[1],
                test_position_ref_offset[2]);
            test_position_ref_offset[3] = test_position_ref[3];
            test_position_ref_offset[4] = test_position_ref[4];
            input_test_pose = transformToBaseFrame(test_position_ref_offset);

            // ── Solve IK ──────────────────────────────────────────────────
            std::unique_ptr<moveit::core::RobotState> ik_state;
            found_test_ik = solveIKWithSeed(input_test_pose, test_joint_values, ik_state, 0.1);

            CollisionInfo col;
            obs_check = false;

            if (found_test_ik && ik_state) {
                col       = checkCollisionWithState(*ik_state);
                obs_check = col.collision;

                // ── Small-depth contact: try masking octomap voxels ───────
                constexpr double kSmallContactDepth = 0.02;
                if (obs_check && col.depth < kSmallContactDepth) {
                    const bool cleared = clearSmallContacts(*ik_state, col);
                    obs_check = col.collision;   // updated by clearSmallContacts
                    if (!cleared) {
                        RCLCPP_WARN(get_logger(),
                            "Small contacts not cleared for idx %zu, continuing sweep", idx);
                        obs_check = true;
                    }
                }
            }

            // ── Pass: IK found and collision-free ─────────────────────────
            if (found_test_ik && !obs_check) {
                RCLCPP_INFO(get_logger(), "IK found at iteration %d", iteration);
                target_position_[idx][3] = test_position_ref[3];
                target_position_[idx][4] = test_position_ref[4];

                // Scan approach path and pre-clear any blocking voxels
                clearApproachPath(test_position_ref_offset, test_joint_values);
                break;
            }

            // ── Adjust roll/pitch for next iteration ──────────────────────
            RCLCPP_WARN(get_logger(),
                "Recompute for pose idx %zu, iteration %d", idx, iteration);

            const double theta   = static_cast<double>(iteration) * M_PI / 5.0;
            const double penalty = static_cast<double>(iteration) / 100.0;

            if (!found_test_ik) {
                RCLCPP_WARN(get_logger(),
                    "No IK found, adjusting roll/pitch via theta sweep");
                const auto [r, p] = estimateRollPitchFromTheta(
                    Eigen::Vector3d(test_position_ref[0],
                                    test_position_ref[1],
                                    test_position_ref[2]),
                    theta, penalty, near_tcp_range_, tcp_range_ + offset_distance_);
                RCLCPP_INFO(get_logger(), "Adjusted roll=%.3f pitch=%.3f", r, p);
                test_position_ref[3] = r;
                test_position_ref[4] = p;
            } else {
                RCLCPP_WARN(get_logger(),
                    "IK in collision (depth=%.4f), adjusting via collision avoidance",
                    col.depth);
                const auto [r, p] = computeRollPitchFromCollision(
                    Eigen::Vector3d(test_position_ref[0],
                                    test_position_ref[1],
                                    test_position_ref[2]),
                    col.position,
                    test_position_ref[3], test_position_ref[4],
                    near_tcp_range_, tcp_range_ + offset_distance_, penalty);
                RCLCPP_INFO(get_logger(), "Adjusted roll=%.3f pitch=%.3f", r, p);
                test_position_ref[3] = r;
                test_position_ref[4] = p;
            }
        }

        if (iteration >= max_iterations)
            RCLCPP_ERROR(get_logger(),
                "Max recompute iterations reached for idx %zu", idx);

        pose_check = found_test_ik && !obs_check;
    }

    // ════════════════════════════════════════════════════════════════════
    //  CLUSTERING
    // ════════════════════════════════════════════════════════════════════
    double poseDistance(const std::array<double, 10>& a, const std::array<double, 10>& b)
    {
        const double dx = a[0] - b[0], dy = a[1] - b[1], dz = a[2] - b[2];
        return std::sqrt(dx*dx + dy*dy + dz*dz);
    }

    std::vector<std::vector<ClusterEntry>> clusterByDistance(
        const std::vector<std::array<double, 10>>& poses, double threshold = 0.15)
    {
        std::vector<std::vector<ClusterEntry>> clusters;
        std::vector<bool> assigned(poses.size(), false);
        for (size_t i = 0; i < poses.size(); ++i) {
            if (assigned[i]) continue;
            std::vector<ClusterEntry> cluster;
            cluster.push_back({poses[i], i});
            assigned[i] = true;
            for (size_t j = i + 1; j < poses.size(); ++j) {
                if (!assigned[j] && poseDistance(poses[i], poses[j]) <= threshold) {
                    cluster.push_back({poses[j], j});
                    assigned[j] = true;
                }
            }
            clusters.push_back(cluster);
        }
        return clusters;
    }

    // ════════════════════════════════════════════════════════════════════
    //  ACTION CLIENTS — callMoveRobot / callMoveToHome / sendGripperCommand
    // ════════════════════════════════════════════════════════════════════
    template<typename ClientT>
    bool ensureActionServerReady(ClientT& client, bool& ready_cache, const char* action_name)
    {
        if (ready_cache) return true;
        if (!client->wait_for_action_server(std::chrono::seconds(5))) {
            RCLCPP_ERROR(get_logger(), "Action server '%s' không khả dụng!", action_name); return false;
        }
        ready_cache = true;
        return true;
    }

    template<typename FutureT>
    bool waitForGoalResponse(FutureT& future_goal, const char* action_name)
    {
        if (future_goal.wait_for(std::chrono::seconds(2)) != std::future_status::ready) {
            RCLCPP_ERROR(get_logger(), "Timeout waiting for '%s' goal response", action_name); return false;
        }
        return true;
    }

    void callMoveToHome(const std::vector<double>& joint_positions, size_t id, size_t pass_permit = 0)
    {
        if (!ensureActionServerReady(move_to_home_client_, home_action_ready_, "move_to_home")) return;
        auto goal_msg = MoveToHome::Goal();
        goal_msg.joint_positions = joint_positions; goal_msg.id = id; goal_msg.pass_permit = pass_permit;
        auto send_goal_options = rclcpp_action::Client<MoveToHome>::SendGoalOptions();
        if ((bypass && id != 9) || (pass_all_ && id != 9)) return;
        auto future_goal = move_to_home_client_->async_send_goal(goal_msg, send_goal_options);
        if (!waitForGoalResponse(future_goal, "move_to_home")) { home_action_ready_ = false; return; }
        auto goal_handle = future_goal.get();
        if (!goal_handle) { RCLCPP_ERROR(get_logger(), "Gửi action goal thất bại!"); return; }
        auto future_result = move_to_home_client_->async_get_result(goal_handle);
        auto result = future_result.get();
        if (result.code == rclcpp_action::ResultCode::SUCCEEDED) {
            RCLCPP_INFO(get_logger(), "Move to home thành công: %s", result.result->message.c_str());
        } else {
            RCLCPP_ERROR(get_logger(), "Move to home thất bại!");
            callMoveRobot(offsetPose(jointStatesToPose(joint_positions), 0.0, 0.5, 0.0),
                          jointStatesToPose(joint_positions), id, 2);
        }
    }

    void callMoveRobot(
        const geometry_msgs::msg::Pose& start_pose,
        const geometry_msgs::msg::Pose& target_pose,
        size_t id, size_t mode)
    {
        if (pass_all_ || bypass) return;
        if (!ensureActionServerReady(move_client_, move_action_ready_, "robot_move_action")) return;
        auto goal_msg = MoveRobot::Goal();
        goal_msg.mode = mode; goal_msg.id = id;
        goal_msg.start_pose = start_pose; goal_msg.target_pose = target_pose;
        auto send_goal_options = rclcpp_action::Client<MoveRobot>::SendGoalOptions();
        auto future_goal = move_client_->async_send_goal(goal_msg, send_goal_options);
        if (!waitForGoalResponse(future_goal, "robot_move_action")) { move_action_ready_ = false; return; }
        auto goal_handle = future_goal.get();
        if (!goal_handle) { RCLCPP_ERROR(get_logger(), "Gửi action goal thất bại!"); return; }
        auto future_result = move_client_->async_get_result(goal_handle);
        auto result = future_result.get();
        move_success_ = false;
        if (result.code == rclcpp_action::ResultCode::SUCCEEDED) {
            RCLCPP_INFO(get_logger(), "Move robot thành công: %s", result.result->message.c_str());
            move_success_ = true;
        } else {
            RCLCPP_ERROR(get_logger(), "Move robot thất bại!");
            if (id == 1)      bypass = true;
            else if (id == 3) callMoveRobot(target_pose, offsetPose(target_pose, 0.0, 0.0, 0.0), 30, 0);
        }
    }

    void sendGripperCommand(double position, size_t id, size_t pass_permit = 0)
    {
        if ((pass_all_ || bypass) && !(bypass && id == 8)) return;
        if (!ensureActionServerReady(gripper_client_, gripper_action_ready_, "gripper_action")) return;
        while (rclcpp::ok()) {
            auto goal_msg = GripperControl::Goal();
            goal_msg.position = position; goal_msg.id = id; goal_msg.pass_permit = pass_permit;
            auto goal_handle_future = gripper_client_->async_send_goal(goal_msg);
            if (!waitForGoalResponse(goal_handle_future, "gripper_action")) { gripper_action_ready_ = false; return; }
            auto goal_handle = goal_handle_future.get();
            if (!goal_handle) { RCLCPP_ERROR(get_logger(), "Gửi lệnh gripper thất bại!"); return; }
            auto result_future = gripper_client_->async_get_result(goal_handle);
            auto result = result_future.get();
            if (result.code == rclcpp_action::ResultCode::SUCCEEDED) {
                RCLCPP_INFO(get_logger(), "Gripper điều khiển thành công."); break;
            }
            RCLCPP_ERROR(get_logger(), "Gripper thất bại");
            resendGripperCommand();
        }
    }

    void resendGripperCommand()
    {
        callMoveRobot(target_pose, target_pose, 1, 0);
        sendGripperCommand(0.8, 2000);
    }

    // ════════════════════════════════════════════════════════════════════
    //  PUBLISHERS
    // ════════════════════════════════════════════════════════════════════
    void publisher_callback(bool flag, double x, bool pause = true, bool skip = false)
    {
        res_msgs::msg::PoseRes res;
        res_msgs::msg::ResFlag flag_msg;
        flag_msg.flag = flag; flag_msg.x = x; flag_msg.pause = pause; flag_msg.skip = skip;
        res.pose_res.push_back(flag_msg);
        publisher_->publish(res);
    }

    void time_publisher(double end_time, bool check = true, int count = 0)
    {
        std::lock_guard<std::mutex> lock(pub_mutex);
        collect_msgs::msg::CollectMsg msg;
        collect_msgs::msg::CollectTime time;
        total_time = 0.0;
        total_time = end_time - start_detection_time - positioning_time - detection_time - temp_total_time;
        temp_total_time = temp_total_time + total_time;
        time.total_time = total_time;
        time.detection_time = detection_time; time.positioning_time = positioning_time;
        temp_total_time = 0.0;
        RCLCPP_WARN(this->get_logger(),
            "DEBUG time_publisher: end_time=%.3f, start_detection_time=%.3f, positioning_time=%.3f, "
            "detection_time=%.3f, temp_total_time=%.3f, total_time=%.3f",
            end_time, start_detection_time, positioning_time, detection_time, temp_total_time, total_time);
        time.check = check; time.count = count;
        msg.collect_msg.push_back(time);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        time_publisher_->publish(msg);
    }

    void publish_depth_signal(bool computing_depth) {
        depth_signal_msgs::msg::DepthSignal msg; msg.computing_depth = computing_depth;
        depth_signal_pub->publish(msg);
    }

    void publish_position_signal(bool computing_position) {
        position_signal_msgs::msg::PositionSignal msg; msg.computing_position = computing_position;
        position_signal_pub->publish(msg);
    }

    void publish_signal(bool signal) { publish_depth_signal(signal); publish_position_signal(signal); }

    void publish_skip_signal(bool skip) {
        skip_signal_msgs::msg::SkipSignal msg; msg.skip = skip; skip_signal_pub->publish(msg);
    }

    void publish_move_signal(bool move) {
        move_signal_msgs::msg::MoveSignal msg; msg.move = move; move_signal_pub->publish(msg);
    }

    // ════════════════════════════════════════════════════════════════════
    //  MISC
    // ════════════════════════════════════════════════════════════════════
    void startConnectionMonitorThread() {
        connection_monitor_thread_ = std::thread([this]() {
            while (rclcpp::ok() && !stop_connection_monitor_) {
                is_server_ready_ = latest_connection_status_;
                is_reset_ = reset_status_;
            }
        });
    }

    void waitForReconnect() {
        while (rclcpp::ok()) {
            RCLCPP_WARN(this->get_logger(), "lost connection. Trying to reconnect...");
            if (is_reset_) { RCLCPP_INFO(this->get_logger(), "Reconnected successfully."); break; }
        }
    }

    void load_setup_params(const std::string& filename)
    {
        RCLCPP_INFO(this->get_logger(), "loading setup params");
        YAML::Node config = YAML::LoadFile(filename);
        auto setup = config["setup"];
        home_position_      = setup["HomePose"].as<std::vector<double>>();
        drop_position_      = setup["DorpPose"].as<std::vector<double>>();
        offset_distance_    = setup["OffSetDistance"].as<double>();
        y_offset_distance_  = setup["YOffSetDistance"].as<double>();
        offset_angle_       = setup["OffSetAngle"].as<double>();
        mul_mode_           = setup["Multi_collect_mode"].as<bool>();
    }

    // ════════════════════════════════════════════════════════════════════
    //  EXECUTE — single-target (original)
    // ════════════════════════════════════════════════════════════════════
    void execute1(const std::shared_ptr<GoalHandleControlRobot> goal_handle)
    {
        success_count = 0;
        auto timeout = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (!target_ready_ || !time_recieved_ || !obs_ready || !config_received_) {
            if (std::chrono::steady_clock::now() > timeout) {
                auto result = std::make_shared<ControlRobot::Result>();
                result->success = false; result->message = "Timeout waiting for target/time";
                RCLCPP_ERROR(get_logger(), "Execute timeout!");
                RCLCPP_ERROR(get_logger(), "target_ready: %s, time_recieved: %s, obs_ready: %s, config_received: %s",
                             target_ready_ ? "true" : "false", time_recieved_ ? "true" : "false",
                             obs_ready ? "true" : "false", config_received_ ? "true" : "false");
                goal_handle->abort(result);
                target_ready_ = false; time_recieved_ = false; obs_ready = false;
                publish_signal(false); return;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }

        auto result   = std::make_shared<ControlRobot::Result>();
        auto feedback = std::make_shared<ControlRobot::Feedback>();
        captureTargetBaseTransform();
        rebuildTargetPoseList();
        clearOctomapCache();

        for (size_t i = 0; i < target_position_.size(); i++) {
            applyOctomap(target_position_[i][6], target_position_[i][7],
                         target_position_[i][8], target_position_[i][9], fx_, fy_, cx_, cy_);
            pass_all_ = false; bypass = false;
            publisher_callback(true, now().seconds(), true, mul_mode_);
            std::this_thread::sleep_for(std::chrono::milliseconds(100));

            while (rclcpp::ok() && !refreshPlanningScene()) {
                RCLCPP_WARN(get_logger(), "Retrying refresh planning scene");
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }

            RCLCPP_INFO(get_logger(), "[Loop Debug] i=%zu / total=%zu | mul_mode=%s | time=%.3f",
                        i, target_position_.size(), mul_mode_ ? "true" : "false", now().seconds());

            posecheck_and_recompute(target_position_[i], home_position_, i);
            target_pose = targetPositionToBasePose(i);

            RCLCPP_INFO(get_logger(), "Updated target: x=%.2f y=%.2f z=%.2f r=%.2f p=%.2f y=%.2f",
                        target_position_[i][0], target_position_[i][1], target_position_[i][2],
                        target_position_[i][3], target_position_[i][4], target_position_[i][5]);

            test_pose = offsetPose(target_pose, 0.0, offset_distance_, 0.0);

            if (pose_check) { RCLCPP_INFO(get_logger(), "DEBUG CONSUME: CAN REACH TO TARGET"); ws_check = true; }
            else            { RCLCPP_ERROR(get_logger(), "DEBUG CONSUME: UNABLE TO REACH TO TARGET"); ws_check = false; }

            obs_check_1 = checkCollisionAtTarget(offsetPose(target_pose, y_offset_distance_, 0.0, 0.0)).collision;
            if (!obs_check_1) {
                next_pose = offsetPose(target_pose, y_offset_distance_, 0.0, 0.0);
                obs_check_2 = true; obs_check_3 = true;
                RCLCPP_ERROR(get_logger(), "CASE 1: Y-Offset");
            } else {
                obs_check_2 = checkCollisionAtTarget(offsetPose(target_pose, 0.0, 0.0, offset_angle_)).collision;
                if (!obs_check_2) {
                    next_pose = offsetPose(target_pose, 0.0, 0.0, offset_angle_);
                    obs_check_3 = true;
                    RCLCPP_ERROR(get_logger(), "CASE 2: Z-Offset");
                } else {
                    obs_check_3 = checkCollisionAtTarget(offsetPose(target_pose, 0.0, 0.0, -offset_angle_)).collision;
                    if (!obs_check_3) {
                        next_pose = offsetPose(target_pose, 0.0, 0.0, -offset_angle_);
                        RCLCPP_ERROR(get_logger(), "CASE 3: -Z-Offset");
                    }
                }
            }

            if (!ws_check) continue;

            feedback->progress = 0.0;
            goal_handle->publish_feedback(feedback);

            ocotmapCombine();
            while (rclcpp::ok() && !refreshPlanningScene()) {
                RCLCPP_WARN(get_logger(), "Retrying refresh planning scene");
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
            callMoveRobot(offsetPose(target_pose, 0.0, offset_distance_, 0.0), target_pose, 1, 2);

            if (!move_success_) {
                RCLCPP_ERROR(get_logger(), "Failed to move to pre-grasp pose, skipping target %zu", i); continue;
            }

            setOctomapCollision(true);
            sendGripperCommand(0.8, 2);
            feedback->progress = 0.10; goal_handle->publish_feedback(feedback);

            callMoveRobot(target_pose, offsetPose(target_pose, 0.0, offset_distance_, 0.0), 3, 1);
            feedback->progress = 0.25; goal_handle->publish_feedback(feedback);

            sendGripperCommand(0.0, 4);
            feedback->progress = 0.40; goal_handle->publish_feedback(feedback);

            callMoveRobot(offsetPose(target_pose, 0.0, offset_distance_, 0.0), next_pose, 5, 1);
            setOctomapCollision(false);
            feedback->progress = 0.55; goal_handle->publish_feedback(feedback);

            callMoveToHome(drop_position_, 6);
            feedback->progress = 0.70; goal_handle->publish_feedback(feedback);

            sendGripperCommand(0.8, 7);
            feedback->progress = 0.85; goal_handle->publish_feedback(feedback);

            sendGripperCommand(0.0, 8);
            feedback->progress = 0.90; goal_handle->publish_feedback(feedback);

            callMoveToHome(home_position_, 9);
            feedback->progress = 1.0; goal_handle->publish_feedback(feedback);

            result->message = "Robot come to home!";
            RCLCPP_INFO(get_logger(), "time: %f", now().seconds());
            result->success = true; goal_handle->succeed(result);
            success_count++; break;
        }

        time_publisher(now().seconds(), true, success_count);
        if (mul_mode_ && success_count != 0) {
            publisher_callback(true, 0.0, false, true);
            publisher_callback(false, now().seconds(), true, mul_mode_);
        } else {
            publisher_callback(true, 0.0, false);
            publisher_callback(false, now().seconds());
        }
        is_robot_moving_ = false; target_ready_ = false; time_recieved_ = false; obs_ready = false;
        if (success_count == 0) { publish_skip_signal(true); publish_move_signal(true); }
        publish_signal(false);
    }

    // ════════════════════════════════════════════════════════════════════
    //  EXECUTE1 — cluster-based multi-target
    // ════════════════════════════════════════════════════════════════════
    void execute(const std::shared_ptr<GoalHandleControlRobot> goal_handle)
    {
        success_count = 0;
        auto timeout = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (!target_ready_ || !time_recieved_ || !obs_ready || !config_received_) {
            if (std::chrono::steady_clock::now() > timeout) {
                auto result = std::make_shared<ControlRobot::Result>();
                result->success = false; result->message = "Timeout waiting for target/time";
                RCLCPP_ERROR(get_logger(), "Execute timeout!");
                goal_handle->abort(result);
                target_ready_ = false; time_recieved_ = false; obs_ready = false;
                publish_signal(false); return;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }

        auto result   = std::make_shared<ControlRobot::Result>();
        auto feedback = std::make_shared<ControlRobot::Feedback>();
        captureTargetBaseTransform();
        rebuildTargetPoseList();
        RCLCPP_INFO(get_logger(), "Estimated max reach of tcp: %.3f m, near_tcp_range_: %.3f m", tcp_range_, near_tcp_range_);

        constexpr double kClusterDistanceMeters = 0.15;
        auto clusters = clusterByDistance(target_position_, kClusterDistanceMeters);
        RCLCPP_INFO(get_logger(), "[Cluster] %zu raw targets -> %zu cluster(s) (threshold=%.3f m)",
                    target_position_.size(), clusters.size(), kClusterDistanceMeters);

        ValidTarget previous_target;
        bool last_cluster_reachable = false;
        clearOctomapCache();

        for (size_t ci = 0; ci < clusters.size(); ++ci) {
            const auto& cluster = clusters[ci];
            RCLCPP_INFO(get_logger(), "[Cluster %zu/%zu] %zu member(s)", ci + 1, clusters.size(), cluster.size());

            if (ci == clusters.size() - 1) {
                RCLCPP_INFO(get_logger(), "[Cluster %zu] Processing last cluster, marking as reachable", ci + 1);
                last_cluster_reachable = true;
            }
            bool cluster_started = false;
            int step_id = 3;

            for (size_t mi = 0; mi < cluster.size(); ++mi) {
                const auto& raw    = cluster[mi].pose;
                const size_t orig_i = cluster[mi].original_idx;

                applyOctomap(
                    static_cast<int>(target_position_[orig_i][6]),
                    static_cast<int>(target_position_[orig_i][7]),
                    static_cast<int>(target_position_[orig_i][8]),
                    static_cast<int>(target_position_[orig_i][9]),
                    fx_, fy_, cx_, cy_);
                pass_all_ = false; bypass = false;
                publisher_callback(true, now().seconds(), true, mul_mode_);
                std::this_thread::sleep_for(std::chrono::milliseconds(100));

                while (rclcpp::ok() && !refreshPlanningScene()) {
                    RCLCPP_WARN(get_logger(), "Retrying refresh planning scene for cluster %zu, member %zu", ci + 1, mi + 1);
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                }

                RCLCPP_INFO(get_logger(), "[Loop Debug] i=%zu / total=%zu | mul_mode=%s | time=%.3f",
                            orig_i, cluster.size(), mul_mode_ ? "true" : "false", now().seconds());

                posecheck_and_recompute(raw, home_position_, orig_i);
                const auto& checked_target = target_position_[orig_i];
                target_pose = targetPositionToBasePose(orig_i);

                RCLCPP_INFO(get_logger(), "Updated target: x=%.2f y=%.2f z=%.2f r=%.2f p=%.2f y=%.2f",
                            checked_target[0], checked_target[1], checked_target[2],
                            checked_target[3], checked_target[4], checked_target[5]);

                if (pose_check) { RCLCPP_INFO(get_logger(), "DEBUG CONSUME: CAN REACH TO TARGET"); ws_check = true; }
                else            { RCLCPP_ERROR(get_logger(), "DEBUG CONSUME: UNABLE TO REACH TO TARGET"); ws_check = false; }

                if (!ws_check) continue;

                const ValidTarget current_target{target_pose, orig_i};

                if (!cluster_started) {
                    if (!go_home_) { callMoveToHome(home_position_, 9); go_home_ = true; }
                    feedback->progress = 0.0; goal_handle->publish_feedback(feedback);

                    ocotmapCombine();

                    while (rclcpp::ok() && !refreshPlanningScene()) {
                        RCLCPP_WARN(get_logger(), "Retrying refresh planning scene");
                        std::this_thread::sleep_for(std::chrono::milliseconds(100));
                    }

                    //setOctomapCollision(true);
                    callMoveRobot(
                        offsetPose(current_target.pose, 0.0, offset_distance_, 0.0),
                        current_target.pose, step_id, 1);

                    if (!move_success_) {
                        for (size_t retry = 1; retry < 4; ++retry) {
                            RCLCPP_WARN(get_logger(), "Retrying move to first target in cluster (attempt %zu)", retry + 1);
                            callMoveRobot(
                                offsetPose(current_target.pose, 0.0, offset_distance_, 0.0),
                                offsetPose(current_target.pose, 0.0, offset_distance_ + retry * 0.1, 0.0), 
                                step_id, 1);
                            if (move_success_) 
                            {   
                                temp_pose = offsetPose(current_target.pose, 0.0, offset_distance_ + retry * 0.1, 0.0);
                                break;
                            }
                        }
                        setOctomapCollision(true);
                        if (move_success_) {
                            callMoveRobot(
                                temp_pose,
                                offsetPose(current_target.pose, 0.0, offset_distance_, 0.0), 
                                step_id, 1);
                            setOctomapCollision(false);
                        }
                    }

                    if (!move_success_) {
                        RCLCPP_ERROR(get_logger(), "Failed to move to first target in cluster, skipping cluster");
                        continue;
                    }

                    applyOctomapTemp();
                    sendGripperCommand(0.8, 2);
                    feedback->progress = 0.10; goal_handle->publish_feedback(feedback);
                    sendGripperCommand(0.0, 2);
                    //setOctomapCollision(false);

                    previous_target = current_target;
                    std::copy_n(raw.begin(), 6, previous_position_.begin());
                    cluster_started = true;
                    success_count++;
                    continue;
                }

                ocotmapCombine();
                while (rclcpp::ok() && !refreshPlanningScene()) {
                    RCLCPP_WARN(get_logger(), "Retrying refresh planning scene");
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                }
                //setOctomapCollision(true);
                callMoveRobot(previous_target.pose, current_target.pose, step_id, 1);
                if (!move_success_) {
                    for (size_t retry = 1; retry < 4; ++retry) {
                        RCLCPP_WARN(get_logger(), "Retrying move to first target in cluster (attempt %zu)", retry + 1);
                        callMoveRobot(
                            offsetPose(current_target.pose, 0.0, offset_distance_, 0.0),
                            offsetPose(current_target.pose, 0.0, offset_distance_ + retry * 0.1, 0.0), 
                            step_id, 1);
                        if (move_success_) 
                        {   
                            temp_pose = offsetPose(current_target.pose, 0.0, offset_distance_ + retry * 0.1, 0.0);
                            break;
                        }
                    }
                    if (move_success_) {
                        callMoveRobot(
                            temp_pose,
                            offsetPose(current_target.pose, 0.0, offset_distance_, 0.0), 
                            step_id, 1);
                        setOctomapCollision(false);
                    }
                }
                if (!move_success_) {
                    RCLCPP_ERROR(get_logger(), "Failed to move to target, skipping remaining targets in cluster");
                    continue;
                }
                applyOctomapTemp();
                sendGripperCommand(0.8, 2);
                sendGripperCommand(0.0, 2);
                //setOctomapCollision(false);

                const float ratio = static_cast<float>(mi + 1) / static_cast<float>(cluster.size());
                feedback->progress = 0.10f + 0.15f * ratio; goal_handle->publish_feedback(feedback);

                previous_target = current_target;
                std::copy_n(raw.begin(), 6, previous_position_.begin());
                success_count++;
            }

            if (last_cluster_reachable) {
                applyOctocmapFromTemp();
                while (rclcpp::ok() && !refreshPlanningScene()) {
                    RCLCPP_WARN(get_logger(), "Retrying refresh planning scene");
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                }
                callMoveToHome(home_position_, step_id++);
                if (!move_success_) {
                    clearCurrentStateContacts();
                    clearApproachPath(previous_position_, home_position_);
                    callMoveToHome(home_position_, step_id++);
                }
                feedback->progress = 1.0; goal_handle->publish_feedback(feedback);
                result->message = "Robot come to home!";
                RCLCPP_INFO(get_logger(), "time: %f", now().seconds());
                result->success = true; goal_handle->succeed(result);
            }
        }

        time_publisher(now().seconds(), true, success_count);
        if (mul_mode_ && success_count != 0) {
            publisher_callback(true, 0.0, false, true);
            publisher_callback(false, now().seconds(), true, mul_mode_);
        } else {
            publisher_callback(true, 0.0, false);
            publisher_callback(false, now().seconds());
        }
        is_robot_moving_ = false; target_ready_ = false; time_recieved_ = false; obs_ready = false;
        if (success_count == 0) { publish_skip_signal(true); publish_move_signal(true); }
        publish_signal(false);
    }
};

// ═══════════════════════════════════════════════════════════════════════
//  main
// ═══════════════════════════════════════════════════════════════════════
int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<MoveItController>();
    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(node->get_node_base_interface());
    std::thread spin_thread([&executor]() { executor.spin(); });
    node->initialize();
    spin_thread.join();
    rclcpp::shutdown();
    return 0;
}
