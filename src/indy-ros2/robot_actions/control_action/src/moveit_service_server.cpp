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

struct CollisionInfo
{
    bool collision = false;

    Eigen::Vector3d position = Eigen::Vector3d::Zero();
    Eigen::Vector3d position_world = Eigen::Vector3d::Zero();
    Eigen::Vector3d normal   = Eigen::Vector3d::Zero();

    double depth = 0.0;

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

class MoveItController : public rclcpp::Node {
   public:
    MoveItController()
        : Node("moveit_controller"),
          //moveit_node_(std::make_shared<rclcpp::Node>("moveit_node")),
          //move_group_interface_(moveit_node_, "indy_manipulator"),
          tf_buffer_(std::make_shared<tf2_ros::Buffer>(this->get_clock())),
          tf_listener_(*tf_buffer_) {

        subscription_ = this->create_subscription<test_msgs::msg::RosYolo>(
            "/ros_yolo", 10, std::bind(&MoveItController::topic_callback, this, std::placeholders::_1));
        connection_ = this->create_subscription<connect_msgs::msg::ConnectMsg>(
            "/connect_msg", 10, std::bind(&MoveItController::connection_callback, this, std::placeholders::_1));
        time_sub_ = this->create_subscription<collect_msgs::msg::CollectMsg>(
            "/collect2_msg", 10, std::bind(&MoveItController::collectmsg_callback, this, std::placeholders::_1));
        publisher_ = this->create_publisher<res_msgs::msg::PoseRes>("/pose_res", 10);
        time_publisher_ = this->create_publisher<collect_msgs::msg::CollectMsg>("/collect3_msg", 10);
//        octomap_sub_ = this->create_subscription<octomap_msgs::msg::Octomap>(
//            "/obstacle_octomap",
//            10,
//            std::bind(&MoveItController::octomapCallback, this, std::placeholders::_1));
        skip_signal_pub     = create_publisher<skip_signal_msgs::msg::SkipSignal>("/skip_signal", 10);
        tomato_octomap_sub_ = this->create_subscription<tomato_octomap_msgs::msg::TomatoOctomaps>(
            "/tomato_octomaps", 10,
            std::bind(&MoveItController::tomatoOctomapCallback, this, std::placeholders::_1));
        config_sub_ = this->create_subscription<config_manager::msg::SystemConfig>(
            "/system_config", 
            rclcpp::QoS(1).transient_local().reliable(),
            std::bind(&MoveItController::config_callback, this, std::placeholders::_1)
        );

        sub_left_cam_ = create_subscription<sensor_msgs::msg::CameraInfo>(
            "/stereo/left/camera_info_calib",
            rclcpp::QoS(1).transient_local().reliable(),
            std::bind(&MoveItController::left_camera_info_callback, this, std::placeholders::_1));

        sub_right_cam_ = create_subscription<sensor_msgs::msg::CameraInfo>(
            "/stereo/right/camera_info_calib",
            rclcpp::QoS(1).transient_local().reliable(),
            std::bind(&MoveItController::right_camera_info_callback, this, std::placeholders::_1));

        depth_signal_pub = create_publisher<depth_signal_msgs::msg::DepthSignal>("/depth_signal", 10);
        position_signal_pub = create_publisher<position_signal_msgs::msg::PositionSignal>("/position_signal", 10);
        move_signal_pub = create_publisher<move_signal_msgs::msg::MoveSignal>("/move_signal", 10);


        action_server_ = rclcpp_action::create_server<ControlRobot>(
            this, "move_robot",
            std::bind(&MoveItController::handle_goal, this, std::placeholders::_1, std::placeholders::_2),
            std::bind(&MoveItController::handle_cancel, this, std::placeholders::_1),
            std::bind(&MoveItController::handle_accepted, this, std::placeholders::_1));
        move_client_ = rclcpp_action::create_client<MoveRobot>(this, "robot_move_action");
        move_to_home_client_ = rclcpp_action::create_client<MoveToHome>(this, "move_to_home");
        gripper_client_ = rclcpp_action::create_client<GripperControl>(this, "gripper_action");
        planning_scene_client_ = this->create_client<moveit_msgs::srv::GetPlanningScene>("/get_planning_scene");
        std::filesystem::path base_path = std::filesystem::current_path(); // sẽ là đường dẫn từ nơi bạn chạy `ros2 run`
        config_path = base_path.string() + "/config/setup.yaml";
        //startConnectionMonitorThread();
        RCLCPP_INFO(this->get_logger(), "MoveIt Action Server started.");
    }

    void initialize() {
        move_group_interface_ = std::make_unique<MoveGroupInterface>(shared_from_this(), "indy_manipulator");
        saveOriginalACM();
        setGripperIgnoreCollision(true);
    }

    void saveOriginalACM()
    {
        using GetPlanningScene = moveit_msgs::srv::GetPlanningScene;

        // Guard check
        if (!planning_scene_client_) {
            RCLCPP_ERROR(this->get_logger(), "planning_scene_client_ is null!");
            return;
        }

        if (!planning_scene_client_->wait_for_service(std::chrono::seconds(5))) {
            RCLCPP_ERROR(this->get_logger(), "Service /get_planning_scene not available");
            return;
        }

        auto request = std::make_shared<GetPlanningScene::Request>();
        request->components.components =
            moveit_msgs::msg::PlanningSceneComponents::ALLOWED_COLLISION_MATRIX;

        std::promise<GetPlanningScene::Response::SharedPtr> promise;
        auto future_result = promise.get_future();

        planning_scene_client_->async_send_request(
            request,
            [&promise](rclcpp::Client<GetPlanningScene>::SharedFuture future) {
                promise.set_value(future.get());
            });

        if (future_result.wait_for(std::chrono::seconds(5)) != std::future_status::ready) {
            RCLCPP_ERROR(this->get_logger(), "Timeout saving original ACM");
            return;
        }

        auto response = future_result.get();
        if (!response) {
            RCLCPP_ERROR(this->get_logger(), "Got null response!");
            return;
        }

        {
            std::lock_guard<std::mutex> lock(acm_mutex_);
            original_acm_ = collision_detection::AllowedCollisionMatrix(
                response->scene.allowed_collision_matrix);
            acm_saved_ = true;
        }
        RCLCPP_INFO(this->get_logger(), "Original ACM saved successfully.");
    }

    void startConnectionMonitorThread() {
        connection_monitor_thread_ = std::thread([this]() {
            while (rclcpp::ok() && !stop_connection_monitor_) {
                is_server_ready_ = latest_connection_status_;
                is_reset_ = reset_status_;
                //std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        });
    }

    ~MoveItController() {
        stop_connection_monitor_ = true;
        if (connection_monitor_thread_.joinable()) {
            connection_monitor_thread_.join();
        }
    }

   private:
    //std::shared_ptr<rclcpp::Node> moveit_node_;
    moveit::planning_interface::PlanningSceneInterface planning_scene_interface_;
    std::unique_ptr<MoveGroupInterface> move_group_interface_;
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    tf2_ros::TransformListener tf_listener_;
    //moveit::core::RobotModelPtr robot_model_;
    std::thread connection_monitor_thread_;
    std::shared_ptr<planning_scene::PlanningScene> cached_scene_;
    //std::shared_ptr<planning_scene::PlanningScene> temp_scene_;
    std::mutex scene_mutex_;
    //std::array<double, 6> target_position_;
    std::mutex pub_mutex;
    std::mutex acm_mutex_;
    std::vector<std::array<double, 10>> target_position_;
    std::array<double, 6> test_position_ref;
    std::array<double, 6> test_position_ref_offset;
    std::array<double, 6> target_idx_position_;
    std::vector<double> home_position_;
    std::vector<double> drop_position_;
    std::size_t success_count;
    std::string config_path;
    collision_detection::AllowedCollisionMatrix original_acm_;
    //rclcpp::TimerBase::SharedPtr save_acm_timer_;
    double offset_distance_;
    double object_offset_;
    double y_offset_distance_;
    double offset_angle_;
    float fx_ = 0.f, fy_ = 0.f, cx_ = 0.f, cy_ = 0.f;
//    double start_time;
    double detection_time;
    double total_time;
    double eef_scale_;
    double start_detection_time;
    double positioning_time;
    double temp_total_time = 0.0;
    //size_t multi_mode_idx = 0;
    bool is_robot_moving_ = false;
    bool is_server_ready_ = false;
    bool stop_connection_monitor_ = false;
    //bool rotate_check_ = false;
    bool is_reset_ = false;
    bool bypass = false;
    bool ws_check = true;
    bool pose_check = false;
    bool recompute = false;
    bool found_test_ik = false;
    bool obs_check = true;
//    bool obs_check = false;
    bool obs_check_1 = false;
    bool obs_check_2 = false;
    bool obs_check_3 = false;
    bool allow_request_ = false;
    bool go_home_ = false;  // Cờ để kiểm tra xem đã về home hay chưa
    bool time_recieved_ = false;
    bool pass_all_ = false;
    bool target_ready_ = false;
    bool obs_ready = false;
    bool config_received_ = false;
    //bool scene_valid_ = false;
    bool target_base_transform_ready_ = false;
    bool move_action_ready_ = false;
    bool home_action_ready_ = false;
    bool gripper_action_ready_ = false;
    std::atomic<bool> acm_saved_{false};
    std::atomic<bool> mul_mode_ = false;
    std::atomic<bool> latest_connection_status_{false};
    std::atomic<bool> reset_status_{false};
    geometry_msgs::msg::Pose target_pose;
    geometry_msgs::msg::Pose pose;
    geometry_msgs::msg::Pose test_pose;
    geometry_msgs::msg::Pose next_pose;
    geometry_msgs::msg::Pose input_test_pose;
    geometry_msgs::msg::TransformStamped target_base_transform_;
    std::vector<geometry_msgs::msg::Pose> target_pose_list_;
    octomap_msgs::msg::Octomap octomap_cache_;
    octomap_msgs::msg::Octomap octomap_temp_;
    octomap_msgs::msg::Octomap octomap_combine_;
    bool octomap_cache_valid_ = false;
    using ControlRobot = control_action::action::MoveRobot;
    using GoalHandleControlRobot = rclcpp_action::ServerGoalHandle<ControlRobot>;
    using MoveRobot = robot_move_action::action::MoveRobot;
    using GoalHandleMoveRobot = rclcpp_action::ClientGoalHandle<MoveRobot>;
    using MoveToHome = robot_home_action::action::MoveToHome;
    using GoalHandleMoveToHome = rclcpp_action::ClientGoalHandle<MoveToHome>;
    using GripperControl = gripper_action::action::GripperControl;
    using GoalHandleGripperControl = rclcpp_action::ClientGoalHandle<GripperControl>;
    rclcpp_action::Server<ControlRobot>::SharedPtr action_server_;
    rclcpp_action::Client<GripperControl>::SharedPtr gripper_client_;
    rclcpp_action::Client<MoveToHome>::SharedPtr move_to_home_client_;
    rclcpp_action::Client<MoveRobot>::SharedPtr move_client_;
    rclcpp::Subscription<test_msgs::msg::RosYolo>::SharedPtr subscription_;
    rclcpp::Subscription<connect_msgs::msg::ConnectMsg>::SharedPtr connection_;
    rclcpp::Subscription<collect_msgs::msg::CollectMsg>::SharedPtr time_sub_;
    rclcpp::Subscription<config_manager::msg::SystemConfig>::SharedPtr config_sub_;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr sub_left_cam_, sub_right_cam_;
    rclcpp::Publisher<depth_signal_msgs::msg::DepthSignal>::SharedPtr depth_signal_pub;
    rclcpp::Publisher<position_signal_msgs::msg::PositionSignal>::SharedPtr position_signal_pub;
    rclcpp::Publisher<res_msgs::msg::PoseRes>::SharedPtr publisher_;
    rclcpp::Publisher<collect_msgs::msg::CollectMsg>::SharedPtr time_publisher_;
    rclcpp::Publisher<skip_signal_msgs::msg::SkipSignal>::SharedPtr         skip_signal_pub;
    rclcpp::Publisher<move_signal_msgs::msg::MoveSignal>::SharedPtr         move_signal_pub;
//    rclcpp::Subscription<octomap_msgs::msg::Octomap>::SharedPtr octomap_sub_;
    //std::map<int, octomap_msgs::msg::Octomap> octomap_map_;
    octomap_msgs::msg::Octomap octomap_single_;
    geometry_msgs::msg::TransformStamped octomap_to_link0_tf_;
    std::mutex octomap_map_mutex_;
    rclcpp::Subscription<tomato_octomap_msgs::msg::TomatoOctomaps>::SharedPtr tomato_octomap_sub_;

    rclcpp::Client<moveit_msgs::srv::GetPlanningScene>::SharedPtr planning_scene_client_;

        // Thêm:
    std::atomic<bool> scene_valid_{false};
    std::atomic<bool> move_success_{false};

    sensor_msgs::msg::CameraInfo::SharedPtr left_camera_info_, right_camera_info_;
    image_geometry::StereoCameraModel model_;

    rclcpp_action::GoalResponse handle_goal(const rclcpp_action::GoalUUID& uuid,
        std::shared_ptr<const ControlRobot::Goal> goal) {
        RCLCPP_INFO(this->get_logger(), "Received action request!");
            (void)goal;
            (void)uuid;
        if (!goal->request_move && !allow_request_) {
            RCLCPP_WARN(this->get_logger(), "Action server not ready. Rejecting goal.");
            return rclcpp_action::GoalResponse::REJECT;
        }
        else {
            allow_request_ = false;
            RCLCPP_INFO(this->get_logger(), "Action goal accepted.");
            return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
        }
    }

    rclcpp_action::CancelResponse handle_cancel(const std::shared_ptr<GoalHandleControlRobot> goal_handle) {
        (void)goal_handle;
        RCLCPP_INFO(this->get_logger(), "Goal canceled!");
        return rclcpp_action::CancelResponse::ACCEPT;
    }

    void handle_accepted(const std::shared_ptr<GoalHandleControlRobot> goal_handle) {
        //if (is_server_ready_) {
        std::thread{std::bind(&MoveItController::execute, this, goal_handle)}.detach();
        //}
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
        //baseline_ = std::abs(static_cast<float>(msg->p[3] / msg->p[0]));
    }

    void update_camera_model() {
        if (left_camera_info_ && right_camera_info_)
            model_.fromCameraInfo(*left_camera_info_, *right_camera_info_);
    }

    void tomatoOctomapCallback(
        const tomato_octomap_msgs::msg::TomatoOctomaps::SharedPtr msg)
    {
        if (!obs_ready) {
            if (msg->octomaps.empty()) return;
            // Giữ octomap gốc ở frame cảm biến để crop bbox đúng hệ tọa độ.
            octomap_single_ = msg->octomaps[0].octomap;
            try {
                octomap_to_link0_tf_ = tf_buffer_->lookupTransform(
                    "link0",
                    octomap_single_.header.frame_id,
                    tf2::TimePointZero,
                    tf2::durationFromSec(1.0));
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

    octomap_msgs::msg::Octomap transformOctomapToLink0(
    const octomap_msgs::msg::Octomap& input_octomap)
    {
        if (input_octomap.header.frame_id == "link0") return input_octomap;

        geometry_msgs::msg::TransformStamped tf_msg;
        try {
            tf_msg = tf_buffer_->lookupTransform(
                "link0", input_octomap.header.frame_id,
                tf2::TimePointZero, tf2::durationFromSec(1.0));
        } catch (const tf2::TransformException& ex) {
            RCLCPP_ERROR(this->get_logger(),
                "Failed to transform octomap: %s", ex.what());
            return input_octomap;
        }

        std::unique_ptr<octomap::AbstractOcTree> abstract_tree(
            octomap_msgs::msgToMap(input_octomap));
        auto* input_tree = dynamic_cast<octomap::OcTree*>(abstract_tree.get());
        if (!input_tree) {
            RCLCPP_ERROR(this->get_logger(), "Failed to convert octomap msg to OcTree");
            return input_octomap;
        }

        tf2::Transform tf;
        tf2::fromMsg(tf_msg.transform, tf);

        octomap::OcTree output_tree(input_tree->getResolution());
        for (auto it = input_tree->begin_leafs(); it != input_tree->end_leafs(); ++it) {
            if (!input_tree->isNodeOccupied(*it)) continue;
            const tf2::Vector3 pt_in(it.getX(), it.getY(), it.getZ());
            const tf2::Vector3 pt_out = tf * pt_in;
            output_tree.updateNode(
                octomap::point3d(pt_out.x(), pt_out.y(), pt_out.z()), true);
        }
        output_tree.updateInnerOccupancy();

        octomap_msgs::msg::Octomap output_octomap;
        octomap_msgs::binaryMapToMsg(output_tree, output_octomap);
        output_octomap.header.frame_id = "link0";
        output_octomap.header.stamp    = input_octomap.header.stamp;
        return output_octomap;
    }

    octomap_msgs::msg::Octomap transformOctomapWithTransform(
        const octomap_msgs::msg::Octomap& input_octomap,
        const geometry_msgs::msg::TransformStamped& tf_msg)
    {
        if (input_octomap.header.frame_id == "link0") return input_octomap;

        std::unique_ptr<octomap::AbstractOcTree> abstract_tree(
            octomap_msgs::msgToMap(input_octomap));
        auto* input_tree = dynamic_cast<octomap::OcTree*>(abstract_tree.get());
        if (!input_tree) {
            RCLCPP_ERROR(this->get_logger(), "Failed to convert octomap msg to OcTree");
            return input_octomap;
        }

        tf2::Transform tf;
        tf2::fromMsg(tf_msg.transform, tf);

        octomap::OcTree output_tree(input_tree->getResolution());
        for (auto it = input_tree->begin_leafs(); it != input_tree->end_leafs(); ++it) {
            if (!input_tree->isNodeOccupied(*it)) continue;
            const tf2::Vector3 pt_in(it.getX(), it.getY(), it.getZ());
            const tf2::Vector3 pt_out = tf * pt_in;
            output_tree.updateNode(
                octomap::point3d(pt_out.x(), pt_out.y(), pt_out.z()), true);
        }
        output_tree.updateInnerOccupancy();

        octomap_msgs::msg::Octomap output_octomap;
        octomap_msgs::binaryMapToMsg(output_tree, output_octomap);
        output_octomap.header.frame_id = "link0";
        output_octomap.header.stamp    = input_octomap.header.stamp;
        return output_octomap;
    }

    octomap_msgs::msg::Octomap cropOctomapByBbox(
        const octomap_msgs::msg::Octomap& input_octomap,
        int x1, int y1, int x2, int y2,
        float fx, float fy, float cx, float cy,
        float z_min = 0.05f, float z_max = 5.0f)
    {
        std::unique_ptr<octomap::AbstractOcTree> abstract_tree(
            octomap_msgs::msgToMap(input_octomap));
        auto* tree = dynamic_cast<octomap::OcTree*>(abstract_tree.get());
        if (!tree) return input_octomap;

        octomap::OcTree out_tree(tree->getResolution());

        for (auto it = tree->begin_leafs(); it != tree->end_leafs(); ++it) {
            if (!tree->isNodeOccupied(*it)) continue;

            const float X = it.getX();  // depth
            const float Y = it.getY();  // -Xcam
            const float Z = it.getZ();  // -Ycam

            bool inside_bbox = false;
            if (X >= z_min && X <= z_max) {
                const float u = fx * (-Y) / X + cx;
                const float v = fy * (-Z) / X + cy;
                inside_bbox = (u >= x1 && u <= x2 && v >= y1 && v <= y2);
            }

            // Giữ toàn bộ điểm ngoài bbox, chỉ xóa điểm nằm trong bbox.
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
        const octomap_msgs::msg::Octomap & input_octomap,
        const Eigen::Vector3d & center_world, 
        double clear_radius) {
        std::unique_ptr < octomap::AbstractOcTree > abstract_tree(octomap_msgs::msgToMap(input_octomap));
        auto * tree = dynamic_cast < octomap::OcTree * > (abstract_tree.get());
        if (!tree) {
            return input_octomap;
        }

        octomap::OcTree out_tree(tree -> getResolution());
        const double radius_sq = clear_radius * clear_radius;
        
        for (auto it = tree -> begin_leafs(); it != tree -> end_leafs(); ++it) {
            if (!tree -> isNodeOccupied( * it)) {
            continue;
            }
            const double dx = it.getX() - center_world.x();
            const double dy = it.getY() - center_world.y();
            const double dz = it.getZ() - center_world.z();
            if ((dx * dx + dy * dy + dz * dz) <= radius_sq) {
            continue;
            }
            out_tree.updateNode(octomap::point3d(it.getX(), it.getY(), it.getZ()), true);
        }

        out_tree.updateInnerOccupancy();
        octomap_msgs::msg::Octomap out_msg;
        octomap_msgs::binaryMapToMsg(out_tree, out_msg);
        out_msg.header = input_octomap.header;
        return out_msg;
    }

    bool applyOctomapMessage(const octomap_msgs::msg::Octomap& octomap_to_apply,
                             const char* success_log)
    {
        moveit_msgs::msg::PlanningScene planning_scene_msg;
        planning_scene_msg.is_diff = true;
        planning_scene_msg.world.octomap.octomap = octomap_to_apply;
        planning_scene_msg.world.octomap.header  = octomap_to_apply.header;
        planning_scene_msg.world.octomap.origin.orientation.w = 1.0;

        auto apply_client = this->create_client<moveit_msgs::srv::ApplyPlanningScene>(
            "/apply_planning_scene");
        if (!apply_client->wait_for_service(std::chrono::seconds(3))) {
            RCLCPP_ERROR(this->get_logger(), "Service /apply_planning_scene not available");
            return false;
        }

        auto request = std::make_shared<moveit_msgs::srv::ApplyPlanningScene::Request>();
        request->scene = planning_scene_msg;

        std::promise<moveit_msgs::srv::ApplyPlanningScene::Response::SharedPtr> promise;
        auto future_result = promise.get_future();
        apply_client->async_send_request(request,
            [&promise](rclcpp::Client<moveit_msgs::srv::ApplyPlanningScene>::SharedFuture f) {
                promise.set_value(f.get());
            });

        if (future_result.wait_for(std::chrono::seconds(5)) != std::future_status::ready) {
            RCLCPP_ERROR(this->get_logger(), "Timeout applying octomap");
            return false;
        }

        planning_scene_interface_.applyPlanningScene(planning_scene_msg);
        RCLCPP_INFO(this->get_logger(), "%s", success_log);
        return true;
    }

    void clearOctomapCache()
    {
        octomap_cache_ = octomap_msgs::msg::Octomap();
        octomap_temp_  = octomap_msgs::msg::Octomap();
        octomap_combine_ = octomap_msgs::msg::Octomap();
        octomap_cache_valid_ = false;
    }

    void applyOctomapTemp()
    {
        octomap_temp_ = octomap_cache_;
    }

    void ocotmapCombine()
    {
        octomap_combine_ = intersectOctomaps(octomap_cache_, octomap_temp_);
        while(rclcpp::ok() && !applyOctomapMessage(octomap_combine_, "Octomap applied from temp"))
        {
            RCLCPP_WARN(this->get_logger(), "Retrying to apply octomap from temp...");
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
    }

    octomap_msgs::msg::Octomap intersectOctomaps(
    const octomap_msgs::msg::Octomap& cache_msg,
    const octomap_msgs::msg::Octomap& temp_msg)
    {
        std::unique_ptr<octomap::AbstractOcTree> cache_abs(
            octomap_msgs::msgToMap(cache_msg));

        std::unique_ptr<octomap::AbstractOcTree> temp_abs(
            octomap_msgs::msgToMap(temp_msg));

        auto* cache_tree =
            dynamic_cast<octomap::OcTree*>(cache_abs.get());

        auto* temp_tree =
            dynamic_cast<octomap::OcTree*>(temp_abs.get());

        if (!cache_tree || !temp_tree)
            return cache_msg;

        octomap::OcTree out_tree(cache_tree->getResolution());

        for (auto it = cache_tree->begin_leafs();
            it != cache_tree->end_leafs();
            ++it)
        {
            if (!cache_tree->isNodeOccupied(*it))
                continue;

            auto* temp_node =
                temp_tree->search(
                    it.getX(),
                    it.getY(),
                    it.getZ());

            if (temp_node &&
                temp_tree->isNodeOccupied(temp_node))
            {
                out_tree.updateNode(
                    octomap::point3d(
                        it.getX(),
                        it.getY(),
                        it.getZ()),
                    true);
            }
        }

        out_tree.updateInnerOccupancy();

        octomap_msgs::msg::Octomap out_msg;
        octomap_msgs::binaryMapToMsg(out_tree, out_msg);
        out_msg.header = cache_msg.header;

        return out_msg;
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

        // Crop trên frame gốc của sensor, sau đó mới transform sang link0.
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

        //if (!applyOctomapMessage(octomap_to_apply, "Octomap applied")) {
        //    octomap_cache_valid_ = false;
        //    return;
        //}
        while(rclcpp::ok() && !applyOctomapMessage(octomap_to_apply, "Octomap applied from msg"))
        {
            RCLCPP_WARN(this->get_logger(), "Retrying to apply octomap from msg...");
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
    }

    void applyOctocmapFromTemp()
    {
        while(rclcpp::ok() && !applyOctomapMessage(octomap_temp_, "Octomap applied from temp"))
        {
            RCLCPP_WARN(this->get_logger(), "Retrying to apply octomap from temp...");
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
        //applyOctomapMessage(octomap_temp_, "Octomap applied from temp");
    }

    bool applyMaskedOctomapFromCache(
        const std::vector<CollisionInfo::ContactSample>& contact_samples,
        double clear_radius = 0.02)
    {
        if (!octomap_cache_valid_) {
            RCLCPP_WARN(this->get_logger(), "No cached octomap available");
            return false;
        }

        auto octomap_to_apply = octomap_cache_;

        if (contact_samples.empty()) {
            return false;
        }

        for (const auto& sample : contact_samples) {
            const Eigen::Vector3d normal_dir =
                sample.normal.norm() > 1e-9 ? sample.normal.normalized()
                                            : Eigen::Vector3d::UnitZ();
            for (int i = -1; i <= 1; ++i) {
                const double offset = static_cast<double>(i) * clear_radius * 0.4;
                octomap_to_apply = maskOctomapAroundPoint( 
                    octomap_to_apply, 
                    sample.position_world + normal_dir * offset, 
                    clear_radius);   // capsule dài 5cm
            }
        }

        octomap_cache_ = octomap_to_apply;
        octomap_cache_valid_ = true;
        while(rclcpp::ok() && !applyOctomapMessage(octomap_to_apply, "Octomap applied from cache"))
        {
            RCLCPP_WARN(this->get_logger(), "Retrying to apply octomap from cache...");
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
        return octomap_cache_valid_;

    }

    // Kept for compatibility with older call sites if any remain.
    void applyMaskedOctomapForContacts(
        const std::vector<CollisionInfo::ContactSample>& contact_samples,
        double clear_radius = 0.02)
    {
        (void)applyMaskedOctomapFromCache(contact_samples, clear_radius);
    }

    void setObstacleCollision(bool allow)
    {
        std::lock_guard<std::mutex> lock(acm_mutex_);
        if (!acm_saved_) {
            RCLCPP_ERROR(this->get_logger(), "Original ACM not saved yet!");
            return;
        }

        moveit_msgs::msg::PlanningScene diff_scene;
        diff_scene.is_diff = true;

        if (allow) {
            // Copy từ original rồi modify
            collision_detection::AllowedCollisionMatrix acm = original_acm_;

            const auto& all_links =
                move_group_interface_->getRobotModel()->getLinkModelNames();
            for (const auto& link : all_links) {
                acm.setDefaultEntry(link, true);
            }

            acm.getMessage(diff_scene.allowed_collision_matrix);
            planning_scene_interface_.applyPlanningScene(diff_scene);
            RCLCPP_INFO(this->get_logger(), "Obstacle collision check: DISABLED");

        } else {
            // Restore về ACM gốc
            original_acm_.getMessage(diff_scene.allowed_collision_matrix);
            planning_scene_interface_.applyPlanningScene(diff_scene);
            RCLCPP_INFO(this->get_logger(), "Obstacle collision check: ENABLED");
        }
    }

    void setGripperIgnoreCollision(bool allow)
    {
        std::lock_guard<std::mutex> lock(acm_mutex_);
        if (!acm_saved_) {
            RCLCPP_ERROR(this->get_logger(), "Original ACM not saved yet!");
            return;
        }

        moveit_msgs::msg::PlanningScene diff_scene;
        diff_scene.is_diff = true;

        if (allow) {
            collision_detection::AllowedCollisionMatrix acm = original_acm_;

            std::vector<std::string> gripper_links = {
                "gripper_left1", "gripper_left2", "gripper_left3",
                "gripper_right1", "gripper_right2", "gripper_right3"
                //"gripper_base"
            };

            const auto& all_links =
                move_group_interface_->getRobotModel()->getLinkModelNames();

            for (const auto& link : gripper_links) {
                for (const auto& other : all_links) {
                    acm.setEntry(link, other, true);
                }
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

    void setOctomapCollision(bool allow_collision)
    {
        std::lock_guard<std::mutex> lock(acm_mutex_);
        if (!acm_saved_) {
            RCLCPP_ERROR(this->get_logger(), "Original ACM not saved yet!");
            return;
        }

        moveit_msgs::msg::PlanningScene diff_scene;
        diff_scene.is_diff = true;

        // Lấy ACM hiện tại (đã có gripper ignore) làm base
        collision_detection::AllowedCollisionMatrix acm = original_acm_;

        // Giữ lại gripper ignore collision
        std::vector<std::string> gripper_links = {
            "gripper_left1", "gripper_left2", "gripper_left3",
            "gripper_right1", "gripper_right2", "gripper_right3"
        };
        const auto& all_links = move_group_interface_->getRobotModel()->getLinkModelNames();
        for (const auto& link : gripper_links) {
            for (const auto& other : all_links) {
                acm.setEntry(link, other, true);
            }
            acm.setDefaultEntry(link, true);
        }

        if (allow_collision) {
            // Disable collision với octomap: setDefaultEntry cho tất cả link = true
            // nghĩa là link vs "<octomap>" sẽ được bỏ qua
            for (const auto& link : all_links) {
                acm.setDefaultEntry(link, true);  // ignore collision với bất kỳ object không tên (octomap)
            }
            RCLCPP_WARN(this->get_logger(), "Octomap collision: DISABLED");
        } else {
            RCLCPP_INFO(this->get_logger(), "Octomap collision: ENABLED");
        }

        acm.getMessage(diff_scene.allowed_collision_matrix);
        planning_scene_interface_.applyPlanningScene(diff_scene);
    }

    void load_setup_params(const std::string &filename) {
        RCLCPP_INFO(this->get_logger(), "loading setup params");
        YAML::Node config = YAML::LoadFile(filename);
        auto setup = config["setup"];
        home_position_ = setup["HomePose"].as<std::vector<double>>();
        drop_position_ = setup["DorpPose"].as<std::vector<double>>();
        offset_distance_ = setup["OffSetDistance"].as<double>();
        y_offset_distance_ = setup["YOffSetDistance"].as<double>();
        offset_angle_ = setup["OffSetAngle"].as<double>();
        mul_mode_ = setup["Multi_collect_mode"].as<bool>();
    }

    void config_callback(const config_manager::msg::SystemConfig::SharedPtr msg)
    {
        RCLCPP_INFO(this->get_logger(), "Load setup");
        config_received_ = false;
        home_position_ = msg->home_pose;
        drop_position_ = msg->drop_pose;
        object_offset_ = msg->object_offset;
        offset_distance_ = msg->offset_distance;
        y_offset_distance_ = msg->y_offset_distance;
        offset_angle_ = msg->offset_angle;
        mul_mode_ = msg->multi_collect_mode;
        config_received_ = true;
    }

    template<typename ClientT>
    bool ensureActionServerReady(
        ClientT& client,
        bool& ready_cache,
        const char* action_name)
    {
        if (ready_cache) return true;
        if (!client->wait_for_action_server(std::chrono::seconds(5))) {
            RCLCPP_ERROR(get_logger(), "Action server '%s' không khả dụng!", action_name);
            return false;
        }
        ready_cache = true;
        return true;
    }

    template<typename FutureT>
    bool waitForGoalResponse(
        FutureT& future_goal,
        const char* action_name)
    {
        if (future_goal.wait_for(std::chrono::seconds(2)) != std::future_status::ready) {
            RCLCPP_ERROR(get_logger(), "Timeout waiting for '%s' goal response", action_name);
            return false;
        }
        return true;
    }

    // Thay toàn bộ hàm:
    void callMoveToHome(
        const std::vector<double>& joint_positions,
        size_t id,
        size_t pass_permit = 0)
    {
        if (!ensureActionServerReady(
                move_to_home_client_, home_action_ready_, "move_to_home")) {
            return;
        }

        auto goal_msg = MoveToHome::Goal();   // ← bỏ typename Controller::
        goal_msg.joint_positions = joint_positions;
        goal_msg.id = id;
        goal_msg.pass_permit = pass_permit;

        auto send_goal_options =
            rclcpp_action::Client<MoveToHome>::SendGoalOptions();   // ← bỏ typename Controller::

        if ((bypass && id != 9) || (pass_all_ && id != 9)) {
            return;
        }

        auto future_goal = move_to_home_client_->async_send_goal(goal_msg, send_goal_options);
        if (!waitForGoalResponse(future_goal, "move_to_home")) {   // ← bỏ self
            home_action_ready_ = false;
            return;
        }
        auto goal_handle = future_goal.get();
        if (!goal_handle) {
            RCLCPP_ERROR(get_logger(), "Gửi action goal thất bại!");
            return;
        }

        auto future_result = move_to_home_client_->async_get_result(goal_handle);
        auto result = future_result.get();
        if (result.code == rclcpp_action::ResultCode::SUCCEEDED) {
            RCLCPP_INFO(get_logger(), "Move to home thành công: %s",
                        result.result->message.c_str());
        } else {
            RCLCPP_ERROR(get_logger(), "Move to home thất bại!");
            callMoveRobot(
                offsetPose(jointStatesToPose(joint_positions), 0.0, 0.5, 0.0),
                jointStatesToPose(joint_positions),
                id, 2);
        }
    }

    void callMoveRobot(
        const geometry_msgs::msg::Pose& start_pose,
        const geometry_msgs::msg::Pose& target_pose,
        size_t id,
        size_t mode)
    {
        if (pass_all_ || bypass) return;

        if (!ensureActionServerReady(
                move_client_, move_action_ready_, "robot_move_action")) {
            return;
        }

        auto goal_msg = MoveRobot::Goal();   // ← bỏ typename Controller::
        goal_msg.mode = mode;
        goal_msg.id = id;
        goal_msg.start_pose = start_pose;
        goal_msg.target_pose = target_pose;

        auto send_goal_options =
            rclcpp_action::Client<MoveRobot>::SendGoalOptions();   // ← bỏ typename Controller::
        auto future_goal = move_client_->async_send_goal(goal_msg, send_goal_options);
        if (!waitForGoalResponse(future_goal, "robot_move_action")) {   // ← bỏ self
            move_action_ready_ = false;
            return;
        }
        auto goal_handle = future_goal.get();
        if (!goal_handle) {
            RCLCPP_ERROR(get_logger(), "Gửi action goal thất bại!");
            return;
        }

        auto future_result = move_client_->async_get_result(goal_handle);
        auto result = future_result.get();
        move_success_ = false;
        if (result.code == rclcpp_action::ResultCode::SUCCEEDED) {
            RCLCPP_INFO(get_logger(), "Move robot thành công: %s",
                        result.result->message.c_str());
            move_success_ = true;
        } else {
            RCLCPP_ERROR(get_logger(), "Move robot thất bại!");
            if (id == 1) {
                bypass = true;
            } else if (id == 3) {
                callMoveRobot(target_pose, offsetPose(target_pose, 0.0, 0.0, 0.0), 30, 0);
            }
        }
    }

    void sendGripperCommand(double position, size_t id, size_t pass_permit = 0)
    {
        if ((pass_all_ || bypass) && !(bypass && id == 8)) return;

        if (!ensureActionServerReady(
                gripper_client_, gripper_action_ready_, "gripper_action")) {
            return;
        }

        while (rclcpp::ok()) {
            auto goal_msg = GripperControl::Goal();   // ← bỏ typename Controller::
            goal_msg.position = position;
            goal_msg.id = id;
            goal_msg.pass_permit = pass_permit;

            auto goal_handle_future = gripper_client_->async_send_goal(goal_msg);
            if (!waitForGoalResponse(goal_handle_future, "gripper_action")) {   // ← bỏ self
                gripper_action_ready_ = false;
                return;
            }
            auto goal_handle = goal_handle_future.get();
            if (!goal_handle) {
                RCLCPP_ERROR(get_logger(), "Gửi lệnh gripper thất bại!");
                return;
            }

            auto result_future = gripper_client_->async_get_result(goal_handle);
            auto result = result_future.get();
            if (result.code == rclcpp_action::ResultCode::SUCCEEDED) {
                RCLCPP_INFO(get_logger(), "Gripper điều khiển thành công.");
                break;
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

    double poseDistance(const std::array<double, 10>& a,
                            const std::array<double, 10>& b)
    {
        const double dx = a[0] - b[0];
        const double dy = a[1] - b[1];
        const double dz = a[2] - b[2];
        return std::sqrt(dx * dx + dy * dy + dz * dz);
    }

    struct ClusterEntry
    {
        std::array<double, 10> pose;
        size_t original_idx;
    };

    std::vector<std::vector<ClusterEntry>> clusterByDistance(
        const std::vector<std::array<double, 10>>& poses,
        double threshold = 0.15)
    {
        std::vector<std::vector<ClusterEntry>> clusters;
        std::vector<bool> assigned(poses.size(), false);

        for (size_t i = 0; i < poses.size(); ++i) {
            if (assigned[i]) {
                continue;
            }

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

    struct ValidTarget
    {
        geometry_msgs::msg::Pose pose;
        size_t original_idx;
    };

    void execute(
        const std::shared_ptr<GoalHandleControlRobot> goal_handle)
    {
        success_count = 0;
        auto timeout = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (!target_ready_ || !time_recieved_ || !obs_ready ||
               !config_received_) {
            if (std::chrono::steady_clock::now() > timeout) {
                auto result = std::make_shared<ControlRobot::Result>();
                result->success = false;
                result->message = "Timeout waiting for target/time";
                RCLCPP_ERROR(get_logger(), "Execute timeout!");
                RCLCPP_ERROR(get_logger(), "target_ready: %s, time_recieved: %s, obs_ready: %s, config_received: %s",
                             target_ready_ ? "true" : "false",
                             time_recieved_ ? "true" : "false",
                             obs_ready ? "true" : "false",
                             config_received_ ? "true" : "false");
                goal_handle->abort(result);
                target_ready_ = false;
                time_recieved_ = false;
                obs_ready = false;
                publish_signal(false);
                return;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }

        auto result   = std::make_shared<ControlRobot::Result>();
        auto feedback = std::make_shared<ControlRobot::Feedback>();
        captureTargetBaseTransform();
        rebuildTargetPoseList();
        clearOctomapCache();

        for (size_t i = 0; i < target_position_.size(); i++) {
            //applyOctomapForIdx(static_cast<int>(i));
            applyOctomap(target_position_[i][6], target_position_[i][7], 
                target_position_[i][8], target_position_[i][9], fx_, fy_, cx_, cy_);
            pass_all_ = false;
            bypass = false;
            publisher_callback(true, now().seconds(), true, mul_mode_);
            std::this_thread::sleep_for(std::chrono::milliseconds(100));

            RCLCPP_INFO(get_logger(),
                        "[Loop Debug] i=%zu / total=%zu | mul_mode=%s | time=%.3f",
                        i,
                        target_position_.size(),
                        mul_mode_ ? "true" : "false",
                        now().seconds());

            posecheck_and_recompute(target_position_[i], home_position_, i);
            target_pose = targetPositionToBasePose(i);

            RCLCPP_INFO(get_logger(),
                        "Updated target: x=%.2f y=%.2f z=%.2f r=%.2f p=%.2f y=%.2f",
                        target_position_[i][0],
                        target_position_[i][1],
                        target_position_[i][2],
                        target_position_[i][3],
                        target_position_[i][4],
                        target_position_[i][5]);

            test_pose = offsetPose(target_pose, 0.0, offset_distance_, 0.0);

            if (!refreshPlanningScene()) {
                RCLCPP_ERROR(get_logger(),
                             "Failed to refresh planning scene, skip iteration %zu", i);
                continue;
            }

            if (pose_check) {
                //&& !checkCollisionAtTarget(test_pose) )//&&
                //!checkCollisionAtTarget(target_pose)) {
                RCLCPP_INFO(get_logger(), "DEBUG CONSUME: CAN REACH TO TARGET");
                ws_check = true;
            } else {
                RCLCPP_ERROR(get_logger(), "DEBUG CONSUME: UNABLE TO REACH TO TARGET");
                ws_check = false;
            }

            obs_check_1 =
                checkCollisionAtTarget(
                    offsetPose(target_pose, y_offset_distance_, 0.0, 0.0)).collision;
            if (!obs_check_1) {
                next_pose =
                    offsetPose(target_pose, y_offset_distance_, 0.0, 0.0);
                obs_check_2 = true;
                obs_check_3 = true;
                RCLCPP_ERROR(get_logger(), "CASE 1: Y-Offset");
            } else {
                obs_check_2 =
                    checkCollisionAtTarget(
                        offsetPose(target_pose, 0.0, 0.0, offset_angle_)).collision;
                if (!obs_check_2) {
                    next_pose =
                        offsetPose(target_pose, 0.0, 0.0, offset_angle_);
                    obs_check_3 = true;
                    RCLCPP_ERROR(get_logger(), "CASE 2: Z-Offset");
                } else {
                    obs_check_3 =
                        checkCollisionAtTarget(
                            offsetPose(target_pose, 0.0, 0.0, -offset_angle_)).collision;
                    if (!obs_check_3) {
                        next_pose =
                            offsetPose(target_pose, 0.0, 0.0, -offset_angle_);
                        RCLCPP_ERROR(get_logger(), "CASE 3: -Z-Offset");
                    }
                }
            }

            if (!ws_check) {
                continue;
            }

            if (!go_home_) {
                callMoveToHome(home_position_, 9);
                go_home_ = true;
            }

            feedback->progress = 0.0;
            goal_handle->publish_feedback(feedback);

            ocotmapCombine();
            callMoveRobot(
                offsetPose(target_pose, 0.0, offset_distance_, 0.0),
                target_pose,
                1,
                2);

            setOctomapCollision(true);

            sendGripperCommand(0.8, 2);
            feedback->progress = 0.10;
            goal_handle->publish_feedback(feedback);

            callMoveRobot(
                target_pose,
                offsetPose(target_pose, 0.0, offset_distance_, 0.0),
                3,
                1);

            feedback->progress = 0.25;
            goal_handle->publish_feedback(feedback);

            sendGripperCommand(0.0, 4);

            feedback->progress = 0.40;
            goal_handle->publish_feedback(feedback);

            callMoveRobot(
                offsetPose(target_pose, 0.0, offset_distance_, 0.0),
                next_pose,
                5,
                1);
            setOctomapCollision(false);
            // checkCollisionAtTarget(next_pose);

            feedback->progress = 0.55;
            goal_handle->publish_feedback(feedback);

            callMoveToHome(drop_position_, 6);
            feedback->progress = 0.70;
            goal_handle->publish_feedback(feedback);

            sendGripperCommand(0.8, 7);
            feedback->progress = 0.85;
            goal_handle->publish_feedback(feedback);

            sendGripperCommand(0.0, 8);
            feedback->progress = 0.90;
            goal_handle->publish_feedback(feedback);

            callMoveToHome(home_position_, 9);
            feedback->progress = 1.0;
            goal_handle->publish_feedback(feedback);

            result->message = "Robot come to home!";
            RCLCPP_INFO(get_logger(), "time: %f", now().seconds());
            result->success = true;
            goal_handle->succeed(result);
            success_count++;
            break;
        }

        time_publisher(now().seconds(), true, success_count);
        if (mul_mode_ && success_count != 0) {
            publisher_callback(true, 0.0, false, true);
            publisher_callback(false, now().seconds(), true, mul_mode_);
        } else {
            publisher_callback(true, 0.0, false);
            publisher_callback(false, now().seconds());
        }

        is_robot_moving_ = false;
        target_ready_ = false;
        time_recieved_ = false;
        obs_ready = false;
        if (success_count == 0) {
            publish_skip_signal(true);
            publish_move_signal(true);
        }
        publish_signal(false);
    }

    void execute1(
        const std::shared_ptr<GoalHandleControlRobot> goal_handle)
    {
        success_count = 0;
        auto timeout = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (!target_ready_ || !time_recieved_ || !obs_ready ||
               !config_received_) {
            if (std::chrono::steady_clock::now() > timeout) {
                auto result = std::make_shared<ControlRobot::Result>();
                result->success = false;
                result->message = "Timeout waiting for target/time";
                RCLCPP_ERROR(get_logger(), "Execute timeout!");
                goal_handle->abort(result);
                target_ready_ = false;
                time_recieved_ = false;
                obs_ready = false;
                publish_signal(false);
                return;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }

        auto result   = std::make_shared<ControlRobot::Result>();
        auto feedback = std::make_shared<ControlRobot::Feedback>();
        captureTargetBaseTransform();
        rebuildTargetPoseList();

        constexpr double kClusterDistanceMeters = 0.15;
        auto clusters = clusterByDistance(target_position_, kClusterDistanceMeters);

        RCLCPP_INFO(get_logger(),
                    "[Cluster] %zu raw targets -> %zu cluster(s) (threshold=%.3f m)",
                    target_position_.size(), clusters.size(), kClusterDistanceMeters);

        ValidTarget previous_target;
        bool last_cluster_reachable = false;
        clearOctomapCache();
        for (size_t ci = 0; ci < clusters.size(); ++ci) {
            const auto& cluster = clusters[ci];
            RCLCPP_INFO(get_logger(), "[Cluster %zu/%zu] %zu member(s)",
                        ci + 1, clusters.size(), cluster.size());

            if (ci == clusters.size() - 1) {
                RCLCPP_INFO(get_logger(),
                            "[Cluster %zu] Processing last cluster, marking as reachable",
                            ci + 1);
                last_cluster_reachable = true;
            }
            bool cluster_started = false;
            int step_id = 3;

            for (size_t mi = 0; mi < cluster.size(); ++mi) {
                const auto& raw = cluster[mi].pose;
                const size_t orig_i = cluster[mi].original_idx;

                //applyOctomapForIdx(static_cast<int>(orig_i));
                //clearOctomapCache();
                applyOctomap(static_cast<int>(target_position_[orig_i][6]), static_cast<int>(target_position_[orig_i][7]), static_cast<int>(
                    target_position_[orig_i][8]), static_cast<int>(target_position_[orig_i][9]), fx_, fy_, cx_, cy_);
                pass_all_ = false;
                bypass = false;
                publisher_callback(true, now().seconds(), true, mul_mode_);
                std::this_thread::sleep_for(std::chrono::milliseconds(100));

                if (!refreshPlanningScene()) {
                    RCLCPP_ERROR(get_logger(),
                                 "Failed to refresh planning scene after octomap apply, skip iteration %zu",
                                 orig_i);
                    continue;
                }

                RCLCPP_INFO(get_logger(),
                            "[Loop Debug] i=%zu / total=%zu | mul_mode=%s | time=%.3f",
                            orig_i, cluster.size(),
                            mul_mode_ ? "true" : "false",
                            now().seconds());

                posecheck_and_recompute(raw, home_position_, orig_i);
                const auto& checked_target = target_position_[orig_i];
                target_pose = targetPositionToBasePose(orig_i);

                RCLCPP_INFO(get_logger(),
                            "Updated target: x=%.2f y=%.2f z=%.2f r=%.2f p=%.2f y=%.2f",
                            checked_target[0], checked_target[1], checked_target[2],
                            checked_target[3], checked_target[4], checked_target[5]);

                if (pose_check) {
                    // && !checkCollisionAtTarget(target_pose)) {
                    RCLCPP_INFO(get_logger(), "DEBUG CONSUME: CAN REACH TO TARGET");
                    ws_check = true;
                } else {
                    RCLCPP_ERROR(get_logger(), "DEBUG CONSUME: UNABLE TO REACH TO TARGET");
                    ws_check = false;
                }

                if (!ws_check) {
                    continue;
                }

                const ValidTarget current_target{target_pose, orig_i};
                if (!cluster_started) {
                    if (!go_home_) {
                        callMoveToHome(home_position_, 9);
                        go_home_ = true;
                    }

                    feedback->progress = 0.0;
                    goal_handle->publish_feedback(feedback);

                    //setOctomapCollision(true);

                    ocotmapCombine();
                    callMoveRobot(
                        offsetPose(current_target.pose, 0.0, offset_distance_, 0.0),
                        current_target.pose,
                        1,
                        2);

                    if (!move_success_) {
                        RCLCPP_ERROR(get_logger(), "Failed to move to first target in cluster, skipping cluster");
                        continue;
                    }

                    applyOctomapTemp();
                    sendGripperCommand(0.8, 2);
                    feedback->progress = 0.10;
                    goal_handle->publish_feedback(feedback);

                    sendGripperCommand(0.0, 2);
                    //setOctomapCollision(false);

                    previous_target = current_target;
                    cluster_started = true;
                    success_count++;
                    //clearOctomapCache();
                    continue;
                }

                //setOctomapCollision(true);
                ocotmapCombine();
                callMoveRobot(previous_target.pose, current_target.pose, step_id, 1);
                if (!move_success_) {
                    RCLCPP_ERROR(get_logger(), "Failed to move to target, skipping remaining targets in cluster");
                    continue;
                }
                applyOctomapTemp();
                sendGripperCommand(0.8, 2);
                sendGripperCommand(0.0, 2);
                //setOctomapCollision(false);

                const float ratio =
                    static_cast<float>(mi + 1) / static_cast<float>(cluster.size());
                feedback->progress = 0.10f + 0.15f * ratio;
                goal_handle->publish_feedback(feedback);

                previous_target = current_target;
                success_count++;
                //clearOctomapCache();
            }

            if (last_cluster_reachable) {
                applyOctocmapFromTemp();
                callMoveToHome(home_position_, step_id++);
                feedback->progress = 1.0;
                goal_handle->publish_feedback(feedback);

                result->message = "Robot come to home!";
                RCLCPP_INFO(get_logger(), "time: %f", now().seconds());
                result->success = true;
                goal_handle->succeed(result);
            }
            //break;
        }

        time_publisher(now().seconds(), true, success_count);
        if (mul_mode_ && success_count != 0) {
            publisher_callback(true, 0.0, false, true);
            publisher_callback(false, now().seconds(), true, mul_mode_);
        } else {
            publisher_callback(true, 0.0, false);
            publisher_callback(false, now().seconds());
        }

        is_robot_moving_ = false;
        target_ready_ = false;
        time_recieved_ = false;
        obs_ready = false;

        if (success_count == 0) {
            publish_skip_signal(true);
            publish_move_signal(true);
        }
        publish_signal(false);
    }

    std::shared_ptr<planning_scene::PlanningScene>
    getSceneSnapshot()
    {
        return std::atomic_load(&cached_scene_);
    }

    bool waitForSceneReady(
                                  std::chrono::milliseconds timeout = std::chrono::milliseconds(1500))
    {
        const auto deadline = std::chrono::steady_clock::now() + timeout;
        while (rclcpp::ok() &&
               !scene_valid_.load(std::memory_order_acquire) &&
               std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        return scene_valid_.load(std::memory_order_acquire);
    }

    void setScene(

        std::shared_ptr<planning_scene::PlanningScene> new_scene)
    {
        std::atomic_store(&cached_scene_, new_scene);
        scene_valid_.store(true, std::memory_order_release);
    }

    bool solveIKWithSeed(
                                const geometry_msgs::msg::Pose& target_pose,
                                const std::vector<double>& joint_values,
                                std::unique_ptr<moveit::core::RobotState>& robot_state,
                                double timeout_seconds = 0.1)
    {
        const auto robot_model = move_group_interface_->getRobotModel();
        if (!robot_model) {
            RCLCPP_ERROR(get_logger(), "Robot model is null.");
            return false;
        }

        const auto* joint_model_group =
            robot_model->getJointModelGroup(move_group_interface_->getName());
        if (!joint_model_group) {
            RCLCPP_ERROR(get_logger(), "Joint model group '%s' not found.",
                         move_group_interface_->getName().c_str());
            return false;
        }

        if (!joint_values.empty() &&
            joint_values.size() != joint_model_group->getVariableCount()) {
            RCLCPP_ERROR(get_logger(),
                         "Mismatch joint count: expected %zu, got %zu.",
                         static_cast<size_t>(joint_model_group->getVariableCount()),
                         joint_values.size());
            return false;
        }

        robot_state = std::make_unique<moveit::core::RobotState>(robot_model);
        robot_state->setToDefaultValues();
        if (!joint_values.empty()) {
            RCLCPP_DEBUG(get_logger(), "Setting seed joint values for IK");
            robot_state->setJointGroupPositions(joint_model_group, joint_values);
        }
        const bool found_ik = robot_state->setFromIK(joint_model_group, target_pose, timeout_seconds);
        if (found_ik) {
            RCLCPP_DEBUG(get_logger(), "IK solution found");
            robot_state->update();
        }
        return found_ik;
    }

    CollisionInfo checkCollisionWithState(

        const moveit::core::RobotState& robot_state)
    {
        CollisionInfo info;

        if (!waitForSceneReady()) {
            RCLCPP_WARN(get_logger(), "Scene not ready");
            return info;
        }

        auto scene_snapshot = getSceneSnapshot();
        if (!scene_snapshot) {
            RCLCPP_WARN(get_logger(), "Scene snapshot null");
            return info;
        }

        collision_detection::CollisionRequest collision_request;
        collision_detection::CollisionResult  collision_result;
        collision_request.contacts     = true;
        collision_request.max_contacts = 10;

        scene_snapshot->checkCollision(collision_request, collision_result, robot_state);

        if (!collision_result.collision) {
            RCLCPP_INFO(get_logger(), "No collision detected");
            return info;
        }

        info.collision = true;
        const auto& contact_pair = *collision_result.contacts.begin();
        info.body1 = contact_pair.first.first;
        info.body2 = contact_pair.first.second;

        const Eigen::Isometry3d& tcp_tf = robot_state.getGlobalLinkTransform("tcp0");
        Eigen::Isometry3d world_to_tcp  = tcp_tf.inverse();

        for (const auto& contact : contact_pair.second) {
            CollisionInfo::ContactSample sample;
            sample.body1 = info.body1;
            sample.body2 = info.body2;
            sample.position_world = contact.pos;
            sample.depth = contact.depth;

            Eigen::Vector3d tcp_pos    = world_to_tcp * contact.pos;
            Eigen::Vector3d tcp_normal = world_to_tcp.linear() * contact.normal;
            if (tcp_normal.norm() > 1e-9) {
                tcp_normal.normalize();
            }

            sample.normal = tcp_normal;
            info.contact_points.push_back(sample);

            info.position = tcp_pos;
            info.position_world = contact.pos;
            info.normal   = tcp_normal;
            info.depth    = contact.depth;
        }

        if (!info.contact_points.empty()) {
            RCLCPP_INFO(get_logger(), "Collision at: %s - %s, contacts=%zu, Depth: %.2f",
                        info.body1.c_str(), info.body2.c_str(),
                        info.contact_points.size(), info.depth);
        }

        return info;
    }

    CollisionInfo checkCollisionWithMaskedContacts(

        moveit_msgs::msg::PlanningScene& scene_msg,
        const moveit::core::RobotState& robot_state
        //,const std::vector<CollisionInfo::ContactSample>& contact_samples,
        //double clear_radius
        )
    {
        CollisionInfo info;

        if (scene_msg.world.octomap.octomap.data.empty()) {
            RCLCPP_WARN(get_logger(), "No octomap in planning scene for masked recheck");
            return info;
        }

        if (!scene_msg.world.octomap.header.frame_id.empty() &&
            scene_msg.world.octomap.header.frame_id != "link0") {
            scene_msg.world.octomap.octomap =
                transformOctomapWithTransform(
                    scene_msg.world.octomap.octomap,
                    octomap_to_link0_tf_);
            scene_msg.world.octomap.header = scene_msg.world.octomap.octomap.header;
        }

        //auto masked_octomap = scene_msg.world.octomap.octomap;

/*
        if (contact_samples.empty()) {
            return info;
        }

        for (const auto& sample : contact_samples) {
            const Eigen::Vector3d normal_dir =
                sample.normal.norm() > 1e-9 ? sample.normal.normalized()
                                            : Eigen::Vector3d::UnitZ();
            for (int i = -1; i <= 1; ++i) {
                const double offset = static_cast<double>(i) * clear_radius * 0.4;
                masked_octomap = maskOctomapAroundPoint(
                    masked_octomap,
                    sample.position_world + normal_dir * offset,
                    clear_radius);
            }
        }
*/
        //scene_msg.world.octomap.octomap = masked_octomap;
        scene_msg.world.octomap.header = scene_msg.world.octomap.octomap.header;
        scene_msg.world.octomap.header = scene_msg.world.octomap.octomap.header;
        if (scene_msg.world.octomap.header.frame_id.empty()) {
            scene_msg.world.octomap.header.frame_id = "link0";
        }

        auto robot_model = move_group_interface_->getRobotModel();
        if (!robot_model) {
            RCLCPP_ERROR(get_logger(), "Robot model is null for masked recheck");
            return info;
        }

        auto temp_scene = getSceneSnapshot()->diff();
        scene_msg.is_diff = true;
        temp_scene->setPlanningSceneDiffMsg(scene_msg);

        collision_detection::CollisionRequest collision_request;
        collision_detection::CollisionResult  collision_result;
        collision_request.contacts     = true;
        collision_request.max_contacts = 10;

        temp_scene->checkCollision(collision_request, collision_result, robot_state);

        if (!collision_result.collision) {
            RCLCPP_INFO(get_logger(), "No collision detected after masked recheck");
            return info;
        }

        info.collision = true;
        const auto& contact_pair = *collision_result.contacts.begin();
        info.body1 = contact_pair.first.first;
        info.body2 = contact_pair.first.second;

        const Eigen::Isometry3d& tcp_tf = robot_state.getGlobalLinkTransform("tcp0");
        Eigen::Isometry3d world_to_tcp  = tcp_tf.inverse();

        for (const auto& contact : contact_pair.second) {
            CollisionInfo::ContactSample sample;
            sample.body1 = info.body1;
            sample.body2 = info.body2;
            sample.position_world = contact.pos;
            sample.depth = contact.depth;
            sample.normal = world_to_tcp.linear() * contact.normal;
            if (sample.normal.norm() > 1e-9) {
                sample.normal.normalize();
            }
            info.contact_points.push_back(sample);

            info.position = world_to_tcp * contact.pos;
            info.position_world = contact.pos;
            info.normal = sample.normal;
            info.depth = contact.depth;
        }

        if (!info.contact_points.empty()) {
            RCLCPP_INFO(get_logger(), "Masked collision at: %s - %s, contacts=%zu, Depth: %.2f",
                        info.body1.c_str(), info.body2.c_str(),
                        info.contact_points.size(), info.depth);
        }

        return info;
    }

    // ────────────────────────────────────────────────────────────────────
    // PosesCheck — giữ nguyên, không liên quan scene
    // ────────────────────────────────────────────────────────────────────
    bool PosesCheck(
                           const geometry_msgs::msg::Pose& input_pose,
                           const std::vector<double>& joint_values)
    {
        std::unique_ptr<moveit::core::RobotState> robot_state;
        const bool found_ik = solveIKWithSeed(input_pose, joint_values, robot_state, 0.1);

        if (found_ik || !recompute)
            RCLCPP_INFO(get_logger(), "Pose is reachable from provided joint state.");
        else
            RCLCPP_WARN(get_logger(), "Pose is NOT reachable from provided joint state.");

        return found_ik;
    }

    // ────────────────────────────────────────────────────────────────────
    // posecheck_and_recompute — giữ nguyên logic
    // ────────────────────────────────────────────────────────────────────
    void posecheck_and_recompute(
                                        const std::array<double, 10>& test_position,
                                        const std::vector<double>& test_joint_values,
                                        std::size_t idx)
    {
        test_position_ref = {
            test_position[0],
            test_position[1],
            test_position[2],
            test_position[3],
            test_position[4],
            test_position[5]
        };
        test_position_ref_offset = test_position_ref;
        found_test_ik = false;
        obs_check     = true;
        recompute     = true;

        const int max_iterations = 10;
        int iteration = 0;

        while (rclcpp::ok() && iteration < max_iterations) {
            iteration++;

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
            std::unique_ptr<moveit::core::RobotState> ik_state;
            found_test_ik =
                solveIKWithSeed(input_test_pose, test_joint_values, ik_state, 0.1);

            CollisionInfo col;
            obs_check = false;
            if (found_test_ik && ik_state) {
                col = checkCollisionWithState(*ik_state);
                obs_check = col.collision;

                constexpr double kSmallContactDepth = 0.1;
                if (obs_check && col.depth < kSmallContactDepth) {
                    //auto scene_snapshot = getSceneSnapshot(self);
            //        if (!scene_snapshot) {
            //            RCLCPP_WARN(get_logger(), "Scene snapshot null for masked recheck");
            //            recompute = false;
            //            break;
            //        } else {
                        //moveit_msgs::msg::PlanningScene masked_scene_msg;
                        //scene_snapshot->getPlanningSceneMsg(masked_scene_msg);

                        const int max_crop_iterations = 100;
                        bool cleared_contact = false;

                        for (int crop_iteration = 0; crop_iteration < max_crop_iterations; ++crop_iteration) {
                            //const auto masked_col = ControlChecks::checkCollisionWithMaskedContacts(
                            //    self, masked_scene_msg, *ik_state 
                            //    ,col.contact_points, 0.00
                            //);
                            if (col.collision) {
                                if (!applyMaskedOctomapFromCache(col.contact_points, 0.02)) {
                                    RCLCPP_WARN(get_logger(),
                                                "Masked octomap apply from cache failed");
                                    recompute = false;
                                    break;
                                }
                                //obs_check = false;
                                if (!refreshPlanningScene()) {
                                    RCLCPP_WARN(get_logger(),
                                                "Failed to refresh planning scene before masked apply");
                                    recompute = false;
                                    break;
                                }
                                //applyOctomapTemp();
                            } else {
                                obs_check = false;
                                cleared_contact = true;
                                RCLCPP_INFO(get_logger(),
                                            "Contact cleared after masked octomap apply at crop iteration %d",
                                            crop_iteration);
                                break;
                            }
                            col = checkCollisionWithState(*ik_state);
                            //col = masked_col;
                        }

                        if (cleared_contact) {
                            target_position_[idx][3] = test_position_ref[3];
                            target_position_[idx][4] = test_position_ref[4];
                            break;
                        }

                        RCLCPP_WARN(get_logger(),
                                    "Small-depth contact did not clear after repeated crop; "
                                    "skip recompute for idx %zu", idx);
                        obs_check = true;
                        recompute = false;
                        break;
                    //}
                }
            }

            if (found_test_ik && !obs_check) {
                RCLCPP_INFO(get_logger(), "IK found at iteration %d", iteration);
                target_position_[idx][3] = test_position_ref[3];
                target_position_[idx][4] = test_position_ref[4];
                break;
            }

            if (!recompute) {
                RCLCPP_WARN(get_logger(),
                            "Recompute converged but IK still not found for idx %zu", idx);
                break;
            }

            RCLCPP_WARN(get_logger(), "Recompute for pose idx %zu, iteration %d",
                        idx, iteration);

            if (!found_test_ik) {
                RCLCPP_WARN(get_logger(), "No IK found, adjusting roll/pitch based on XYZ offset");
                auto [test_roll, test_pitch] = computeRollPitchFromXYZ(
                    test_position_ref[0], test_position_ref[1], test_position_ref[2],
                    test_position_ref[3], test_position_ref[4]);
                test_position_ref[3] = test_roll;
                test_position_ref[4] = test_pitch;
            } else {
                RCLCPP_WARN(get_logger(), "IK found but in collision, adjusting roll/pitch based on contact normal");
                auto [test_roll, test_pitch] = computeRollPitchFromCollision(
                    col.normal, col.depth,
                    test_position_ref[3], test_position_ref[4]);
                test_position_ref[3] = test_roll;
                test_position_ref[4] = test_pitch;
            }
        }

        if (iteration >= max_iterations)
            RCLCPP_ERROR(get_logger(), "Max recompute iterations reached for idx %zu", idx);

        pose_check = found_test_ik && !obs_check;
        recompute  = false;
    }

    // ────────────────────────────────────────────────────────────────────
    // refreshPlanningScene — lock-free, build ngoài rồi atomic swap
    // ────────────────────────────────────────────────────────────────────
    bool refreshPlanningScene()
    {
        using GetPlanningScene = moveit_msgs::srv::GetPlanningScene;

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

        // Build scene mới hoàn toàn ngoài lock
        auto new_scene = std::make_shared<planning_scene::PlanningScene>(robot_model);
        new_scene->setPlanningSceneDiffMsg(response->scene);

        // ✅ Atomic swap — không có mutex, không block checkCollisionAtTarget
        setScene(new_scene);

        return true;
    }

    // ────────────────────────────────────────────────────────────────────
    // checkCollisionAtTarget — lock-free hoàn toàn
    // ────────────────────────────────────────────────────────────────────
    CollisionInfo checkCollisionAtTarget(

        const geometry_msgs::msg::Pose& target_pose,
        double ik_timeout_seconds = 0.1)
    {
        CollisionInfo info;

        std::unique_ptr<moveit::core::RobotState> robot_state;
        if (!solveIKWithSeed(target_pose, std::vector<double>{}, robot_state, ik_timeout_seconds)) {
            RCLCPP_WARN(get_logger(), "IK not found for target pose");
            return info;
        }

        info = checkCollisionWithState(*robot_state);
        if (info.collision) {
            RCLCPP_WARN(get_logger(), "Collision detected at target pose!");
        }

        return info;
    }

    std::tuple<double, double> computeRollPitchFromXYZ(
        double x, double y, double z, 
        double roll_prev = 0.0, double pitch_prev = 0.0)
    {
        Eigen::Vector3d normal(x, y, z);
        if (normal.norm() < 1e-6) return {0.0, 0.0};
        normal.normalize();

        double roll_nor  = -std::atan2(normal(1), normal(2));
        double pitch_nor =  std::atan2(normal(0), normal(2));

        double roll  = 0.5 * roll_prev  + 0.5 * roll_nor;
        double pitch = 0.5 * pitch_prev + 0.5 * pitch_nor;

        if (std::abs(roll_nor - roll) <= M_PI/90 && 
            std::abs(pitch_nor - pitch) <= M_PI/90) {
            recompute = false;
            RCLCPP_INFO(this->get_logger(), "Last recompute reached. Stop recompute.");
        }
        return {roll, pitch};
    }

    std::tuple<double, double>
    computeRollPitchFromCollision(
        const Eigen::Vector3d& tcp_normal,   // trong TCP frame, body2→body1
        double depth,
        double roll_prev  = 0.0,
        double pitch_prev = 0.0)
    {
        // =====================================================
        // body2 = robot → đảo normal để lấy hướng tránh
        // =====================================================
        Eigen::Vector3d avoid_dir = -tcp_normal.normalized();

        // safety
        if (tcp_normal.norm() < 1e-6) {
            recompute = false;
            return {roll_prev, pitch_prev};
        }

        // =====================================================
        // TCP z-axis hiện tại trong TCP frame = (0, 0, 1)
        // Mục tiêu: xoay TCP sao cho z-axis align với avoid_dir
        //
        // Tính rotation từ (0,0,1) → avoid_dir
        // dùng cross product + angle-axis
        // =====================================================
        const Eigen::Vector3d tcp_z(0.0, 0.0, 1.0);

        Eigen::Vector3d axis  = tcp_z.cross(avoid_dir);
        double          sin_a = axis.norm();
        double          cos_a = tcp_z.dot(avoid_dir);

        // nếu gần như song song (không cần xoay)
        if (sin_a < 1e-6) {
            if (cos_a > 0) {
                // cùng hướng → không cần correction
                recompute = false;
                return {roll_prev, pitch_prev};
            } else {
                // ngược hướng hoàn toàn → xoay 180° quanh x
                axis = Eigen::Vector3d(1.0, 0.0, 0.0);
                sin_a = 0.0;
                cos_a = -1.0;
            }
        }

        axis.normalize();
        double angle = std::atan2(sin_a, cos_a);  // góc cần xoay [0, π]

        // =====================================================
        // depth scaling — depth càng lớn correction càng mạnh
        // depth thường 0.001~0.02m → scale lên
        // =====================================================
        double gain = std::clamp(depth * 80.0, 0.05, 1.0);
        angle *= gain;

        // =====================================================
        // clamp góc tối đa mỗi lần recompute
        // =====================================================
        constexpr double max_angle = M_PI / 12.0;  // 15 deg
        angle = std::clamp(angle, 0.0, max_angle);

        // =====================================================
        // Tính rotation matrix từ angle-axis
        // =====================================================
        Eigen::AngleAxisd aa(angle, axis);
        Eigen::Matrix3d   R = aa.toRotationMatrix();

        // =====================================================
        // Extract roll/pitch từ rotation matrix
        // R = Rz(yaw) * Ry(pitch) * Rx(roll)  (ZYX convention)
        // roll  = atan2(R(2,1), R(2,2))
        // pitch = atan2(-R(2,0), sqrt(R(2,1)^2 + R(2,2)^2))
        // =====================================================
        double roll_target  = std::atan2( R(2,1), R(2,2));
        double pitch_target = std::atan2(-R(2,0), std::sqrt(R(2,1)*R(2,1) + R(2,2)*R(2,2)));

        // =====================================================
        // smoothing
        // =====================================================
        constexpr double alpha = 0.4;  // trust new value hơn prev
        double roll  = alpha * roll_prev  + (1.0 - alpha) * roll_target;
        double pitch = alpha * pitch_prev + (1.0 - alpha) * pitch_target;

        // =====================================================
        // convergence: khi correction đủ nhỏ thì dừng
        // =====================================================
        constexpr double thresh = M_PI / 180.0;  // 1 deg
        if (std::abs(roll_target  - roll)  < thresh &&
            std::abs(pitch_target - pitch) < thresh)
        {
            recompute = false;
            RCLCPP_INFO(this->get_logger(), "Collision recompute converged.");
        }

        RCLCPP_INFO(this->get_logger(),
            "avoid_dir=(%.3f %.3f %.3f) axis=(%.3f %.3f %.3f) "
            "angle_deg=%.2f depth=%.4f gain=%.3f "
            "roll_target=%.3f pitch_target=%.3f "
            "-> roll=%.3f pitch=%.3f",
            avoid_dir.x(), avoid_dir.y(), avoid_dir.z(),
            axis.x(), axis.y(), axis.z(),
            angle * 180.0 / M_PI, depth, gain,
            roll_target, pitch_target, roll, pitch);

        return {roll, pitch};
    }

    void compute_offset_position(
        double x, double y, double z,
        double roll, double pitch, double yaw,
        double offset_distance,
        double& x_out, double& y_out, double& z_out)  // float → double
    {
        double cr = std::cos(roll),  sr = std::sin(roll);
        double cp = std::cos(pitch), sp = std::sin(pitch);
        double cy = std::cos(yaw),   sy = std::sin(yaw);

        double R02 = cy*sp*cr + sy*sr;
        double R12 = sy*sp*cr - cy*sr;
        double R22 = cp*cr;

        x_out = x -offset_distance * R02;
        y_out = y -offset_distance * R12;
        z_out = z -offset_distance * R22;
    }

    geometry_msgs::msg::Pose offsetPose(const geometry_msgs::msg::Pose& input_pose, double y_offset, double z_offset, double  yaw_offset) {
        // Tính offset Z theo hướng Z của TCP
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
        pose.orientation = tf2::toMsg(q_new);  // Chuyển đổi quaternion về msg

        return pose;
    }
/*
    void topic_callback(const test_msgs::msg::RosYolo& msg) {
        if (!is_robot_moving_ && !msg.ros_yolo.empty()) {
            is_robot_moving_ = true;
            allow_request_ = true;
            const auto& result = msg.ros_yolo.front();
            target_position_ = {result.x, result.y, result.z, result.roll, result.pitch, result.yall};
            RCLCPP_INFO(this->get_logger(), "Updated target: x=%.2f y=%.2f z=%.2f r=%.2f p=%.2f y=%.2f", result.x,
                        result.y, result.z, result.roll, result.pitch, result.yall);
            target_pose = transformToBaseFrame(target_position_);
            test_pose = offsetPose(target_pose, 0.0, offset_distance_, 0.0 );  // Lưu lại pose để sử dụng sau này
            ws_check = PosesCheck(test_pose, home_position_);
        }
    }
*/
    void topic_callback(const test_msgs::msg::RosYolo& msg)
    {
        if (!target_ready_ && !msg.ros_yolo.empty()) {
            // Đánh dấu robot bắt đầu xử lý
            allow_request_ = true;
            // Ghi toàn bộ danh sách pose vào target_position_
            target_position_.clear();
            target_pose_list_.clear();
            target_base_transform_ready_ = false;
            for (const auto& result : msg.ros_yolo) {
                target_position_.push_back({
                    result.x,
                    result.y,
                    result.z,
                    result.roll,
                    result.pitch,
                    result.yall,
                    static_cast<double>(result.x1),
                    static_cast<double>(result.y1),
                    static_cast<double>(result.x2),
                    static_cast<double>(result.y2)
                });
            }
            is_robot_moving_ = true;
            target_ready_ = true;
            //Log chi tiết toàn bộ danh sách đã nhận
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
            start_detection_time = 0.0;  // Reset start_time after publishing
            detection_time = 0.0;  // Reset detection_time after publishing
            positioning_time = 0.0;
            const auto& time = msg.collect_msg.front();
            //start_time = time.start_time;
            start_detection_time = time.start_detection;
            detection_time = time.detection_time;
            positioning_time = time.positioning_time;
            time_recieved_ = true;
            RCLCPP_WARN(this->get_logger(), "DEBUG time_sub: start_detection_time=%.3f, positioning_time=%.3f, detection_time=%.3f", start_detection_time, positioning_time, detection_time);
        }
    }

    void waitForReconnect() {
        while (rclcpp::ok()) {
            RCLCPP_WARN(this->get_logger(), "lost connection. Trying to reconnect...");
            if(//is_server_ready_ || 
                is_reset_){
                RCLCPP_INFO(this->get_logger(), "Reconnected successfully.");
                break;
            }
        }
    } 

    void publisher_callback(bool flag, double x, bool pause = true, bool skip = false) {
        res_msgs::msg::PoseRes res;
        res_msgs::msg::ResFlag flag_msg;
        flag_msg.flag = flag;
        flag_msg.x = x;
        flag_msg.pause = pause;  // Set y to 0.0 as per your requirement
        flag_msg.skip = skip;
        res.pose_res.push_back(flag_msg);
        publisher_->publish(res);
    } 

    void time_publisher(double end_time, bool check = true, int count = 0) {
        std::lock_guard<std::mutex> lock(pub_mutex);
        collect_msgs::msg::CollectMsg msg;
        collect_msgs::msg::CollectTime time;
        total_time = 0.0;  // Reset total_time after publishing
        total_time = end_time - start_detection_time - positioning_time - detection_time - temp_total_time;
        temp_total_time = temp_total_time + total_time;
        time.total_time = total_time;
//        if (!mul_mode) {
        time.detection_time = detection_time;
        time.positioning_time = positioning_time;
        temp_total_time = 0.0;
//        } else {
//            time.detection_time = 0.0;
//            time.positioning_time = 0.0;
//        }
        RCLCPP_WARN(this->get_logger(), "DEBUG time_publisher: end_time=%.3f, start_detection_time=%.3f, positioning_time=%.3f, detection_time=%.3f, temp_total_time=%.3f, total_time=%.3f",
                    end_time, start_detection_time, positioning_time, detection_time, temp_total_time, total_time);
        ///time.detection_time = detection_time;
        time.check = check;
        time.count = count;
        msg.collect_msg.push_back(time);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        time_publisher_->publish(msg);
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

    void publish_signal(bool signal) {
        publish_depth_signal(signal);
        publish_position_signal(signal);
    }

    void publish_skip_signal(bool skip) {
        skip_signal_msgs::msg::SkipSignal msg;
        msg.skip = skip;
        skip_signal_pub->publish(msg);
    }

    void publish_move_signal(bool move) {
        move_signal_msgs::msg::MoveSignal msg;
        msg.move = move;
        move_signal_pub->publish(msg);
    }


    geometry_msgs::msg::Pose jointStatesToPose(const std::vector<double>& joint_values)
    {
        auto robot_model = move_group_interface_->getRobotModel();
        const moveit::core::JointModelGroup* jmg =
            robot_model->getJointModelGroup("indy_manipulator");

        moveit::core::RobotState robot_state(robot_model);
        robot_state.setJointGroupPositions(jmg, joint_values);
        robot_state.update();  // tính FK

        // Lấy pose của link cuối (EEF)
        const std::string& eef_link = move_group_interface_->getEndEffectorLink();
        const Eigen::Isometry3d& transform = robot_state.getGlobalLinkTransform(eef_link);

        // Convert Eigen → geometry_msgs
        geometry_msgs::msg::Pose pose;
        pose.position.x = transform.translation().x();
        pose.position.y = transform.translation().y();
        pose.position.z = transform.translation().z();

        Eigen::Quaterniond q(transform.rotation());
        pose.orientation.x = q.x();
        pose.orientation.y = q.y();
        pose.orientation.z = q.z();
        pose.orientation.w = q.w();

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

    geometry_msgs::msg::Pose transformToBaseFrame(const std::array<double, 6>& position) {
        geometry_msgs::msg::Pose msg;
        msg.position.x = position[0];
        msg.position.y = position[1];
        msg.position.z = position[2];
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
            target[0], target[1], target[2],
            target[3], target[4], target[5],
            object_offset_,
            target_idx_position_[0], target_idx_position_[1], target_idx_position_[2]);
        target_idx_position_[3] = target[3];
        target_idx_position_[4] = target[4];
        target_idx_position_[5] = target[5];

        auto converted_pose = transformToBaseFrame(target_idx_position_);
        if (idx >= target_pose_list_.size()) {
            target_pose_list_.resize(idx + 1);
        }
        target_pose_list_[idx] = converted_pose;
        return converted_pose;
    }

    void rebuildTargetPoseList()
    {
        target_pose_list_.resize(target_position_.size());
        for (std::size_t i = 0; i < target_position_.size(); ++i) {
            target_pose_list_[i] = targetPositionToBasePose(i);
        }
    }
};

int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);
    
    auto node = std::make_shared<MoveItController>();
    
    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(node->get_node_base_interface());
    
    // Spin trên thread riêng TRƯỚC khi gọi initialize()
    std::thread spin_thread([&executor]() {
        executor.spin();
    });
    
    // Bây giờ initialize() mới có executor đang chạy để xử lý timer/service callback
    node->initialize();
    
    spin_thread.join();
    rclcpp::shutdown();
    return 0;
}
