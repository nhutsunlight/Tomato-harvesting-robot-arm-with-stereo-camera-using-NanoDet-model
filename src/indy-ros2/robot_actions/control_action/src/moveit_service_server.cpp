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

#include <octomap_msgs/msg/octomap.hpp>
#include <octomap_msgs/conversions.h>
#include "tomato_octomap_msgs/msg/tomato_octomaps.hpp"
#include "tomato_octomap_msgs/msg/tomato_octomap.hpp"
#include <moveit_msgs/srv/apply_planning_scene.hpp>
#include "depth_signal_msgs/msg/depth_signal.hpp"
#include "position_signal_msgs/msg/position_signal.hpp"
#include "skip_signal_msgs/msg/skip_signal.hpp"
#include "move_signal_msgs/msg/move_signal.hpp"
#include "control_action/workflow.hpp"

using moveit::planning_interface::MoveGroupInterface;

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
    friend struct control_action::ActionRequests;
    friend struct control_action::ControlChecks;
    friend struct control_action::Workflow;
    friend struct control_action::ExecuteWorkflow;
    friend struct control_action::TestExecuteWorkflow;

    //std::shared_ptr<rclcpp::Node> moveit_node_;
    moveit::planning_interface::PlanningSceneInterface planning_scene_interface_;
    std::unique_ptr<MoveGroupInterface> move_group_interface_;
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    tf2_ros::TransformListener tf_listener_;
    //moveit::core::RobotModelPtr robot_model_;
    std::thread connection_monitor_thread_;
    std::shared_ptr<planning_scene::PlanningScene> cached_scene_;
    std::mutex scene_mutex_;
    //std::array<double, 6> target_position_;
    std::mutex pub_mutex;
    std::mutex acm_mutex_;
    std::vector<std::array<double, 6>> target_position_;
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
    bool scene_valid_ = false;
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
    rclcpp::Publisher<depth_signal_msgs::msg::DepthSignal>::SharedPtr depth_signal_pub;
    rclcpp::Publisher<position_signal_msgs::msg::PositionSignal>::SharedPtr position_signal_pub;
    rclcpp::Publisher<res_msgs::msg::PoseRes>::SharedPtr publisher_;
    rclcpp::Publisher<collect_msgs::msg::CollectMsg>::SharedPtr time_publisher_;
    rclcpp::Publisher<skip_signal_msgs::msg::SkipSignal>::SharedPtr         skip_signal_pub;
    rclcpp::Publisher<move_signal_msgs::msg::MoveSignal>::SharedPtr         move_signal_pub;
//    rclcpp::Subscription<octomap_msgs::msg::Octomap>::SharedPtr octomap_sub_;
    std::map<int, octomap_msgs::msg::Octomap> octomap_map_;
    std::mutex octomap_map_mutex_;
    rclcpp::Subscription<tomato_octomap_msgs::msg::TomatoOctomaps>::SharedPtr tomato_octomap_sub_;

    rclcpp::Client<moveit_msgs::srv::GetPlanningScene>::SharedPtr planning_scene_client_;

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

    void tomatoOctomapCallback(
        const tomato_octomap_msgs::msg::TomatoOctomaps::SharedPtr msg)
    {
        if (!obs_ready) {
            std::lock_guard<std::mutex> lock(octomap_map_mutex_);
            octomap_map_.clear();
            for (const auto& to : msg->octomaps) {
                octomap_map_[to.idx] = transformOctomapToLink0(to.octomap);
                RCLCPP_INFO(this->get_logger(), "Stored octomap for idx %d", to.idx);
            }
            obs_ready = true;  // đánh dấu octomap đã sẵn sàng
        }
    }

    octomap_msgs::msg::Octomap transformOctomapToLink0(
        const octomap_msgs::msg::Octomap& input_octomap)
    {
        if (input_octomap.header.frame_id == "link0") {
            return input_octomap;
        }

        geometry_msgs::msg::TransformStamped tf_msg;
        try {
            tf_msg = tf_buffer_->lookupTransform(
                "link0",
                input_octomap.header.frame_id,
                tf2::TimePointZero,
                tf2::durationFromSec(1.0));
        } catch (const tf2::TransformException& ex) {
            RCLCPP_ERROR(this->get_logger(),
                "Failed to transform octomap from %s to link0: %s",
                input_octomap.header.frame_id.c_str(), ex.what());
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
            if (!input_tree->isNodeOccupied(*it)) {
                continue;
            }

            const tf2::Vector3 point_in(
                it.getX(),
                it.getY(),
                it.getZ());
            const tf2::Vector3 point_out = tf * point_in;
            output_tree.updateNode(
                octomap::point3d(
                    point_out.x(),
                    point_out.y(),
                    point_out.z()),
                true);
        }
        output_tree.updateInnerOccupancy();

        octomap_msgs::msg::Octomap output_octomap;
        octomap_msgs::binaryMapToMsg(output_tree, output_octomap);
        output_octomap.header.frame_id = "link0";
        output_octomap.header.stamp = input_octomap.header.stamp;
        return output_octomap;
    }

    void applyOctomapForIdx(int idx)
    {
        octomap_msgs::msg::Octomap octomap_to_apply;
        {
            std::lock_guard<std::mutex> lock(octomap_map_mutex_);
            auto it = octomap_map_.find(idx);
            if (it == octomap_map_.end()) {
                RCLCPP_WARN(this->get_logger(), "No octomap for idx %d", idx);
                return;
            }
            octomap_to_apply = it->second;
        }

        moveit_msgs::msg::PlanningScene planning_scene_msg;
        planning_scene_msg.is_diff = true;
        planning_scene_msg.world.octomap.octomap = octomap_to_apply;
        planning_scene_msg.world.octomap.header = octomap_to_apply.header;
        planning_scene_msg.world.octomap.origin.orientation.w = 1.0;

        // Dùng service — đồng bộ, đợi move_group confirm xong mới return
        auto apply_client = this->create_client<moveit_msgs::srv::ApplyPlanningScene>(
            "/apply_planning_scene");

        if (!apply_client->wait_for_service(std::chrono::seconds(3))) {
            RCLCPP_ERROR(this->get_logger(), "Service /apply_planning_scene not available");
            return;
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
            RCLCPP_ERROR(this->get_logger(), "Timeout applying octomap for idx %d", idx);
            return;
        }

        RCLCPP_INFO(this->get_logger(), "Octomap applied for idx %d", idx);
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

    void callMoveToHome(const std::vector<double>& joint_positions,
                        size_t id,
                        size_t pass_permit = 0)
    {
        control_action::ActionRequests::callMoveToHome(
            *this, joint_positions, id, pass_permit);
    }

    void callMoveRobot(const geometry_msgs::msg::Pose& start_pose,
                       const geometry_msgs::msg::Pose& target_pose,
                       size_t id,
                       size_t mode)
    {
        control_action::ActionRequests::callMoveRobot(
            *this, start_pose, target_pose, id, mode);
    }

    void sendGripperCommand(double position, size_t id, size_t pass_permit = 0)
    {
        control_action::ActionRequests::sendGripperCommand(
            *this, position, id, pass_permit);
    }

    void resendGripperCommand()
    {
        control_action::ActionRequests::resendGripperCommand(*this);
    }

    void execute(const std::shared_ptr<GoalHandleControlRobot> goal_handle)
    {
        control_action::Workflow::execute(*this, goal_handle);
    }

    bool PosesCheck(const geometry_msgs::msg::Pose& input_pose,
                    const std::vector<double>& joint_values)
    {
        return control_action::ControlChecks::PosesCheck(
            *this, input_pose, joint_values);
    }

    void posecheck_and_recompute(const std::array<double, 6>& test_position,
                                 const std::vector<double>& test_joint_values,
                                 std::size_t idx)
    {
        control_action::ControlChecks::posecheck_and_recompute(
            *this, test_position, test_joint_values, idx);
    }

    bool refreshPlanningScene()
    {
        return control_action::ControlChecks::refreshPlanningScene(*this);
    }

    bool checkCollisionAtTarget(const geometry_msgs::msg::Pose& target_pose)
    {
        return control_action::ControlChecks::checkCollisionAtTarget(
            *this, target_pose);
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
                    result.yall
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

    void time_publisher(double end_time, bool check = true) {
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
