#include <moveit/move_group_interface/move_group_interface.h>
#include <chrono>
#include <functional>
#include <future>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <memory>
#include <moveit_msgs/msg/constraints.hpp>
#include <moveit_msgs/msg/orientation_constraint.hpp>
#include <rclcpp/rclcpp.hpp>
#include <string>
#include <vector>
#include <moveit/trajectory_processing/iterative_time_parameterization.h>
#include <moveit/robot_trajectory/robot_trajectory.h>

#include "robot_move_action/action/move_robot.hpp"
#include "rclcpp_action/rclcpp_action.hpp"
#include "connect_msgs/msg/connect_msg.hpp"


using moveit::planning_interface::MoveGroupInterface;

class MoveItController : public rclcpp::Node {
   public:
    MoveItController()
        : Node("moveit_controller")
          //moveit_node_(std::make_shared<rclcpp::Node>("moveit_node"))
          //move_group_interface_(moveit_node_, "indy_manipulator") 
    {

//        connection_ = this->create_subscription<connect_msgs::msg::ConnectMsg>(
//            "/connect_msg", 10, std::bind(&MoveItController::connection_callback, this, std::placeholders::_1));


        action_server_ = rclcpp_action::create_server<MoveRobot>(
            this, "robot_move_action",
            std::bind(&MoveItController::handle_goal, this, std::placeholders::_1, std::placeholders::_2),
            std::bind(&MoveItController::handle_cancel, this, std::placeholders::_1),
            std::bind(&MoveItController::handle_accepted, this, std::placeholders::_1));

        //startConnectionMonitorThread();

        RCLCPP_INFO(this->get_logger(), "MoveIt Action Server started.");
    }

    void initialize() {
        move_group_interface_ = std::make_unique<MoveGroupInterface>(shared_from_this(), "indy_manipulator");
    }

    void startUpdatePoseThread(const std::string& frame_id) {
        if (update_pose_thread_.joinable()) return;  // Nếu thread đang chạy thì không tạo mới
        pose_promise_ = std::promise<bool>();        // Reset promise
        auto future_stop = pose_promise_.get_future();
        update_pose_thread_ = std::thread([this, future_stop = std::move(future_stop), frame_id]() mutable {
            RCLCPP_INFO(this->get_logger(), "Getting current pose infformation...");
            while (future_stop.wait_for(std::chrono::milliseconds(10)) == std::future_status::timeout && rclcpp::ok()) {
                auto state = move_group_interface_->getCurrentState(1.0);
                if (state) {
                    move_group_interface_->setStartState(*state);
                    break;
                }
            }
            RCLCPP_INFO(this->get_logger(), "Current pose information retrieved.");
        });
    }

    void stopUpdatePoseThread() {
        pose_promise_.set_value(true);  // Gửi tín hiệu dừng
        if (update_pose_thread_.joinable()) {
            update_pose_thread_.join();  // Đợi thread kết thúc
        }
    }

    void startConnectionMonitorThread() {
        connection_monitor_thread_ = std::thread([this]() {
            while (rclcpp::ok() && !stop_connection_monitor_) {
                is_server_ready_ = latest_connection_status_;
                is_reset_ = reset_status_;
            }
        });
    }

    ~MoveItController() {
        stopUpdatePoseThread();  // Đảm bảo thread dừng khi node bị hủy
        stop_connection_monitor_ = true;
        if (connection_monitor_thread_.joinable()) {
            connection_monitor_thread_.join();
        }
    }

   private:
    //std::shared_ptr<rclcpp::Node> moveit_node_;
    std::unique_ptr<MoveGroupInterface> move_group_interface_;
    std::thread update_pose_thread_;
    std::thread connection_monitor_thread_;
    std::promise<bool> pose_promise_;
    std::array<double, 6> target_position_;
    bool is_server_ready_ = false;
    bool is_reset_ = false;
    bool stop_connection_monitor_ = false;
    bool success_ = false;
//    size_t res_id;
//    size_t req_id;
    size_t mode_;
    std::atomic<bool> latest_connection_status_{false};
    std::atomic<bool> reset_status_{false};
    geometry_msgs::msg::Pose target_pose;
    geometry_msgs::msg::Pose start_pose;
    geometry_msgs::msg::Pose current_pose;
    geometry_msgs::msg::Pose pose;
    geometry_msgs::msg::Pose next_pose_;
    using MoveRobot = robot_move_action::action::MoveRobot;
    using GoalHandleMoveRobot = rclcpp_action::ServerGoalHandle<MoveRobot>;
    rclcpp_action::Server<MoveRobot>::SharedPtr action_server_;
    rclcpp::Subscription<connect_msgs::msg::ConnectMsg>::SharedPtr connection_;

    rclcpp_action::GoalResponse handle_goal(const rclcpp_action::GoalUUID& uuid,
                                            std::shared_ptr<const MoveRobot::Goal> goal) {
        RCLCPP_INFO(this->get_logger(), "Received action request!");
        (void)goal;
        (void)uuid;
        mode_ = goal->mode;
        //req_id = goal->id;
        if (mode_ == 0 && !is_reset_) {
            target_pose = goal->target_pose;
            return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
        }
        else if (mode_ == 1 && !is_reset_) {
            start_pose = goal->start_pose;
            target_pose = goal->target_pose;
            return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
        }
        else if (mode_ == 2 && !is_reset_) {
            target_pose = goal->target_pose;
            next_pose_ = goal->start_pose;  // tái dùng start_pose để truyền target2
            return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
        } else {
            return rclcpp_action::GoalResponse::REJECT;
        }
    }

    rclcpp_action::CancelResponse handle_cancel(const std::shared_ptr<GoalHandleMoveRobot> goal_handle) {
        (void)goal_handle;
        RCLCPP_INFO(this->get_logger(), "Goal canceled!");
        return rclcpp_action::CancelResponse::ACCEPT;
    }

    void handle_accepted(const std::shared_ptr<GoalHandleMoveRobot> goal_handle) {
        std::thread{std::bind(&MoveItController::execute, this, goal_handle)}.detach();

    }

    void updateStartState(const std::string& frame_id) {
        move_group_interface_->clearPoseTargets();
        move_group_interface_->clearPathConstraints();
        startUpdatePoseThread(frame_id);
        stopUpdatePoseThread();
    }

    // Hàm tìm IK solution tương thích cho cả 2 target
    bool findConsistentIK(
        const geometry_msgs::msg::Pose& target1,
        const geometry_msgs::msg::Pose& target2,
        std::vector<double>& out_joints_t1)
    {
        auto robot_model = move_group_interface_->getRobotModel();
        const auto* jmg = robot_model->getJointModelGroup("indy_manipulator");

        const int MAX_ATTEMPTS = 20;
        double best_score = std::numeric_limits<double>::max();
        bool found = false;

        for (int i = 0; i < MAX_ATTEMPTS; ++i) {
            // Random seed mỗi lần để lấy IK solution khác nhau
            moveit::core::RobotState seed_state(robot_model);
            seed_state.setToRandomPositions(jmg);

            // IK cho target1 với random seed
            if (!seed_state.setFromIK(jmg, target1, 0.05)) continue;

            std::vector<double> joints_t1;
            seed_state.copyJointGroupPositions(jmg, joints_t1);

            // IK cho target2 dùng joints_t1 làm seed
            moveit::core::RobotState state_t2(robot_model);
            state_t2.setJointGroupPositions(jmg, joints_t1);
            if (!state_t2.setFromIK(jmg, target2, 0.05)) continue;

            std::vector<double> joints_t2;
            state_t2.copyJointGroupPositions(jmg, joints_t2);

            // Tính displacement giữa 2 joint state
            double score = 0.0;
            for (size_t j = 0; j < joints_t1.size(); ++j) {
                double d = joints_t1[j] - joints_t2[j];
                score += d * d;
            }

            if (score < best_score) {
                best_score = score;
                out_joints_t1 = joints_t1;
                found = true;
            }
        }

        if (found)
            RCLCPP_INFO(this->get_logger(), "Consistent IK found, score=%.4f", best_score);
        else
            RCLCPP_ERROR(this->get_logger(), "No consistent IK found after %d attempts", MAX_ATTEMPTS);

        return found;
    }

    // Validate Cartesian path từ joints_t1 đến target2
    bool validateCartesianFromJoints(
        const std::vector<double>& joints_t1,
        const geometry_msgs::msg::Pose& target2)
    {
        auto robot_model = move_group_interface_->getRobotModel();
        const auto* jmg = robot_model->getJointModelGroup("indy_manipulator");

        moveit::core::RobotState start_state(robot_model);
        start_state.setJointGroupPositions(jmg, joints_t1);
        move_group_interface_->setStartState(start_state);

        std::vector<geometry_msgs::msg::Pose> waypoints = {target2};
        moveit_msgs::msg::RobotTrajectory trajectory;

        double fraction = move_group_interface_->computeCartesianPath(
            waypoints, 0.01, 0.0, trajectory);

        // Reset start state về current
        move_group_interface_->setStartStateToCurrentState();

        RCLCPP_INFO(this->get_logger(),
            "Cartesian validation fraction: %.2f%%", fraction * 100.0);
        return fraction >= 0.95;
    }

    void execute(const std::shared_ptr<GoalHandleMoveRobot> goal_handle) {
        //auto feedback = std::make_shared<MoveRobot::Feedback>();
        //auto result = std::make_shared<MoveRobot::Result>();
        //result->success = true; 
        if (mode_ == 0) {
            executePlan(target_pose, goal_handle);
        }
        else if (mode_ ==1) {
            moveFocusStraightCartesian(start_pose, target_pose, goal_handle);
        } 
        else if (mode_ == 2) {
            executePlan(target_pose, goal_handle);
        }
        /*if (success_) {
            result->success = true;
            goal_handle->succeed(result);
        } else {
            result->success = false;
            goal_handle->abort(result);
        }*/
    }

    void moveFocusStraightCartesian(const geometry_msgs::msg::Pose& start_pose,
                                    const geometry_msgs::msg::Pose& end_pose,
                                    const std::shared_ptr<GoalHandleMoveRobot> goal_handle) {
        auto feedback = std::make_shared<MoveRobot::Feedback>();
        auto result = std::make_shared<MoveRobot::Result>();

        move_group_interface_->clearPoseTargets();
        move_group_interface_->clearPathConstraints();

        std::vector<geometry_msgs::msg::Pose> waypoints = {start_pose, end_pose};
        moveit_msgs::msg::RobotTrajectory trajectory;

        double fraction = move_group_interface_->computeCartesianPath(
            waypoints, 0.001, 0.0, trajectory);

        RCLCPP_INFO(this->get_logger(), "Cartesian path: %.2f%%", fraction * 100.0);

        if (fraction <= 0.95) {
            RCLCPP_ERROR(this->get_logger(), 
                "Cartesian path failed (%.2f%%)", fraction * 100.0);
            success_ = false;
            result->success = false;
            goal_handle->abort(result);
            return;
        }

        // Time parameterization
        robot_trajectory::RobotTrajectory rt(
            move_group_interface_->getRobotModel(), "indy_manipulator");
        rt.setRobotTrajectoryMsg(
            *move_group_interface_->getCurrentState(), trajectory);

        trajectory_processing::IterativeParabolicTimeParameterization iptp;
        if (!iptp.computeTimeStamps(rt, 0.3, 0.3)) {
            RCLCPP_ERROR(this->get_logger(), "Time parameterization failed!");
            success_ = false;
            result->success = false;
            goal_handle->abort(result);
            return;
        }

        rt.getRobotTrajectoryMsg(trajectory);

        moveit::planning_interface::MoveGroupInterface::Plan plan;
        plan.trajectory_ = trajectory;

        // Execute với timeout
        auto exec_promise = std::make_shared<std::promise<moveit::core::MoveItErrorCode>>();
        auto exec_future = exec_promise->get_future();

        std::thread exec_thread([this, plan, exec_promise]() {
            try {
                exec_promise->set_value(move_group_interface_->execute(plan));
            } catch (const std::exception& e) {
                RCLCPP_ERROR(this->get_logger(), "Execute crash: %s", e.what());
                try {
                    exec_promise->set_value(moveit::core::MoveItErrorCode::FAILURE);
                } catch (...) {}
            }
        });

        if (exec_future.wait_for(std::chrono::seconds(30)) != std::future_status::ready) {
            RCLCPP_ERROR(this->get_logger(), "Execute timeout!");
            exec_thread.detach();
            success_ = false;
            result->success = false;
            goal_handle->abort(result);
            return;
        }

        exec_thread.join();

        if (exec_future.get() != moveit::core::MoveItErrorCode::SUCCESS) {
            RCLCPP_ERROR(this->get_logger(), "Execute failed!");
            success_ = false;
            result->success = false;
            goal_handle->abort(result);
            return;
        }

        success_ = true;
        result->success = true;
        goal_handle->succeed(result);
    }

    void executePlan(const geometry_msgs::msg::Pose& execute_pose,
                    const std::shared_ptr<GoalHandleMoveRobot> goal_handle)
    {
        auto result = std::make_shared<MoveRobot::Result>();
        move_group_interface_->clearPoseTargets();
        move_group_interface_->clearPathConstraints();

        moveit::planning_interface::MoveGroupInterface::Plan plan;
        bool plan_success = false;

        // Kiểm tra xem có next_pose để seed IK không
        //bool has_next_pose = (next_pose_.orientation.w != 0.0);  // pose mặc định w=0

        if (mode_ == 2) {
            // Tìm IK solution tương thích với cả target1 và target2
            std::vector<double> best_joints;
            if (findConsistentIK(execute_pose, next_pose_, best_joints)) {
                // Validate Cartesian path trước
                if (validateCartesianFromJoints(best_joints, next_pose_)) {
                    // Dùng joint-space goal thay vì pose goal
                    move_group_interface_->setJointValueTarget(best_joints);

                    for (int attempt = 0; attempt < 5; attempt++) {
                        auto [success, p] = planMotion();
                        if (success && !p.trajectory_.joint_trajectory.points.empty()) {
                            plan = p;
                            plan_success = true;
                            break;
                        }
                        RCLCPP_WARN(this->get_logger(),
                            "Joint-space plan attempt %d/5 failed", attempt + 1);
                    }
                } else {
                    RCLCPP_WARN(this->get_logger(),
                        "Best IK still fails Cartesian validation, fallback to pose target");
                }
            }
        }

        // Fallback: plan bình thường nếu không có next_pose hoặc IK không tìm được
        if (!plan_success) {
            move_group_interface_->setPoseTarget(execute_pose);
            for (int attempt = 0; attempt < 5; attempt++) {
                auto [success, p] = planMotion();
                if (success && !p.trajectory_.joint_trajectory.points.empty()) {
                    plan = p;
                    plan_success = true;
                    break;
                }
                RCLCPP_WARN(this->get_logger(),
                    "Fallback plan attempt %d/5 failed", attempt + 1);
            }
        }

        if (!plan_success) {
            RCLCPP_ERROR(this->get_logger(), "All plan attempts failed");
            result->success = false;
            goal_handle->abort(result);
            return;
        }

        // Execute với timeout (giữ nguyên như cũ)
        auto exec_promise = std::make_shared<std::promise<moveit::core::MoveItErrorCode>>();
        auto exec_future  = exec_promise->get_future();
        std::thread exec_thread([this, plan, exec_promise]() {
            try {
                exec_promise->set_value(move_group_interface_->execute(plan));
            } catch (const std::exception& e) {
                RCLCPP_ERROR(this->get_logger(), "Execute crash: %s", e.what());
                try { exec_promise->set_value(moveit::core::MoveItErrorCode::FAILURE); } catch (...) {}
            }
        });

        if (exec_future.wait_for(std::chrono::seconds(60)) != std::future_status::ready) {
            RCLCPP_ERROR(this->get_logger(), "Execute timeout!");
            exec_thread.detach();
            result->success = false;
            goal_handle->abort(result);
            return;
        }
        exec_thread.join();

        if (exec_future.get() == moveit::core::MoveItErrorCode::SUCCESS) {
            result->success = true;
            goal_handle->succeed(result);
        } else {
            RCLCPP_ERROR(this->get_logger(), "Execute failed!");
            result->success = false;
            goal_handle->abort(result);
        }
    }

    std::pair<bool, MoveGroupInterface::Plan> planMotion() {
        //move_group_interface_->setEndEffectorLink("tcp0");
        move_group_interface_->setPlanningTime(5.0);
        move_group_interface_->setPlannerId("RRTConnectkConfigDefault");
        move_group_interface_->setMaxVelocityScalingFactor(1.0);
        move_group_interface_->setMaxAccelerationScalingFactor(1.0);
        MoveGroupInterface::Plan plan;
        bool success = static_cast<bool>(move_group_interface_->plan(plan));
        return {success, plan};
    }
};

int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::executors::SingleThreadedExecutor executor;
    auto node = std::make_shared<MoveItController>();
    node->initialize();
    executor.add_node(node->get_node_base_interface());
    executor.spin();
    rclcpp::shutdown();
    return 0;
}
