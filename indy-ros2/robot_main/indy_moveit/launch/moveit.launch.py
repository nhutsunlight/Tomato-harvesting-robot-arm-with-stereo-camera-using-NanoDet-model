# import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from indy_moveit.launch_common import load_yaml
from launch_ros.descriptions import ComposableNode
from launch_ros.actions import Node, ComposableNodeContainer


def launch_setup(context, *args, **kwargs):
    description_package = FindPackageShare('indy_description')
    moveit_config_package = FindPackageShare('indy_moveit')

    # Initialize Arguments
    name = LaunchConfiguration("name")
    indy_type = LaunchConfiguration("indy_type")
    indy_eye = LaunchConfiguration("indy_eye")
    servo_mode = LaunchConfiguration("servo_mode")
    prefix = LaunchConfiguration("prefix")
    launch_rviz_moveit = LaunchConfiguration("launch_rviz_moveit")
    use_sim_time = LaunchConfiguration("use_sim_time")
    only_robot = LaunchConfiguration("only_robot")
    camera_test = LaunchConfiguration("camera_test")

    robot_description_content = Command(
        [
            PathJoinSubstitution([FindExecutable(name="xacro")]),
            " ",
            PathJoinSubstitution([description_package, "urdf", "indy.urdf.xacro"]),
            " ",
            "name:=",
            name,
            " ",
            "indy_type:=",
            indy_type,
            " ",
            "indy_eye:=",
            indy_eye,
            " ",
            "prefix:=",
            prefix,
        ]
    )
    robot_description = {"robot_description": robot_description_content}

    # MoveIt Configuration
    robot_description_semantic_content = Command(
        [
            PathJoinSubstitution([FindExecutable(name="xacro")]),
            " ",
            PathJoinSubstitution([moveit_config_package, "srdf", "indy.srdf.xacro"]),
            " ",
            "name:=",
            name,
            " ",
            "indy_type:=",
            indy_type,
            " ",
            "indy_eye:=",
            indy_eye,
            " ",
            "prefix:=",
            prefix,
        ]
    )
    robot_description_semantic = {"robot_description_semantic": robot_description_semantic_content}

    robot_description_kinematics = PathJoinSubstitution(
        [moveit_config_package, "moveit_config", "kinematics.yaml"]
    )
    
    ompl_planning_pipeline_config = {
        "move_group": {
            "planning_plugin": "ompl_interface/OMPLPlanner",
            "request_adapters": """default_planner_request_adapters/AddTimeOptimalParameterization default_planner_request_adapters/FixWorkspaceBounds default_planner_request_adapters/FixStartStateBounds default_planner_request_adapters/FixStartStateCollision default_planner_request_adapters/FixStartStatePathConstraints""",
            "start_state_max_bounds_error": 0.5,
        }
    }
    ompl_planning_yaml = load_yaml("indy_moveit", "moveit_config/ompl_planning.yaml")
    ompl_planning_pipeline_config["move_group"].update(ompl_planning_yaml)

    # Trajectory Execution Configuration
    if (indy_type.perform(context) == 'indyrp2') or (indy_type.perform(context) == 'indyrp2_v2'):
        controllers_yaml = load_yaml("indy_moveit", "moveit_config/controllers_7dof.yaml")
    else:
        controllers_yaml = load_yaml("indy_moveit", "moveit_config/controllers_6dof.yaml")

    moveit_controllers = {
        "moveit_simple_controller_manager": controllers_yaml,
        "moveit_controller_manager": "moveit_simple_controller_manager/MoveItSimpleControllerManager",
    }

    trajectory_execution = {
        "moveit_manage_controllers": False,
        "trajectory_execution.allowed_execution_duration_scaling": 1.2,
        "trajectory_execution.allowed_goal_duration_margin": 1.0,
        "trajectory_execution.allowed_start_tolerance": 0.5,
        "trajectory_execution.trajectory_duration_monitoring": False
    }

    planning_scene_monitor_parameters = {
        "publish_planning_scene": True,
        "publish_geometry_updates": True,
        "publish_state_updates": True,
        "publish_transforms_updates": True,
    }

    # Start the actual move_group node/action server
    move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[
            robot_description,
            robot_description_semantic,
            robot_description_kinematics,
            ompl_planning_pipeline_config,
            trajectory_execution,
            moveit_controllers,
            planning_scene_monitor_parameters,
            {"use_sim_time": use_sim_time},
        ],
    )

    # rviz with moveit configuration
    if servo_mode.perform(context) == 'true':
        rviz_config_file = PathJoinSubstitution(
            [moveit_config_package, "rviz_config", "indy_servo.rviz"]
        )    
    else:
        rviz_config_file = PathJoinSubstitution(
            [moveit_config_package, "rviz_config", "indy_moveit.rviz"]
        )

    rviz_node = Node(
        condition=IfCondition(launch_rviz_moveit),
        package="rviz2",
        executable="rviz2",
        name="rviz2_moveit",
        output="log",
        arguments=["-d", rviz_config_file],
        parameters=[
            robot_description,
            robot_description_semantic,
            ompl_planning_pipeline_config,
            robot_description_kinematics,
        ],
    )

    # Servo node for realtime control
    servo_yaml = load_yaml("indy_moveit", "moveit_config/indy_servo.yaml")
    servo_params = {"moveit_servo": servo_yaml}
    servo_node = Node(
        package="moveit_servo",
        executable="servo_node_main",
        parameters=[
            servo_params,
            robot_description,
            robot_description_semantic,
            robot_description_kinematics
        ],
        output="screen",
    )

    move_to_home_action = Node(
        name="move_to_home_node",
        package="robot_home_action",
        executable="move_to_home_server",
        output="screen",
        parameters=[
            robot_description,
            robot_description_semantic,
            robot_description_kinematics,
            {"use_sim_time": use_sim_time},
            #{"use_sim_time": True},
        ],
        arguments=['--ros-args', '--log-level', 'info'],  # chỉ log khi có lỗi
    )

    robot_move_action = Node(
        name="robot_move_action_node",
        package="robot_move_action",
        executable="robot_move_server",
        output="screen",
        parameters=[
            robot_description,
            robot_description_semantic,
            robot_description_kinematics,
            {"use_sim_time": use_sim_time},
            #{"use_sim_time": True},
        ],
        arguments=['--ros-args', '--log-level', 'info'],  # chỉ log khi có lỗi
    )

    control_action = Node(
        name="moveit_controller_node",
        package="control_action",
        executable="moveit_service_server",
        output="screen",
        parameters=[
            robot_description,
            robot_description_semantic,
            robot_description_kinematics,
            {"use_sim_time": use_sim_time},
            #{"use_sim_time": True},
        ],
        arguments=['--ros-args', '--log-level', 'info'],
    )

    gripper_action = Node(
        name="gripper_node",
        package="gripper_action",
        executable="gripper_action_server",
        output="screen",
        parameters=[
            robot_description,
            robot_description_semantic,
            robot_description_kinematics,
            {"use_sim_time": use_sim_time},
            #{"use_sim_time": True},
        ],
        arguments=['--ros-args', '--log-level', 'info'],  # chỉ log khi có lỗi
    )

    stereo_camera_info_node = Node(
        package="stereo_camera_info_cpp",
        executable="stereo_camera_info_node",
        name="stereo_camera_info",
        output="screen",
        parameters=[
            {"use_sim_time": use_sim_time},
            #{"use_sim_time": True},
        ],
        arguments=['--ros-args', '--log-level', 'info'],  # chỉ log khi có lỗi
    )

    stereo_depth_cpp_node = Node(
        package="stereo_depth_cpp",
        executable="stereo_depth_node",
        name="stereo_depth_cpp",
        output="screen",
        parameters=[
            {"use_sim_time": use_sim_time},
            #{"use_sim_time": True},
        ],
        arguments=['--ros-args', '--log-level', 'info'],  # chỉ log khi có lỗi
    )

    stereo_rectify_node = Node(
        package="stereo_rectify_cpp",
        executable="stereo_rectify_node",
        name="stereo_rectify",
        output="screen",
        parameters=[
            {"use_sim_time": use_sim_time},
            #{"use_sim_time": True},
        ],
        arguments=['--ros-args', '--log-level', 'info'],  # chỉ log khi có lỗi
    )

    depth_map_node = Node(
        package="depth_map",
        executable="depth_publisher_node",
        name="depth_map_node",
        output="screen",
        parameters=[
            {"use_sim_time": use_sim_time},
            #{"use_sim_time": True},
        ],
        arguments=['--ros-args', '--log-level', 'info'],  # chỉ log khi có lỗi
    )

    camera_rectify_node = Node(
        package="camera_rectify_cpp",
        executable="camera_rectify_node",
        name="camera_rectify",
        output="screen",
        parameters=[
            {"use_sim_time": use_sim_time},
            #{"use_sim_time": True},
        ],
        arguments=['--ros-args', '--log-level', 'info'],  # chỉ log khi có lỗi
    )

    detector_node = Node(
        package="cpp_pubsub",
        executable="tomato_3d_detector_node",
        name="detector_node",
        output="screen",
        parameters=[
            {"use_sim_time": use_sim_time},
            #{"use_sim_time": True},
        ],
        arguments=['--ros-args', '--log-level', 'info'],  # chỉ log khi có lỗi
    )

    serial_node = Node(
        package="serial_hello_world",
        executable="hello_uart_node",
        name="serial_node",
        output="screen",
        parameters=[
            {"use_sim_time": use_sim_time},
            #{"use_sim_time": True},
        ],
        arguments=['--ros-args', '--log-level', 'info'],  # chỉ log khi có lỗi
    )

    start_service = Node(
        package="start_request_service",
        executable="start_request_server",
        name="start_request_node",
        output="screen",
        parameters=[
            {"use_sim_time": use_sim_time},
            #{"use_sim_time": True},
        ],
        arguments=['--ros-args', '--log-level', 'info'],  # chỉ log khi có lỗi
    )

    collect_logger_node = Node(
        package="harvet_info",
        executable="collect_logger_node",
        name="collect_logger_node",
        output="screen",
        parameters=[
            {"use_sim_time": use_sim_time},
            #{"use_sim_time": True},
        ],
        arguments=['--ros-args', '--log-level', 'info'],  # chỉ log khi có lỗi
    )

    recognition_py_node = Node(
        package="yolobot_recognition_py",
        executable="recognition_node",
        name="recognition_py_node",
        output="screen",
        parameters=[
            {"use_sim_time": use_sim_time},
            #{"use_sim_time": True},
        ],
        arguments=['--ros-args', '--log-level', 'info'],  # chỉ log khi có lỗi
    )

    config_manager_node = Node(
        package="config_manager",
        executable="config_manager_node",
        name="config_manager_node",
        output="screen",
        parameters=[
            {"use_sim_time": use_sim_time},
            #{"use_sim_time": True},
        ],
        arguments=['--ros-args', '--log-level', 'info'],  # chỉ log khi có lỗi
    )

    container = ComposableNodeContainer(
        name="moveit_servo_container",
        namespace="/",
        package="rclcpp_components",
        executable="component_container",
        composable_node_descriptions=[
            # ComposableNode(
            #     package="indy_moveit",
            #     plugin="moveit_servo::JoyToServoPub",
            #     name="controller_to_servo_node",
            # ),
            ComposableNode(
                package="joy",
                plugin="joy::Joy",
                name="joy_node",
                # extra_arguments=[{
                #     "use_intra_process_comms": True,
                #     "deadzone": 0.35
                # }],
            )
        ],
        output="screen",
    )

    gripper_attacher = Node(
        name="gripper_attacher_node",
        package="gripper_attacher",
        executable="gripper_attacher_node",
        output="screen",
        parameters=[
            robot_description,
            robot_description_semantic,
            robot_description_kinematics,
            {"use_sim_time": use_sim_time},
            #{"use_sim_time": True},
        ],
        arguments=['--ros-args', '--log-level', 'info'],  # chỉ log khi có lỗi
    )

    if servo_mode.perform(context) == 'true':        
        nodes_to_start = [servo_node, container, rviz_node]
        # nodes_to_start = [servo_node, rviz_node]
    elif only_robot.perform(context) == 'true':
        nodes_to_start = [
            move_group_node,
            rviz_node,
        ]
    elif camera_test.perform(context) == 'true':
        nodes_to_start = [
            move_group_node,
            rviz_node,
            stereo_camera_info_node,
            stereo_depth_cpp_node,
            stereo_rectify_node,
            camera_rectify_node,
            depth_map_node,
            recognition_py_node,
            config_manager_node,
        ]
    else:
        nodes_to_start = [
            move_group_node, 
            rviz_node, 
            #move_group_service,
            #move_group_action,
            control_action,
            stereo_camera_info_node,
            stereo_depth_cpp_node,
            stereo_rectify_node,
            camera_rectify_node,
            depth_map_node,
            move_to_home_action,
            robot_move_action,
            gripper_action,
            detector_node,
            #serial_node,
            start_service,
            collect_logger_node,
            recognition_py_node,
            #gripper_attacher,
            config_manager_node,
        ]


    return nodes_to_start


def generate_launch_description():
    declared_arguments = []

    declared_arguments.append(
        DeclareLaunchArgument(
            "name",
            default_value="indy"
        )
    )

    declared_arguments.append(
        DeclareLaunchArgument(
            "indy_type",
            default_value="indy7",
            description="Type of Indy robot.",
            choices=["indy7", "indy7_v2" , "indy12", "indy12_v2", "indyrp2", "indyrp2_v2", "icon7l", "icon3", "nuri3s", "nuri4s", "nuri7c", "nuri20c", "opti5"]
        )
    )

    declared_arguments.append(
        DeclareLaunchArgument(
            "indy_eye",
            default_value="false",
            description="Work with Indy Eye",
        )
    )

    declared_arguments.append(
        DeclareLaunchArgument(
            "servo_mode",
            default_value="false",
            description="Servoing mode",
        )
    )

    declared_arguments.append(
        DeclareLaunchArgument(
            "prefix",
            default_value='""',
            description="Prefix of the joint names, useful for multi-robot setup. \
            If changed than also joint names in the controllers configuration have to be updated."
        )
    )

    declared_arguments.append(
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="true",
            description="Make MoveIt to use simulation time. This is needed for the trajectory planing in simulation.",
        )
    )

    declared_arguments.append(
        DeclareLaunchArgument(
            "only_robot",
            default_value="false",
            description="Launch only robot model without MoveIt and RViz.",
        )
    )

    declared_arguments.append(
        DeclareLaunchArgument(
            "camera_test",
            default_value="false",
            description="Launch only robot model without MoveIt and RViz.",
        )
    )

    declared_arguments.append(
        DeclareLaunchArgument("launch_rviz_moveit", default_value="true", description="Launch RViz?")
    )

    return LaunchDescription(declared_arguments + [OpaqueFunction(function=launch_setup)])
