import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, RegisterEventHandler, OpaqueFunction, SetEnvironmentVariable
from launch.event_handlers import OnProcessExit
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def launch_setup(context, *args, **kwargs):
    description_package = FindPackageShare('indy_description')
    gazebo_package = FindPackageShare('indy_gazebo')
    gazebo_worlds_path = 'worlds'

    # ✅ Set ngay lập tức trong Python process
    pkg_share = FindPackageShare('indy_gazebo').find('indy_gazebo')
    gazebo_models_path = os.path.join(pkg_share, 'models')
    
    # Append thay vì ghi đè
    current = os.environ.get('GAZEBO_MODEL_PATH', '')
    os.environ['GAZEBO_MODEL_PATH'] = gazebo_models_path + (':' + current if current else '')
    
    # Debug: in ra để kiểm tra path có đúng không
    print(f"\n[DEBUG] GAZEBO_MODEL_PATH = {os.environ['GAZEBO_MODEL_PATH']}\n")
    print(f"[DEBUG] models folder exists: {os.path.exists(gazebo_models_path)}\n")

    # Initialize Arguments
    name = LaunchConfiguration("name")
    indy_type = LaunchConfiguration("indy_type")
    indy_eye = LaunchConfiguration("indy_eye")
    prefix = LaunchConfiguration("prefix")
    launch_rviz = LaunchConfiguration("launch_rviz")
    world = LaunchConfiguration('world')

    world_path = PathJoinSubstitution([
        gazebo_package,
        gazebo_worlds_path,
        world
    ])

    if (indy_type.perform(context) == 'indyrp2') or (indy_type.perform(context) == 'indyrp2_v2'):
        initial_joint_controllers = PathJoinSubstitution(
            [gazebo_package, "controller", "indy_controllers_7dof.yaml"]
        )
    else:
        initial_joint_controllers = PathJoinSubstitution(
            [gazebo_package, "controller", "indy_controllers_6dof.yaml"]
        )

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
            " ",
            "sim_gazebo:=true",
            " ",
            "simulation_controllers:=",
            initial_joint_controllers,
        ]
    )
    robot_description = {"robot_description": robot_description_content}

    rviz_config_file = PathJoinSubstitution(
        [description_package, "rviz_config", "indy.rviz"]
    )

    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[{"use_sim_time": True}, robot_description],
    )

    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster", "--controller-manager", "/controller_manager"],
        output="screen",
    )

    joint_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_trajectory_controller", "-c", "/controller_manager"],
    )

    # Start gripper action controller
    #start_gripper_action_controller_cmd = ExecuteProcess(
    #    cmd=['ros2', 'control', 'load_controller', '--set-state', 'active',
    #         'gripper_action_controller'],
    #    output='screen')

    start_gripper_action_controller_cmd = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["gripper_action_controller", "-c", "/controller_manager"],
    )

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [FindPackageShare("gazebo_ros"), "/launch", "/gazebo.launch.py"]
        ),
        launch_arguments={
            'world': world_path,
            'verbose': 'false',
            'pause': 'false',
#            'server_required': 'true',   # đảm bảo có server
#            'gui_required': 'true',      # nếu bạn muốn mở giao diện
        }.items(),
    )

    # Spawn robot
    gazebo_spawn_robot = Node(
        package="gazebo_ros",
        executable="spawn_entity.py",
        name="spawn_indy",
        arguments=[
            "-entity", "indy", 
            "-topic", 
            "robot_description",
            "-x", "0.6",
            "-y", "0.8",
            "-z", "0.5",
            "-R", "0",
            "-P", "0",
            "-Y", "3.14"
        ],
        output="screen",
    )

    rviz_node = Node(
        condition=IfCondition(launch_rviz),
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="log",
        arguments=["-d", rviz_config_file],
    )

    # Delay start joint_state_broadcaster
    delay_joint_state_broadcaster_spawner = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=gazebo_spawn_robot,
            on_exit=[joint_state_broadcaster_spawner],
        )
    )

    # Delay start of robot_controller
    delay_robot_controller_spawner = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=joint_state_broadcaster_spawner,
            on_exit=[joint_controller_spawner],
        )
    )

    #Delay gripper spawner
    delay_gripper_spawner = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=joint_controller_spawner,
            on_exit=[start_gripper_action_controller_cmd],
        )
    )

    # Delay rviz
    delay_rviz2_spawner = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=start_gripper_action_controller_cmd,
            on_exit=[rviz_node],
        )
    )

    nodes_to_start = [
        gazebo,
        gazebo_spawn_robot,

        robot_state_publisher_node,

        #joint_state_broadcaster_spawner,
        #joint_controller_spawner,

        delay_joint_state_broadcaster_spawner,
        delay_robot_controller_spawner,
        delay_gripper_spawner,
        #delay_rviz2_spawner,
    ]

    return nodes_to_start


def generate_launch_description():
    #default_world_file = 'test2.world'
    default_world_file = 'tomato-farm.world'
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
            "prefix",
            default_value='""',
            description="Prefix of the joint names, useful for multi-robot setup. \
            If changed than also joint names in the controllers configuration have to be updated."
        )
    )

    declared_arguments.append(
        DeclareLaunchArgument(
            "launch_rviz", 
            default_value="true", 
            description="Launch RViz?"
        )
    )

    declared_arguments.append(
        DeclareLaunchArgument(
            'world',
            default_value=default_world_file,
            description='SDF world file'
        )
    )

    return LaunchDescription(declared_arguments + [OpaqueFunction(function=launch_setup)])
    