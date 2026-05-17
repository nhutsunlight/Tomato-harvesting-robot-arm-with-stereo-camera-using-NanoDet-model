cmake_minimum_required(VERSION 3.8)
project(control_action)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(moveit_core REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)
find_package(test_msgs REQUIRED)
find_package(res_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(action_msgs REQUIRED)
find_package(robot_home_action REQUIRED)
find_package(gripper_action REQUIRED)
find_package(connect_msgs REQUIRED)
find_package(robot_move_action REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(collect_msgs REQUIRED)
find_package(stereo_msgs REQUIRED)
find_package(octomap_msgs REQUIRED)
find_package(moveit_ros_planning REQUIRED)
find_package(moveit_msgs REQUIRED)
find_package(tomato_octomap_msgs REQUIRED)
find_package(config_manager REQUIRED)
find_package(depth_signal_msgs REQUIRED)
find_package(position_signal_msgs)
find_package(skip_signal_msgs REQUIRED)
find_package(move_signal_msgs REQUIRED)

# Generate service interface
rosidl_generate_interfaces(${PROJECT_NAME}
  "action/MoveRobot.action"
  DEPENDENCIES builtin_interfaces res_msgs
)

# Ensure headers are found
include_directories(include ${CMAKE_CURRENT_BINARY_DIR}/rosidl_generator_cpp)
include_directories(
  ${moveit_ros_planning_INCLUDE_DIRS}
  ${moveit_core_INCLUDE_DIRS}
)

# Add server executable
add_executable(moveit_service_server src/moveit_service_server.cpp)
target_compile_definitions(moveit_service_server PRIVATE CONTROL_ACTION_USE_TEST_WORKFLOW)
target_link_libraries(moveit_service_server yaml-cpp)
ament_target_dependencies(moveit_service_server 
  rclcpp 
  moveit_core 
  moveit_ros_planning_interface 
  test_msgs 
  res_msgs 
  connect_msgs
  rosidl_default_runtime
  rclcpp_action
  robot_home_action
  gripper_action
  robot_move_action
  yaml-cpp
  collect_msgs
  stereo_msgs
  octomap_msgs
  moveit_ros_planning
  moveit_msgs
  tomato_octomap_msgs
  config_manager
  depth_signal_msgs
  position_signal_msgs
  skip_signal_msgs
  move_signal_msgs
)

# Link service interface with server
rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} "rosidl_typesupport_cpp")
target_link_libraries(moveit_service_server ${cpp_typesupport_target})

# Add client executable
add_executable(moveit_service_client src/moveit_service_client.cpp)
ament_target_dependencies(moveit_service_client 
  rclcpp 
  rosidl_default_runtime
  rclcpp_action
)

# Link service interface with client
rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} "rosidl_typesupport_cpp")
target_link_libraries(moveit_service_client ${cpp_typesupport_target})

# Install executables
install(TARGETS
  moveit_service_server
  moveit_service_client
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY 
	  launch
    DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY
  include/
  DESTINATION include
)

ament_package()
