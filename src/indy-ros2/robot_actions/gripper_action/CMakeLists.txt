cmake_minimum_required(VERSION 3.8)
project(gripper_action)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(control_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)
find_package(connect_msgs REQUIRED)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

rosidl_generate_interfaces(${PROJECT_NAME}
  "action/GripperControl.action"
  DEPENDENCIES builtin_interfaces
)
ament_export_dependencies(rosidl_default_runtime)

rosidl_get_typesupport_target(cpp_typesupport_target gripper_action "rosidl_typesupport_cpp")

add_executable(gripper_action_server src/gripper_action_server.cpp)
ament_target_dependencies(gripper_action_server rclcpp rclcpp_action std_msgs moveit_ros_planning_interface connect_msgs)

add_executable(gripper_action_client src/gripper_action_client.cpp)
ament_target_dependencies(gripper_action_client rclcpp rclcpp_action)

add_dependencies(gripper_action_server ${cpp_typesupport_target})
add_dependencies(gripper_action_client ${cpp_typesupport_target})

target_link_libraries(gripper_action_server ${cpp_typesupport_target})
target_link_libraries(gripper_action_client ${cpp_typesupport_target})

install(TARGETS
  gripper_action_server
  gripper_action_client
  DESTINATION lib/${PROJECT_NAME})

ament_package()
