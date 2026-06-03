cmake_minimum_required(VERSION 3.8)
project(robot_home_action)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)
find_package(moveit_msgs REQUIRED)
find_package(connect_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# Generate action interfaces
rosidl_generate_interfaces(robot_home_action
  "action/MoveToHome.action"
  DEPENDENCIES builtin_interfaces
)

ament_export_dependencies(rosidl_default_runtime)

# Get the type support target
rosidl_get_typesupport_target(cpp_typesupport_target robot_home_action "rosidl_typesupport_cpp")

# Define executables
add_executable(move_to_home_server src/move_to_home_server.cpp)
add_executable(move_to_home_client src/move_to_home_client.cpp)

# Set dependencies for executables
ament_target_dependencies(move_to_home_server rclcpp rclcpp_action moveit_ros_planning_interface moveit_msgs connect_msgs)
ament_target_dependencies(move_to_home_client rclcpp rclcpp_action)

# Ensure actions are generated before compiling
add_dependencies(move_to_home_server ${cpp_typesupport_target})
add_dependencies(move_to_home_client ${cpp_typesupport_target})

# Link action type support library
target_link_libraries(move_to_home_server ${cpp_typesupport_target})
target_link_libraries(move_to_home_client ${cpp_typesupport_target})

# Install executables
install(TARGETS
  move_to_home_server
  move_to_home_client
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
