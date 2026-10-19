cmake_minimum_required(VERSION 3.8)
project(robot_move_action)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(moveit_core REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(action_msgs REQUIRED)
find_package(connect_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

# Generate service interface
rosidl_generate_interfaces(${PROJECT_NAME}
  "action/MoveRobot.action"
  DEPENDENCIES builtin_interfaces geometry_msgs
)

# Ensure headers are found
include_directories(include ${CMAKE_CURRENT_BINARY_DIR}/rosidl_generator_cpp)

# Add server executable
add_executable(robot_move_server src/moveit_service_server.cpp)
ament_target_dependencies(robot_move_server 
  rclcpp 
  moveit_core 
  moveit_ros_planning_interface 
  connect_msgs
  rosidl_default_runtime
  rclcpp_action
)

# Link service interface with server
rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} "rosidl_typesupport_cpp")
target_link_libraries(robot_move_server ${cpp_typesupport_target})

# Add client executable
add_executable(robot_move_client src/moveit_service_client.cpp)
ament_target_dependencies(robot_move_client 
  rclcpp 
  rosidl_default_runtime
  rclcpp_action
)

# Link service interface with client
rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} "rosidl_typesupport_cpp")
target_link_libraries(robot_move_client ${cpp_typesupport_target})

# Install executables
install(TARGETS
  robot_move_server
  robot_move_client
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY 
	  launch
    DESTINATION share/${PROJECT_NAME}
)

ament_package()
