cmake_minimum_required(VERSION 3.8)
project(harvest_flag_bridge)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(connect_msgs REQUIRED)
find_package(yolov8_msgs REQUIRED)

add_executable(harvest_flag_bridge_node src/harvest_flag_bridge_node.cpp)
ament_target_dependencies(harvest_flag_bridge_node
  rclcpp
  connect_msgs
  yolov8_msgs
)

install(TARGETS
  harvest_flag_bridge_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
