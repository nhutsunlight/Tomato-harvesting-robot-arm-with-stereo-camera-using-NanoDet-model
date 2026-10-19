cmake_minimum_required(VERSION 3.8)
project(cpp_pubsub)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(stereo_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(yolov8_msgs REQUIRED)
find_package(test_msgs REQUIRED)
find_package(res_msgs REQUIRED)
find_package(control_action REQUIRED)
find_package(collect_msgs REQUIRED)
find_package(connect_msgs REQUIRED)
find_package(OpenCV REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(PCL REQUIRED COMPONENTS common filters io kdtree segmentation)
find_package(yaml-cpp REQUIRED)
find_package(image_geometry REQUIRED)
find_package(octomap REQUIRED)
find_package(octomap_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(tomato_octomap_msgs REQUIRED)
find_package(config_manager REQUIRED)
find_package(position_signal_msgs REQUIRED)
find_package(skip_signal_msgs REQUIRED)
find_package(depth_signal_msgs REQUIRED)
find_package(move_signal_msgs REQUIRED)

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

add_executable(tomato_3d_detector_node src/subscriber_member_function.cpp)
target_link_libraries(tomato_3d_detector_node yaml-cpp  octomap octomath)
ament_target_dependencies(tomato_3d_detector_node
  rclcpp
  rclcpp_action
  sensor_msgs
  stereo_msgs
  yolov8_msgs
  res_msgs
  test_msgs
  collect_msgs
  connect_msgs
  cv_bridge
  control_action
  PCL
  yaml-cpp
  image_geometry
  pcl_conversions
  octomap_msgs
  tomato_octomap_msgs
  config_manager
  position_signal_msgs
  skip_signal_msgs
  depth_signal_msgs
  move_signal_msgs
)

include_directories(
  ${PCL_INCLUDE_DIRS}
)

target_link_libraries(tomato_3d_detector_node
  ${PCL_LIBRARIES}
)

# Install target
install(TARGETS
  tomato_3d_detector_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
