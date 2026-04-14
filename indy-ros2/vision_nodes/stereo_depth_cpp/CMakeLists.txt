cmake_minimum_required(VERSION 3.8)
project(stereo_depth_cpp)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

#set(OpenCV_DIR "OpenCV_DIR=~/ros2_ws/install/opencv-4.12.0/lib/cmake/opencv4")

# find dependencies
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(stereo_msgs REQUIRED)
find_package(message_filters REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc ximgproc)
find_package(ament_index_cpp REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(image_transport REQUIRED)
find_package(image_geometry REQUIRED)
find_package(res_msgs REQUIRED)
find_package(collect_msgs REQUIRED)
find_package(yolov8_msgs REQUIRED)
find_package(config_manager REQUIRED)

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

# ==== Build libELAS từ external ====

include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
)

add_executable(stereo_depth_node src/stereo_depth_node.cpp)
target_include_directories(stereo_depth_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(stereo_depth_node
  ${OpenCV_LIBS}
  yaml-cpp
)

ament_target_dependencies(stereo_depth_node
  rclcpp
  sensor_msgs
  stereo_msgs
  message_filters
  cv_bridge
  ament_index_cpp
  image_transport
  image_geometry
  res_msgs
  collect_msgs
  yolov8_msgs 
  config_manager
)

install(TARGETS stereo_depth_node
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config)

ament_package()
