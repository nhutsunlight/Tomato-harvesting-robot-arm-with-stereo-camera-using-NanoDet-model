cmake_minimum_required(VERSION 3.8)
project(stereo_rectify_cpp)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(message_filters REQUIRED)
find_package(image_transport REQUIRED)
find_package(OpenCV REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(res_msgs REQUIRED)
find_package(yolov8_msgs REQUIRED)

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

add_executable(stereo_rectify_node src/stereo_rectify_node.cpp)

ament_target_dependencies(stereo_rectify_node
  rclcpp
  sensor_msgs
  cv_bridge
  message_filters
  image_transport
  OpenCV
  res_msgs
  yolov8_msgs
)

target_link_libraries(stereo_rectify_node
  ${OpenCV_LIBRARIES}
)

install(TARGETS
  stereo_rectify_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
