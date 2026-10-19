cmake_minimum_required(VERSION 3.8)
project(stereo_camera_info_cpp)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(camera_info_manager REQUIRED)
find_package(stereo_image_proc REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(res_msgs REQUIRED)

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

add_executable(stereo_camera_info_node src/stereo_camera_info_node.cpp)
ament_target_dependencies(stereo_camera_info_node rclcpp sensor_msgs camera_info_manager stereo_image_proc rclcpp_action res_msgs)

target_link_libraries(stereo_camera_info_node
  rclcpp::rclcpp
  camera_info_manager::camera_info_manager
)

install(TARGETS
  stereo_camera_info_node
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY config
  DESTINATION share/${PROJECT_NAME})

install(DIRECTORY 
	  launch
    DESTINATION share/${PROJECT_NAME}
)

ament_package()
