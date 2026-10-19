cmake_minimum_required(VERSION 3.8)
project(start_request_service)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(connect_msgs REQUIRED)
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

find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/StartRequest.srv"
)

ament_export_dependencies(rosidl_default_runtime)

# ===== Server =====
add_executable(start_request_server src/start_request_server.cpp)
ament_target_dependencies(start_request_server rclcpp connect_msgs res_msgs)

rosidl_get_typesupport_target(typesupport_target ${PROJECT_NAME} "rosidl_typesupport_cpp")
target_link_libraries(start_request_server ${typesupport_target})

# ===== Client =====
add_executable(start_request_client src/start_request_client.cpp)
ament_target_dependencies(start_request_client rclcpp)

target_link_libraries(start_request_client ${typesupport_target})

# ===== Install =====
install(TARGETS
  start_request_server
  start_request_client
  DESTINATION lib/${PROJECT_NAME})

ament_package()
