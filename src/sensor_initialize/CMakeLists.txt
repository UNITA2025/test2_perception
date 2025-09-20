cmake_minimum_required(VERSION 3.8)
cmake_policy(SET CMP0074 NEW)
project(sensor_initialize)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_sensor_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)

find_package(visualization_msgs REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io)

add_executable(lidar_to_camera_broadcaster src/lidar_to_camera_broadcaster.cpp)

ament_target_dependencies(lidar_to_camera_broadcaster
  rclcpp
  tf2
  tf2_ros
  tf2_sensor_msgs
  sensor_msgs
  std_msgs
  pcl_conversions
  yaml-cpp
)

target_link_libraries(lidar_to_camera_broadcaster
  yaml-cpp
)

install(TARGETS
  lidar_to_camera_broadcaster
  DESTINATION lib/${PROJECT_NAME}
)

# preprocessing 노드 추가
add_executable(lidar_preprocessing_node src/lidar_preprocessing_node.cpp)

ament_target_dependencies(lidar_preprocessing_node
  rclcpp
  sensor_msgs
  pcl_conversions
)

install(TARGETS
  lidar_preprocessing_node
  DESTINATION lib/${PROJECT_NAME}
)

# launch 디렉토리 설치
install(
  DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

# config 디렉토리 설치
install(DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config
)



# ── cluster_node ───────────────────────────────────────────
add_executable(cluster_node src/cluster_node.cpp)
ament_target_dependencies(cluster_node
  rclcpp sensor_msgs visualization_msgs pcl_conversions)

target_include_directories(cluster_node SYSTEM PUBLIC ${PCL_INCLUDE_DIRS})
target_link_directories  (cluster_node PUBLIC        ${PCL_LIBRARY_DIRS})
target_link_libraries    (cluster_node ${PCL_LIBRARIES})

install(TARGETS cluster_node            
  DESTINATION lib/${PROJECT_NAME})


  

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

ament_package()
