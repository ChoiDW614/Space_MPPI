cmake_minimum_required(VERSION 3.5)
project(ets_vii)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(urdf REQUIRED)

install(DIRECTORY
  config
  launch
  worlds
  rviz
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
