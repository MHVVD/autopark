cmake_minimum_required(VERSION 3.8)
project(autopark_msgs)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/ParkingSlot.msg"
  "msg/ParkingSlotArray.msg"
  "msg/ParkingPath.msg"
  "msg/ControlStatus.msg"
  "msg/ParkingStatus.msg"
  "srv/ResetScenario.srv"
  "srv/PlanParking.srv"
  "srv/SetPose.srv"
  DEPENDENCIES std_msgs geometry_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
