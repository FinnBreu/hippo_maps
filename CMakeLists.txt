cmake_minimum_required(VERSION 3.8)
project(hippo_maps)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/OccupancyGrid3D.msg"
  DEPENDENCIES std_msgs geometry_msgs
)
ament_export_dependencies(rosidl_default_runtime)

install(PROGRAMS scripts/generate_occupancy_grid.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME generate_occupancy_grid
)
install(PROGRAMS scripts/view_occupancy_grid.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME view_occupancy_grid
)
install(PROGRAMS scripts/occupancy_grid_bridge.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME occupancy_grid_bridge
)
install(FILES scripts/occupancy_grid.py
  DESTINATION lib/${PROJECT_NAME}
)
install(DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  add_test(NAME test_generate_occupancy_grid
    COMMAND "${Python3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/test/test_generate_occupancy_grid.py")
  add_test(NAME test_occupancy_grid
    COMMAND "${Python3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/test/test_occupancy_grid.py")
  set_tests_properties(test_occupancy_grid PROPERTIES
    ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}/rosidl_generator_py:$ENV{PYTHONPATH}")
endif()

ament_package()
