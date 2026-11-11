cmake_minimum_required(VERSION 3.8)
project(hippo_maps)

find_package(ament_cmake REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(rclcpp REQUIRED)
find_package(foxglove_msgs REQUIRED)
find_package(hippo_guidance_msgs REQUIRED)
find_package(yaml-cpp REQUIRED)

add_executable(occupancy_grid_bridge src/occupancy_grid_bridge.cpp)
ament_target_dependencies(occupancy_grid_bridge
  ament_index_cpp rclcpp foxglove_msgs hippo_guidance_msgs)
target_link_libraries(occupancy_grid_bridge yaml-cpp)
target_compile_features(occupancy_grid_bridge PRIVATE cxx_std_17)
install(TARGETS occupancy_grid_bridge DESTINATION lib/${PROJECT_NAME})

install(PROGRAMS scripts/generate_occupancy_grid.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME generate_occupancy_grid
)
install(PROGRAMS scripts/view_occupancy_grid.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME view_occupancy_grid
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
endif()

ament_package()
