cmake_minimum_required(VERSION 3.8)
project(hippo_maps)

find_package(ament_cmake REQUIRED)
find_package(hippo_guidance_msgs REQUIRED)

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
endif()

ament_package()
