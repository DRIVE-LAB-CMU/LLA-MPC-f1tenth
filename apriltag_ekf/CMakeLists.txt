cmake_minimum_required(VERSION 3.5)
# FIX 1: Change this to your actual package name! 
# (It must match the <name> tag in package.xml)
project(apriltag_ekf) 

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

# NOTE: You do NOT need to find_package() python dependencies like rclpy, tf2_ros, etc.
# Those are handled by package.xml <exec_depend>

# ---------------------------------------------------------
# C++ SECTION (Delete this block if you don't have C++ code)
# ---------------------------------------------------------
# If you DO have C++ code, uncomment these, but ensure the files actually exist!
# find_package(rclcpp REQUIRED)
# add_executable(some_cpp_node src/some_cpp_node.cpp)
# ament_target_dependencies(some_cpp_node rclcpp)
# install(TARGETS some_cpp_node DESTINATION lib/${PROJECT_NAME})
# ---------------------------------------------------------

# ---------------------------------------------------------
# PYTHON SECTION
# ---------------------------------------------------------

# 1. Install Python modules (if you have a folder named 'apriltag_ekf' with __init__.py)
# ament_python_install_package(${PROJECT_NAME})

# 2. Install the script
install(PROGRAMS
  scripts/apriltag_ekf_node.py  # FIX 2: Corrected spelling (was aprilatag)
  DESTINATION lib/${PROJECT_NAME}
)

# 3. Install config/launch files (highly recommended for your YAML)
install(DIRECTORY
  launch
  config
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()