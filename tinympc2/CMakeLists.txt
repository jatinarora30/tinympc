cmake_minimum_required(VERSION 3.10)
project(TinyMPC2)

set(CMAKE_CXX_STANDARD 17)

# Eigen include path (system-installed)
include_directories(/usr/include/eigen3)

# OSQP manually built include paths
include_directories(
    /home/jatinarora/osqp/include       # contains osqp.h
    /home/jatinarora/osqp/build/include      # contains osqp_configure.h
)

# OSQP manually built lib path
link_directories(/home/jatinarora/osqp/build/out)

# Find Python libraries (requires python3-dev installed)
find_package(PythonLibs 3 REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})

# Optional: print Python info
message(STATUS "Found Python libraries: ${PYTHON_LIBRARIES}")
message(STATUS "Found Python include dirs: ${PYTHON_INCLUDE_DIRS}")

# Build target
add_executable(tinympc2 src/mpc.cpp)

# Link OSQP and Python libraries
target_link_libraries(tinympc2 osqp ${PYTHON_LIBRARIES})
