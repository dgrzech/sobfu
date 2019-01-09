# ---------------------------------------------------------------------------- #
# CMAKE INIT CONFIGURATIONS
# ---------------------------------------------------------------------------- #
cmake_minimum_required(VERSION 2.8.8)

# ---------------------------------------------------------------------------- #
# SET THE PROJECT
# ---------------------------------------------------------------------------- #
project(kfusion C CXX)

# ---------------------------------------------------------------------------- #
# UTILITY FUNCTIONS (MACROS) AND PATHS
# ---------------------------------------------------------------------------- #
LIST(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/")
include(Utils)
include(Targets)

# ---------------------------------------------------------------------------- #
# CMAKE SETTINGS
# ---------------------------------------------------------------------------- #
# Use both Debug and Release versions
SET(CMAKE_CONFIGURATION_TYPES "Debug;Release")

# Set CXX to compile with C++11 standard
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -MP -MD -pthread -fpermissive")

# ---------------------------------------------------------------------------- #
# DEPENDENCIES
# ---------------------------------------------------------------------------- #
find_package(Boost 1.58.0 REQUIRED COMPONENTS filesystem program_options)
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core viz highgui calib3d)
find_package(PCL 1.8 REQUIRED)
find_package(VTK REQUIRED)

# ---------------------------------------------------------------------------- #
# CUDA SETTINGS
# ---------------------------------------------------------------------------- #
SET(HAVE_CUDA 1)
LIST(APPEND CUDA_NVCC_FLAGS
   "-gencode;arch=compute_61,code=sm_61;
    --ftz=true;
    --prec-div=false;
    --prec-sqrt=false;
    --expt-relaxed-constexpr"
)

if(UNIX OR APPLE)
  LIST(APPEND CUDA_NVCC_FLAGS "-Xcompiler;-fPIC")
endif()

# ---------------------------------------------------------------------------- #
# SET INCLUDE PATHS
# ---------------------------------------------------------------------------- #
include_directories(include
  ${Boost_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${VTK_INCLUDE_DIRS}
)

# ---------------------------------------------------------------------------- #
# BUILD SRC DIRECTORY
# ---------------------------------------------------------------------------- #
add_subdirectory(src)

# ---------------------------------------------------------------------------- #
# BUILD TESTS
# ---------------------------------------------------------------------------- #
if(BUILD_TESTS)
  enable_testing()
  add_subdirectory(test)
endif()
