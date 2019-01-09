# ---------------------------------------------------------------------------- #
# GOOGLE TEST DEPENDECIES
# ---------------------------------------------------------------------------- #

# Implementation heavily based upon:
# https://github.com/kaizouman/gtest-cmake-example

# Enable ExternalProject CMake module
include(ExternalProject)

# Download and install GoogleTest
ExternalProject_Add(
    gtest
    URL https://github.com/google/googletest/archive/release-1.8.1.zip
    PREFIX ${CMAKE_CURRENT_BINARY_DIR}/gtest
    # Disable install step
    INSTALL_COMMAND ""
)

# Get GTest source and binary directories from CMake project
ExternalProject_Get_Property(gtest source_dir binary_dir)

# Create a libgtest target to be used as a dependency by test programs
add_library(libgtest IMPORTED STATIC GLOBAL)
add_dependencies(libgtest gtest)

# Set libgtest properties
set_target_properties(libgtest PROPERTIES
    "IMPORTED_LOCATION" "${binary_dir}/googlemock/gtest/libgtest.a"
    "IMPORTED_LINK_INTERFACE_LIBRARIES" "${CMAKE_THREAD_LIBS_INIT}"
)

# Create a libgmock target to be used as a dependency by test programs
add_library(libgmock IMPORTED STATIC GLOBAL)
add_dependencies(libgmock gtest)

# Set libgmock properties
set_target_properties(libgmock PROPERTIES
    "IMPORTED_LOCATION" "${binary_dir}/googlemock/libgmock.a"
    "IMPORTED_LINK_INTERFACE_LIBRARIES" "${CMAKE_THREAD_LIBS_INIT}"
)


# ---------------------------------------------------------------------------- #
# TEST INCLUDE DIRECTORIES
# ---------------------------------------------------------------------------- #
include_directories("${source_dir}/googletest/include"
                    "${source_dir}/googlemock/include")

# ---------------------------------------------------------------------------- #
# CREATE TESTS
# ---------------------------------------------------------------------------- #
# Force test build
set(gtest_build_samples ON)

# Add quaternion Test
CREATE_TEST(sobfu_test)

# Link all libraries for testing
target_link_libraries(sobfu_test
  sobfu
  ${CUDA_CUDART_LIBRARY}
  ${OpenCV_LIBS}
  ${PCL_LIBRARIES}
)
