################################################################################################
# short command to setup source group
function(kf_source_group group)
  cmake_parse_arguments(VW_SOURCE_GROUP "" "" "GLOB" ${ARGN})
  file(GLOB srcs ${VW_SOURCE_GROUP_GLOB})
  #list(LENGTH ${srcs} ___size)
  #if (___size GREATER 0)
    source_group(${group} FILES ${srcs})
  #endif()
endfunction()


################################################################################################
# short command for declaring includes from other modules
macro(declare_deps_includes)
  foreach(__arg ${ARGN})
    get_filename_component(___path "${CMAKE_SOURCE_DIR}/modules/${__arg}/include" ABSOLUTE)
    if (EXISTS ${___path})
      include_directories(${___path})
    endif()
  endforeach()

  unset(___path)
  unset(__arg)
endmacro()


################################################################################################
# short command for setting defeault target properties
function(default_properties target)
  set_target_properties(${target} PROPERTIES
    DEBUG_POSTFIX "d"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

    if (NOT ${target} MATCHES "^test_")
      install(TARGETS ${the_target} RUNTIME DESTINATION ".")
    endif()
endfunction()

function(test_props target)
  #os_project_label(${target} "[test]")
  if(USE_PROJECT_FOLDERS)
    set_target_properties(${target} PROPERTIES FOLDER "Tests")
  endif()
endfunction()

function(app_props target)
  #os_project_label(${target} "[app]")
  if(USE_PROJECT_FOLDERS)
    set_target_properties(${target} PROPERTIES FOLDER "Apps")
  endif()
endfunction()


################################################################################################
# short command for setting defeault target properties
function(default_properties target)
  set_target_properties(${target} PROPERTIES
    DEBUG_POSTFIX "d"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

    if (NOT ${target} MATCHES "^test_")
      install(TARGETS ${the_target} RUNTIME DESTINATION ".")
    endif()
endfunction()


################################################################################################
# short command for adding library module
macro(add_module_library name)
  set(module_name ${name})

  FILE(GLOB_RECURSE sources *.cpp *.cu)

  cuda_add_library(${module_name} STATIC ${sources})

  if(MSVC)
    set_target_properties(${module_name} PROPERTIES DEFINE_SYMBOL KFUSION_API_EXPORTS)
  else()
    add_definitions(-DKFUSION_API_EXPORTS)
  endif()

  default_properties(${module_name})
endmacro()


################################################################################################
# short command for adding application module
macro(add_application target)
  FILE(GLOB_RECURSE sources *.cpp)
  add_executable(${target} ${sources})
  default_properties(${target})
endmacro()


################################################################################################
# short command for adding test target
macro(CREATE_TEST target)
  FILE(GLOB SRCS *.cpp)
  ADD_EXECUTABLE(${target} ${SRCS})
  TARGET_LINK_LIBRARIES(${target} libgtest libgmock)
  add_test(NAME ${target} COMMAND ${target})
  default_properties(${target})
endmacro()
