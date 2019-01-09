# Build all kfusion source files into library kfusion
add_module_library(kfusion)

# Link to kfusion library
target_link_libraries(kfusion
  ${CUDA_CUDART_LIBRARY}
  ${PCL_LIBRARIES}
)
