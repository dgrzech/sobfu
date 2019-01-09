# Build all sobfu source files into library dynfy
add_module_library(sobfu)

# Link to sobfu library
target_link_libraries(sobfu
    kfusion
    gcov
    ${Boost_LIBRARIES}
    ${CUDA_CUDART_LIBRARY}
    ${PCL_LIBRARIES}
)
