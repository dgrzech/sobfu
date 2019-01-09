# Link source files to executables
add_application(app)

# Link to all libraries
target_link_libraries(app
    sobfu
    kfusion
    png
    Boost::program_options
    ${Boost_LIBRARIES}
    ${CUDA_CUDA_LIBRARY}
    ${CUDA_CUDART_LIBRARY}
    ${OpenCV_LIBS}
    ${PCL_LIBRARIES}
    ${VTK_LIBRARIES}
)
