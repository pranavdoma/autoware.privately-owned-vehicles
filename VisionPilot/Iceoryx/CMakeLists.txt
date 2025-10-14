cmake_minimum_required(VERSION 3.16)
project(iceoryx_demo)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find dependencies
find_package(iceoryx_posh REQUIRED)
find_package(iceoryx_binding_c REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core highgui imgproc)
find_package(CUDA REQUIRED)

# Add our common libraries from the VisionPilot project
# Adjust the relative paths if necessary
set(COMMON_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../common)

# GStreamer Engine Library
add_library(gstreamer_engine STATIC
    ${COMMON_DIR}/sensors/gstreamer_engine.cpp
)
target_include_directories(gstreamer_engine PUBLIC ${COMMON_DIR}/include)
# GStreamer requires pkg-config to find its libraries
find_package(PkgConfig REQUIRED)
pkg_check_modules(GSTREAMER REQUIRED gstreamer-1.0 gstreamer-app-1.0)
target_include_directories(gstreamer_engine PUBLIC ${GSTREAMER_INCLUDE_DIRS})
target_link_libraries(gstreamer_engine PUBLIC ${GSTREAMER_LIBRARIES})
target_compile_definitions(gstreamer_engine PRIVATE LOG_TYPE=0) # Use printf for logging

# AutoSpeed TensorRT Backend Library
add_library(autospeed_backend STATIC
    ${COMMON_DIR}/backends/autospeed/tensorrt_engine.cpp
)
target_include_directories(autospeed_backend PUBLIC 
    ${COMMON_DIR}/include
    ${CUDA_INCLUDE_DIRS}
    /usr/include/x86_64-linux-gnu/ # For TensorRT headers
)
target_link_libraries(autospeed_backend PUBLIC
    gstreamer_engine
    ${CUDA_LIBRARIES}
    nvinfer
    nvparsers
    nvonnxparser
)
target_compile_definitions(autospeed_backend PRIVATE LOG_TYPE=0) # Use printf for logging

# Set include directory for the topic header
include_directories(include)

# Publisher Executable
add_executable(publisher src/publisher_node.cpp)
target_link_libraries(publisher
    PRIVATE
    autospeed_backend
    gstreamer_engine
    iceoryx_posh::iceoryx_posh
    iceoryx_binding_c::iceoryx_binding_c
    ${OpenCV_LIBS}
)
target_compile_options(publisher PRIVATE "-Wno-unused-parameter")

# Subscriber Executable
add_executable(subscriber src/subscriber_node.cpp)
target_link_libraries(subscriber
    PRIVATE
    iceoryx_posh::iceoryx_posh
    iceoryx_binding_c::iceoryx_binding_c
    ${OpenCV_LIBS}
)
target_compile_options(subscriber PRIVATE "-Wno-unused-parameter")

# Install executables
install(TARGETS publisher subscriber
    RUNTIME DESTINATION bin
)
