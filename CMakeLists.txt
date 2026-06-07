cmake_minimum_required(VERSION 3.16)

project(edge_pose_cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(
    RKNN_API_ROOT
    "/home/orangepi/Projects/rknn-toolkit2/rknpu2/runtime/Linux/librknn_api"
)

set(
    RKNN_INCLUDE_DIR
    "${RKNN_API_ROOT}/include"
)

set(
    RKNN_RUNTIME_LIB
    "${RKNN_API_ROOT}/aarch64/librknnrt.so"
)

find_package(OpenCV REQUIRED)

add_executable(edge_pose_cpp
    src/main.cpp
    src/video_source.cpp
    src/frame_processor.cpp
    src/fps_meter.cpp
    src/hdmi_sink.cpp
    src/pose_estimator.cpp
    src/pose_renderer.cpp
)

target_include_directories(edge_pose_cpp PRIVATE
    include
    ${OpenCV_INCLUDE_DIRS}
    ${RKNN_INCLUDE_DIR}
)
find_library(
    RKNN_RUNTIME_LIB
    NAMES rknnrt
    PATHS
        /usr/lib
        /usr/lib/aarch64-linux-gnu
    REQUIRED
)

target_link_libraries(edge_pose_cpp PRIVATE
    ${OpenCV_LIBS}
    ${RKNN_RUNTIME_LIB}
)
set_target_properties(edge_pose_cpp PROPERTIES
    BUILD_RPATH "${RKNN_API_ROOT}/aarch64"
)
target_compile_options(edge_pose_cpp PRIVATE
    -Wall
    -Wextra
    -Wpedantic
)