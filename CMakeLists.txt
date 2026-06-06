cmake_minimum_required(VERSION 3.16)

project(edge_pose_cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

add_executable(edge_pose_cpp
    src/main.cpp
    src/video_source.cpp
    src/frame_processor.cpp
    src/fps_meter.cpp
)

target_include_directories(edge_pose_cpp PRIVATE
    include
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(edge_pose_cpp PRIVATE
    ${OpenCV_LIBS}
)

target_compile_options(edge_pose_cpp PRIVATE
    -Wall
    -Wextra
    -Wpedantic
)