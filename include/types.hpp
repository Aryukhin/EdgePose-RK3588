#pragma once

#include <opencv2/opencv.hpp>
#include <cstdint>

struct FramePacket {
    cv::Mat frame;
    std::uint64_t frame_id = 0;
    double timestamp_ms = 0.0;

    double read_ms = 0.0;
    double convert_ms = 0.0;
    double process_ms = 0.0;
    double write_ms = 0.0;
};