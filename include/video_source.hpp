#pragma once

#include "types.hpp"

#include <opencv2/opencv.hpp>
#include <string>
#include <cstdint>

class VideoSource {
public:
    explicit VideoSource(const std::string& source);

    bool open();
    bool read(FramePacket& packet);

    int width() const;
    int height() const;
    double fps() const;

private:
    std::string source_;
    cv::VideoCapture cap_;
    std::uint64_t frame_id_;
};