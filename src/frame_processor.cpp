#include "frame_processor.hpp"

#include <opencv2/opencv.hpp>
#include <sstream>

FrameProcessor::FrameProcessor() {}

bool FrameProcessor::process(FramePacket& packet) {
    if (packet.frame.empty()) {
        return false;
    }

    // Здесь потом будет:
    // 1. preprocess
    // 2. pose estimation inference
    // 3. postprocess keypoints
    // 4. draw skeleton

    draw_debug_info(packet);

    return true;
}

void FrameProcessor::draw_debug_info(FramePacket& packet) {
    std::ostringstream text;
    text << "Frame: " << packet.frame_id;

    cv::putText(
        packet.frame,
        text.str(),
        cv::Point(30, 40),
        cv::FONT_HERSHEY_SIMPLEX,
        1.0,
        cv::Scalar(0, 255, 0),
        2
    );

    cv::circle(
        packet.frame,
        cv::Point(packet.frame.cols / 2, packet.frame.rows / 2),
        10,
        cv::Scalar(0, 0, 255),
        -1
    );
}