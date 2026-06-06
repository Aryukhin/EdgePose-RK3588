#include "frame_processor.hpp"

#include <chrono>
#include <opencv2/opencv.hpp>
#include <sstream>

FrameProcessor::FrameProcessor(const std::string& model_path)
    : pose_estimator_(model_path),
      pose_renderer_(0.3F) {}

bool FrameProcessor::initialize() {
    return pose_estimator_.initialize();
}

bool FrameProcessor::process(FramePacket& packet) {
    if (packet.frame.empty()) {
        return false;
    }

    const auto process_start = std::chrono::steady_clock::now();

    if (!pose_estimator_.infer(
            packet.frame,
            packet.poses,
            packet.inference_ms,
            packet.postprocess_ms)) {
        return false;
    }

    pose_renderer_.draw(packet.frame, packet.poses);
    draw_debug_info(packet);

    const auto process_end = std::chrono::steady_clock::now();

    packet.process_ms =
        std::chrono::duration<double, std::milli>(
            process_end - process_start
        ).count();

    return true;
}

void FrameProcessor::draw_debug_info(FramePacket& packet) {
    std::ostringstream text;
    text << "Frame: " << packet.frame_id
         << " Poses: " << packet.poses.size();

    cv::putText(
        packet.frame,
        text.str(),
        cv::Point(30, 40),
        cv::FONT_HERSHEY_SIMPLEX,
        0.8,
        cv::Scalar(0, 255, 0),
        2
    );
}