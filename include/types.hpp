#pragma once

#include <opencv2/opencv.hpp>

#include <cstdint>
#include <vector>

struct Keypoint {
    float x = 0.0F;
    float y = 0.0F;
    float confidence = 0.0F;
};

struct BoundingBox {
    float x1 = 0.0F;
    float y1 = 0.0F;
    float x2 = 0.0F;
    float y2 = 0.0F;
    float confidence = 0.0F;
};

struct HumanPose {
    BoundingBox box;
    std::vector<Keypoint> keypoints;
};

struct FramePacket {
    cv::Mat frame;
    std::uint64_t frame_id = 0;
    double timestamp_ms = 0.0;

    double read_ms = 0.0;
    double convert_ms = 0.0;
    double process_ms = 0.0;

    double preprocess_ms = 0.0;
    double inference_ms = 0.0;
    double postprocess_ms = 0.0;

    std::vector<HumanPose> poses;
};