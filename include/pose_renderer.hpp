#pragma once

#include "types.hpp"

#include <opencv2/opencv.hpp>

#include <vector>

class PoseRenderer {
public:
    explicit PoseRenderer(float keypoint_threshold = 0.3F);

    void draw(
        cv::Mat& frame,
        const std::vector<HumanPose>& poses
    ) const;

private:
    float keypoint_threshold_;
};