#pragma once

#include "pose_estimator.hpp"
#include "pose_renderer.hpp"
#include "types.hpp"

#include <string>

class FrameProcessor {
public:
    explicit FrameProcessor(const std::string& model_path);

    bool initialize();
    bool process(FramePacket& packet);

private:
    void draw_debug_info(FramePacket& packet);

private:
    PoseEstimator pose_estimator_;
    PoseRenderer pose_renderer_;
};