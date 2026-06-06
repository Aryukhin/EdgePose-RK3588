#pragma once

#include "types.hpp"

#include <opencv2/opencv.hpp>

#include <string>
#include <vector>

class PoseEstimator {
public:
    explicit PoseEstimator(const std::string& model_path);
    ~PoseEstimator();

    PoseEstimator(const PoseEstimator&) = delete;
    PoseEstimator& operator=(const PoseEstimator&) = delete;

    bool initialize();
    bool infer(
        const cv::Mat& frame,
        std::vector<HumanPose>& poses,
        double& inference_ms,
        double& postprocess_ms
    );

private:
    bool preprocess(const cv::Mat& frame, cv::Mat& input);
    bool run_inference(const cv::Mat& input);
    bool postprocess(
        const cv::Size& original_size,
        std::vector<HumanPose>& poses
    );

private:
    std::string model_path_;
    bool initialized_ = false;

    int input_width_ = 640;
    int input_height_ = 640;

    // Позже здесь будут:
    // rknn_context context_;
    // input/output attributes;
    // buffers;
};