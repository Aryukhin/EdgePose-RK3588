#include "pose_estimator.hpp"

#include <chrono>
#include <iostream>

PoseEstimator::PoseEstimator(const std::string& model_path)
    : model_path_(model_path) {}

PoseEstimator::~PoseEstimator() {
    // Позже здесь будет rknn_destroy(context_).
}

bool PoseEstimator::initialize() {
    if (model_path_.empty()) {
        std::cerr << "Pose model path is empty" << std::endl;
        return false;
    }

    std::cout << "PoseEstimator model: " << model_path_ << std::endl;

    // Позже:
    // 1. прочитать .rknn файл;
    // 2. rknn_init;
    // 3. запросить input/output attributes.

    initialized_ = true;
    return true;
}

bool PoseEstimator::infer(
    const cv::Mat& frame,
    std::vector<HumanPose>& poses,
    double& inference_ms,
    double& postprocess_ms
) {
    if (!initialized_) {
        std::cerr << "PoseEstimator is not initialized" << std::endl;
        return false;
    }

    if (frame.empty()) {
        std::cerr << "PoseEstimator received an empty frame" << std::endl;
        return false;
    }

    cv::Mat input;

    if (!preprocess(frame, input)) {
        return false;
    }

    const auto inference_start = std::chrono::steady_clock::now();

    if (!run_inference(input)) {
        return false;
    }

    const auto inference_end = std::chrono::steady_clock::now();

    const auto postprocess_start = inference_end;

    poses.clear();

    if (!postprocess(frame.size(), poses)) {
        return false;
    }

    const auto postprocess_end = std::chrono::steady_clock::now();

    inference_ms =
        std::chrono::duration<double, std::milli>(
            inference_end - inference_start
        ).count();

    postprocess_ms =
        std::chrono::duration<double, std::milli>(
            postprocess_end - postprocess_start
        ).count();

    return true;
}

bool PoseEstimator::preprocess(const cv::Mat& frame, cv::Mat& input) {
    cv::resize(
        frame,
        input,
        cv::Size(input_width_, input_height_)
    );

    return !input.empty();
}

bool PoseEstimator::run_inference(const cv::Mat& input) {
    if (input.empty()) {
        return false;
    }

    // Заглушка до подключения RKNN.
    return true;
}

bool PoseEstimator::postprocess(
    const cv::Size& original_size,
    std::vector<HumanPose>& poses
) {
    static_cast<void>(original_size);
    poses.clear();

    // Заглушка до получения выходов модели.
    return true;
}