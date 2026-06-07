#pragma once

#include "types.hpp"

#include <opencv2/opencv.hpp>
#include <rknn_api.h>

#include <string>
#include <vector>

class PoseEstimator {
public:
    explicit PoseEstimator(std::string model_path);
    ~PoseEstimator();

    PoseEstimator(const PoseEstimator&) = delete;
    PoseEstimator& operator=(const PoseEstimator&) = delete;

    bool initialize();

    bool infer(
        const cv::Mat& frame,
        std::vector<HumanPose>& poses,
        double& preprocess_ms,
        double& inference_ms,
        double& postprocess_ms
    );

private:
    bool load_model();

    cv::Mat letterbox(
        const cv::Mat& frame,
        float& scale,
        int& pad_x,
        int& pad_y
    ) const;

    bool preprocess(const cv::Mat& frame, cv::Mat& input);
    bool run_inference(const cv::Mat& input);

    bool postprocess(
        const cv::Size& original_size,
        std::vector<HumanPose>& poses
    );

    void print_tensor_attributes();

private:
    std::string model_path_;
    std::vector<unsigned char> model_data_;

    rknn_context context_ = 0;
    rknn_input_output_num io_num_{};

    std::vector<rknn_tensor_attr> input_attrs_;
    std::vector<rknn_tensor_attr> output_attrs_;
    std::vector<rknn_output> outputs_;

    int input_width_ = 640;
    int input_height_ = 640;

    float input_scale_ = 1.0F;
    int input_pad_x_ = 0;
    int input_pad_y_ = 0;

    bool initialized_ = false;
};
