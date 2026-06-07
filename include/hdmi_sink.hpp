#pragma once

#include <opencv2/opencv.hpp>

class HdmiSink {
public:
    HdmiSink(
        int input_width,
        int input_height,
        int output_width = 1920,
        int output_height = 1080,
        double fps = 30.0
    );

    bool open();
    bool write(const cv::Mat& frame);
    void release();

private:
    int input_width_;
    int input_height_;
    int output_width_;
    int output_height_;
    double fps_;

    cv::VideoWriter writer_;
};