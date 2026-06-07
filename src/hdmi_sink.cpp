#include "hdmi_sink.hpp"

#include <iostream>
#include <sstream>

HdmiSink::HdmiSink(
    int input_width,
    int input_height,
    int output_width,
    int output_height,
    double fps
)
    : input_width_(input_width),
      input_height_(input_height),
      output_width_(output_width),
      output_height_(output_height),
      fps_(fps) {}

bool HdmiSink::open() {
    std::ostringstream pipeline;

    pipeline
        << "appsrc is-live=true format=time do-timestamp=true ! "
        << "video/x-raw,format=BGR,width=" << input_width_
        << ",height=" << input_height_
        << ",framerate=10/1 ! "
        << "queue leaky=downstream max-size-buffers=1 ! "
        << "videoconvert ! "
        << "videoscale ! "
        << "video/x-raw,format=BGRx,width=" << output_width_
        << ",height=" << output_height_ << " ! "
        << "kmssink force-modesetting=true sync=false";

    std::cout << "Opening HDMI pipeline:\n"
              << pipeline.str() << std::endl;

    writer_.open(
        pipeline.str(),
        cv::CAP_GSTREAMER,
        0,
        10.0,
        cv::Size(input_width_, input_height_),
        true
    );

    if (!writer_.isOpened()) {
        std::cerr << "Failed to open HDMI sink" << std::endl;
        return false;
    }

    return true;
}

bool HdmiSink::write(const cv::Mat& frame) {
    if (!writer_.isOpened()) {
        return false;
    }

    if (frame.empty()) {
        return false;
    }

    writer_.write(frame);
    return true;
}

void HdmiSink::release() {
    if (writer_.isOpened()) {
        writer_.release();
    }
}