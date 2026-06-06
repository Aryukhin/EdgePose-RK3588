#include "video_source.hpp"

#include <iostream>
#include <chrono>
#include <cctype>

VideoSource::VideoSource(const std::string& source)
    : source_(source),
      frame_id_(0) {}

bool VideoSource::open() {
    if (source_ == "csi") {
        std::string pipeline =
            "v4l2src device=/dev/video11 io-mode=mmap ! "
            "video/x-raw,format=NV12,width=1280,height=720,framerate=30/1 ! "
            "queue leaky=downstream max-size-buffers=1 ! "
            "appsink drop=true max-buffers=1 sync=false";


        std::cout << "Opening CSI camera via GStreamer pipeline:\n"
                  << pipeline << std::endl;

        cap_.open(pipeline, cv::CAP_GSTREAMER);
    }
    else if (source_.size() == 1 &&
             std::isdigit(static_cast<unsigned char>(source_[0]))) {
        int camera_index = std::stoi(source_);
        cap_.open(camera_index);
        std::cout << "Opening camera index: " << camera_index << std::endl;
    }
    else {
        cap_.open(source_);
        std::cout << "Opening source: " << source_ << std::endl;
    }

    if (!cap_.isOpened()) {
        std::cerr << "Failed to open video source: " << source_ << std::endl;
        return false;
    }

    std::cout << "Video source opened successfully" << std::endl;
    std::cout << "Width: " << width()
              << ", Height: " << height()
              << ", FPS: " << fps()
              << std::endl;

    return true;
}

bool VideoSource::read(FramePacket& packet) {
    auto t0 = std::chrono::steady_clock::now();

    cv::Mat raw_frame;

    if (!cap_.read(raw_frame) || raw_frame.empty()) {
        return false;
    }

    auto t1 = std::chrono::steady_clock::now();

    cv::Mat output_frame;

    if (source_ == "csi") {
        if (frame_id_ == 0) {
            std::cout << "CSI raw frame: "
                      << raw_frame.cols << "x" << raw_frame.rows
                      << ", channels=" << raw_frame.channels()
                      << ", type=" << raw_frame.type()
                      << std::endl;
        }

        if (raw_frame.channels() == 1) {
            cv::cvtColor(raw_frame, output_frame, cv::COLOR_YUV2BGR_NV12);
        } else {
            output_frame = raw_frame;
        }
    } else {
        output_frame = raw_frame;
    }

    auto t2 = std::chrono::steady_clock::now();

    auto now = std::chrono::steady_clock::now();
    auto now_ms = std::chrono::duration<double, std::milli>(
        now.time_since_epoch()
    ).count();

    packet.frame = output_frame;
    packet.frame_id = frame_id_++;
    packet.timestamp_ms = now_ms;

    packet.read_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // Временно сюда пишем время конвертации.
    packet.convert_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();

    return true;
}

int VideoSource::width() const {
    return static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_WIDTH));
}

int VideoSource::height() const {
    return static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_HEIGHT));
}

double VideoSource::fps() const {
    return cap_.get(cv::CAP_PROP_FPS);
}