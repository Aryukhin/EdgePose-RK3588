#include "fps_meter.hpp"

FpsMeter::FpsMeter()
    : start_time_(std::chrono::steady_clock::now()),
      last_time_(start_time_),
      frame_count_(0),
      fps_(0.0) {}

void FpsMeter::tick() {
    frame_count_++;

    auto now = std::chrono::steady_clock::now();
    double elapsed_sec = std::chrono::duration<double>(now - start_time_).count();

    if (elapsed_sec > 0.0) {
        fps_ = static_cast<double>(frame_count_) / elapsed_sec;
    }

    last_time_ = now;
}

double FpsMeter::fps() const {
    return fps_;
}

double FpsMeter::elapsed_ms() const {
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(now - start_time_).count();
}