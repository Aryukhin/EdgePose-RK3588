#pragma once

#include <chrono>

class FpsMeter {
public:
    FpsMeter();

    void tick();
    double fps() const;
    double elapsed_ms() const;

private:
    std::chrono::steady_clock::time_point start_time_;
    std::chrono::steady_clock::time_point last_time_;
    std::uint64_t frame_count_;
    double fps_;
};