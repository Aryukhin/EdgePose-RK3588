#pragma once

#include "types.hpp"

class FrameProcessor {
public:
    FrameProcessor();

    bool process(FramePacket& packet);

private:
    void draw_debug_info(FramePacket& packet);
};