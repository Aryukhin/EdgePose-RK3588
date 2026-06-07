#include "video_source.hpp"
#include "frame_processor.hpp"
#include "fps_meter.hpp"
#include "hdmi_sink.hpp"

#include <opencv2/opencv.hpp>

#include <iostream>
#include <memory>
#include <sstream>
#include <string>

int main(int argc, char** argv) {
    std::string source = "csi";
    std::string model_path = "../models/yolo11x-pose-fp16.rknn";

    if (argc > 1) {
        source = argv[1];
    }

    if (argc > 2) {
        model_path = argv[2];
    }

    std::cout << "Selected source: " << source << std::endl;
    std::cout << "Selected model: " << model_path << std::endl;

    VideoSource video_source(source);

    if (!video_source.open()) {
        return 1;
    }

    FrameProcessor processor(model_path);

    if (!processor.initialize()) {
        std::cerr << "Failed to initialize frame processor" << std::endl;
        return 1;
    }

    FpsMeter fps_meter;
    FramePacket packet;

    std::unique_ptr<HdmiSink> hdmi_sink;
    bool hdmi_enabled = true;

    while (true) {
        if (!video_source.read(packet)) {
            std::cout << "End of stream or failed to read frame" << std::endl;
            break;
        }

        if (!processor.process(packet)) {
            std::cerr << "Failed to process frame "
                      << packet.frame_id << std::endl;
            continue;
        }

        fps_meter.tick();

        std::ostringstream fps_text;
        fps_text << "FPS: " << static_cast<int>(fps_meter.fps());

        cv::putText(
            packet.frame,
            fps_text.str(),
            cv::Point(30, 80),
            cv::FONT_HERSHEY_SIMPLEX,
            1.0,
            cv::Scalar(255, 0, 0),
            2
        );

        if (packet.frame_id % 30 == 0) {
            std::cout
                << "Processed frame: " << packet.frame_id
                << ", size: " << packet.frame.cols
                << "x" << packet.frame.rows
                << ", fps: " << fps_meter.fps()
                << ", read_ms: " << packet.read_ms
                << ", convert_ms: " << packet.convert_ms
                << ", preprocess_ms: " << packet.preprocess_ms
                << ", inference_ms: " << packet.inference_ms
                << ", postprocess_ms: " << packet.postprocess_ms
                << ", process_ms: " << packet.process_ms
                << ", poses: " << packet.poses.size()
                << std::endl;
        }

        /*
         * Инициализируем HDMI-вывод после получения первого
         * успешно обработанного кадра, когда уже известен его размер.
         */
        if (!hdmi_sink && hdmi_enabled) {
            cv::imwrite("debug_frame.jpg", packet.frame);

            hdmi_sink = std::make_unique<HdmiSink>(
                packet.frame.cols,
                packet.frame.rows,
                1920,
                1080,
                30.0
            );

            if (!hdmi_sink->open()) {
                std::cerr << "HDMI output disabled" << std::endl;
                hdmi_sink.reset();
                hdmi_enabled = false;
            }
        }

        if (hdmi_enabled && hdmi_sink) {
            if (!hdmi_sink->write(packet.frame)) {
                std::cerr << "Failed to write frame to HDMI" << std::endl;
                hdmi_sink->release();
                hdmi_sink.reset();
                hdmi_enabled = false;
            }
        }

        // if (packet.frame_id >= 300) {
        //     std::cout << "Reached max frames limit" << std::endl;
        //     break;
        // }
    }

    if (hdmi_sink) {
        hdmi_sink->release();
    }

    return 0;
}