#include "video_source.hpp"
#include "frame_processor.hpp"
#include "fps_meter.hpp"

#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <sstream>

int main(int argc, char** argv) {
    std::string source = "0";

    if (argc > 1) {
        source = argv[1];
    }

    std::cout << "Selected source: " << source << std::endl;

    VideoSource video_source(source);

    if (!video_source.open()) {
        return 1;
    }

    FrameProcessor processor;
    FpsMeter fps_meter;

    FramePacket packet;

    while (true) {
        bool ok = video_source.read(packet);

        if (!ok) {
            std::cout << "End of stream or failed to read frame" << std::endl;
            break;
        }

        if (!processor.process(packet)) {
            std::cerr << "Failed to process frame" << std::endl;
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

        // cv::imshow("edge_pose_cpp", packet.frame);

        // int key = cv::waitKey(1);
        // if (key == 'q' || key == 27) {
        //     std::cout << "Exit requested" << std::endl;
        //     break;
        // }
        if (packet.frame_id % 30 == 0) {
            std::cout << "Processed frame: " << packet.frame_id
                    << ", size: " << packet.frame.cols
                    << "x" << packet.frame.rows
                    << ", fps: " << fps_meter.fps()
                    << ", read_ms: " << packet.read_ms
                    << ", convert_ms: " << packet.convert_ms
                    << ", process_ms: " << packet.process_ms
                    << std::endl;
        }
        if (packet.frame_id == 0) {
            cv::imwrite("debug_frame.jpg", packet.frame);
        }

            // cv::imwrite("debug_frame.jpg", packet.frame);
        //}

        if (packet.frame_id >= 300) {
            std::cout << "Reached max frames limit" << std::endl;
            break;
        }
    }

    cv::destroyAllWindows();

    return 0;
}