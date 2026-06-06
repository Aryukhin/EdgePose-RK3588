#include "pose_renderer.hpp"

#include <array>
#include <utility>

namespace {

constexpr std::array<std::pair<int, int>, 12> kSkeleton = {{
    {5, 7},
    {7, 9},
    {6, 8},
    {8, 10},
    {5, 6},
    {5, 11},
    {6, 12},
    {11, 12},
    {11, 13},
    {13, 15},
    {12, 14},
    {14, 16}
}};

}  // namespace

PoseRenderer::PoseRenderer(float keypoint_threshold)
    : keypoint_threshold_(keypoint_threshold) {}

void PoseRenderer::draw(
    cv::Mat& frame,
    const std::vector<HumanPose>& poses
) const {
    for (const HumanPose& pose : poses) {
        const cv::Rect box(
            cv::Point(
                static_cast<int>(pose.box.x1),
                static_cast<int>(pose.box.y1)
            ),
            cv::Point(
                static_cast<int>(pose.box.x2),
                static_cast<int>(pose.box.y2)
            )
        );

        cv::rectangle(frame, box, cv::Scalar(0, 255, 0), 2);

        for (const Keypoint& keypoint : pose.keypoints) {
            if (keypoint.confidence < keypoint_threshold_) {
                continue;
            }

            cv::circle(
                frame,
                cv::Point(
                    static_cast<int>(keypoint.x),
                    static_cast<int>(keypoint.y)
                ),
                3,
                cv::Scalar(0, 0, 255),
                -1
            );
        }

        for (const auto& [first_index, second_index] : kSkeleton) {
            if (first_index >= static_cast<int>(pose.keypoints.size()) ||
                second_index >= static_cast<int>(pose.keypoints.size())) {
                continue;
            }

            const Keypoint& first = pose.keypoints[first_index];
            const Keypoint& second = pose.keypoints[second_index];

            if (first.confidence < keypoint_threshold_ ||
                second.confidence < keypoint_threshold_) {
                continue;
            }

            cv::line(
                frame,
                cv::Point(
                    static_cast<int>(first.x),
                    static_cast<int>(first.y)
                ),
                cv::Point(
                    static_cast<int>(second.x),
                    static_cast<int>(second.y)
                ),
                cv::Scalar(255, 0, 0),
                2
            );
        }
    }
}