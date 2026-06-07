#include "pose_estimator.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <utility>

namespace {

float intersection_over_union(
    const BoundingBox& first,
    const BoundingBox& second
) {
    const float left = std::max(first.x1, second.x1);
    const float top = std::max(first.y1, second.y1);
    const float right = std::min(first.x2, second.x2);
    const float bottom = std::min(first.y2, second.y2);

    const float intersection_width = std::max(0.0F, right - left);
    const float intersection_height = std::max(0.0F, bottom - top);
    const float intersection = intersection_width * intersection_height;

    const float first_area =
        std::max(0.0F, first.x2 - first.x1) *
        std::max(0.0F, first.y2 - first.y1);

    const float second_area =
        std::max(0.0F, second.x2 - second.x1) *
        std::max(0.0F, second.y2 - second.y1);

    const float union_area = first_area + second_area - intersection;

    if (union_area <= 0.0F) {
        return 0.0F;
    }

    return intersection / union_area;
}

}  // namespace

PoseEstimator::PoseEstimator(std::string model_path)
    : model_path_(std::move(model_path)) {}

PoseEstimator::~PoseEstimator() {
    if (context_ != 0) {
        rknn_destroy(context_);
        context_ = 0;
    }
}

bool PoseEstimator::load_model() {
    std::ifstream model_file(model_path_, std::ios::binary);

    if (!model_file) {
        std::cerr << "Cannot open RKNN model: "
                  << model_path_ << std::endl;
        return false;
    }

    model_file.seekg(0, std::ios::end);
    const std::streamsize model_size = model_file.tellg();
    model_file.seekg(0, std::ios::beg);

    if (model_size <= 0) {
        std::cerr << "RKNN model is empty" << std::endl;
        return false;
    }

    model_data_.resize(static_cast<std::size_t>(model_size));

    if (!model_file.read(
            reinterpret_cast<char*>(model_data_.data()),
            model_size)) {
        std::cerr << "Failed to read RKNN model" << std::endl;
        return false;
    }

    const int ret = rknn_init(
        &context_,
        model_data_.data(),
        static_cast<uint32_t>(model_data_.size()),
        0,
        nullptr
    );

    if (ret != RKNN_SUCC) {
        std::cerr << "rknn_init failed: " << ret << std::endl;
        return false;
    }

    return true;
}

bool PoseEstimator::initialize() {
    if (!load_model()) {
        return false;
    }

    int ret = rknn_query(
        context_,
        RKNN_QUERY_IN_OUT_NUM,
        &io_num_,
        sizeof(io_num_)
    );

    if (ret != RKNN_SUCC) {
        std::cerr << "RKNN_QUERY_IN_OUT_NUM failed: "
                  << ret << std::endl;
        return false;
    }

    input_attrs_.resize(io_num_.n_input);
    output_attrs_.resize(io_num_.n_output);

    for (uint32_t i = 0; i < io_num_.n_input; ++i) {
        std::memset(&input_attrs_[i], 0, sizeof(rknn_tensor_attr));
        input_attrs_[i].index = i;

        ret = rknn_query(
            context_,
            RKNN_QUERY_INPUT_ATTR,
            &input_attrs_[i],
            sizeof(rknn_tensor_attr)
        );

        if (ret != RKNN_SUCC) {
            std::cerr << "Failed to query input " << i << std::endl;
            return false;
        }
    }

    for (uint32_t i = 0; i < io_num_.n_output; ++i) {
        std::memset(&output_attrs_[i], 0, sizeof(rknn_tensor_attr));
        output_attrs_[i].index = i;

        ret = rknn_query(
            context_,
            RKNN_QUERY_OUTPUT_ATTR,
            &output_attrs_[i],
            sizeof(rknn_tensor_attr)
        );

        if (ret != RKNN_SUCC) {
            std::cerr << "Failed to query output " << i << std::endl;
            return false;
        }
    }

    print_tensor_attributes();

    const rknn_tensor_attr& input = input_attrs_.front();

    // Для исходного ONNX ожидаем NCHW: [1, 3, 640, 640].
    if (input.n_dims == 4 && input.fmt == RKNN_TENSOR_NCHW) {
        input_height_ = input.dims[2];
        input_width_ = input.dims[3];
    } else if (input.n_dims == 4 && input.fmt == RKNN_TENSOR_NHWC) {
        input_height_ = input.dims[1];
        input_width_ = input.dims[2];
    }

    std::cout << "Model input size: "
              << input_width_ << "x" << input_height_
              << std::endl;

    initialized_ = true;
    return true;
}

void PoseEstimator::print_tensor_attributes() {
    std::cout << "RKNN inputs: " << io_num_.n_input
              << ", outputs: " << io_num_.n_output
              << std::endl;

    for (const rknn_tensor_attr& attr : input_attrs_) {
        std::cout << "Input " << attr.index
                  << ", dims:";

        for (uint32_t i = 0; i < attr.n_dims; ++i) {
            std::cout << " " << attr.dims[i];
        }

        std::cout << ", type=" << attr.type
                  << ", fmt=" << attr.fmt
                  << ", size=" << attr.size
                  << std::endl;
    }

    for (const rknn_tensor_attr& attr : output_attrs_) {
        std::cout << "Output " << attr.index
                  << ", dims:";

        for (uint32_t i = 0; i < attr.n_dims; ++i) {
            std::cout << " " << attr.dims[i];
        }

        std::cout << ", type=" << attr.type
                  << ", fmt=" << attr.fmt
                  << ", elements=" << attr.n_elems
                  << ", size=" << attr.size
                  << std::endl;
    }
}

cv::Mat PoseEstimator::letterbox(
    const cv::Mat& frame,
    float& scale,
    int& pad_x,
    int& pad_y
) const {
    scale = std::min(
        static_cast<float>(input_width_) /
            static_cast<float>(frame.cols),
        static_cast<float>(input_height_) /
            static_cast<float>(frame.rows)
    );

    const int resized_width =
        static_cast<int>(std::round(frame.cols * scale));

    const int resized_height =
        static_cast<int>(std::round(frame.rows * scale));

    pad_x = (input_width_ - resized_width) / 2;
    pad_y = (input_height_ - resized_height) / 2;

    cv::Mat resized;
    cv::resize(
        frame,
        resized,
        cv::Size(resized_width, resized_height)
    );

    cv::Mat padded(
        input_height_,
        input_width_,
        CV_8UC3,
        cv::Scalar(114, 114, 114)
    );

    resized.copyTo(
        padded(
            cv::Rect(
                pad_x,
                pad_y,
                resized_width,
                resized_height
            )
        )
    );

    return padded;
}

bool PoseEstimator::preprocess(
    const cv::Mat& frame,
    cv::Mat& input
) {
    if (frame.empty()) {
        return false;
    }

    cv::Mat padded = letterbox(
        frame,
        input_scale_,
        input_pad_x_,
        input_pad_y_
    );

    if (padded.empty()) {
        return false;
    }

    // OpenCV хранит изображение в BGR, YOLO обычно ожидает RGB.
    cv::cvtColor(padded, input, cv::COLOR_BGR2RGB);

    if (!input.isContinuous()) {
        input = input.clone();
    }

    return true;
}


bool PoseEstimator::run_inference(const cv::Mat& input) {
    if (input.empty() || !input.isContinuous()) {
        std::cerr << "Invalid RKNN input image" << std::endl;
        return false;
    }

    rknn_input rknn_input_data{};
    rknn_input_data.index = 0;
    rknn_input_data.buf = input.data;
    rknn_input_data.size =
        static_cast<uint32_t>(input.total() * input.elemSize());
    rknn_input_data.type = RKNN_TENSOR_UINT8;
    rknn_input_data.fmt = RKNN_TENSOR_NHWC;
    rknn_input_data.pass_through = 0;

    int ret = rknn_inputs_set(
        context_,
        1,
        &rknn_input_data
    );

    if (ret != RKNN_SUCC) {
        std::cerr << "rknn_inputs_set failed: "
                  << ret << std::endl;
        return false;
    }

    ret = rknn_run(context_, nullptr);

    if (ret != RKNN_SUCC) {
        std::cerr << "rknn_run failed: "
                  << ret << std::endl;
        return false;
    }

    outputs_.clear();
    outputs_.resize(io_num_.n_output);

    for (uint32_t i = 0; i < io_num_.n_output; ++i) {
        std::memset(
            &outputs_[i],
            0,
            sizeof(rknn_output)
        );

        outputs_[i].index = i;

        // Пока запрашиваем float для более простого postprocess.
        outputs_[i].want_float = 1;
        outputs_[i].is_prealloc = 0;
    }

    ret = rknn_outputs_get(
        context_,
        io_num_.n_output,
        outputs_.data(),
        nullptr
    );

    if (ret != RKNN_SUCC) {
        std::cerr << "rknn_outputs_get failed: "
                  << ret << std::endl;
        outputs_.clear();
        return false;
    }

    static bool printed_once = false;

    if (!printed_once) {
        for (uint32_t i = 0; i < io_num_.n_output; ++i) {
            std::cout << "Runtime output " << i
                      << ": size=" << outputs_[i].size
                      << " bytes"
                      << std::endl;
        }

        printed_once = true;
    }

    return true;
}


bool PoseEstimator::postprocess(
    const cv::Size& original_size,
    std::vector<HumanPose>& poses
) {
    poses.clear();

    if (outputs_.empty() || outputs_[0].buf == nullptr) {
        std::cerr << "RKNN outputs are empty" << std::endl;
        return false;
    }

    const float confidence_threshold = 0.35F;
    const float nms_threshold = 0.45F;

    constexpr int kValuesPerPrediction = 56;
    constexpr int kKeypointCount = 17;
    constexpr int kKeypointValues = 3;

    const auto& output_attr = output_attrs_[0];

    if (output_attr.n_elems % kValuesPerPrediction != 0) {
        std::cerr
            << "Unexpected YOLO pose output size: "
            << output_attr.n_elems
            << std::endl;

        rknn_outputs_release(
            context_,
            io_num_.n_output,
            outputs_.data()
        );
        outputs_.clear();

        return false;
    }

    const int prediction_count =
        static_cast<int>(output_attr.n_elems) /
        kValuesPerPrediction;

    const float* output =
        static_cast<const float*>(outputs_[0].buf);

    if (output == nullptr) {
        std::cerr << "Output buffer is null" << std::endl;
        return false;
    }

    /*
     * YOLO/Ultralytics часто выдаёт [1, 56, 8400]:
     *
     * output[channel * prediction_count + prediction_index]
     *
     * Но иногда после конвертации может получиться [1, 8400, 56].
     * Здесь определяем layout по dims.
     */
    bool channels_first = true;

    if (output_attr.n_dims >= 2) {
        const int last_dim =
            output_attr.dims[output_attr.n_dims - 1];

        if (last_dim == kValuesPerPrediction) {
            channels_first = false;
        }
    }

    auto get_value =
        [&](int prediction_index, int value_index) -> float {
            if (channels_first) {
                return output[
                    value_index * prediction_count +
                    prediction_index
                ];
            }

            return output[
                prediction_index * kValuesPerPrediction +
                value_index
            ];
        };

    std::vector<HumanPose> candidates;
    candidates.reserve(100);

    for (int prediction_index = 0;
         prediction_index < prediction_count;
         ++prediction_index) {

        const float confidence =
            get_value(prediction_index, 4);

        if (confidence < confidence_threshold) {
            continue;
        }

        const float center_x =
            get_value(prediction_index, 0);

        const float center_y =
            get_value(prediction_index, 1);

        const float width =
            get_value(prediction_index, 2);

        const float height =
            get_value(prediction_index, 3);

        /*
         * Координаты модели относятся к letterbox-изображению.
         * Возвращаем их в координаты исходного кадра.
         */
        float x1 =
            (center_x - width * 0.5F - input_pad_x_) /
            input_scale_;

        float y1 =
            (center_y - height * 0.5F - input_pad_y_) /
            input_scale_;

        float x2 =
            (center_x + width * 0.5F - input_pad_x_) /
            input_scale_;

        float y2 =
            (center_y + height * 0.5F - input_pad_y_) /
            input_scale_;

        x1 = std::clamp(
            x1,
            0.0F,
            static_cast<float>(original_size.width - 1)
        );

        y1 = std::clamp(
            y1,
            0.0F,
            static_cast<float>(original_size.height - 1)
        );

        x2 = std::clamp(
            x2,
            0.0F,
            static_cast<float>(original_size.width - 1)
        );

        y2 = std::clamp(
            y2,
            0.0F,
            static_cast<float>(original_size.height - 1)
        );

        if (x2 <= x1 || y2 <= y1) {
            continue;
        }

        HumanPose pose;

        pose.box.x1 = x1;
        pose.box.y1 = y1;
        pose.box.x2 = x2;
        pose.box.y2 = y2;
        pose.box.confidence = confidence;

        pose.keypoints.reserve(kKeypointCount);

        for (int keypoint_index = 0;
             keypoint_index < kKeypointCount;
             ++keypoint_index) {

            const int base =
                5 + keypoint_index * kKeypointValues;

            float keypoint_x =
                get_value(prediction_index, base);

            float keypoint_y =
                get_value(prediction_index, base + 1);

            const float keypoint_confidence =
                get_value(prediction_index, base + 2);

            keypoint_x =
                (keypoint_x - input_pad_x_) /
                input_scale_;

            keypoint_y =
                (keypoint_y - input_pad_y_) /
                input_scale_;

            keypoint_x = std::clamp(
                keypoint_x,
                0.0F,
                static_cast<float>(original_size.width - 1)
            );

            keypoint_y = std::clamp(
                keypoint_y,
                0.0F,
                static_cast<float>(original_size.height - 1)
            );

            pose.keypoints.push_back(
                Keypoint{
                    keypoint_x,
                    keypoint_y,
                    keypoint_confidence
                }
            );
        }

        candidates.push_back(std::move(pose));
    }

    /*
     * Сортировка перед NMS:
     * сначала наиболее уверенные детекции.
     */
    std::sort(
        candidates.begin(),
        candidates.end(),
        [](const HumanPose& first, const HumanPose& second) {
            return first.box.confidence >
                   second.box.confidence;
        }
    );

    std::vector<bool> suppressed(
        candidates.size(),
        false
    );

    for (std::size_t i = 0; i < candidates.size(); ++i) {
        if (suppressed[i]) {
            continue;
        }

        poses.push_back(candidates[i]);

        for (std::size_t j = i + 1;
             j < candidates.size();
             ++j) {

            if (suppressed[j]) {
                continue;
            }

            const float iou = intersection_over_union(
                candidates[i].box,
                candidates[j].box
            );

            if (iou > nms_threshold) {
                suppressed[j] = true;
            }
        }
    }

    rknn_outputs_release(
        context_,
        io_num_.n_output,
        outputs_.data()
    );

    outputs_.clear();

    return true;
}

bool PoseEstimator::infer(
    const cv::Mat& frame,
    std::vector<HumanPose>& poses,
    double& preprocess_ms,
    double& inference_ms,
    double& postprocess_ms
) {
    if (!initialized_) {
        std::cerr << "PoseEstimator is not initialized" << std::endl;
        return false;
    }

    if (frame.empty()) {
        std::cerr << "Empty input frame" << std::endl;
        return false;
    }

    const auto preprocess_start = std::chrono::steady_clock::now();

    cv::Mat input;
    if (!preprocess(frame, input)) {
        return false;
    }

    const auto preprocess_end = std::chrono::steady_clock::now();

    preprocess_ms =
        std::chrono::duration<double, std::milli>(
            preprocess_end - preprocess_start
        ).count();

    const auto inference_start = std::chrono::steady_clock::now();

    if (!run_inference(input)) {
        return false;
    }

    const auto inference_end = std::chrono::steady_clock::now();

    inference_ms =
        std::chrono::duration<double, std::milli>(
            inference_end - inference_start
        ).count();

    const auto postprocess_start = std::chrono::steady_clock::now();

    poses.clear();

    if (!postprocess(frame.size(), poses)) {
        return false;
    }

    const auto postprocess_end = std::chrono::steady_clock::now();

    postprocess_ms =
        std::chrono::duration<double, std::milli>(
            postprocess_end - postprocess_start
        ).count();

    return true;
}