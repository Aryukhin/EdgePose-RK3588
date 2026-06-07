# EdgePose-RK3588

Real-time human pose estimation pipeline for RK3588-based edge devices.

The application captures video from a CSI camera, runs YOLO11n-Pose inference on the RK3588 NPU, draws detected human skeletons, and outputs the processed video stream to an HDMI display.

## Current pipeline

```text
CSI camera
    ↓
V4L2 / GStreamer
    ↓
NV12 frame
    ↓
OpenCV NV12 → BGR conversion
    ↓
Letterbox preprocessing
    ↓
YOLO11n-Pose RKNN inference
    ↓
Pose postprocessing and NMS
    ↓
Skeleton rendering
    ↓
GStreamer appsrc
    ↓
kmssink
    ↓
HDMI display
```

## Features

* CSI camera capture through GStreamer and V4L2
* Support for Rockchip `rkisp_mainpath`
* NV12 to BGR conversion with OpenCV
* YOLO11n-Pose inference using RKNN Runtime
* RK3588 NPU execution
* Letterbox preprocessing
* Human bounding-box decoding
* COCO 17-keypoint pose decoding
* Non-Maximum Suppression
* Skeleton and keypoint rendering
* HDMI output through GStreamer `kmssink`
* Runtime profiling:

  * frame capture time
  * color conversion time
  * preprocessing time
  * inference time
  * postprocessing time
  * total processing time
  * average FPS

## Hardware

The project is currently tested on:

* Orange Pi 5 Plus
* Rockchip RK3588
* Ubuntu Server 22.04
* Linux kernel `5.10.160-rockchip-rk3588`
* CSI camera connected through Rockchip ISP
* HDMI display

Current camera node:

```text
/dev/video11
```

Current camera stream configuration:

```text
Format: NV12
Resolution: 1280x720
Requested framerate: 30 FPS
```

The actual camera framerate may depend on the sensor mode, ISP configuration, exposure settings, and RKAIQ availability.

## Software dependencies

* C++17
* CMake
* OpenCV
* GStreamer
* V4L2
* RKNN Runtime
* RKNN Toolkit2
* YOLO11n-Pose

Install the main system dependencies:

```bash
sudo apt update

sudo apt install -y \
    build-essential \
    cmake \
    pkg-config \
    libopencv-dev \
    v4l-utils \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly
```

## Project structure

```text
EdgePose-RK3588/
├── CMakeLists.txt
├── README.md
├── include/
│   ├── types.hpp
│   ├── video_source.hpp
│   ├── frame_processor.hpp
│   ├── fps_meter.hpp
│   ├── pose_estimator.hpp
│   ├── pose_renderer.hpp
│   └── hdmi_sink.hpp
├── src/
│   ├── main.cpp
│   ├── video_source.cpp
│   ├── frame_processor.cpp
│   ├── fps_meter.cpp
│   ├── pose_estimator.cpp
│   ├── pose_renderer.cpp
│   └── hdmi_sink.cpp
├── models/
│   ├── yolo11n-pose.onnx
│   └── yolo11n-pose-fp16.rknn
├── scripts/
│   └── convert_rknn.py
├── third_party/
│   └── rknn/
└── build/
```

## Model preparation

The project uses YOLO11n-Pose.

The original model is exported to ONNX and then converted to RKNN for RK3588.

Expected model path:

```text
models/yolo11n-pose-fp16.rknn
```

### Minimal RKNN conversion

```python
from pathlib import Path
from rknn.api import RKNN


PROJECT_DIR = Path(__file__).resolve().parent.parent

ONNX_MODEL = PROJECT_DIR / "models" / "yolo11n-pose.onnx"
RKNN_MODEL = PROJECT_DIR / "models" / "yolo11n-pose-fp16.rknn"


def main():
    rknn = RKNN(verbose=True)

    try:
        ret = rknn.config(
            target_platform="rk3588",
            mean_values=[[0, 0, 0]],
            std_values=[[255, 255, 255]],
        )
        if ret != 0:
            raise RuntimeError(f"rknn.config failed: {ret}")

        ret = rknn.load_onnx(model=str(ONNX_MODEL))
        if ret != 0:
            raise RuntimeError(f"rknn.load_onnx failed: {ret}")

        ret = rknn.build(do_quantization=False)
        if ret != 0:
            raise RuntimeError(f"rknn.build failed: {ret}")

        ret = rknn.export_rknn(str(RKNN_MODEL))
        if ret != 0:
            raise RuntimeError(f"rknn.export_rknn failed: {ret}")

        print(f"Saved: {RKNN_MODEL}")

    finally:
        rknn.release()


if __name__ == "__main__":
    main()
```

Run:

```bash
python3 scripts/convert_rknn.py
```

The model is currently built without INT8 quantization.

## RKNN Runtime

The project requires:

```text
rknn_api.h
librknnrt.so
```

The Linux ARM64 runtime from RKNN Toolkit2 is used:

```text
rknpu2/runtime/Linux/librknn_api/include/rknn_api.h
rknpu2/runtime/Linux/librknn_api/aarch64/librknnrt.so
```

The RKNN header and runtime library must be compatible with the RKNN Toolkit2 version used to build the model.

An incompatible runtime may produce an error such as:

```text
Invalid RKNN model version
```

## Building

Create a build directory:

```bash
mkdir -p build
cd build
```

Configure the project:

```bash
cmake .. -DCMAKE_BUILD_TYPE=Release
```

Build:

```bash
cmake --build . -j$(nproc)
```

For debugging:

```bash
cmake .. -DCMAKE_BUILD_TYPE=Debug
cmake --build . -j$(nproc)
```

## Running

Run with the CSI camera and the default model:

```bash
./edge_pose_cpp csi ../models/yolo11n-pose-fp16.rknn
```

Arguments:

```text
argv[1] — video source
argv[2] — path to RKNN model
```

Example with a video file:

```bash
./edge_pose_cpp ../data/video.mp4 ../models/yolo11n-pose-fp16.rknn
```

## CSI input pipeline

The current CSI pipeline is:

```text
v4l2src device=/dev/video11 io-mode=mmap
    !
video/x-raw,format=NV12,width=1280,height=720,framerate=30/1
    !
queue leaky=downstream max-size-buffers=1
    !
appsink drop=true max-buffers=1 sync=false
```

The RKISP camera may produce warnings such as:

```text
libv4l2: error getting pixformat: Invalid argument
```

If the pipeline successfully enters the playing state and frames are received, this warning may be non-fatal.

## HDMI output

Processed frames are sent to HDMI through:

```text
appsrc
    !
videoconvert
    !
videoscale
    !
video/x-raw,format=BGRx,width=1920,height=1080
    !
kmssink force-modesetting=true sync=false
```

Since Ubuntu Server displays a console on HDMI, the console may compete with `kmssink`.

To temporarily stop the first virtual console:

```bash
sudo systemctl stop getty@tty1.service
```

Run the application:

```bash
sudo -E ./edge_pose_cpp csi ../models/yolo11n-pose-fp16.rknn
```

Restore the console after testing:

```bash
sudo systemctl start getty@tty1.service
```

## Performance

Example measurements on the RK3588:

```text
Input frame: 1280x720
Preprocessing: approximately 2 ms
RKNN inference: approximately 60–110 ms
Postprocessing: depends on candidate count
Total FPS: approximately 7–9 FPS
```

The main bottleneck is currently RKNN inference and pose postprocessing.

Performance depends on:

* RKNN model graph
* RKNN Runtime version
* NPU core configuration
* model input resolution
* output tensor format
* confidence threshold
* NMS implementation
* camera framerate
* HDMI rendering pipeline

## Pose output

The model predicts:

* one `person` class
* bounding box
* confidence score
* 17 COCO keypoints

Each keypoint contains:

```text
x
y
confidence
```

The COCO pose keypoints are:

```text
0  nose
1  left eye
2  right eye
3  left ear
4  right ear
5  left shoulder
6  right shoulder
7  left elbow
8  right elbow
9  left wrist
10 right wrist
11 left hip
12 right hip
13 left knee
14 right knee
15 left ankle
16 right ankle
```

## Known issues

### Excessive pose detections

If hundreds of skeletons are detected in a single frame, check:

* input normalization
* RGB/BGR channel order
* output tensor layout
* confidence index
* confidence threshold
* whether sigmoid is required
* model output dimensions
* RKNN output type
* compatibility between ONNX export and postprocessing

The expected output layout for the current decoder is usually one of:

```text
[1, 56, 8400]
```

or:

```text
[1, 8400, 56]
```

The exact tensor dimensions must be verified through `rknn_query`.

### HDMI flickering

Possible causes:

* framebuffer console and `kmssink` using the display simultaneously
* missing `force-modesetting=true`
* wrong HDMI connector
* incorrect appsrc timestamps
* output FPS not matching the processing FPS

### Camera framerate below 30 FPS

Although the caps may report `30/1`, the actual camera stream may operate at 10–15 FPS.

This may depend on:

* camera sensor mode
* ISP configuration
* exposure time
* lighting conditions
* pixel format
* RKAIQ service
* upstream sensor resolution


## Future improvements

Potential application-level features:

* exercise repetition counting
* fall detection
* posture monitoring
* action recognition
* keypoint smoothing
* multi-person tracking
* unsafe-zone monitoring
* event logging

## License

The project source code can be distributed under a license selected by the repository owner.

YOLO, RKNN Toolkit2, OpenCV, and GStreamer are external dependencies and remain subject to their respective licenses.
