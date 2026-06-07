from pathlib import Path
from rknn.api import RKNN


ONNX_MODEL = Path("./models/yolo11m-pose.onnx")
RKNN_MODEL = Path("./models/yolo11m-pose-fp16.rknn")
TARGET_PLATFORM = "rk3588"


def main() -> None:
    if not ONNX_MODEL.exists():
        raise FileNotFoundError(f"ONNX model not found: {ONNX_MODEL}")

    RKNN_MODEL.parent.mkdir(parents=True, exist_ok=True)

    rknn = RKNN(verbose=False)

    try:
        print("--> Config RKNN")

        ret = rknn.config(
            target_platform="rk3588",
            mean_values=[[0, 0, 0]],
            std_values=[[255, 255, 255]],
        )
        if ret != 0:
            raise RuntimeError(f"rknn.config failed, code: {ret}")

        print("--> Load ONNX model")

        ret = rknn.load_onnx(
            model=str(ONNX_MODEL),
        )
        if ret != 0:
            raise RuntimeError(f"rknn.load_onnx failed, code: {ret}")

        print("--> Build RKNN model without quantization")

        ret = rknn.build(
            do_quantization=False,
        )
        if ret != 0:
            raise RuntimeError(f"rknn.build failed, code: {ret}")

        print("--> Export RKNN model")

        ret = rknn.export_rknn(
            str(RKNN_MODEL),
        )
        if ret != 0:
            raise RuntimeError(f"rknn.export_rknn failed, code: {ret}")

        print(f"Done: {RKNN_MODEL}")

    finally:
        rknn.release()


if __name__ == "__main__":
    main()