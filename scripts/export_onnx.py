from ultralytics import YOLO

model = YOLO("yolo11m-pose.pt")
model.export(
    format="onnx",
    imgsz=640,
    opset=12,
    simplify=True,
    dynamic=False
)