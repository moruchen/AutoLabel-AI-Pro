# scripts/demo_yolo.py
from src.detection import YOLODetector

detector = YOLODetector(model_name="yolov8n", device="cpu")
result = detector.detect(                    # ← 单数 result
    source=r"D:\pythonproject\day1\test1.jpg",
    save=True,
    conf=0.25
)
print(f"✅ 检测到 {len(result.boxes)} 个目标，耗时 {result.latency_ms}ms")
for box in result.boxes:
    print(f"  - {box.class_name} (置信度 {box.confidence:.2f})")