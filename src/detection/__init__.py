"""Detection 模块：YOLOv8 目标检测"""
from .yolo_detector import (
    AnnotationExporter,
    BBox,
    DetectionResult,
    HybridDetector,
    SalientYOLODetector,
    YOLODetector,
)

__all__ = [
    "AnnotationExporter",
    "BBox",
    "DetectionResult",
    "HybridDetector",
    "SalientYOLODetector",
    "YOLODetector",
]
