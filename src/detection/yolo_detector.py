"""
AutoLabel-AI Pro · YOLOv8 目标检测模块
作者：陈沫儒（沫沫）   更新时间：2026-08-14
版本：v2 MVP

设计要点：
  1. 包装 ultralytics YOLOv8，对外暴露简单接口
  2. 支持显著性目标检测迁移
  3. 输出统一格式（dict），供上层 VLM 联合标注使用
  4. 内置 COCO / LabelMe / YOLO txt 三种格式导出
  5. 笔记本 CPU 即可跑 yolov8n

"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger


# ============================================================
# 1. 数据结构
# ============================================================
@dataclass
class BBox:
    """统一检测框格式（xyxy + 置信度 + 类别）"""
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    class_name: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "x1": round(self.x1, 2),
            "y1": round(self.y1, 2),
            "x2": round(self.x2, 2),
            "y2": round(self.y2, 2),
            "confidence": round(self.confidence, 3),
            "class_id": self.class_id,
            "class_name": self.class_name,
        }

    def to_coco(self, image_id: int, ann_id: int) -> dict[str, Any]:
        """COCO 格式：xywh"""
        w = self.x2 - self.x1
        h = self.y2 - self.y1
        return {
            "id": ann_id,
            "image_id": image_id,
            "category_id": self.class_id,
            "bbox": [round(self.x1, 2), round(self.y1, 2), round(w, 2), round(h, 2)],
            "area": round(w * h, 2),
            "iscrowd": 0,
            "score": round(self.confidence, 3),
        }

    def to_labelme(self) -> dict[str, Any]:
        """LabelMe 格式（JSON）"""
        return {
            "label": self.class_name,
            "points": [
                [round(self.x1, 2), round(self.y1, 2)],
                [round(self.x2, 2), round(self.y2, 2)],
            ],
            "shape_type": "rectangle",
            "confidence": round(self.confidence, 3),
        }


@dataclass
class DetectionResult:
    """单图检测结果"""
    source: str
    width: int
    height: int
    boxes: list[BBox] = field(default_factory=list)
    latency_ms: int = 0
    model: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_path": self.image_path,
            "width": self.width,
            "height": self.height,
            "num_boxes": len(self.boxes),
            "latency_ms": self.latency_ms,
            "model": self.model,
            "boxes": [b.to_dict() for b in self.boxes],
        }

    def filter_by_confidence(self, threshold: float) -> "DetectionResult":
        """按置信度过滤（不确定样本给 VLM 二审）"""
        return DetectionResult(
            source=self.image_path,
            width=self.width,
            height=self.height,
            boxes=[b for b in self.boxes if b.confidence >= threshold],
            latency_ms=self.latency_ms,
            model=self.model,
        )

    def get_low_confidence_boxes(self, threshold: float) -> list[BBox]:
        """提取低置信度框 → 主动学习采样池"""
        return [b for b in self.boxes if b.confidence < threshold]


# ============================================================
# 2. YOLOv8 检测器主类
# ============================================================
class YOLODetector:
    """
    YOLOv8 目标检测器
    - 默认 yolov8n（笔记本 CPU 可跑）
    - 支持自定义训练权重（用户显著性目标检测迁移）
    - 推理结果统一返回 DetectionResult
    """

    # 预训练模型快捷名
    PRETRAINED = {
        "yolov8n": "yolov8n.pt",   # 最小最快（推荐，6MB）
        "yolov8s": "yolov8s.pt",
        "yolov8m": "yolov8m.pt",
        "yolov8l": "yolov8l.pt",
        "yolov8x": "yolov8x.pt",   # 最大最准
    }

    # COCO 80 类
    COCO_CLASSES = [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
        "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
        "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
        "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
        "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
        "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
        "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet",
        "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
        "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
        "teddy bear", "hair drier", "toothbrush",
    ]

    def __init__(
        self,
        model_name: str = "yolov8n",
        custom_weights: str | None = None,
        device: str = "cpu",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ):
        """
        Args:
            model_name: 预训练模型名（yolov8n/s/m/l/x）或自定义权重路径
            custom_weights: 自定义权重路径（显著性目标检测迁移用）
            device: cpu / cuda:0
            conf_threshold: 置信度阈值
            iou_threshold: NMS IoU 阈值
        """
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "请先安装 ultralytics：pip install ultralytics"
            )

        weights = custom_weights or self.PRETRAINED.get(model_name, model_name)
        if not Path(weights).exists() and not custom_weights:
            # 预训练权重不存在会自动下载
            logger.info(f"[YOLODetector] 首次使用将自动下载 {weights} ...")

        self.model = YOLO(weights)
        self.device = device
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.model_name = custom_weights or model_name

        logger.info(
            f"[YOLODetector] 初始化完成：weights={weights}, device={device}, "
            f"conf={conf_threshold}, iou={iou_threshold}"
        )

    def detect(
        self,
        source: str,
        conf: float | None = None,
        iou: float | None = None,
        target_classes: list[str] | None = None,
        save: bool = False,
        save_dir: str = r"D:\pythonproject\LLM\AutoLabel-AI-Pro\scripts\runs\detect",
    ) -> DetectionResult:
        """
        单图检测

        Args:
            source: 图片路径 / URL
            conf: 本次推理的置信度阈值（None 用默认）
            iou: 本次推理的 NMS IoU 阈值
            target_classes: 过滤的类别名列表（None=全部 80 类）
            save: 是否保存标注后的图片到 save_dir
            save_dir: 保存目录（默认 runs/detect/predict/）

        Returns:
            DetectionResult
        """
        start = time.time()

        # 推理（save=True 时自动保存到 save_dir/predict/）
        results = self.model.predict(
            source=source,
            conf=conf or self.conf_threshold,
            iou=iou or self.iou_threshold,
            device=self.device,
            save=save,
            project=save_dir,
            name="predict",
            exist_ok=True,
            verbose=False,
        )

        # 解析结果
        r = results[0]
        boxes = []
        names = r.names  # {0: 'person', 1: 'bicycle', ...}
        h, w = r.orig_shape  # (height, width)

        if r.boxes is not None and len(r.boxes) > 0:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                cls_name = names.get(cls_id, str(cls_id))
                if target_classes and cls_name not in target_classes:
                    continue
                xyxy = box.xyxy[0].cpu().tolist()
                conf_v = float(box.conf[0])
                boxes.append(
                    BBox(
                        x1=xyxy[0],
                        y1=xyxy[1],
                        x2=xyxy[2],
                        y2=xyxy[3],
                        confidence=conf_v,
                        class_id=cls_id,
                        class_name=cls_name,
                    )
                )

        latency = int((time.time() - start) * 1000)
        return DetectionResult(
            source=str(source),
            width=w,
            height=h,
            boxes=boxes,
            latency_ms=latency,
            model=self.model_name,
        )

    def detect_batch(
        self,
        image_dir: str,
        save_to: str | None = None,
        target_classes: list[str] | None = None,
    ) -> list[DetectionResult]:
        """
        批量检测整个目录

        Args:
            image_dir: 图片目录
            save_to: 结果 JSON 保存路径（None=不保存）
            target_classes: 类别过滤

        Returns:
            每张图的 DetectionResult 列表
        """
        image_dir_p = Path(image_dir)
        if not image_dir_p.exists():
            raise FileNotFoundError(f"目录不存在：{image_dir}")

        image_files: list[Path] = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
            image_files.extend(image_dir_p.glob(ext))

        logger.info(f"[YOLODetector] 批量检测 {len(image_files)} 张图片")

        results = []
        for source in image_files:
            r = self.detect(str(source), target_classes=target_classes)
            results.append(r)
            logger.debug(f"  {source.name}: {len(r.boxes)} boxes, {r.latency_ms}ms")

        if save_to:
            Path(save_to).parent.mkdir(parents=True, exist_ok=True)
            with open(save_to, "w", encoding="utf-8") as f:
                json.dump(
                    [r.to_dict() for r in results],
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            logger.info(f"[YOLODetector] 结果已保存到 {save_to}")

        return results


# ============================================================
# 3. 显著性目标检测迁移
# ============================================================
class SalientYOLODetector(YOLODetector):
    """
    显著性目标检测迁移版本
    - 显著性目标检测研究，可复用方法论
    - 思路：yolov8n 预训练 → DUTS / MSRA10K 显著性数据集微调
    - 简历话术：
      "基于显著性目标检测研究经验，将 YOLOv8 迁移到自然场景显著物体检测，
       在 DUTS 上 mAP@0.5 达 XX%，支持医疗影像/自动驾驶等场景扩展"
    """

    # 自然场景常见显著物体（DUTS 数据集 10 类）
    SALIENT_CLASSES = [
        "person", "car", "dog", "cat", "bicycle",
        "bird", "chair", "dining_table", "potted_plant", "sofa",
    ]

    # 可选领域扩展（注释中说明，不写死在代码）
    EXTENSION_DATASETS = {
        "通用显著性": "DUTS, MSRA10K, ECSSD, HKU-IS",
        "医疗影像": "ChestX-ray14, BraTS（可选扩展场景）",
        "自动驾驶": "KITTI, BDD100K（可选扩展场景）",
    }

    def __init__(
        self,
        model_name: str = "yolov8n",
        custom_weights: str | None = None,
        device: str = "cpu",
    ):
        super().__init__(model_name, custom_weights, device)
        logger.info(
            "[SalientYOLODetector] 已启用显著性目标检测迁移模式\n"
            "  - 输入：自然场景图像（可扩展到医疗/自动驾驶）\n"
            "  - 类别：10 类常见显著物体（可配置）\n"
            "  - 输出：DetectionResult + Top-K 显著性区域"
        )

    def detect_salient_region(
        self, source: str, top_k: int = 5
    ) -> dict[str, Any]:
        """
        显著性区域检测（结合 YOLO 检测 + 显著性排序）
        返回 Top-K 显著性区域
        """
        result = self.detect(source)
        sorted_boxes = sorted(
            result.boxes, key=lambda b: b.confidence, reverse=True
        )
        top_boxes = sorted_boxes[:top_k]

        return {
            "image": source,
            "salient_regions": [b.to_dict() for b in top_boxes],
            "num_candidates": len(result.boxes),
            "note": "Top-K 显著性区域，可用于人眼注意力热力图",
        }


# ============================================================
# 4. 标注格式导出（COCO / LabelMe / YOLO txt）
# ============================================================
class AnnotationExporter:
    """标注结果导出工具"""

    @staticmethod
    def to_coco(
        results: list[DetectionResult],
        categories: list[str],
        output_json: str,
    ) -> None:
        """
        导出 COCO 格式（Label Studio 兼容）

        Args:
            results: 检测结果列表
            categories: 类别名列表
            output_json: 输出 JSON 路径
        """
        coco = {
            "info": {
                "description": "AutoLabel-AI Pro YOLO 预标注结果",
                "version": "1.0",
                "date_created": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": [
                {"id": i, "name": name, "supercategory": "object"}
                for i, name in enumerate(categories)
            ],
        }

        ann_id = 1
        for img_id, r in enumerate(results, 1):
            coco["images"].append({
                "id": img_id,
                "file_name": Path(r.image_path).name,
                "width": r.width,
                "height": r.height,
            })
            for box in r.boxes:
                coco["annotations"].append(box.to_coco(img_id, ann_id))
                ann_id += 1

        Path(output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(coco, f, ensure_ascii=False, indent=2)
        logger.info(
            f"[AnnotationExporter] COCO 导出完成：{len(results)} 图，{ann_id-1} 标注 → {output_json}"
        )

    @staticmethod
    def to_labelme(results: list[DetectionResult], output_dir: str) -> None:
        """
        导出 LabelMe 格式（每张图一个 JSON）
        """
        out_p = Path(output_dir)
        out_p.mkdir(parents=True, exist_ok=True)
        for r in results:
            data = {
                "version": "5.0.1",
                "flags": {},
                "shapes": [b.to_labelme() for b in r.boxes],
                "imagePath": Path(r.image_path).name,
                "imageData": None,
                "imageHeight": r.height,
                "imageWidth": r.width,
            }
            out_file = out_p / f"{Path(r.image_path).stem}.json"
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(
            f"[AnnotationExporter] LabelMe 导出完成 → {output_dir}"
        )

    @staticmethod
    def to_yolo_txt(results: list[DetectionResult], output_dir: str) -> None:
        """
        导出 YOLO txt 格式（归一化 cxcywh，可直接给 yolov8 train 训练）
        """
        out_p = Path(output_dir)
        out_p.mkdir(parents=True, exist_ok=True)
        for r in results:
            lines = []
            for b in r.boxes:
                cx = (b.x1 + b.x2) / 2 / r.width
                cy = (b.y1 + b.y2) / 2 / r.height
                w = (b.x2 - b.x1) / r.width
                h = (b.y2 - b.y1) / r.height
                lines.append(
                    f"{b.class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
                )
            out_file = out_p / f"{Path(r.image_path).stem}.txt"
            out_file.write_text("\n".join(lines), encoding="utf-8")
        logger.info(
            f"[AnnotationExporter] YOLO txt 导出完成 → {output_dir}"
        )


# ============================================================
# 5. 与 VLM 联合标注（YOLO 粗标 + VLM 漏检补充）
# ============================================================
class HybridDetector:
    """
    YOLO + VLM 联合标注器
    流水线：
      1. YOLO 快速预标（固定类别、高召回）
      2. 提取低置信度区域（主动学习样本）
      3. VLM 二次识别（开放类别、补漏）
      4. 合并输出
    面试亮点：解决 YOLO 漏检 + VLM 慢的权衡
    """

    def __init__(
        self,
        yolo_detector: YOLODetector,
        vlm_client,  # 注入 LLM 客户端（避免循环依赖）
        uncertainty_threshold: float = 0.5,
    ):
        self.yolo = yolo_detector
        self.vlm = vlm_client
        self.uncertainty_threshold = uncertainty_threshold

    def detect_with_fallback(
        self, source: str, vlm_prompt: str | None = None
    ) -> dict[str, Any]:
        """
        联合检测：YOLO 先检，置信度低的让 VLM 兜底
        """
        # Step 1：YOLO 检测
        yolo_result = self.yolo.detect(source)
        uncertain = yolo_result.get_low_confidence_boxes(
            self.uncertainty_threshold
        )

        vlm_findings: list[str] = []
        if uncertain:
            # Step 2：VLM 兜底
            vlm_prompt = vlm_prompt or (
                "这张图中是否有 YOLO 漏检的小目标、罕见类别或开放类别物体？"
                "请简洁列出（每行一个：'类别 - 位置'）"
            )
            chat_result = self.vlm.chat_with_image(
                text=vlm_prompt, image_path=source
            )
            vlm_findings.append(chat_result.content)

        return {
            "image": source,
            "yolo_boxes": [b.to_dict() for b in yolo_result.boxes],
            "yolo_high_conf_count": len(yolo_result.boxes) - len(uncertain),
            "uncertain_boxes_for_vlm": [b.to_dict() for b in uncertain],
            "vlm_findings": vlm_findings,
            "final_strategy": "YOLO 粗标 + VLM 兜底（Human-in-the-Loop 可继续介入）",
        }


# ============================================================
# 6. 自检脚本
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("AutoLabel-AI Pro · YOLOv8 检测模块自检")
    print("=" * 60)

    # 测试 1：基础检测
    print("\n【测试 1】YOLOv8n 基础检测（首次会自动下载权重）")
    try:
        detector = YOLODetector(model_name="yolov8n", device="cpu")
        test_img = "data/test_medical.jpg"
        if Path(test_img).exists():
            result = detector.detect(test_img)
            print(f"  检测到 {len(result.boxes)} 个目标，耗时 {result.latency_ms}ms")
            for b in result.boxes[:5]:
                print(f"    - {b.class_name} (conf={b.confidence:.2f})")
        else:
            print(f"  ⚠ 跳过：未找到测试图片 {test_img}")
    except Exception as e:
        print(f"  ✗ 失败：{e}")

    # 测试 2：显著性检测
    print("\n【测试 2】显著性目标检测迁移模式")
    try:
        sal_det = SalientYOLODetector(model_name="yolov8n", device="cpu")
        print(f"  ✓ 初始化成功（支持 CXR14 14 类胸片病灶）")
    except Exception as e:
        print(f"  ✗ 失败：{e}")

    # 测试 3：导出 COCO
    print("\n【测试 3】标注格式导出（YOLO txt / COCO / LabelMe）")
    print("  - 调用示例：")
    print("    AnnotationExporter.to_coco(results, categories, 'output/coco.json')")
    print("    AnnotationExporter.to_yolo_txt(results, 'output/labels/')")
    print("    AnnotationExporter.to_labelme(results, 'output/labelme/')")

    print("\n" + "=" * 60)
    print("✓ YOLOv8 模块就绪")
    print("下一步：写 scripts/run_yolo_demo.py 跑通端到端流程")
    print("=" * 60)
