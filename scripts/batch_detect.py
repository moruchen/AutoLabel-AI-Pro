"""
批量检测 COCO val 子集
- 输入：data/raw/coco_val/sample_list.json（100 条）
- 抽样 50 条（固定种子 42，可复现）
- 用 YOLODetector 检测每张图
- 出图保存到 docs/images/coco_val_samples/predict/
- JSON 标注保存到 data/annotations/coco_val_pred.json
- token 消耗：0
"""
import json
import random
import time
from pathlib import Path

from src.detection import YOLODetector

# 路径
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
SAMPLE_LIST = PROJECT_ROOT / "data" / "raw" / "coco_val" / "sample_list.json"
IMG_DIR = PROJECT_ROOT / "data" / "raw" / "coco_val" / "samples"
OUTPUT_JSON = PROJECT_ROOT / "data" / "annotations" / "coco_val_pred.json"
OUTPUT_IMG_DIR = PROJECT_ROOT / "docs" / "images" / "coco_val_samples"


def main():
    # 1. 读 sample_list
    with open(SAMPLE_LIST, "r", encoding="utf-8") as f:
        samples = json.load(f)
    print(f"[batch_detect] 总共 {len(samples)} 张候选图")

    # 2. 抽 50 张
    random.seed(42)
    selected = random.sample(samples, 50)
    print(f"[batch_detect] 抽样 50 张（固定 seed=42，可复现）")

    # 3. 初始化检测器
    detector = YOLODetector(model_name="yolov8n", device="cpu")

    # 4. 创建输出目录
    OUTPUT_IMG_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    # 5. 批量检测
    all_results = []
    total_start = time.time()

    for i, sample in enumerate(selected, 1):
        img_path = IMG_DIR / sample["file_name"]
        if not img_path.exists():
            print(f"  [{i}/50] ⚠️  跳过（文件不存在）：{img_path.name}")
            continue

        result = detector.detect(
            source=str(img_path),
            save=True,
            save_dir=str(OUTPUT_IMG_DIR),
        )

        # 累积结果
        all_results.append({
            "image_id": sample["id"],
            "file_name": sample["file_name"],
            "width": sample["width"],
            "height": sample["height"],
            "num_boxes": len(result.boxes),
            "latency_ms": result.latency_ms,
            "boxes": [b.to_dict() for b in result.boxes],
        })

        print(
            f"  [{i}/50] {sample['file_name']}: "
            f"{len(result.boxes)} boxes, {result.latency_ms}ms"
        )

    # 6. 保存 JSON
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    total_sec = time.time() - total_start
    print(f"\n[batch_detect] ✅ 完成")
    print(f"  检测图片：{len(all_results)} 张")
    print(f"  总耗时：{total_sec:.1f} 秒（平均 {total_sec / len(all_results) * 1000:.0f} ms/张）")
    print(f"  标注图：{OUTPUT_IMG_DIR / 'predict'}")
    print(f"  标注 JSON：{OUTPUT_JSON}")


if __name__ == "__main__":
    main()