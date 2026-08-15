"""
通用数据集抽样下载脚本
目标：硬件友好 + 0 token 优先，总样本控制在 100-500 条

使用：
    python scripts/download_datasets.py cifar10
    python scripts/download_datasets.py coco_val
"""
from pathlib import Path
from torchvision.datasets import CIFAR10
from datasets import load_dataset
import pickle
import numpy as np
from PIL import Image
import urllib.request
import zipfile
import json
import random
import sys

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
DATA_DIR = PROJECT_ROOT / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# 数据集配置
DATASET_CONFIG = {
    "cifar10": {
        "type": "image_classification",
        "samples": 100,
        "description": "CIFAR-10 子集，10 类，32x32 RGB",
    },
    "coco_val": {
        "type": "object_detection",
        "samples": 100,
        "description": "COCO val 2017 子集",
    },
    "coco_captions": {
        "type": "image_captioning",
        "samples": 50,
        "description": "COCO Captions 2017 子集",
    },
    "chinese_sentiment": {
        "type": "text_classification",
        "samples": 100,
        "description": "chinese_sentiment 中文分类/匹配子集",
    },
    "alpaca_zh": {
        "type": "instruction_tuning",
        "samples": 100,
        "description": "alpaca-zh 中文指令微调数据",
    },
}



def download_cifar10(n_samples: int = 100) -> int:
    """下载 CIFAR-10 子集（0 token，torchvision 一行代码）"""

    save_dir = DATA_DIR / "cifar10"
    save_dir.mkdir(parents=True, exist_ok=True)

    # 下载完整数据集到 data/raw/cifar10/full
    full_dir = save_dir / "full"
    if not full_dir.exists():
        print(f"⬇️  下载 CIFAR-10 完整数据集到 {full_dir}（~170MB）...")
        CIFAR10(root=str(full_dir), train=True, download=True)

    # 抽样 n_samples 张图
    src_dir = full_dir / "cifar-10-batches-py"
    img_dir = save_dir / "samples"
    img_dir.mkdir(parents=True, exist_ok=True)

    print(f"📦 抽样 {n_samples} 张 CIFAR-10 图片到 {img_dir}")
    # 这里用 unpickle 读 + 抽样 + 保存图片

    with open(src_dir / "data_batch_1", "rb") as f:
        batch = pickle.load(f, encoding="bytes")
    images = batch[b"data"]
    labels = batch[b"labels"]

    n = min(n_samples, len(images))
    for i in range(n):
        img = images[i].reshape(3, 32, 32).transpose(1, 2, 0)
        img = Image.fromarray(img)
        label = labels[i]
        img.save(img_dir / f"cifar10_{i:04d}_class{label}.png")

    print(f"[cifar10] ✅ 完成: {n} 张图 → {img_dir}")
    return n


def download_coco_val(n_samples: int = 100) -> None:
    """下载 COCO val2017 子集：标注 zip + n_samples 张图片。"""
    coco_dir = DATA_DIR / "coco_val"
    coco_dir.mkdir(parents=True, exist_ok=True)

    # 1. 下载标注 zip（如果还没下）
    ann_zip = coco_dir / "annotations_trainval2017.zip"
    if not ann_zip.exists():
        url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
        print(f"[COCO] 下载标注: {url}")
        urllib.request.urlretrieve(url, ann_zip)
        print(f"[COCO] 标注下载完成: {ann_zip.stat().st_size / 1024 / 1024:.1f} MB")

    # 2. 解压（如果还没解压）
    ann_dir = coco_dir / "annotations"
    if not ann_dir.exists():
        print(f"[COCO] 解压标注...")
        with zipfile.ZipFile(ann_zip, "r") as zf:
            zf.extractall(coco_dir)
        print(f"[COCO] 标注解压完成")

    # 3. 读 instances_val2017.json 抽 100 个 image_id
    instances_json = ann_dir / "instances_val2017.json"
    with open(instances_json, "r", encoding="utf-8") as f:
        instances = json.load(f)

    all_images = instances["images"]
    random.seed(42)  # 固定种子，结果可复现
    sampled = random.sample(all_images, min(n_samples, len(all_images)))

    # 4. 下载图片
    img_dir = coco_dir / "samples"
    img_dir.mkdir(exist_ok=True)
    sample_list = []

    for i, img_meta in enumerate(sampled, 1):
        file_name = img_meta["file_name"]
        img_path = img_dir / file_name
        if not img_path.exists():
            url = f"http://images.cocodataset.org/val2017/{file_name}"
            urllib.request.urlretrieve(url, img_path)
        sample_list.append(
            {
                "id": img_meta["id"],
                "file_name": file_name,
                "url": f"http://images.cocodataset.org/val2017/{file_name}",
                "width": img_meta["width"],
                "height": img_meta["height"],
            }
        )
        print(f"[COCO] {i}/{len(sampled)} {file_name}")

    # 5. 保存 sample_list.json
    with open(coco_dir / "sample_list.json", "w", encoding="utf-8") as f:
        json.dump(sample_list, f, ensure_ascii=False, indent=2)

    print(f"[COCO] ✅ 完成: {len(sample_list)} 张图 → {img_dir}")


def download_coco_captions(n_samples: int = 50) -> int:
    """从 D8-B 已下的 COCO 标注中抽 n_samples 条 caption（0 token）"""
    coco_ann = DATA_DIR / "coco_val" / "annotations" / "captions_val2017.json"
    if not coco_ann.exists():
        print(f"[coco_captions] ❌ 标注未找到，请先跑 coco_val")
        return 0

    with open(coco_ann, "r", encoding="utf-8") as f:
        captions = json.load(f)

    out_dir = DATA_DIR / "coco_captions"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_anns = captions["annotations"]
    random.seed(42)
    sampled = random.sample(all_anns, min(n_samples, len(all_anns)))

    out_data = [
        {"image_id": ann["image_id"], "caption": ann["caption"]} for ann in sampled
    ]

    with open(out_dir / "samples.json", "w", encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=False, indent=2)

    print(
        f"[coco_captions] ✅ 完成: {len(out_data)} 条 → {out_dir / 'samples.json'}"
    )
    return len(out_data)

def download_chinese_sentiment(n_samples: int = 100) -> int:
    """下载 chnsenticorp 中文情感分类子集（积极/消极，0 token）"""
    out_dir = DATA_DIR / "chinese_sentiment"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[chinese_sentiment] 下载 chnsenticorp（中文情感分类）...")
    ds = load_dataset("lansinuote/chnsenticorp", split=f"train[:{n_samples}]")

    out_data = [
        {"text": item["text"], "label": item["label"]}
        for item in ds
    ]

    with open(out_dir / "samples.json", "w", encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=False, indent=2)

    print(f"[chinese_sentiment] ✅ 完成: {len(out_data)} 条 → {out_dir / 'samples.json'}")
    return len(out_data)

def download_alpaca_zh(n_samples: int = 100) -> int:
    """下载 alpaca-zh 中文指令微调子集（0 token）"""
    out_dir = DATA_DIR / "alpaca_zh"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[alpaca_zh] 下载 alpaca-zh 中文指令数据...")
    ds = load_dataset("shibing624/alpaca-zh", split=f"train[:{n_samples}]")

    out_data = [
        {
            "instruction": item["instruction"],
            "input": item["input"],
            "output": item["output"],
        }
        for item in ds
    ]

    with open(out_dir / "samples.json", "w", encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=False, indent=2)

    print(f"[alpaca_zh] ✅ 完成: {len(out_data)} 条 → {out_dir / 'samples.json'}")
    return len(out_data)


# 数据集函数映射（避免 main 里写一堆 if/elif）
DATASET_FUNCS = {
    "cifar10": download_cifar10,
    "coco_val": download_coco_val,
    "coco_captions": download_coco_captions,
    "chinese_sentiment": download_chinese_sentiment,
    "alpaca_zh": download_alpaca_zh,
}



def main():
    if len(sys.argv) < 2:
        print("用法：python scripts/download_datasets.py <dataset_name>")
        print("可选：", ", ".join(DATASET_CONFIG.keys()))
        return

    name = sys.argv[1]
    if name not in DATASET_CONFIG:
        print(f"❌ 不支持的数据集：{name}")
        return

    cfg = DATASET_CONFIG[name]
    n = cfg["samples"]

    func = DATASET_FUNCS[name]
    func(n_samples=n)


if __name__ == "__main__":
    main()