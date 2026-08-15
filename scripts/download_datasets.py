"""
通用数据集抽样下载脚本
目标：硬件友好 + 0 token 优先，总样本控制在 100-500 条

使用：
    python scripts/download_datasets.py cifar10
    python scripts/download_datasets.py coco_val
"""
import os
from pathlib import Path
import yaml

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
    "clue": {
        "type": "text_classification",
        "samples": 100,
        "description": "CLUE 中文分类/匹配子集",
    },
    "alpaca_zh": {
        "type": "instruction_tuning",
        "samples": 100,
        "description": "alpaca-zh 中文指令微调数据",
    },
}


def download_cifar10(n_samples: int = 100) -> int:
    """下载 CIFAR-10 子集（0 token，torchvision 一行代码）"""
    from torchvision.datasets import CIFAR10

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
    import pickle
    import numpy as np
    from PIL import Image

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

    print(f"✅ CIFAR-10 抽样完成：{n} 张图片存到 {img_dir}")
    return n


def main():
    import sys
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

    if name == "cifar10":
        download_cifar10(n_samples=n)
    else:
        print(f"⏳ {name} 还没实现（下一阶段加）")


if __name__ == "__main__":
    main()