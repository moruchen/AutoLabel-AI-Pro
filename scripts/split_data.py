"""
D11 LoRA 微调 - 数据划分脚本
输入：data/annotations/chinese_sentiment_pred_v3.json
输出：data/processed/train.jsonl / val.jsonl / test.jsonl
零 token，10 秒跑完
"""
import json
from pathlib import Path

# ============== 路径 ==============
PROJECT_ROOT = Path(r"D:\pythonproject\LLM\AutoLabel-AI-Pro")
SRC = PROJECT_ROOT / "data" / "annotations" / "chinese_sentiment_pred_v3.json"
OUT_DIR = PROJECT_ROOT / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ============== 划分配置 ==============
TRAIN_N = 70
VAL_N = 13
TEST_N = 13
TOTAL_N = TRAIN_N + VAL_N + TEST_N  # = 96

# ============== 读 v3 ==============
with open(SRC, "r", encoding="utf-8") as f:
    data = json.load(f)

# 抽 96 条 high
records = [r for r in data["results"] if r.get("confidence") == "high"]
print(f"[1] 读 v3.json -> {len(data['results'])} 条总记录")
print(f"[2] 抽 high 置信 -> {len(records)} 条")

# 按 idx 排序（保证分层稳定）
records.sort(key=lambda r: r.get("idx", 0))

# 检查总数
assert len(records) >= TOTAL_N, f"高置信数 {len(records)} < 96"
records = records[:TOTAL_N]

# ============== 划分 ==============
train = records[:TRAIN_N]
val = records[TRAIN_N : TRAIN_N + VAL_N]
test = records[TRAIN_N + VAL_N : TRAIN_N + VAL_N + TEST_N]

print(f"[3] 划分 -> train={len(train)}, val={len(val)}, test={len(test)}")

# ============== 写出 jsonl ==============
for name, subset in [("train", train), ("val", val), ("test", test)]:
    out_path = OUT_DIR / f"{name}.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for r in subset:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    # 抽样 1 条展示
    sample = subset[0]
    print(f"  -> {out_path.name}  示例: idx={sample['idx']}  true={sample['true_label']}  pred={sample['final']}  text='{sample['text'][:30]}...'")

print(f"\n[OK] 数据划分完成，输出目录：{OUT_DIR}")
