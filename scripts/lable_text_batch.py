"""
智谱 GLM-4.5-air 批量文本标注
任务: 对 chinese_sentiment 100 条做情感分类，统计一致率
"""
import os
import json
import time
from pathlib import Path
from dotenv import load_dotenv
from zhipuai import ZhipuAI

# ===== 第 1 块：初始化 =====
load_dotenv()
api_key = os.getenv("ZHIPUAI_API_KEY")
assert api_key, "❌ 请检查 .env"
client = ZhipuAI(api_key=api_key)
MODEL = "glm-4.5-air"

# ===== 第 2 块：读样本 =====
project_root = Path(__file__).parent.parent.resolve()
sample_path = project_root / "data" / "raw" / "chinese_sentiment" / "samples.json"
with open(sample_path, "r", encoding="utf-8") as f:
    samples = json.load(f)
print(f"📚 加载样本: {len(samples)} 条")

# ===== 第 3 块：批量标注 =====
results = []
total_tokens = 0
start = time.time()

for i, item in enumerate(samples, 1):
    text = item["text"]
    true_label = item["label"]
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "你是情感分析专家。请判断用户评论的情感极性，输出 0 或 1：0 = 负面，1 = 正面。只输出一个数字，不要任何其他文字。"},
                {"role": "user", "content": f"评论：{text}\n请输出 0 或 1："}
            ],
            temperature=0,
        )
        pred = resp.choices[0].message.content.strip()
        tokens = resp.usage.total_tokens
        total_tokens += tokens
        is_match = pred == str(true_label)
        results.append({"idx": i, "text": text[:50] + "...", "true_label": true_label, "pred_label": pred, "is_match": is_match, "tokens": tokens})
        if i % 10 == 0:
            print(f"  进度 {i}/{len(samples)} | 累计 token: {total_tokens}")
        time.sleep(0.1)  # 防限流
    except Exception as e:
        print(f"  ❌ 第 {i} 条失败: {e}")
        results.append({"idx": i, "text": text[:50] + "...", "true_label": true_label, "pred_label": "ERROR", "is_match": False, "error": str(e)})
        time.sleep(1)

elapsed = time.time() - start
print(f"\n✅ 标注完成: {len(results)} 条 | 耗时: {elapsed:.1f}s | 总 token: {total_tokens}")

# ===== 第 4 块：统计评估 =====
match_count = sum(1 for r in results if r["is_match"])
accuracy = match_count / len(results)
print(f"\n📊 评估结果:")
print(f"  一致条数: {match_count}/{len(results)}")
print(f"  一致率: {accuracy * 100:.1f}%")
print(f"  平均 token/条: {total_tokens // len(results)}")

errors = [r for r in results if not r["is_match"]]
if errors:
    print(f"\n❌ 不一致 {len(errors)} 条（前 5 条）:")
    for r in errors[:5]:
        print(f"  #{r['idx']}: 真实={r['true_label']} 预测={r['pred_label']} | {r['text']}")

# ===== 第 5 块：保存结果 =====
output_path = project_root / "data" / "annotations" / "chinese_sentiment_pred.json"
output_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_path, "w", encoding="utf-8") as f:
    json.dump({"model": MODEL, "total_samples": len(results), "total_tokens": total_tokens, "accuracy": accuracy, "elapsed_sec": elapsed, "results": results}, f, ensure_ascii=False, indent=2)
print(f"\n💾 已保存: {output_path}")