"""
优化版 Prompt 批量标注
任务: 用 few-shot + 错误补丁 prompt 重跑 100 条，对比 lable_text_batch 一致率
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

# ===== 第 2 块：优化版 Prompt（含 few-shot + 错误补丁） =====
SYSTEM_PROMPT = """你是中文情感分析专家。请严格按以下标准判断用户评论的情感极性，输出 0 或 1。

【分类标准】
- 0 = 负面：用户整体表达不满、失望、批评、抱怨
- 1 = 正面：用户整体表达满意、喜欢、推荐、认可

【关键判断原则】
1. 关注整体态度，不要被单个负面细节带偏（如"有蚊子"不否定"位置好"的整体正面）
2. 关注语气和场景，"累死我了"在"女儿要听故事"语境下是正面
3. 关注结果是否解决了用户问题，忽略过程描述

【Few-shot 例子】
例子 1（输出 1）：今天才知道这书还有第6卷...有点郁闷：为什么同一套书有两种版本呢？→ 虽然有"郁闷"一词，但整体是惊喜发现新版本，应为正面
例子 2（输出 0）：今天装声卡装好了，接着装显卡，装完显卡电脑就没声音了 → 明确表达问题未解决，应为负面
例子 3（输出 1）：表皮看上去不错很精致，但是看得出来是盗的。但是里面内容真的不错 → 整体满意内容（"真的不错"），但抱怨盗版，应关注主要诉求（内容）

【输出格式】只输出 0 或 1，不要任何其他字符、标点、解释。"""

# ===== 第 3 块：读样本 =====
project_root = Path(__file__).parent.parent.resolve()
sample_path = project_root / "data" / "raw" / "chinese_sentiment" / "samples.json"
with open(sample_path, "r", encoding="utf-8") as f:
    samples = json.load(f)
print(f"📚 加载样本: {len(samples)} 条")

# ===== 第 4 块：批量标注 =====
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
                {"role": "system", "content": SYSTEM_PROMPT},
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
        time.sleep(0.1)
    except Exception as e:
        print(f"  ❌ 第 {i} 条失败: {e}")
        results.append({"idx": i, "text": text[:50] + "...", "true_label": true_label, "pred_label": "ERROR", "is_match": False, "error": str(e)})
        time.sleep(1)

elapsed = time.time() - start
print(f"\n✅ 标注完成: {len(results)} 条 | 耗时: {elapsed:.1f}s | 总 token: {total_tokens}")

# ===== 第 5 块：评估 + 对比 D9 =====
match_count = sum(1 for r in results if r["is_match"])
accuracy = match_count / len(results)
D9_ACCURACY = 0.85  # D9 baseline
print(f"\n📊 评估结果:")
print(f"  D10 一致率: {accuracy * 100:.1f}% (D9 baseline: {D9_ACCURACY*100:.0f}%)")
print(f"  提升: +{(accuracy - D9_ACCURACY) * 100:.1f}%")

errors = [r for r in results if not r["is_match"]]
if errors:
    print(f"\n❌ 仍不一致 {len(errors)} 条（前 5 条）:")
    for r in errors[:5]:
        print(f"  #{r['idx']}: 真实={r['true_label']} 预测={r['pred_label']} | {r['text']}")

# ===== 第 6 块：保存结果 =====
output_path = project_root / "data" / "annotations" / "chinese_sentiment_pred_v2.json"
output_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_path, "w", encoding="utf-8") as f:
    json.dump({"model": MODEL, "version": "v2-fewshot-prompt", "total_samples": len(results), "total_tokens": total_tokens, "accuracy": accuracy, "d9_baseline": D9_ACCURACY, "elapsed_sec": elapsed, "results": results}, f, ensure_ascii=False, indent=2)
print(f"\n💾 已保存: {output_path}")