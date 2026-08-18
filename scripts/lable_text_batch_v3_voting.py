"""
lable_text_batch_v3_voting.py
D10 阶段 2：3 次投票 + 置信度过滤
基于 v2 优化版 prompt，每条跑 3 次（temperature=0.7），3 次一致为高置信
对比 D9 baseline (85%) 和 D10 v2 (88%)
"""

import os
import json
import time
from pathlib import Path
from dotenv import load_dotenv
from zhipuai import ZhipuAI
from collections import Counter

# ===== 第 1 块：初始化 =====
load_dotenv()
api_key = os.getenv("ZHIPUAI_API_KEY")
assert api_key, "❌ 请检查 .env"
client = ZhipuAI(api_key=api_key)
MODEL = "glm-4.5-air"
ROUNDS = 3
TEMPERATURE = 0.7

# ===== 第 2 块：优化版 Prompt（跟 v2 一致） =====
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
print(
    f"📚 加载样本: {len(samples)} 条 | 投票轮数: {ROUNDS} | temperature: {TEMPERATURE}"
)

# ===== 第 4 块：3 次投票标注 =====
results = []
total_tokens = 0
start = time.time()

for i, item in enumerate(samples, 1):
    text = item["text"]
    true_label = item["label"]
    votes = []
    sample_tokens = 0
    error_msg = None

    for r in range(ROUNDS):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"评论：{text}\n请输出 0 或 1："},
                ],
                temperature=TEMPERATURE,
            )
            pred = resp.choices[0].message.content.strip()
            tokens = resp.usage.total_tokens
            total_tokens += tokens
            sample_tokens += tokens
            votes.append(int(pred))
        except Exception as e:
            error_msg = str(e)
            votes.append(-1)
        time.sleep(0.05)

    # 投票决策
    valid_votes = [v for v in votes if v in (0, 1)]
    counter = Counter(valid_votes)
    most_common = counter.most_common(1)[0] if counter else (None, 0)
    final = most_common[0]

    # 置信度：3 一致=high，2:1=low，1:1:1=reject
    distinct = set(valid_votes)
    if len(distinct) == 1:
        confidence = "high"
    elif len(distinct) == 2 and most_common[1] >= 2:
        confidence = "low"
    else:
        confidence = "reject"

    is_match = final == true_label
    record = {
        "idx": i,
        "text": text[:50] + "...",
        "true_label": true_label,
        "votes": votes,
        "final": final,
        "confidence": confidence,
        "is_match": is_match,
        "tokens": sample_tokens,
    }
    if error_msg:
        record["error"] = error_msg
    results.append(record)

    if i % 10 == 0:
        print(
            f"  进度 {i}/{len(samples)} | 累计 token: {total_tokens} | 耗时: {time.time() - start:.0f}s"
        )
    time.sleep(0.1)

elapsed = time.time() - start
print(
    f"\n✅ 投票完成: {len(results)} 条 | 耗时: {elapsed:.1f}s | 总 token: {total_tokens}"
)

# ===== 第 5 块：评估 + 三方对比 =====
D9_ACCURACY = 0.85
D10_V2_ACCURACY = 0.88

# 整体
match_count = sum(1 for r in results if r["is_match"])
overall_acc = match_count / len(results)

# 按置信度分组
by_conf = {"high": [], "low": [], "reject": []}
for r in results:
    by_conf[r["confidence"]].append(r)

print(f"\n📊 评估结果:")
print(f"  D9 baseline:   {D9_ACCURACY * 100:.0f}%")
print(f"  D10 v2:        {D10_V2_ACCURACY * 100:.0f}%")
print(f"  D10 v3 投票后: {overall_acc * 100:.1f}% (整体)")
print(f"  提升 vs D9:    +{(overall_acc - D9_ACCURACY) * 100:.1f}pp")
print(f"  提升 vs v2:    +{(overall_acc - D10_V2_ACCURACY) * 100:.1f}pp")

print(f"\n🎯 置信度分布:")
for conf in ["high", "low", "reject"]:
    items = by_conf[conf]
    if items:
        acc = sum(1 for r in items if r["is_match"]) / len(items)
        print(f"  {conf:7s}: {len(items):3d} 条 | 一致率 {acc * 100:.1f}%")

# 错误案例
errors = [r for r in results if not r["is_match"]]
if errors:
    print(f"\n❌ 仍不一致 {len(errors)} 条（前 5 条）:")
    for r in errors[:5]:
        print(
            f"  #{r['idx']}: 真={r['true_label']} 投={r['votes']} 终={r['final']} 置信={r['confidence']} | {r['text']}"
        )

# ===== 第 6 块：保存 v3 JSON =====
output_path = project_root / "data" / "annotations" / "chinese_sentiment_pred_v3.json"
output_path.parent.mkdir(parents=True, exist_ok=True)
save_data = {
    "model": MODEL,
    "version": "v3-voting",
    "rounds": ROUNDS,
    "temperature": TEMPERATURE,
    "system_prompt": SYSTEM_PROMPT,
    "total_samples": len(results),
    "total_tokens": total_tokens,
    "elapsed_sec": elapsed,
    "overall_accuracy": overall_acc,
    "d9_baseline": D9_ACCURACY,
    "d10_v2": D10_V2_ACCURACY,
    "by_confidence": {
        conf: {
            "count": len(items),
            "accuracy": sum(1 for r in items if r["is_match"]) / len(items)
            if items
            else 0,
        }
        for conf, items in by_conf.items()
    },
    "results": results,
}
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(save_data, f, ensure_ascii=False, indent=2)
print(f"\n💾 已保存: {output_path}")
