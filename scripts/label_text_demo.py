"""
D9 阶段 1: 智谱 GLM-4 文本标注 Demo
任务: 对 chinese_sentiment 第 1 条做情感分类（0=负面 / 1=正面）
"""
import os
import json
from pathlib import Path
from dotenv import load_dotenv
from zhipuai import ZhipuAI

# ===== 第 1 块：加载 API Key =====
load_dotenv()
api_key = os.getenv("ZHIPUAI_API_KEY")
assert api_key, "❌ 请检查 .env 里的 ZHIPUAI_API_KEY 是否配置"

# ===== 第 2 块：初始化智谱客户端 =====
client = ZhipuAI(api_key=api_key)

# ===== 第 3 块：读 1 条样本 =====
project_root = Path(__file__).parent.parent.resolve()
sample_path = project_root / "data" / "raw" / "chinese_sentiment" / "samples.json"
with open(sample_path, "r", encoding="utf-8") as f:
    samples = json.load(f)
text = samples[0]["text"]
true_label = samples[0]["label"]
print(f"📝 待标注文本: {text}")
print(f"🏷️  真实标签: {true_label} ({'正面' if true_label == 1 else '负面'})")
print("=" * 60)

# ===== 第 4 块：调智谱 GLM-4 Flash =====
response = client.chat.completions.create(
    model="glm-4.5-air",
    messages=[
        {"role": "system", "content": "你是情感分析专家。请判断用户评论的情感极性，输出 0 或 1：0 = 负面，1 = 正面。只输出一个数字，不要任何其他文字。"},
        {"role": "user", "content": f"评论：{text}\n请输出 0 或 1："}
    ],
    temperature=0,
)

# ===== 第 5 块：解析结果 =====
pred_label = response.choices[0].message.content.strip()
print(f"🤖 模型预测: {pred_label}")
print(f"✅ 是否一致: {pred_label == str(true_label)}")
print(f"💰 Token 消耗: {response.usage.total_tokens} (prompt + completion)")