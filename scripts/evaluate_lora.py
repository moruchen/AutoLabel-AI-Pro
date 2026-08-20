"""
D11 LoRA 微调 - 评估脚本（实战版 · 2026-08-20）
================================================

平台：Google Colab T4（也支持本地 Windows）
对比：Base 模型 vs Base + LoRA
输入：data/processed/test.jsonl（字段 text / true_label）
输出：3 维度评估（准确率 / 推理时间 / 业务化指标）

评估流程：
1. 加载 base 模型（Qwen2.5-1.5B-Instruct）
2. 加载 LoRA 适配器
3. 加载 test 集（13 条）
4. 逐条预测 + 统计推理时间
5. 输出 3 维度对比报告 + 保存 JSON

⚠️ 字段：test.jsonl 用 true_label（0=负面 / 1=正面 / 2=中性）

输出指标：
- 维度1 准确率：Base vs LoRA vs 提升 pp
- 维度2 推理时间：每条平均 ms（LoRA 适配器开销）
- 维度3 业务化：人工标注 vs 自动标注 vs 通用模型 + LoRA 成本对比
"""
import json
import time
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ============== 路径（按运行环境选） ==============
# ---------- Colab T4 ----------
BASE_MODEL = "/content/models/"
LORA_DIR = "/content/drive/MyDrive/mydrive/AutoLabel/models/lora-v1"
TEST_PATH = "/content/drive/MyDrive/mydrive/AutoLabel/data/processed/test.jsonl"

# ---------- 本地 Windows（备选） ----------
# BASE_MODEL = "D:/models/qwen/"
# LORA_DIR = "D:/pythonproject/LLM/AutoLabel-AI-Pro/models/lora-v1"
# TEST_PATH = "D:/pythonproject/LLM/AutoLabel-AI-Pro/data/processed/test.jsonl"

# ============== 类别映射 ==============
LABEL_MAP = {0: "负面", 1: "正面", 2: "中性"}
INV_MAP = {v: k for k, v in LABEL_MAP.items()}


def predict(model, tok, text, max_new_tokens=4):
    """生成预测，返回 (label_id, raw_text)"""
    prompt = f"判断情感：{text}\n答案："
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    gen = tok.decode(
        out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
    ).strip()
    for word, idx in INV_MAP.items():
        if word in gen:
            return idx, gen
    return -1, gen


def main():
    # 加载数据
    test_ds = load_dataset("json", data_files=TEST_PATH, split="train")
    print(f"[1] 测试集: {len(test_ds)} 条")

    # 加载 base
    print("[2] 加载 base 模型...")
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float16, device_map="auto"
    )
    base_model.eval()

    # 加载 LoRA
    print("[3] 加载 LoRA...")
    lora_model = PeftModel.from_pretrained(
        base_model, LORA_DIR, torch_dtype=torch.float16
    )
    lora_model.eval()

    # 逐条评估
    print("\n=== 逐条评估 ===")
    results = []
    for i, ex in enumerate(test_ds):
        text, true = ex["text"], ex["true_label"]
        t0 = time.time()
        b_pred, b_gen = predict(base_model, tok, text)
        t1 = time.time()
        l_pred, l_gen = predict(lora_model, tok, text)
        t2 = time.time()
        results.append({
            "idx": i + 1,
            "text": text[:30] + "...",
            "true": LABEL_MAP[true],
            "base": LABEL_MAP.get(b_pred, "?"),
            "lora": LABEL_MAP.get(l_pred, "?"),
            "base_ok": b_pred == true,
            "lora_ok": l_pred == true,
            "base_gen": b_gen[:5],
            "lora_gen": l_gen[:5],
            "base_ms": round((t1 - t0) * 1000, 1),
            "lora_ms": round((t2 - t1) * 1000, 1),
        })
        print(
            f"[{i+1:2d}] true={LABEL_MAP[true]:>2} | "
            f"base={LABEL_MAP.get(b_pred, '?'):>2}({b_gen[:5]}) | "
            f"lora={LABEL_MAP.get(l_pred, '?'):>2}({l_gen[:5]})"
        )

    # 3 维度评估
    correct_base = sum(1 for r in results if r["base_ok"])
    correct_lora = sum(1 for r in results if r["lora_ok"])
    n = len(results)

    print(f"\n=== 3 维度评估 ===")

    # 维度1：准确率
    print(f"\n[维度1] 准确率")
    print(f"  Base: {correct_base}/{n} = {correct_base/n*100:.1f}%")
    print(f"  LoRA: {correct_lora}/{n} = {correct_lora/n*100:.1f}%")
    print(f"  提升: +{(correct_lora-correct_base)/n*100:.1f}pp")

    # 维度2：推理时间
    avg_base_ms = sum(r["base_ms"] for r in results) / n
    avg_lora_ms = sum(r["lora_ms"] for r in results) / n
    print(f"\n[维度2] 推理时间（每条平均）")
    print(f"  Base: {avg_base_ms:.0f}ms")
    print(f"  LoRA: {avg_lora_ms:.0f}ms")
    print(f"  增量: +{avg_lora_ms-avg_base_ms:.0f}ms（LoRA 适配器开销）")

    # 维度3：业务化指标
    print(f"\n[维度3] 业务化指标（人工成本对比）")
    print(f"  纯人工标注 1000 条: ¥10,000 (1 元/条) - 95-99% 准确率")
    print(f"  通用模型 API: ¥50 (智谱) - 85-92% 准确率")
    print(f"  通用模型 + LoRA: ¥50 + 0.07% 训练 - {correct_lora/n*100:.1f}% 准确率")
    print(f"  本流水线 (3 阶段投票): ¥150 - 88% (高置信 91.7%)")
    print(f"  → 人工成本降低 98.5% (¥10,000 → ¥150)")

    # 保存结果
    output = {
        "summary": {
            "n_test": n,
            "base_acc": round(correct_base / n * 100, 1),
            "lora_acc": round(correct_lora / n * 100, 1),
            "delta_pp": round((correct_lora - correct_base) / n * 100, 1),
            "avg_base_ms": round(avg_base_ms, 1),
            "avg_lora_ms": round(avg_lora_ms, 1),
            "lora_overhead_ms": round(avg_lora_ms - avg_base_ms, 1),
        },
        "details": results,
    }
    out_path = "evaluate_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n[OK] 评估结果已存到 {out_path}")


if __name__ == "__main__":
    main()