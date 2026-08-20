# ===== LoRA 微调训练（Colab T4 + 本地 Qwen2.5-1.5B）=====
import os, json
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType

# 1. 路径
MODEL_DIR = "/content/models/"
TRAIN_PATH = "/content/drive/MyDrive/mydrive/AutoLabel/data/processed/train.jsonl"
VAL_PATH   = "/content/drive/MyDrive/mydrive/AutoLabel/data/processed/val.jsonl"
DRIVE_OUT  = "/content/drive/MyDrive/mydrive/AutoLabel/models/lora-v1"
os.makedirs(DRIVE_OUT, exist_ok=True)

# 2. 加载数据
train_ds = load_dataset("json", data_files=TRAIN_PATH, split="train")
val_ds   = load_dataset("json", data_files=VAL_PATH,   split="train")
print(f"[1] 训练集: {len(train_ds)} 条  验证集: {len(val_ds)} 条")

# 3. 加载模型（之前 cell 已经加载过，这里直接复用 globals）
# 实际上 from_pretrained 是幂等的，但保险起见再 load 一次
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR, trust_remote_code=True,
    torch_dtype="auto", device_map="auto"
)

# 4. LoRA 配置
lora_config = LoraConfig(
    r=8, lora_alpha=16, lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 5. 构造 prompt + 标签
def to_text(example):
    label_map = {0: "负面", 1: "正面", 2: "中性"}
    return {"text": f"判断情感：{example['text']}\n答案：{label_map[example['true_label']]}"}
train_ds = train_ds.map(to_text)
val_ds   = val_ds.map(to_text)

# 6. Tokenize
def tokenize_fn(example):
    return tokenizer(example["text"], truncation=True, max_length=128, padding="max_length")

train_tok = train_ds.map(tokenize_fn, batched=False).remove_columns([c for c in train_ds.column_names if c not in ["input_ids","attention_mask","labels"]])
val_tok   = val_ds.map(tokenize_fn, batched=False).remove_columns([c for c in val_ds.column_names if c not in ["input_ids","attention_mask","labels"]])

# 7. 训练
args = TrainingArguments(
    output_dir="/content/lora-output",
    num_train_epochs=5,
    per_device_train_batch_size=2,
    gradient_checkpointing=True,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_strategy="no",
    optim="paged_adamw_8bit",
    report_to="none"
)
trainer = Trainer(
    model=model, args=args,
    train_dataset=train_tok, eval_dataset=val_tok,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)

print("[2] 开始训练...")
trainer.train()
print("[3] 训练完成")

# 8. 保存 LoRA 权重到 Drive
model.save_pretrained(DRIVE_OUT)
tokenizer.save_pretrained(DRIVE_OUT)
print(f"[4] LoRA 权重已存到 {DRIVE_OUT}")
