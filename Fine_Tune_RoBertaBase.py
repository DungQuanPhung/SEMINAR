import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
from sklearn.metrics import accuracy_score, classification_report

# ====================== CẤU HÌNH ======================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f" Using device: {device}")

# ====================== ĐỌC DỮ LIỆU ======================
goal_path = "70% sample.csv"
llm_path = "goal.xlsx"

goal_df = pd.read_csv(goal_path)
llm_predict_df = pd.read_excel(llm_path)

# Đảm bảo có cột đúng định dạng
assert {"clause", "category"}.issubset(goal_df.columns), "goal.xlsx thiếu cột 'clause' hoặc 'category'"
assert {"clause", "category"}.issubset(llm_predict_df.columns), "llm_predict.csv thiếu cột 'clause' hoặc 'category'"

# ====================== CHUẨN BỊ DỮ LIỆU ======================
# goal_df = ground truth
# llm_predict_df = mô hình cũ dự đoán (có thể đúng/sai)
required_cols = ["clause", "category"]
train_df = goal_df[required_cols].copy()
test_df = llm_predict_df[required_cols].copy()

# ====================== TOKENIZER + ENCODING ======================
model_name = "roberta-base"
tokenizer_cat = AutoTokenizer.from_pretrained(model_name)

label_list = sorted(goal_df["category"].unique().tolist())
label2id = {l: i for i, l in enumerate(label_list)}
id2label = {i: l for l, i in label2id.items()}

def encode_fn(batch):
    enc = tokenizer_cat(batch["clause"], truncation=True, padding="max_length", max_length=128)
    enc["labels"] = label2id[batch["category"]]
    return enc

train_ds = Dataset.from_pandas(train_df)
test_ds = Dataset.from_pandas(test_df)
train_ds = train_ds.map(encode_fn)
test_ds = test_ds.map(encode_fn)
train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
test_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# ====================== HÀM ĐÁNH GIÁ ======================
def evaluate_model(model, dataset):
    model.eval()
    model.to(device)
    preds, trues = [], []
    for i in range(len(dataset)):
        inputs = {k: v.unsqueeze(0).to(device) for k, v in dataset[i].items() if k in ["input_ids", "attention_mask"]}
        with torch.no_grad():
            outputs = model(**inputs)
            pred = torch.argmax(outputs.logits, dim=1).item()
        preds.append(pred)
        trues.append(dataset[i]["labels"].item())
    print(classification_report(trues, preds, target_names=label_list, zero_division=0))
    return accuracy_score(trues, preds)

model_pre = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(label_list),
    ignore_mismatched_sizes=True
).to(device)

# ====================== CẤU HÌNH LoRA ======================
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["query", "value"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_CLS
)

model_cat = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(label_list),
    ignore_mismatched_sizes=True
)
model_cat = get_peft_model(model_cat, lora_config)
model_cat.to(device)

# ====================== HUẤN LUYỆN ======================
args = TrainingArguments(
    output_dir="./roberta_lora_goal",
    report_to="none",
    per_device_train_batch_size=8,       # ↑ tăng batch size giúp gradient ổn định hơn
    gradient_accumulation_steps=4,       # tích lũy gradient → giả lập batch lớn hơn (8×4=32)
    num_train_epochs=80,                 # ↑ học lâu hơn (từ 50 → 80)
    learning_rate=1.5e-4,                # giảm nhẹ learning rate để tránh overfit khi tăng epoch
    warmup_ratio=0.1,                    # warmup giúp ổn định giai đoạn đầu
    lr_scheduler_type="cosine",          # scheduler mượt hơn
    weight_decay=0.05,                   # tăng nhẹ regularization để chống overfit
    save_strategy="no",                  # vẫn không lưu giữa chừng
    logging_steps=10,
    fp16=torch.cuda.is_available(),      # giữ nguyên để tận dụng GPU
)

trainer = Trainer(
    model=model_cat,
    args=args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    tokenizer=tokenizer_cat,
)

# ... (code trainer) ...
trainer.train()

# 1. Merge LoRA weights vào base model và unload adapter
# Cần gọi .base_model vì trainer.model là một đối tượng PeftModel
merged_model = model_cat.merge_and_unload()

print(f"Đang lưu mô hình ĐẦY ĐỦ (merged) vào thư mục: {args.output_dir}")

# 2. Lưu mô hình đầy đủ (merged model)
merged_model.save_pretrained(args.output_dir)

# 3. Lưu cả tokenizer vào cùng thư mục
# (Rất quan trọng để app streamlit tải đúng)
tokenizer_cat.save_pretrained(args.output_dir)

print(f"--- Đã lưu xong mô hình và tokenizer vào {args.output_dir} ---")
# --- KẾT THÚC SỬA ĐỔI ---