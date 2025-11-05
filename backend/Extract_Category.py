import torch
from typing import List, Dict, Any

MAX_LENGTH = 128

def extract_categories(
    clauses: List[Dict[str, Any]],
    model_cat: Any,
    tokenizer_cat: Any,
    id2label: Dict[int, str],
    device: torch.device
) -> List[Dict[str, Any]]:
    """
    Hàm chính: Dự đoán Category (phân loại) cho một danh sách các mệnh đề (clauses).
    """
    
    if model_cat is None or tokenizer_cat is None:
        print("Lỗi: Mô hình Category chưa được tải.")
        for c in clauses:
            c["category"] = "Error (Model not loaded)"
        return clauses

    if model_cat.training:
        model_cat.eval()
    model_cat.to(device)
    
    # Bỏ print() để làm sạch output của Streamlit
    # print(f"Bắt đầu dự đoán Category cho {len(clauses)} mệnh đề...")

    for c in clauses:
        text = str(c.get("clause", "")).strip()
        
        if not text:
            c["category"] = "Unknown"
            continue

        try:
            inputs = tokenizer_cat(
                text,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=MAX_LENGTH 
            ).to(device)

            with torch.no_grad():
                outputs = model_cat(**inputs)
                pred_id = torch.argmax(outputs.logits, dim=1).item()
                c["category"] = id2label.get(pred_id, "Unknown")
                
        except Exception as e:
            print(f" Lỗi khi xử lý category: '{text}': {e}")
            c["category"] = "Unknown"

    # print("Hoàn thành dự đoán Category.")
    return clauses

def get_predicted_categories(clauses: List[Dict[str, Any]]) -> List[str]:
    """
    Hàm phụ: Trả về danh sách các Category đã được dự đoán.
    """
    categories = []
    for c in clauses:
        category = c.get("category")
        if category is not None:
            categories.append(str(category))
            
    return categories