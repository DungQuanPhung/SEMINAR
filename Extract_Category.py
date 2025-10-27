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
    
    Hàm này được tối ưu để chạy inference (dự đoán) bằng PyTorch/Transformers.

    Args:
        clauses (List[Dict[str, Any]]): Danh sách các mệnh đề, mỗi mệnh đề 
                                         là một dict phải chứa key 'clause'.
        model_cat (Any): Mô hình phân loại PyTorch đã được load và chuyển sang eval mode.
        tokenizer_cat (Any): Tokenizer tương ứng với mô hình.
        id2label (Dict[int, str]): Ánh xạ ID dự đoán sang nhãn Category.
        device (torch.device): Thiết bị để chạy mô hình ('cuda' hoặc 'cpu').

    Returns:
        List[Dict[str, Any]]: Danh sách các mệnh đề đã được cập nhật thêm key 'category'.
    """
    
    # Đảm bảo mô hình ở chế độ đánh giá và trên đúng thiết bị
    if model_cat.training:
        model_cat.eval()
    model_cat.to(device)
    
    print(f"Bắt đầu dự đoán Category cho {len(clauses)} mệnh đề...")

    for c in clauses:
        text = str(c.get("clause", "")).strip()
        
        # 1. Xử lý trường hợp chuỗi rỗng
        if not text:
            c["category"] = "Unknown"
            continue

        try:
            # 2. Tokenize input
            inputs = tokenizer_cat(
                text,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=MAX_LENGTH 
            ).to(device)

            # 3. Dự đoán (Inference)
            with torch.no_grad():
                outputs = model_cat(**inputs)
                
                # Lấy ID dự đoán
                pred_id = torch.argmax(outputs.logits, dim=1).item()
                
                # 4. Ánh xạ ID sang nhãn Category
                c["category"] = id2label.get(pred_id, "Unknown")
                
        except Exception as e:
            # Xử lý lỗi (ví dụ: GPU OOM)
            print(f" Lỗi khi xử lý clause='{text}': {e}")
            c["category"] = "Unknown"

    print("Hoàn thành dự đoán Category.")
    return clauses

def get_predicted_categories(clauses: List[Dict[str, Any]]) -> List[str]:
    """
    Hàm phụ: Trả về danh sách các Category đã được dự đoán từ danh sách clauses.
    
    Args:
        clauses (List[Dict[str, Any]]): Danh sách các mệnh đề đã có key 'category'.

    Returns:
        List[str]: Danh sách các Category (bao gồm cả giá trị lặp lại).
    """
    categories = []
    for c in clauses:
        # Sử dụng .get() để an toàn nếu trường 'category' không tồn tại
        category = c.get("category")
        if category is not None:
            categories.append(str(category))

    return categories