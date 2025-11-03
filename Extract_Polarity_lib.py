from typing import List, Dict, Any

def run_detect_polarity(clauses: List[Dict[str, Any]], polarity_classifier: Any) -> List[Dict[str, Any]]:
    """
    Phát hiện polarity, nhận classifier đã được tải làm tham số.
    """
    
    if polarity_classifier is None:
        print("Lỗi: Mô hình Polarity chưa được tải.")
        for item in clauses:
            item["polarity"] = "Error (Model not loaded)"
            item["polarity_score"] = 0.0
        return clauses
        
    results = []
    
    # Bỏ tqdm vì Streamlit có st.spinner
    for item in clauses: 
        clause = str(item.get("clause", "")).strip()
        if clause == "":
            item["polarity"] = "Neutral"
            item["polarity_score"] = 0.0
            results.append(item)
            continue

        try:
            # Chạy mô hình pipeline
            res = polarity_classifier(clause)
            
            # Xử lý định dạng output (đôi khi pipeline trả về list trong list)
            if isinstance(res, list) and isinstance(res[0], list):
                res = res[0]
                
            # Tìm label có score cao nhất
            top = max(res, key=lambda x: x["score"]) 
            item["polarity"] = top["label"].capitalize() 
            item["polarity_score"] = round(top["score"], 4)
            
        except Exception as e:
            print(f" Lỗi khi xử lý polarity: '{clause}': {e}") 
            item["polarity"] = "Neutral"
            item["polarity_score"] = 0.0

        results.append(item)
    return results