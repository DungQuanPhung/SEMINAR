from tqdm import tqdm
from typing import List, Dict, Any

def run_detect_polarity(clauses: List[Dict[str, Any]], polarity_classifier: Any) -> List[Dict[str, Any]]:
    """
    Phát hiện polarity, nhận classifier đã được tải làm tham số.
   
    """
    results = []
    # tqdm sẽ không hiển thị đẹp trong Streamlit, nhưng ta có thể để đó cho logic
    for item in clauses: 
        clause = str(item.get("clause", "")).strip()
        if clause == "":
            item["polarity"] = "Neutral"
            item["polarity_score"] = 0.0
            results.append(item)
            continue

        try:
            res = polarity_classifier(clause)
            if isinstance(res, list) and isinstance(res[0], list):
                res = res[0]
            top = max(res, key=lambda x: x["score"]) #
            item["polarity"] = top["label"].capitalize() #
            item["polarity_score"] = round(top["score"], 4)
        except Exception as e:
            print(f" Lỗi khi xử lý polarity: '{clause}': {e}") # Giữ print để debug
            item["polarity"] = "Neutral"
            item["polarity_score"] = 0.0

        results.append(item)
    return results