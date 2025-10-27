import re
from typing import List, Dict, Any, Callable

def run_extract_opinions(clauses: List[Dict[str, Any]], chat_func: Callable, max_new_tokens: int = 200) -> List[Dict[str, Any]]:
    """
    Trích xuất Opinion, sử dụng hàm chat_func được truyền vào.
   
    """
    final_clauses = []

    for c in clauses:
        clause_text = c["clause"]
        term = c.get("term", "")
        sentence_original = c.get("sentence_original", "")
        prompt = f"""
        You are an expert in Aspect-Based Sentiment Analysis (ABSA).
        Task: Extract all **opinion expressions** about the aspect/term "{term}" from the following clause.
        Strict rules:
        1. Only extract opinion words or short opinion phrases that appear **exactly** in the clause.
        2. Extract only opinions that clearly describe or evaluate the main term "{term}".
        3. Do **NOT** paraphrase, translate, or invent new words.
        4. Do **NOT** include explanations, reasoning, or labels.
        5. If there is no clear opinion, output an empty string.
        6. Output format: comma-separated list — e.g., "very helpful, friendly".
        Clause:
        "{clause_text}"
        Answer:
        """ #
        
        messages = [{"role": "user", "content": prompt}]
        # Gọi hàm chat được truyền từ app.py
        opinion_text = chat_func(messages, max_new_tokens=max_new_tokens).strip()

        opinion_text = opinion_text.replace("<|im_end|>", "").replace("\n", " ").strip()
        opinions = [o.strip() for o in re.split(r",", opinion_text) if o.strip()]

        # Logic lọc opinion
        valid_opinions = [
            o for o in opinions if re.search(rf"\b{re.escape(o)}\b", sentence_original, re.IGNORECASE)
        ]

        new_c = c.copy()
        new_c["opinion"] = ", ".join(valid_opinions) if valid_opinions else ""
        final_clauses.append(new_c)

    return final_clauses