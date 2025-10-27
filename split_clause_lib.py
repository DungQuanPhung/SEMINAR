import re
from typing import List, Dict, Any, Callable

def chat_llm(model: Any, tokenizer: Any, messages: List[Dict[str, str]], max_new_tokens: int = 100) -> str:
    """
    Hàm chat cơ bản. Nhận model và tokenizer làm tham số.
   
    """
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:])

def run_split_and_term(sentence: str, chat_func: Callable, max_new_tokens: int = 300) -> List[Dict[str, Any]]:
    """
    Tách câu và trích xuất Term, sử dụng hàm chat_func được truyền vào.
   
    """
    prompt = (
    "You are an expert linguist working on Aspect-Based Sentiment Analysis (ABSA).\n"
    # ... (Toàn bộ prompt dài của bạn ở đây, tôi rút gọn cho dễ nhìn) ...
    "10️. If a clause refers to the same entity as a previous one but does not repeat it explicitly, "
    "**propagate the term from the previous clause**.\n\n"
    "11. (CRITICAL) Always split sentences on major conjunctions like 'but', 'however', and 'although' when they separate different opinions or topics.\n\n"

    "==================== OUTPUT FORMAT ====================\n"
    "==================== OUTPUT FORMAT ====================\n"
    "Clause: <clause text> | Term: <term1,term2,...>\n\n"
    "==================== RESPONSE INSTRUCTION ====================\n"
    "Respond ONLY with the clauses and terms exactly in the format shown above.\n"
    "Do NOT include any explanation, reasoning, or commentary.\n"
    "Do NOT include quotation marks, markdown, or extra text.\n\n"
    f"Now process this sentence WITHOUT changing any words:\n{sentence}"
    ) #

    messages = [{"role": "user", "content": prompt}]
    # Gọi hàm chat được truyền từ app.py
    response = chat_func(messages, max_new_tokens=max_new_tokens).strip()

    # --- Làm sạch output (Logic từ split_clause.py) ---
    result = []
    last_term = ""
    for line in response.split("\n"):
        line = line.strip()
        if not line: continue
        if "| Term:" in line:
            parts = line.split("| Term:")
            clause_text = parts[0].replace("Clause:", "").strip()
            term = parts[1].strip()
            if term == "":
                term = last_term
            else:
                last_term = term
        else:
            clause_text = line.replace("Clause:", "").strip()
            term = last_term
        result.append({"clause": clause_text, "term": term, "sentence_original": sentence})
    
    # --- Logic lọc term (Logic từ split_clause.py) ---
    if result:
        last_clause = result[-1]
        if "term" in last_clause:
            last_clause["term"] = last_clause["term"].replace("<|im_end|>", "").strip()
    for c in result:
        terms = [t.strip() for t in c.get("term", "").split(",") if t.strip()]
        terms = [t for t in terms if t.lower() in c["sentence_original"].lower()]
        c["term"] = ",".join(terms)
        
    return result