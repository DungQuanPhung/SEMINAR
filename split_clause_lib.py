from pyexpat import model
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
    # Giữ lại phần phản hồi mới sinh ra và loại bỏ special tokens để giảm nhiễu.
    return tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    ).strip()

def run_split_and_term(sentence: str, chat_func: Callable, max_new_tokens: int = 300) -> List[Dict[str, Any]]:
    """
    Tách câu và trích xuất Term, sử dụng hàm chat_func được truyền vào.
   
    """
    prompt = (
        "You are an expert linguist working on Aspect-Based Sentiment Analysis (ABSA).\n"
        "Your task is to split the following review sentence into smaller clauses and identify the aspect/term discussed in each clause.\n\n"

        "==================== STRICT RULES ====================\n"
        "1️. DO NOT add, remove, translate, explain, or modify ANY words, symbols, or punctuation in the original sentence.\n"
        "   • Every clause must be a **continuous substring** of the original sentence.\n"
        "   • The output must cover **all parts of the sentence** — no content should be ignored or missing.\n"
        "2️. Only split the sentence where it makes sense semantically — typically around conjunctions ('and', 'but', 'while', 'although', etc.) "
        "or when the opinion changes.\n"
        "   •Do NOT split phrases that grammatically or logically belong to the same subject. "
        "   • If a descriptive phrase does not have a clear term in the sentence, keep it as a separate clause but leave Term blank."
        "3️. Keep the exact original wording and order in each clause. Do NOT reorder, paraphrase, or summarize.\n"
        "4️. Each clause must express a clear **opinion or evaluative meaning**, either explicit (e.g., 'dirty', 'perfect') or implicit "
        "(e.g., 'gave us many tips' implies helpfulness, 'helped us with departure' implies good service).\n"
        "5️. Do NOT separate adverbs (e.g., 'really', 'very', 'so', 'too', 'quite', 'extremely', 'absolutely', "
        "'rather', 'fairly', 'pretty', 'incredibly', 'particularly', 'deeply', 'highly') from the words they modify.\n"
        "6️. Keep negative or limiting words such as 'nothing', 'none', 'nobody', 'no one', 'nowhere', 'never', "
        "'hardly', 'barely', 'scarcely', 'without', 'no', 'not' **inside the same clause** — they must not be removed or separated.\n"
        "7️. Identify the **TERM** being discussed in each clause.\n"
        "   • TERM: the main aspect or entity being described (e.g., 'staff', 'room', 'hotel').\n"
        "   • If no clear term appears, leave it blank.\n"
        "8️. Avoid creating meaningless or redundant clauses.\n"
        "9️. If multiple terms appear in the same clause, separate them with commas.\n"
        "10️. If a clause refers to the same entity as a previous one but does not repeat it explicitly, "
        "**propagate the term from the previous clause**.\n\n"

        "==================== COVERAGE REQUIREMENT ====================\n"
        " Every part of the original sentence must appear in at least one clause.\n"
        " Do NOT skip, shorten, or drop any meaningful phrase, even if it lacks an explicit sentiment word.\n"
        " Clauses that describe actions, experiences, or behaviors with clear positive/negative implications "
        "must be included (e.g., 'gave us many tips', 'helped us with departure').\n\n"

        "==================== OUTPUT FORMAT ====================\n"
        "Clause: <clause text> | Term: <term1,term2,...>\n\n"

        "==================== EXAMPLES ====================\n"
        "Input: The apartment was fully furnished, great facilities, everything was cleaned and well prepared.\n"
        "Output:\n"
        "Clause: The apartment was fully furnished | Term: apartment\n"
        "Clause: great facilities | Term: facilities\n"
        "Clause: everything was cleaned and well prepared | Term: room,facility\n\n"

        "Input: diny was really helpful, he gave us many tips and helped us with departure.\n"
        "Output:\n"
        "Clause: diny was really helpful | Term: staff\n"
        "Clause: he gave us many tips | Term: staff\n"
        "Clause: helped us with departure | Term: staff\n\n"

        "Input: i can definitely recommend it!.\n"
        "Output:\n"
        "Clause: i can definitely recommend it! | Term: \n\n"

        "==================== RESPONSE INSTRUCTION ====================\n"
        "Respond ONLY with the clauses and terms exactly in the format shown above.\n"
        "Do NOT include any explanation, reasoning, or commentary.\n"
        "Do NOT include quotation marks, markdown, or extra text.\n\n"

        f"Now process this sentence WITHOUT changing any words:\n{sentence}"
    )

    messages = [{"role": "user", "content": prompt}]
    # Gọi hàm chat được truyền từ app.py
    response = chat_func(messages, max_new_tokens=max_new_tokens).strip()
    # --- Làm sạch output ---
    result = []
    last_term = ""
    for line in response.split("\n"):
        line = line.strip()
        if not line:
            continue
        if "| Term:" in line:
            clause_text, term = line.split("| Term:")
            clause_text = clause_text.replace("Clause:", "").strip()
            term = term.strip()
            if term == "":
                term = last_term  # propagate term
            else:
                last_term = term
        else:
            clause_text = line
            term = last_term.replace("<|im_end|>", "").strip()
        result.append({"clause": clause_text, "term": term, "sentence_original": sentence})

    return result