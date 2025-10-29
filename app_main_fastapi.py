# Tên file: app_fastapi.py
# (File này thay thế cho app(2).py và pipeline_orchestrator.py
#  để chạy một API backend)

import uvicorn
import pandas as pd
import os
import torch
import re
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig,
    AutoModelForSequenceClassification,
    pipeline
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# =============================================================================
# BƯỚC 1: IMPORT CÁC HÀM "WORKER" TỪ CÁC FILE LIB
# =============================================================================
try:
    # Import các hàm worker (chỉ chứa logic, không tải model)
    from split_clause_lib import run_split_and_term, chat_llm
    from Term_Opinion_lib import run_extract_opinions
    from Extract_Category import extract_categories
    from Extract_Polarity_lib import run_detect_polarity
except ImportError as e:
    print(f"Lỗi Import: {e}")
    print("Hãy đảm bảo các file _lib.py nằm cùng thư mục.")
    exit()

# =============================================================================
# BƯỚC 2: CÁC HÀM TẢI MÔ HÌNH (Loại bỏ code UI)
# =============================================================================

def load_qwen_model():
    """Tải mô hình Qwen-LLM (1.5B) cho (Clause, Term, Opinion)."""
    print("Đang tải mô hình Qwen LLM (4-bit)...")
    model_id = "Qwen/Qwen2-1.5B-Instruct" # Dùng model 1.5B để tránh lỗi VRAM
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    ) #
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=64, lora_alpha=16, target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1, bias="none", task_type="CAUSAL_LM"
    ) #
    model = get_peft_model(model, lora_config)
    print("Tải xong mô hình Qwen LLM.")
    return model, tokenizer

def load_category_model():
    """Tải mô hình RoBERTa đã fine-tune cho (Category)."""
    print("Đang tải mô hình Category (RoBERTa)...")
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, "roberta_lora_goal")
        
        model_cat = AutoModelForSequenceClassification.from_pretrained(model_path)
        tokenizer_cat = AutoTokenizer.from_pretrained(model_path)
        
        id2label = {
            0: "Amenity", 1: "Branding", 2: "Experience",
            3: "Facility", 4: "Loyalty", 5: "Service"
        } #
        print("Tải xong mô hình Category.")
        return model_cat, tokenizer_cat, id2label
    except Exception as e:
        print(f"Lỗi: Không thể tải mô hình Category từ '{model_path}'.")
        print(f"Bạn đã chạy script 'Fine_Tune_RoBertaBase.py' chưa? Lỗi chi tiết: {e}")
        return None, None, None

def load_polarity_model():
    """Tải mô hình DeBERTa cho (Polarity)."""
    print("Đang tải mô hình Polarity (DeBERTa)...")
    polarity_classifier = pipeline(
        "text-classification",
        model="yangheng/deberta-v3-base-absa-v1.1",
        top_k=None, truncation=True,
        device=0 if torch.cuda.is_available() else -1
    ) #
    print("Tải xong mô hình Polarity.")
    return polarity_classifier

# =============================================================================
# BƯỚC 3: HÀM PIPELINE (Loại bỏ code UI)
# =============================================================================

def run_full_pipeline(sentence: str, models: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Hàm này chạy toàn bộ pipeline 5 bước và trả về List[Dict].
    """
    # Trích xuất các mô hình từ biến toàn cục
    qwen_model = models["qwen_model"]
    qwen_tokenizer = models["qwen_tokenizer"]
    cat_model = models["cat_model"]
    cat_tokenizer = models["cat_tokenizer"]
    cat_id2label = models["cat_id2label"]
    polarity_classifier = models["polarity_classifier"]
    device = models["device"]

    # Tạo hàm chat_func (logic từ app(2).py)
    def chat_func(messages, max_new_tokens):
        return chat_llm(qwen_model, qwen_tokenizer, messages, max_new_tokens)

    # Chạy 4 bước pipeline
    print("Bước 1/4: Tách Clause và Term...")
    clauses_terms = run_split_and_term(sentence, chat_func)
    
    print("Bước 2/4: Trích xuất Opinion...")
    clauses_opinions = run_extract_opinions(clauses_terms, chat_func)
    
    print("Bước 3/4: Dự đoán Category...")
    clauses_categories = extract_categories(
        clauses_opinions, cat_model, cat_tokenizer, cat_id2label, device
    )
    
    print("Bước 4/4: Phát hiện Polarity...")
    final_results = run_detect_polarity(clauses_categories, polarity_classifier)
    
    print("Hoàn thành pipeline!")
    
    # Trả về List[Dict] thay vì DataFrame
    return final_results

# =============================================================================
# BƯỚC 4: KHỞI TẠO FASTAPI VÀ CÁC BIẾN TOÀN CỤC
# =============================================================================

# Khởi tạo app FastAPI
app = FastAPI(
    title="ABSA Pipeline API",
    description="API để chạy pipeline 5 bước (Clause, Term, Opinion, Category, Polarity)",
    version="1.0.0"
)

# Biến toàn cục để giữ các mô hình đã tải
MODELS = {}

# Định nghĩa Pydantic models (kiểu dữ liệu cho input/output)
class SentenceRequest(BaseModel):
    sentence: str = Field(
        ..., 
        example="The food was great and the staff was very friendly, but the room was a bit small."
    )

class ClauseResponse(BaseModel):
    clause: str
    term: str
    opinion: str
    category: str
    polarity: str
    polarity_score: float
    sentence_original: str

class PipelineResponse(BaseModel):
    results: List[ClauseResponse]

# =============================================================================
# BƯỚC 5: SỰ KIỆN STARTUP (TẢI MÔ HÌNH KHI KHỞI ĐỘNG)
# =============================================================================

@app.on_event("startup")
def load_models_on_startup():
    """Tải tất cả 3 mô hình khi FastAPI khởi động."""
    global MODELS
    try:
        qwen_model, qwen_tokenizer = load_qwen_model()
        cat_model, cat_tokenizer, cat_id2label = load_category_model()
        polarity_classifier = load_polarity_model()
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if cat_model:
            cat_model.to(device).eval()
            
        MODELS = {
            "qwen_model": qwen_model,
            "qwen_tokenizer": qwen_tokenizer,
            "cat_model": cat_model,
            "cat_tokenizer": cat_tokenizer,
            "cat_id2label": cat_id2label,
            "polarity_classifier": polarity_classifier,
            "device": device
        }
        print("--- TẤT CẢ MÔ HÌNH ĐÃ SẴN SÀNG ---")
    except Exception as e:
        print(f"Lỗi nghiêm trọng khi tải mô hình: {e}")
        MODELS = None # Đánh dấu là tải lỗi

# =============================================================================
# BƯỚC 6: TẠO API ENDPOINT
# =============================================================================

@app.post("/analyze", response_model=PipelineResponse)
def analyze_sentence(request: SentenceRequest):
    """
    Endpoint chính để phân tích một câu.
    """
    if not MODELS or not all(MODELS.values()):
        raise HTTPException(status_code=503, detail="Models are not loaded or failed to load. Please check the server logs.")
    
    try:
        # Chạy pipeline
        # FastAPI sẽ tự động chạy hàm sync này trong một thread pool
        results_list = run_full_pipeline(request.sentence, MODELS)
        
        # Trả về kết quả
        return PipelineResponse(results=results_list)
        
    except Exception as e:
        print(f"Lỗi pipeline: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/", include_in_schema=False)
def root():
    return {"message": "ABSA Pipeline API is running. Truy cập /docs để xem tài liệu API."}

# =============================================================================
# BƯỚC 7: CHẠY APP VỚI UVICORN
# =============================================================================

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)