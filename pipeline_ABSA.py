import streamlit as st
import os
import pandas as pd
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig,
    AutoModelForSequenceClassification,
    pipeline
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from typing import List, Dict, Any

# =============================================================================
# BƯỚC 1: IMPORT CÁC HÀM "WORKER" TỪ CÁC FILE LIB
# =============================================================================
try:
    from split_clause_lib import run_split_and_term, chat_llm
    from Term_Opinion_lib import run_extract_opinions
    from Extract_Category import extract_categories
    from Extract_Polarity_lib import run_detect_polarity
except ImportError as e:
    st.error(f"Lỗi Import (pipeline_orchestrator.py): {e}")
    st.stop()

# =============================================================================
# BƯỚC 2: CÁC HÀM TẢI MÔ HÌNH (CACHE TỐI ƯU)
# (Logic tải mô hình từ Colab 
#  và app(2).py)
# =============================================================================

@st.cache_resource
def load_qwen_model():
    st.info("Đang tải mô hình Qwen LLM (4-bit)... (Chỉ tải 1 lần)")
    model_id = "Qwen/Qwen3-4B-Instruct-2507"

    # Cấu hình 4-bit QLoRA
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto"
    )

    # Chuẩn bị LoRA
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=64,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    
    st.success("Tải xong mô hình Qwen LLM.")
    return model, tokenizer

@st.cache_resource
def load_category_model():
    """Tải mô hình RoBERTa đã fine-tune cho (Category)."""
    st.info("Đang tải mô hình Category (RoBERTa)... (Chỉ tải 1 lần)")
    
    try:
        # Lấy đường dẫn tuyệt đối
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, "roberta_lora_goal")
        
        # Tải mô hình đã được merge đầy đủ (từ Fine_Tune_RoBertaBase.py)
        model_cat = AutoModelForSequenceClassification.from_pretrained(model_path)
        tokenizer_cat = AutoTokenizer.from_pretrained(model_path)
        
        # Ánh xạ ID sang Label (đã sửa từ lần trước)
        id2label = {
            0: "Amenity",
            1: "Branding",
            2: "Experience",
            3: "Facility",
            4: "Loyalty",
            5: "Service"
        }
        st.success("Tải xong mô hình Category.")
        
        return model_cat, tokenizer_cat, id2label
    
    except Exception as e:
        st.error(f"Lỗi: Không thể tải mô hình Category từ '{model_path}'.")
        st.error(f"Bạn đã chạy script 'Fine_Tune_RoBertaBase.py' chưa? Lỗi chi tiết: {e}")
        return None, None, None

@st.cache_resource
def load_polarity_model():
    """Tải mô hình DeBERTa cho (Polarity)."""
    st.info("Đang tải mô hình Polarity (DeBERTa)... (Chỉ tải 1 lần)")
    
    # Tải pipeline
    polarity_classifier = pipeline(
        "text-classification",
        model="yangheng/deberta-v3-base-absa-v1.1",
        top_k=None,
        truncation=True,
        device=0 if torch.cuda.is_available() else -1
    ) 
    st.success("Tải xong mô hình Polarity.")
    return polarity_classifier

# =============================================================================
# BƯỚC 3: HÀM ĐIỀU PHỐI (GỌI TỪ APP CHÍNH)
# =============================================================================

def load_all_models():
    """
    Hàm này được app_main.py gọi để tải và cache tất cả 3 mô hình.
    """
    
    # Khởi tạo tất cả các biến là None để tránh NameError (từ lần sửa lỗi trước)
    qwen_model, qwen_tokenizer = None, None
    cat_model, cat_tokenizer, cat_id2label = None, None, None
    polarity_classifier = None
    device = None

    try:
        qwen_model, qwen_tokenizer = load_qwen_model()
        cat_model, cat_tokenizer, cat_id2label = load_category_model()
        polarity_classifier = load_polarity_model()
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if cat_model:
            cat_model.to(device).eval()
            
    except Exception as e:
        st.error(f"Lỗi nghiêm trọng trong quá trình tải mô hình: {e}")
    
    # Đóng gói tất cả mô hình vào một dict để trả về
    models = {
        "qwen_model": qwen_model,
        "qwen_tokenizer": qwen_tokenizer,
        "cat_model": cat_model,
        "cat_tokenizer": cat_tokenizer,
        "cat_id2label": cat_id2label,
        "polarity_classifier": polarity_classifier,
        "device": device
    }
    return models

def run_full_pipeline(sentence: str, models: Dict[str, Any]) -> pd.DataFrame:
    """
    Hàm này được app_main.py gọi để chạy toàn bộ pipeline 5 bước.
    """
    # Trích xuất các mô hình đã tải
    qwen_model = models["qwen_model"]
    qwen_tokenizer = models["qwen_tokenizer"]
    cat_model = models["cat_model"]
    cat_tokenizer = models["cat_tokenizer"]
    cat_id2label = models["cat_id2label"]
    polarity_classifier = models["polarity_classifier"]
    device = models["device"]
    
    # Tạo hàm chat_func (đã sửa lỗi 'max_new_tokens')
    def chat_func(messages, max_new_tokens):
        return chat_llm(qwen_model, qwen_tokenizer, messages, max_new_tokens)

    # Chạy 4 bước pipeline (gọi các hàm worker từ lib)
    with st.spinner("Bước 1/4: Tách Clause và Term (sử dụng Qwen LLM)..."):
        clauses_terms = run_split_and_term(sentence, chat_func)
    
    with st.spinner("Bước 2/4: Trích xuất Opinion (sử dụng Qwen LLM)..."):
        clauses_opinions = run_extract_opinions(clauses_terms, chat_func)
    
    with st.spinner("Bước 3/4: Dự đoán Category (sử dụng RoBERTa)..."):
        clauses_categories = extract_categories(
            clauses_opinions, cat_model, cat_tokenizer, cat_id2label, device
        )
    
    st.info("Bước 4/4: Phát hiện Polarity (sử dụng DeBERTa)...")
    final_results = run_detect_polarity(clauses_categories, polarity_classifier)
    
    st.success("Hoàn thành pipeline!")
    
    # Định dạng kết quả
    df = pd.DataFrame(final_results)
    columns_order = [
        "clause", "term", "opinion", "category", "polarity", 
        "polarity_score", "sentence_original"
    ]
    final_columns = [col for col in columns_order if col in df.columns]
    return df[final_columns]