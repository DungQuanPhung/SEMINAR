# Tên file: pipeline_orchestrator.py
# (File này chứa toàn bộ logic tải model và chạy pipeline
#  được copy từ app(2).py)

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
# (Đây là các file lib bạn đã cung cấp)
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
# (Các hàm này được chuyển từ app(2).py sang đây)
# =============================================================================

@st.cache_resource
def load_qwen_model():
    """Tải mô hình Qwen-LLM cho (Clause, Term, Opinion)."""
    st.info("Đang tải mô hình Qwen LLM (4-bit)... (Chỉ tải 1 lần)")
    
    model_id = "Qwen/Qwen3-4B-Instruct-2507" #

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
        
        model_cat = AutoModelForSequenceClassification.from_pretrained(model_path)
        tokenizer_cat = AutoTokenizer.from_pretrained(model_path)
        
        # Ánh xạ ID sang Label (đã sửa)
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
    
    polarity_classifier = pipeline(
        "text-classification",
        model="yangheng/deberta-v3-base-absa-v1.1",
        top_k=None,
        truncation=True,
        device=0 if torch.cuda.is_available() else -1
    ) #
    st.success("Tải xong mô hình Polarity.")
    return polarity_classifier

# =============================================================================
# BƯỚC 3: HÀM ĐIỀU PHỐI (GỌI TỪ APP CHÍNH)
# =============================================================================

# (Bên trong file pipeline_orchestrator.py)

def load_all_models():
    """
    Hàm này được app(2).py gọi để tải và cache tất cả 3 mô hình.
    """
    
    # --- SỬA CODE Ở ĐÂY ---
    # Khởi tạo tất cả các biến là None để tránh NameError
    qwen_model, qwen_tokenizer = None, None
    cat_model, cat_tokenizer, cat_id2label = None, None, None
    polarity_classifier = None
    device = None
    # --- KẾT THÚC SỬA ĐỔI ---

    try:
        qwen_model, qwen_tokenizer = load_qwen_model()
        cat_model, cat_tokenizer, cat_id2label = load_category_model()
        polarity_classifier = load_polarity_model()
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if cat_model:
            cat_model.to(device).eval()
            
    except Exception as e:
        st.error(f"Lỗi nghiêm trọng trong quá trình tải mô hình: {e}")
        # Vẫn tiếp tục để trả về dict, app(2).py sẽ xử lý
    
    # Đóng gói tất cả mô hình vào một dict để trả về
    models = {
        "qwen_model": qwen_model,
        "qwen_tokenizer": qwen_tokenizer,
        "cat_model": cat_model,
        "cat_tokenizer": cat_tokenizer, # Dòng này sẽ không còn lỗi
        "cat_id2label": cat_id2label,
        "polarity_classifier": polarity_classifier,
        "device": device
    }
    return models

def run_full_pipeline(sentence: str, models: Dict[str, Any]) -> pd.DataFrame:
    """
    Hàm này được app(2).py gọi để chạy toàn bộ pipeline 5 bước.
    (Chứa logic xử lý được chuyển từ app(2).py sang)
    """
    # Trích xuất các mô hình đã tải
    qwen_model = models["qwen_model"]
    qwen_tokenizer = models["qwen_tokenizer"]
    cat_model = models["cat_model"]
    cat_tokenizer = models["cat_tokenizer"]
    cat_id2label = models["cat_id2label"]
    polarity_classifier = models["polarity_classifier"]
    device = models["device"]
    
    # Kiểm tra nếu mô hình load thất bại
    if not all([qwen_model, cat_model, polarity_classifier]):
        st.error("Một hoặc nhiều mô hình chưa được tải. Vui lòng kiểm tra lỗi bên trên.")
        return pd.DataFrame()

    # Tạo hàm chat_func (logic từ app(2).py)
    def chat_func(messages, max_new_tokens):
        return chat_llm(qwen_model, qwen_tokenizer, messages, max_new_tokens)

    # Chạy 4 bước pipeline (gọi các hàm worker)
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
    
    # Định dạng kết quả (logic từ app(2).py)
    df = pd.DataFrame(final_results)
    columns_order = [
        "clause", "term", "opinion", "category", "polarity", 
        "polarity_score", "sentence_original"
    ]
    final_columns = [col for col in columns_order if col in df.columns]
    return df[final_columns]