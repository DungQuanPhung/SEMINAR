# Tên file: app_gradio.py
# (File này thay thế cho app(2).py và pipeline_orchestrator.py)

import gradio as gr
import pandas as pd
import os
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

# =Lỗi Import: Không tìm thấy file thư viện. Hãy đảm bảo các file 'split_clause_lib.py', 'Term_Opinion_lib.py', 'Extract_Category.py', 'Extract_Polarity_lib.py' nằm cùng thư mục.
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
# BƯỚC 2: TẢI TẤT CẢ MÔ HÌNH (TẢI TOÀN CỤC - CHỈ 1 LẦN)
# (Đây là logic từ pipeline_orchestrator.py, nhưng đã loại bỏ
#  tất cả code của Streamlit)
# =============================================================================

def load_qwen_model():
    print("Đang tải mô hình Qwen LLM (4-bit)...")
    model_id = "Qwen/Qwen3-4B-Instruct-2507"
    
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

# --- TẢI MÔ HÌNH NGAY KHI APP KHỞI ĐỘNG ---
MODELS = {}
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
# BƯỚC 3: HÀM PIPELINE (ĐƯỢC GRADIO GỌI MỖI LẦN CLICK)
# =============================================================================

def run_pipeline_gradio(sentence: str, progress=gr.Progress(track_tqdm=True)):
    """
    Hàm này được Gradio gọi để chạy toàn bộ pipeline 5 bước.
    Nó sử dụng các mô hình đã được tải ở BƯỚC 2.
    """
    if not MODELS or not all(MODELS.values()):
        raise gr.Error("Mô hình chưa được tải thành công. Vui lòng kiểm tra console.")
    
    try:
        # Trích xuất các mô hình từ biến toàn cục
        qwen_model = MODELS["qwen_model"]
        qwen_tokenizer = MODELS["qwen_tokenizer"]
        cat_model = MODELS["cat_model"]
        cat_tokenizer = MODELS["cat_tokenizer"]
        cat_id2label = MODELS["cat_id2label"]
        polarity_classifier = MODELS["polarity_classifier"]
        device = MODELS["device"]

        # Tạo hàm chat_func (logic từ app(2).py)
        def chat_func(messages, max_new_tokens):
            return chat_llm(qwen_model, qwen_tokenizer, messages, max_new_tokens)

        # Chạy 4 bước pipeline
        progress(0.25, desc="Bước 1/4: Tách Clause và Term (Qwen LLM)...")
        clauses_terms = run_split_and_term(sentence, chat_func)
        
        progress(0.50, desc="Bước 2/4: Trích xuất Opinion (Qwen LLM)...")
        clauses_opinions = run_extract_opinions(clauses_terms, chat_func)
        
        progress(0.75, desc="Bước 3/4: Dự đoán Category (RoBERTa)...")
        clauses_categories = extract_categories(
            clauses_opinions, cat_model, cat_tokenizer, cat_id2label, device
        )
        
        progress(0.90, desc="Bước 4/4: Phát hiện Polarity (DeBERTa)...")
        final_results = run_detect_polarity(clauses_categories, polarity_classifier)
        
        # Định dạng kết quả
        df = pd.DataFrame(final_results)
        columns_order = [
            "clause", "term", "opinion", "category", "polarity", 
            "polarity_score", "sentence_original"
        ]
        final_columns = [col for col in columns_order if col in df.columns]
        return df[final_columns]
    
    except Exception as e:
        print(f"Lỗi pipeline: {e}")
        raise gr.Error(f"Đã xảy ra lỗi trong quá trình xử lý: {e}")

# =============================================================================
# BƯỚC 4: XÂY DỰNG GIAO DIỆN GRADIO
# =============================================================================

# Lấy ví dụ từ file Colab của bạn
example_sentences = [
    "The food was great and the staff was very friendly, but the room was a bit small.",
    "Quy trình check-in rất suôn sẻ và nhân viên vô cùng nhiệt tình. Phòng ốc của chúng tôi rộng rãi và chiếc giường rất thoải mái, mặc dù wifi trong phòng lại chậm một cách khủng khiếp. Chúng tôi rất thích hồ bơi, nhưng thiết bị phòng gym có vẻ đã khá cũ và lỗi thời. Dù là thành viên thân thiết, chúng tôi hơi thất vọng vì không được nâng hạng phòng, nhưng nhìn chung trải nghiệm tổng thể rất tuyệt vời và chắc chắn chúng tôi sẽ quay lại.",
    "Nhà hàng trong khách sạn có đồ ăn rất ngon, nhưng dịch vụ tại bàn lại chậm. Quầy bar trên tầng thượng có tầm nhìn tuyệt đẹp, đó thực sự là một trải nghiệm đáng nhớ. Tuy nhiên, khu vực sảnh chờ (lobby) lại khá ồn ào. Thương hiệu này luôn làm tôi hài lòng về sự sạch sẽ."
]

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🚀 Pipeline Phân tích Cảm xúc (ABSA)
        Nhập một câu đánh giá (review) và ứng dụng sẽ chạy pipeline 5 bước để trích xuất: **Clause, Term, Opinion, Category, Polarity**.
        """
    )
    
    with gr.Column():
        text_input = gr.Textbox(
            lines=5,
            label="Nhập câu đánh giá của bạn:",
            placeholder="Ví dụ: The food was great and the staff was very friendly..."
        )
        analyze_button = gr.Button("Phân tích (Run Pipeline)", variant="primary")
        
        gr.Markdown("---")
        
        output_df = gr.DataFrame(
            label="Kết quả Phân tích Pipeline",
            wrap=True
        )
        
        gr.Examples(
            examples=example_sentences,
            inputs=text_input
        )
        
    # Liên kết nút bấm với hàm xử lý
    analyze_button.click(
        fn=run_pipeline_gradio,
        inputs=text_input,
        outputs=output_df
    )

# Chạy ứng dụng
if __name__ == "__main__":
    demo.queue().launch(share=True)