import streamlit as st
import os
import pandas as pd
import torch
import gc
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig,
    AutoModelForSequenceClassification,
    pipeline
)
# Thêm thư viện để kiểm tra Flash Attention
from transformers.utils import is_flash_attn_2_available
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from typing import List, Dict, Any

# Import cấu hình
try:
    from config import (
        QWEN_MODEL_ID, MAX_GPU_MEMORY, CATEGORY_BATCH_SIZE, POLARITY_BATCH_SIZE,
        MAX_INPUT_LENGTH, MAX_NEW_TOKENS_SPLIT, MAX_NEW_TOKENS_OPINION,
        USE_4BIT_QUANTIZATION, UNLOAD_QWEN_AFTER_LLM, DEBUG_MODE
    )
except ImportError:
    # Giá trị mặc định nếu không tìm thấy config.py
    QWEN_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
    MAX_GPU_MEMORY = "2.0GB"
    CATEGORY_BATCH_SIZE = 4
    POLARITY_BATCH_SIZE = 4
    MAX_INPUT_LENGTH = 128
    MAX_NEW_TOKENS_SPLIT = 50
    MAX_NEW_TOKENS_OPINION = 10
    USE_4BIT_QUANTIZATION = False
    UNLOAD_QWEN_AFTER_LLM = False
    DEBUG_MODE = False

# =============================================================================
# BƯỚC 1: IMPORT CÁC HÀM "WORKER" TỪ CÁC FILE LIB
# =============================================================================
try:
    # Vẫn import các hàm gốc, nhưng chúng ta sẽ thay thế 2 hàm cuối
    # bằng phiên bản batch của riêng mình
    from split_clause_lib import run_split_and_term, chat_llm
    from Term_Opinion_lib import run_extract_opinions
    
    # Chúng ta sẽ không dùng 2 hàm này nữa, nhưng giữ import để tránh lỗi
    # nếu các file lib của bạn có liên kết chéo
    from Extract_Category import extract_categories
    from Extract_Polarity_lib import run_detect_polarity
except ImportError as e:
    st.error(f"Lỗi Import (pipeline_ABSA.py): {e}")
    st.stop()

# =============================================================================
# BƯỚC 2: CÁC HÀM TẢI MÔ HÌNH (CACHE TỐI ƯU)
# =============================================================================

# Xác định kiểu dữ liệu 16-bit tối ưu
dtype_16bit = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

# === TỐI ƯU: KIỂM TRA FLASH ATTENTION 2 ===
# Tự động dùng "flash_attention_2" nếu có, nếu không thì dùng "sdpa" (Pytorch 2.0+)
# Giúp mô hình chạy nhanh hơn và tiết kiệm VRAM
ATTN_IMPL = "flash_attention_2" if is_flash_attn_2_available() else "sdpa"

@st.cache_resource(show_spinner=True)
def load_qwen_model():
    model_id = QWEN_MODEL_ID
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        # Giới hạn context để tiết kiệm VRAM
        try:
            tokenizer.model_max_length = min(getattr(tokenizer, "model_max_length", 4096) or 4096, MAX_INPUT_LENGTH)
        except Exception:
            pass

        # Kiểm tra xem có GPU không
        has_cuda = torch.cuda.is_available()
        
        # Thử tải với 4-bit quantization nếu có GPU và config cho phép
        if has_cuda and USE_4BIT_QUANTIZATION:
            try:            
                # Cấu hình 4-bit quantization với FP32 CPU offload
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    llm_int8_enable_fp32_cpu_offload=True
                )
                
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    quantization_config=bnb_config,
                    device_map="auto",
                    low_cpu_mem_usage=True,
                    torch_dtype=torch.float16,
                    attn_implementation=ATTN_IMPL,
                    max_memory={0: MAX_GPU_MEMORY}  # Giới hạn sử dụng VRAM để tránh OOM
                )
                
                # Bật gradient checkpointing để tiết kiệm VRAM (nếu hỗ trợ)
                if hasattr(model, 'gradient_checkpointing_enable'):
                    model.gradient_checkpointing_enable()
                
                return model, tokenizer
                
            except Exception as quant_error:
                error_msg = str(quant_error)
                
                # Nếu lỗi liên quan đến _is_hf_initialized, thử 8-bit
                if "_is_hf_initialized" in error_msg or "Params4bit" in error_msg:
                    try:                    
                        bnb_config_8bit = BitsAndBytesConfig(
                            load_in_8bit=True,
                            llm_int8_enable_fp32_cpu_offload=True
                        )
                        
                        model = AutoModelForCausalLM.from_pretrained(
                            model_id,
                            quantization_config=bnb_config_8bit,
                            device_map="auto",
                            low_cpu_mem_usage=True,
                            torch_dtype=torch.float16,
                            attn_implementation=ATTN_IMPL,
                            max_memory={0: MAX_GPU_MEMORY}
                        )

                        return model, tokenizer
                        
                    except Exception as quant8_error:
                        pass
        
        # Cấu hình tải mô hình
        model_kwargs = {
            "torch_dtype": dtype_16bit,
            "low_cpu_mem_usage": True,
            "attn_implementation": ATTN_IMPL
        }
        
        if has_cuda:
            model_kwargs["device_map"] = "auto"
            model_kwargs["max_memory"] = {0: MAX_GPU_MEMORY}
        else:
            model_kwargs["device_map"] = "cpu"
        
        model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)

        return model, tokenizer

    except Exception as e:
        # Thử fallback cuối cùng: CPU với float32
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float32,
                device_map="cpu",
                low_cpu_mem_usage=True
            )

            return model, tokenizer
        except Exception as final_error:
            return None, None

@st.cache_resource
def load_category_model():   
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, "roberta_lora_goal")
        
        model_cat = AutoModelForSequenceClassification.from_pretrained(
            model_path, 
            torch_dtype=dtype_16bit,
            # === TỐI ƯU: THÊM ATTENTION IMPLEMENTATION ===
            attn_implementation=ATTN_IMPL
        )
        
        tokenizer_cat = AutoTokenizer.from_pretrained(model_path)
        
        id2label = {
            0: "Amenity", 1: "Branding", 2: "Experience",
            3: "Facility", 4: "Loyalty", 5: "Service"
        }
        
        return model_cat, tokenizer_cat, id2label
    
    except Exception as e:
        return None, None, None

@st.cache_resource
def load_polarity_model():
    model_kwargs = {
        "torch_dtype": dtype_16bit,
        "attn_implementation": "eager" # <-- THAY ĐỔI CHÍNH: Đổi từ ATTN_IMPL sang "eager"
    }
    
    try:
        polarity_classifier = pipeline(
            "text-classification",
            model="yangheng/deberta-v3-base-absa-v1.1",
            model_kwargs=model_kwargs,
            top_k=None,
            truncation=True,
            device=0 if torch.cuda.is_available() else -1
        ) 
    except Exception as e:
        return None
    return polarity_classifier

# =============================================================================
# BƯỚC 3: CÁC HÀM PIPELINE ĐÃ TỐI ƯU (BATCHED)
# =============================================================================

# === TỐI ƯU: HÀM NÀY THAY THẾ CHO Extract_Category.py ===
# Nó xử lý tất cả các mệnh đề trong một batch thay vì lặp qua từng câu
def _run_categories_batched(
    clauses: List[Dict[str, Any]],
    model_cat: Any,
    tokenizer_cat: Any,
    id2label: Dict[int, str],
    device: torch.device,
    batch_size: int = 32 # Có thể điều chỉnh batch size
) -> List[Dict[str, Any]]:
    
    if model_cat.training:
        model_cat.eval()
    
    # Lấy danh sách các mệnh đề
    clause_texts = [str(c.get("clause", "")).strip() for c in clauses]
    # Chuẩn bị list kết quả
    results_categories = ["Unknown"] * len(clause_texts)

    # Lọc ra các câu rỗng, xử lý riêng
    non_empty_indices = [i for i, text in enumerate(clause_texts) if text]
    non_empty_texts = [clause_texts[i] for i in non_empty_indices]

    if not non_empty_texts: # Nếu không có gì để xử lý
        for c in clauses:
            c["category"] = "Unknown"
        return clauses

    try:
        with torch.no_grad():
            # Xử lý theo batch
            for i in range(0, len(non_empty_texts), batch_size):
                batch_texts = non_empty_texts[i : i + batch_size]
                
                inputs = tokenizer_cat(
                    batch_texts,
                    return_tensors="pt",
                    truncation=True,
                    padding=True, # Pad theo batch, không phải max_length cố định
                    max_length=tokenizer_cat.model_max_length,
                ).to(device)

                outputs = model_cat(**inputs)
                pred_ids = torch.argmax(outputs.logits, dim=1).cpu().tolist()
                
                # Map dự đoán trở lại list kết quả
                batch_indices_original = non_empty_indices[i : i + batch_size]
                for j, pred_id in enumerate(pred_ids):
                    original_index = batch_indices_original[j]
                    results_categories[original_index] = id2label.get(pred_id, "Unknown")
                    
    except Exception as e:
        # Đánh dấu lỗi nếu có
        for idx in non_empty_indices:
             if results_categories[idx] == "Unknown":
                results_categories[idx] = "Error"

    # Gán kết quả category trở lại dict
    for i, c in enumerate(clauses):
        c["category"] = results_categories[i]
        
    return clauses

# === TỐI ƯU: HÀM NÀY THAY THẾ CHO Extract_Polarity_lib.py ===
# Nó sử dụng khả năng xử lý batch gốc của HfPipeline
def _run_polarity_batched(
    clauses: List[Dict[str, Any]], 
    polarity_classifier: Any,
    batch_size: int = 32 # pipeline cũng chấp nhận batch_size
) -> List[Dict[str, Any]]:
    
    clause_texts = [str(c.get("clause", "")).strip() for c in clauses]
    
    # Chuẩn bị danh sách kết quả cuối cùng theo đúng thứ tự
    final_results = [None] * len(clauses)
    
    non_empty_texts = []
    # Map từ index của batch về index của list gốc
    indices_map = [] 
    
    for i, text in enumerate(clause_texts):
        if not text:
            # Ghi nhận kết quả cho câu rỗng
            item = clauses[i].copy()
            item["polarity"] = "Neutral"
            item["polarity_score"] = 0.0
            final_results[i] = item
        else:
            non_empty_texts.append(text)
            indices_map.append(i) # (index_batch_0 -> index_goc_1, ...)

    if not non_empty_texts: # Nếu tất cả đều rỗng
         return final_results

    try:
        # Chạy pipeline ở chế độ batch (nhanh hơn nhiều)
        pipeline_outputs = polarity_classifier(
            non_empty_texts, 
            batch_size=batch_size
        )
        
        # pipeline_outputs là list[list[dict]]
        for i, res_list in enumerate(pipeline_outputs):
            original_clause_index = indices_map[i]
            item = clauses[original_clause_index].copy()
            
            # res_list là [ {'label': 'Positive', 'score': 0.9}, {'label': 'Negative', ...} ]
            if isinstance(res_list, list): 
                 top = max(res_list, key=lambda x: x["score"])
                 item["polarity"] = top["label"].capitalize()
                 item["polarity_score"] = round(top["score"], 4)
            else:
                item["polarity"] = "Neutral"
                item["polarity_score"] = 0.0
            
            final_results[original_clause_index] = item

    except Exception as e:
        st.error(f"Lỗi khi xử lý polarity (batch): {e}")
        # Ghi nhận lỗi cho các câu đang xử lý
        for i in indices_map:
            if final_results[i] is None: # Nếu chưa được gán
                item = clauses[i].copy()
                item["polarity"] = "Error"
                item["polarity_score"] = 0.0
                final_results[i] = item

    return final_results

# =============================================================================
# BƯỚC 4: HÀM ĐIỀU PHỐI (GỌI TỪ APP CHÍNH)
# =============================================================================

def load_all_models():
    """
    Hàm này được app_main.py gọi để tải và cache tất cả 3 mô hình.
    """
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
            # Chuyển mô hình 16-bit lên GPU và đặt ở chế độ eval
            cat_model.to(device).eval() 
            
    except Exception as e:
        pass
    
    models = {
        "qwen_model": qwen_model, "qwen_tokenizer": qwen_tokenizer,
        "cat_model": cat_model, "cat_tokenizer": cat_tokenizer,
        "cat_id2label": cat_id2label,
        "polarity_classifier": polarity_classifier,
        "device": device
    }
    return models

def run_full_pipeline(sentence: str, models: Dict[str, Any]) -> pd.DataFrame:
    """
    Hàm này được app_main.py gọi để chạy toàn bộ pipeline 5 bước.
    (ĐÃ ĐƯỢC CẬP NHẬT ĐỂ GỌI CÁC HÀM BATCHED)
    """
    qwen_model = models["qwen_model"]
    qwen_tokenizer = models["qwen_tokenizer"]
    cat_model = models["cat_model"]
    cat_tokenizer = models["cat_tokenizer"]
    cat_id2label = models["cat_id2label"]
    polarity_classifier = models["polarity_classifier"]
    device = models["device"]
    
    def chat_func(messages, max_new_tokens):
        return chat_llm(qwen_model, qwen_tokenizer, messages, max_new_tokens)

    # Bước 1 và 2: Vẫn dùng LLM (không thay đổi)
    with st.spinner("Bước 1/4: Tách Clause và Term (sử dụng Qwen LLM)..."):
        clauses_terms = run_split_and_term(sentence, chat_func)
    
    with st.spinner("Bước 2/4: Trích xuất Opinion (sử dụng Qwen LLM)..."):
        clauses_opinions = run_extract_opinions(clauses_terms, chat_func)
    
    # === QUAN TRỌNG: GIẢI PHÓNG MÔ HÌNH QWEN ĐỂ TIẾT KIỆM VRAM ===
    if UNLOAD_QWEN_AFTER_LLM:
        if qwen_model is not None:
            try:
                # Xóa tất cả các tham chiếu đến mô hình
                del qwen_model
                models["qwen_model"] = None
                
                # Xóa cache và garbage collection
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()  # Đợi tất cả các thao tác GPU hoàn thành
                gc.collect()
            except Exception as e:
                pass
    else:
        # Vẫn xóa cache nhưng không unload mô hình
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    with st.spinner("Bước 3/4: Dự đoán Category (sử dụng RoBERTa - Chế độ Batch)..."):
        clauses_categories = _run_categories_batched(
            clauses_opinions, cat_model, cat_tokenizer, cat_id2label, device,
            batch_size=CATEGORY_BATCH_SIZE
        )
    
    # Giải phóng bộ nhớ sau Category
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    # === TỐI ƯU: GỌI HÀM BATCHED ===
    final_results = _run_polarity_batched(
        clauses_categories, polarity_classifier,
        batch_size=POLARITY_BATCH_SIZE
    )

    # === GIẢI PHÓNG BỘ NHỚ ===
    # (Giữ nguyên, rất tốt)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    # Định dạng kết quả (giữ nguyên)
    if not final_results:
        return pd.DataFrame()
        
    df = pd.DataFrame(final_results)
    columns_order = [
        "clause", "term", "opinion", "category", "polarity", 
        "polarity_score", "sentence_original"
    ]
    # Lọc ra các cột thực sự có trong DF để tránh lỗi
    final_columns = [col for col in columns_order if col in df.columns]
    
    if not final_columns:
         return pd.DataFrame(final_results) # Trả về DF thô nếu không khớp cột

    return df[final_columns]