# Tên file: app(2).py
# (File chính đã được refactor - Chỉ chứa UI và gọi hàm)

import streamlit as st

# =============================================================================
# BƯỚC 1: IMPORT CÁC HÀM ĐIỀU PHỐI
# =============================================================================
try:
    from pipeline_orchestrator import load_all_models, run_full_pipeline
except ImportError as e:
    st.error(f"Lỗi: Không tìm thấy file 'pipeline_orchestrator.py'. {e}")
    st.stop()

# =============================================================================
# BƯỚC 2: GIAO DIỆN WEB APP
# =============================================================================

st.set_page_config(page_title="ABSA Pipeline", layout="wide")
st.title("🚀 Pipeline Phân tích Cảm xúc (ABSA) - Modular")
st.markdown("Nhập một câu đánh giá (review) và ứng dụng sẽ chạy pipeline 5 bước để trích xuất: **Clause, Term, Opinion, Category, Polarity**.")

# --- Tải tất cả mô hình ---
# (Chỉ gọi 1 hàm duy nhất)
models = load_all_models()

# --- Khu vực nhập liệu ---
default_sentence = "The food was great and the staff was very friendly, but the room was a bit small."
sentence = st.text_area("Nhập câu đánh giá của bạn:", default_sentence, height=100)

if st.button("Phân tích", type="primary"):
    if not sentence.strip():
        st.warning("Vui lòng nhập một câu.")
    else:
        try:
            # --- Chạy Pipeline ---
            # (Chỉ gọi 1 hàm duy nhất)
            results_df = run_full_pipeline(sentence, models)
            
            # --- Hiển thị kết quả ---
            if not results_df.empty:
                st.subheader("Kết quả Phân tích Pipeline")
                st.dataframe(results_df, use_container_width=True)
            
        except Exception as e:
            st.error(f"Đã xảy ra lỗi trong quá trình xử lý pipeline: {e}")
            st.exception(e)