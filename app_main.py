import streamlit as st
import pandas as pd

# =============================================================================
# BƯỚC 1: IMPORT CÁC HÀM ĐIỀU PHỐI
# =============================================================================
try:
    # Import 2 hàm chính từ file điều phối
    from pipeline_ABSA import load_all_models, run_full_pipeline
except ImportError as e:
    st.error(f"Lỗi: Không tìm thấy file 'pipeline_orchestrator.py'. {e}")
    st.stop()

# =============================================================================
# BƯỚC 2: GIAO DIỆN WEB APP
# =============================================================================

st.set_page_config(page_title="ABSA Pipeline", layout="wide")
st.title("🚀 Pipeline Phân tích Cảm xúc (ABSA)")
st.markdown("Nhập một câu đánh giá (review) và ứng dụng sẽ chạy pipeline 5 bước để trích xuất: **Clause, Term, Opinion, Category, Polarity**.")

# --- Tải tất cả mô hình ---
# (Chỉ gọi 1 hàm duy nhất để tải và cache)
models = load_all_models()

# --- Khu vực nhập liệu ---
default_sentence = "The food was great and the staff was very friendly, but the room was a bit small."
sentence = st.text_area("Nhập câu đánh giá của bạn:", default_sentence, height=100)

if st.button("Phân tích", type="primary"):
    if not sentence.strip():
        st.warning("Vui lòng nhập một câu.")
    # Kiểm tra xem mô hình đã tải thành công chưa (không phải là None)
    elif not all(models.values()):
        st.error("Một hoặc nhiều mô hình chưa được tải. Vui lòng kiểm tra lỗi bên trên khi khởi động app.")
    else:
        try:
            # --- Chạy Pipeline ---
            # (Chỉ gọi 1 hàm duy nhất để chạy toàn bộ 4 bước)
            results_df = run_full_pipeline(sentence, models)
            
            # --- Hiển thị kết quả ---
            if not results_df.empty:
                st.subheader("Kết quả Phân tích Pipeline")
                st.dataframe(results_df, use_container_width=True)
            
        except Exception as e:
            st.error(f"Đã xảy ra lỗi trong quá trình xử lý pipeline: {e}")
            st.exception(e)