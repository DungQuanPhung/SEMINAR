import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# =============================================================================
# 1️⃣ TẢI HÀM XỬ LÝ PIPELINE
# =============================================================================
try:
    from pipeline_ABSA import load_all_models, run_full_pipeline
except ImportError as e:
    st.error(f"Lỗi: Không tìm thấy file 'pipeline_ABSA.py'. {e}")
    st.stop()

# =============================================================================
# 2️⃣ GIAO DIỆN CHÍNH (KIỂU TEXT2DATA)
# =============================================================================
st.set_page_config(page_title="Automatic Labelling Engine", layout="wide")
st.title("🧠 Automatic Labelling Engine — ABSA Sentiment Demo")
st.caption("Giao diện mô phỏng phong cách [text2data.com/Demo](https://text2data.com/Demo)")

# Sidebar cấu hình
st.sidebar.header("⚙️ Tùy chọn")
language = st.sidebar.selectbox("Ngôn ngữ", ["English", "Vietnamese"])
analysis_mode = st.sidebar.selectbox("Chế độ phân tích", ["ABSA Pipeline (Clause-Level)", "Sentiment Overview"])
st.sidebar.info("Ứng dụng chạy pipeline ABSA gồm 5 bước: Clause, Term, Opinion, Category, Polarity.")

# Tải model 1 lần duy nhất
models = load_all_models()

# =============================================================================
# 3️⃣ INPUT ZONE
# =============================================================================
default_sentence = "The food was great and the staff was friendly, but the room was small and dirty."
text_input = st.text_area("✍️ Nhập review hoặc đoạn văn để phân tích:", default_sentence, height=150)

col1, col2 = st.columns([1, 1])
analyze = col1.button("🔍 Phân tích")
clear = col2.button("🧹 Xóa")

if clear:
    st.experimental_rerun()

# =============================================================================
# 4️⃣ XỬ LÝ PIPELINE & HIỂN THỊ
# =============================================================================
if analyze and text_input.strip():
    if not all(models.values()):
        st.error("Một hoặc nhiều mô hình chưa được tải. Kiểm tra file pipeline_ABSA.py.")
    else:
        try:
            st.info("⏳ Đang chạy pipeline, vui lòng đợi...")
            results_df = run_full_pipeline(text_input, models)

            if results_df.empty:
                st.warning("Không có kết quả từ pipeline.")
                st.stop()

            st.success("✅ Phân tích hoàn tất!")

            # ---------------------------------------------------
            # 🎯 PHẦN 1 — Tổng quan Sentiment
            # ---------------------------------------------------
            st.subheader("🎯 Tổng quan cảm xúc")
            # Tính điểm sentiment tổng hợp
            pos_count = sum(results_df["polarity"].str.lower() == "positive")
            neg_count = sum(results_df["polarity"].str.lower() == "negative")
            neu_count = sum(results_df["polarity"].str.lower() == "neutral")
            total = pos_count + neg_count + neu_count
            score = ((pos_count - neg_count) / max(total, 1) + 1) * 50

            colA, colB = st.columns([1, 2])
            with colA:
                st.metric("Positive", pos_count)
                st.metric("Negative", neg_count)
                st.metric("Neutral", neu_count)
                st.metric("Sentiment Score", f"{score:.1f}/100")

            with colB:
                fig, ax = plt.subplots(figsize=(4, 2))
                color = "green" if score > 60 else "red" if score < 40 else "gray"
                ax.barh(["Score"], [score], color=color)
                ax.set_xlim(0, 100)
                ax.set_xlabel("0 – 100 sentiment score")
                st.pyplot(fig)

            # ---------------------------------------------------
            # 📋 PHẦN 2 — Bảng chi tiết ABSA
            # ---------------------------------------------------
            st.subheader("📋 Chi tiết từng Clause")
            st.dataframe(results_df, use_container_width=True)

            # ---------------------------------------------------
            # ☁️ PHẦN 3 — Word Cloud
            # ---------------------------------------------------
            st.subheader("☁️ Word Cloud (Từ khóa nổi bật)")
            full_text = " ".join(results_df["opinion"].dropna().astype(str))
            if full_text.strip():
                wc = WordCloud(width=800, height=300, background_color="white").generate(full_text)
                plt.figure(figsize=(10, 4))
                plt.imshow(wc, interpolation="bilinear")
                plt.axis("off")
                st.pyplot(plt)
            else:
                st.info("Không có từ khóa cảm xúc để hiển thị wordcloud.")

            # ---------------------------------------------------
            # 💾 PHẦN 4 — Xuất kết quả
            # ---------------------------------------------------
            csv = results_df.to_csv(index=False).encode("utf-8")
            st.download_button("💾 Tải CSV kết quả", csv, "absa_results.csv", "text/csv")

        except Exception as e:
            st.error(f"Đã xảy ra lỗi trong quá trình xử lý pipeline: {e}")
            st.exception(e)
elif analyze:
    st.warning("Vui lòng nhập văn bản để phân tích.")

st.markdown("---")
st.caption("Được mô phỏng lại bằng ❤️ từ text2data.com, xây dựng bằng Streamlit.")