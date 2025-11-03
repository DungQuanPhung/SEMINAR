import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from io import StringIO

# =============================================================================
# 1️⃣ TẢI HÀM XỬ LÝ PIPELINE
# =============================================================================
try:
    from pipeline_ABSA import load_all_models, run_full_pipeline
except ImportError as e:
    st.error(f"Lỗi: Không tìm thấy file 'pipeline_ABSA.py'. {e}")
    st.stop()

# =============================================================================
# 2️⃣ GIAO DIỆN CHÍNH (KIỂU DASHBOARD)
# =============================================================================
st.set_page_config(page_title="ABSA Sentiment Dashboard", layout="wide")
st.title("🧠 ABSA Sentiment Analysis — Dashboard")
st.caption("Phân tích cảm xúc & khía cạnh (Aspect-Based Sentiment Analysis)")

# =============================================================================
# 3️⃣ LOAD MODELS (CACHE)
# =============================================================================
@st.cache_resource
def get_models():
    return load_all_models()

models = get_models()
if not all(models.values()):
    st.error("Một hoặc nhiều mô hình chưa được tải. Kiểm tra file pipeline_ABSA.py.")
    st.stop()

# =============================================================================
# 4️⃣ HÀM HIỂN THỊ DASHBOARD
# =============================================================================
def render_dashboard(results_df: pd.DataFrame):
    # --- Chuẩn hóa cột ---
    for col in ["category", "Category"]:
        if col in results_df.columns:
            results_df["Category"] = results_df[col]
    for col in ["polarity", "Polarity"]:
        if col in results_df.columns:
            results_df["Polarity"] = results_df[col]

    st.markdown("### 🎯 Tổng quan cảm xúc")
    pos = sum(results_df["Polarity"].str.lower() == "positive")
    neg = sum(results_df["Polarity"].str.lower() == "negative")
    neu = sum(results_df["Polarity"].str.lower() == "neutral")
    total = pos + neg + neu
    score = ((pos - neg) / max(total, 1) + 1) * 50

    # --- Nhóm 3 biểu đồ cạnh nhau ---
    st.markdown("### 📊 Tổng quan thống kê cảm xúc")

    # Tạo 3 cột song song
    col1, col2, col3 = st.columns(3)

    # --- 1️⃣ Sentiment Score ---
    with col1:
        st.markdown("#### 🎯 Sentiment Score")
        fig1, ax1 = plt.subplots(figsize=(2.5, 2))  # giảm kích thước
        color = "green" if score > 60 else "red" if score < 40 else "gray"
        ax1.barh(["Score"], [score], color=color)
        ax1.set_xlim(0, 100)
        ax1.set_xlabel("0–100 sentiment score")
        st.pyplot(fig1)

    # --- 2️⃣ Category ---
    with col2:
        st.markdown("#### 📂 Category")
        if "Category" in results_df.columns:
            cat_counts = results_df["Category"].value_counts()
            fig2, ax2 = plt.subplots(figsize=(2.5, 2))
            ax2.bar(cat_counts.index, cat_counts.values, color="skyblue")
            ax2.set_xticklabels(cat_counts.index, rotation=45, ha="right", fontsize=8)
            ax2.set_ylabel("")
            st.pyplot(fig2)
        else:
            st.info("Không có dữ liệu Category")

    # --- 3️⃣ Polarity ---
    with col3:
        st.markdown("#### 🎭 Polarity")
        if "Polarity" in results_df.columns:
            pol_counts = results_df["Polarity"].value_counts()
            colors = ["green" if i.lower() == "positive" else "gray" if i.lower() == "neutral" else "red"
                    for i in pol_counts.index]
            fig3, ax3 = plt.subplots(figsize=(2.5, 2))
            ax3.bar(pol_counts.index, pol_counts.values, color=colors)
            ax3.set_xticklabels(pol_counts.index, rotation=0, ha="center", fontsize=8)
            ax3.set_ylabel("")
            st.pyplot(fig3)
        else:
            st.info("Không có dữ liệu Polarity")

    # --- WordCloud Term ---
    st.markdown("### ☁️ Word Cloud — Term nổi bật")
    if "term" in results_df.columns or "Term" in results_df.columns:
        text_terms = " ".join(results_df.get("term", results_df.get("Term", pd.Series())).dropna().astype(str))
        if text_terms.strip():
            wc = WordCloud(width=800, height=300, background_color="white", colormap="viridis").generate(text_terms)
            plt.figure(figsize=(8, 4))
            plt.imshow(wc, interpolation="bilinear")
            plt.axis("off")
            st.pyplot(plt)
        else:
            st.info("Không có Term để hiển thị WordCloud.")
    else:
        st.info("Không có cột Term trong dữ liệu.")

    # --- WordCloud Opinion ---
    st.markdown("### 💭 Word Cloud — Opinion nổi bật")
    if "opinion" in results_df.columns or "Opinion" in results_df.columns:
        text_ops = " ".join(results_df.get("opinion", results_df.get("Opinion", pd.Series())).dropna().astype(str))
        if text_ops.strip():
            wc = WordCloud(width=800, height=300, background_color="white", colormap="cool").generate(text_ops)
            plt.figure(figsize=(8, 4))
            plt.imshow(wc, interpolation="bilinear")
            plt.axis("off")
            st.pyplot(plt)
        else:
            st.info("Không có Opinion để hiển thị WordCloud.")
    else:
        st.info("Không có cột Opinion trong dữ liệu.")

    # --- Bảng dữ liệu chi tiết ---
    st.markdown("### 📋 Bảng chi tiết kết quả")
    st.dataframe(results_df, use_container_width=True)

    # --- Xuất file CSV ---
    csv = results_df.to_csv(index=False).encode("utf-8")
    st.download_button("💾 Tải CSV kết quả", csv, "absa_results.csv", "text/csv")

# =============================================================================
# 5️⃣ TAB: SINGLE & BATCH
# =============================================================================
tab1, tab2 = st.tabs(["📝 Phân tích 1 review", "📤 Phân tích file hàng loạt"])

# --- TAB 1 ---
with tab1:
    st.subheader("✍️ Nhập câu review")
    default_sentence = "The food was great and the staff was friendly, but the room was small and dirty."
    text_input = st.text_area("Nhập nội dung:", default_sentence, height=150)
    col1, col2 = st.columns([1, 1])
    analyze = col1.button("🔍 Phân tích", use_container_width=True, key="single_btn")
    clear = col2.button("🧹 Xóa", use_container_width=True, key="clear_btn")
    if clear:
        st.experimental_rerun()

    if analyze and text_input.strip():
        try:
            st.info("⏳ Đang chạy pipeline...")
            df = run_full_pipeline(text_input, models)
            if df.empty:
                st.warning("Không có kết quả từ pipeline.")
            else:
                st.success("✅ Phân tích hoàn tất!")
                render_dashboard(df)
        except Exception as e:
            st.error(f"Lỗi: {e}")
            st.exception(e)

# --- TAB 2 ---
with tab2:
    st.subheader("📂 Tải file .txt (mỗi dòng là 1 review)")
    uploaded_file = st.file_uploader("Chọn file", type=["txt"])
    if st.button("🚀 Chạy phân tích hàng loạt", key="batch_btn"):
        if uploaded_file is None:
            st.warning("Vui lòng tải lên file.")
        else:
            try:
                stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
                reviews = [line.strip() for line in stringio.readlines() if line.strip()]
            except Exception as e:
                st.error(f"Lỗi đọc file: {e}")
                st.stop()

            if not reviews:
                st.warning("File trống hoặc không hợp lệ.")
            else:
                all_results = []
                st.info(f"Đang xử lý {len(reviews)} dòng...")
                progress = st.progress(0)
                for i, review in enumerate(reviews):
                    try:
                        df = run_full_pipeline(review, models)
                        if not df.empty:
                            df["review_line"] = i + 1
                            df["review_text"] = review
                            all_results.append(df)
                    except Exception as e:
                        st.error(f"Lỗi dòng {i+1}: {e}")
                    progress.progress((i + 1) / len(reviews))
                progress.empty()

                if not all_results:
                    st.warning("Không tìm thấy khía cạnh nào.")
                else:
                    final_df = pd.concat(all_results, ignore_index=True)
                    st.success(f"✅ Hoàn tất! {len(final_df)} khía cạnh được tìm thấy.")
                    render_dashboard(final_df)

st.markdown("---")
st.caption("✨ Dashboard ABSA hoàn thiện: Category + Polarity chart + WordCloud Term/Opinion.")