# Cấu hình cho ứng dụng ABSA
# Chỉnh sửa các giá trị này để phù hợp với phần cứng của bạn

# ============== CẤU HÌNH MÔ HÌNH ==============

# Mô hình LLM chính (Qwen)
QWEN_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

# ============== CẤU HÌNH BỘ NHỚ ==============

# Giới hạn VRAM tối đa cho mô hình Qwen (GB)
MAX_GPU_MEMORY = "2.0GB"

# Batch size cho các mô hình phân loại
CATEGORY_BATCH_SIZE = 2  # Giảm xuống nếu vẫn bị OOM
POLARITY_BATCH_SIZE = 2  # Giảm xuống nếu vẫn bị OOM

# ============== CẤU HÌNH TOKEN ==============

# Số token tối đa cho input
MAX_INPUT_LENGTH = 128  # Giảm xuống 128 hoặc 64 nếu vẫn OOM

# Số token mới sinh ra tối đa cho mỗi bước
MAX_NEW_TOKENS_SPLIT = 50  # Bước 1: Tách clause và term
MAX_NEW_TOKENS_OPINION = 10  # Bước 2: Trích xuất opinion

# ============== CẤU HÌNH QUANTIZATION ==============
USE_4BIT_QUANTIZATION = False

# ============== CẤU HÌNH KHÁC ==============
UNLOAD_QWEN_AFTER_LLM = False

# Có hiển thị thông tin debug không?
DEBUG_MODE = False