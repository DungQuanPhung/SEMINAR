"""
Hugging Face Spaces Application - ABSA Pipeline
Gradio interface for Aspect-Based Sentiment Analysis
Optimized for GPU deployment with caching and error handling
"""

import gradio as gr
import pandas as pd
import os
import torch
import logging
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig,
    AutoModelForSequenceClassification,
    pipeline
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from typing import List, Dict, Any
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# IMPORT WORKER FUNCTIONS
# =============================================================================
try:
    from split_clause_lib import run_split_and_term, chat_llm
    from Term_Opinion_lib import run_extract_opinions
    from Extract_Category import extract_categories
    from Extract_Polarity_lib import run_detect_polarity
except ImportError as e:
    logger.error(f"Import error: {e}")
    raise

# =============================================================================
# MODEL LOADING FUNCTIONS
# =============================================================================

def load_qwen_model():
    """Load Qwen LLM with 4-bit quantization"""
    logger.info("Loading Qwen LLM (4-bit quantization)...")
    model_id = "Qwen/Qwen2.5-3B-Instruct"
    
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
        device_map="auto",
        trust_remote_code=True
    )
    
    # Prepare for LoRA
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
    
    logger.info("Qwen LLM loaded successfully")
    return model, tokenizer

def load_category_model():
    """Load fine-tuned RoBERTa for category classification"""
    logger.info("Loading Category model (RoBERTa)...")
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, "roberta_lora_goal")
        
        model_cat = AutoModelForSequenceClassification.from_pretrained(model_path)
        tokenizer_cat = AutoTokenizer.from_pretrained(model_path)
        
        id2label = {
            0: "Amenity",
            1: "Branding",
            2: "Experience",
            3: "Facility",
            4: "Loyalty",
            5: "Service"
        }
        
        logger.info("Category model loaded successfully")
        return model_cat, tokenizer_cat, id2label
    except Exception as e:
        logger.error(f"Failed to load category model: {e}")
        return None, None, None

def load_polarity_model():
    """Load DeBERTa for polarity detection"""
    logger.info("Loading Polarity model (DeBERTa)...")
    polarity_classifier = pipeline(
        "text-classification",
        model="yangheng/deberta-v3-base-absa-v1.1",
        top_k=None,
        truncation=True,
        device=0 if torch.cuda.is_available() else -1
    )
    logger.info("Polarity model loaded successfully")
    return polarity_classifier

# =============================================================================
# LOAD ALL MODELS AT STARTUP
# =============================================================================
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
    logger.info("All models loaded and ready")
except Exception as e:
    logger.error(f"Critical error loading models: {e}")
    MODELS = None

# =============================================================================
# PIPELINE FUNCTION
# =============================================================================

def run_pipeline(sentence: str, progress=gr.Progress(track_tqdm=True)):
    """
    Run the full ABSA pipeline:
    1. Split clauses and extract terms (Qwen)
    2. Extract opinions (Qwen)
    3. Classify categories (RoBERTa)
    4. Detect polarity (DeBERTa)
    """
    if not MODELS or not all(MODELS.values()):
        raise gr.Error("Models not loaded successfully. Please check logs.")
    
    try:
        start_time = time.time()
        
        # Extract models
        qwen_model = MODELS["qwen_model"]
        qwen_tokenizer = MODELS["qwen_tokenizer"]
        cat_model = MODELS["cat_model"]
        cat_tokenizer = MODELS["cat_tokenizer"]
        cat_id2label = MODELS["cat_id2label"]
        polarity_classifier = MODELS["polarity_classifier"]
        device = MODELS["device"]

        # Create chat function
        def chat_func(messages, max_new_tokens):
            return chat_llm(qwen_model, qwen_tokenizer, messages, max_new_tokens)

        # Run pipeline steps
        progress(0.25, desc="Step 1/4: Splitting clauses and extracting terms...")
        clauses_terms = run_split_and_term(sentence, chat_func)
        
        progress(0.50, desc="Step 2/4: Extracting opinions...")
        clauses_opinions = run_extract_opinions(clauses_terms, chat_func)
        
        progress(0.75, desc="Step 3/4: Classifying categories...")
        clauses_categories = extract_categories(
            clauses_opinions, cat_model, cat_tokenizer, cat_id2label, device
        )
        
        progress(0.90, desc="Step 4/4: Detecting polarity...")
        final_results = run_detect_polarity(clauses_categories, polarity_classifier)
        
        # Format results
        df = pd.DataFrame(final_results)
        columns_order = [
            "clause", "term", "opinion", "category", "polarity", 
            "polarity_score", "sentence_original"
        ]
        final_columns = [col for col in columns_order if col in df.columns]
        
        elapsed_time = time.time() - start_time
        logger.info(f"Pipeline completed in {elapsed_time:.2f}s")
        
        return df[final_columns]
    
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        raise gr.Error(f"Error during processing: {str(e)}")

def health_check():
    """Health check endpoint for monitoring"""
    if MODELS and all(MODELS.values()):
        return {"status": "healthy", "gpu_available": torch.cuda.is_available()}
    return {"status": "unhealthy", "error": "Models not loaded"}

# =============================================================================
# GRADIO INTERFACE
# =============================================================================

# Example sentences
examples = [
    "The food was great and the staff was very friendly, but the room was a bit small.",
    "The hotel was clean and modern, with excellent service.",
    "Quy trình check-in rất suôn sẻ và nhân viên vô cùng nhiệt tình. Phòng ốc của chúng tôi rộng rãi và chiếc giường rất thoải mái, mặc dù wifi trong phòng lại chậm một cách khủng khiếp.",
    "Nhà hàng trong khách sạn có đồ ăn rất ngon, nhưng dịch vụ tại bàn lại chậm. Quầy bar trên tầng thượng có tầm nhìn tuyệt đẹp.",
]

with gr.Blocks(
    theme=gr.themes.Soft(),
    title="ABSA Pipeline - Aspect-Based Sentiment Analysis"
) as demo:
    gr.Markdown(
        """
        # 🚀 ABSA Pipeline - Aspect-Based Sentiment Analysis
        
        This application analyzes customer reviews using a multi-model pipeline:
        - **Qwen LLM**: Clause splitting and term/opinion extraction
        - **RoBERTa**: Aspect category classification
        - **DeBERTa**: Sentiment polarity detection
        
        Enter a review sentence and click "Analyze" to extract:
        **Clause, Term, Opinion, Category, Polarity**
        """
    )
    
    with gr.Row():
        with gr.Column(scale=2):
            text_input = gr.Textbox(
                lines=5,
                label="Enter your review:",
                placeholder="Example: The food was great and the staff was very friendly..."
            )
            analyze_button = gr.Button("🔍 Analyze", variant="primary", size="lg")
            
            gr.Markdown("### Try these examples:")
            gr.Examples(
                examples=examples,
                inputs=text_input,
                label=""
            )
        
        with gr.Column(scale=3):
            output_df = gr.DataFrame(
                label="Analysis Results",
                wrap=True,
                interactive=False
            )
    
    gr.Markdown(
        """
        ---
        ### About the Pipeline
        
        1. **Clause Splitting**: Breaks down the review into meaningful clauses
        2. **Term Extraction**: Identifies the aspect/entity being discussed
        3. **Opinion Extraction**: Extracts opinion words describing each aspect
        4. **Category Classification**: Classifies into: Amenity, Branding, Experience, Facility, Loyalty, Service
        5. **Polarity Detection**: Determines sentiment: Positive, Negative, or Neutral
        
        **API Access**: This Space exposes an API endpoint that can be called programmatically.
        See the "Use via API" button above for details.
        """
    )
    
    # Connect button to pipeline function
    analyze_button.click(
        fn=run_pipeline,
        inputs=text_input,
        outputs=output_df
    )

# =============================================================================
# LAUNCH APPLICATION
# =============================================================================
if __name__ == "__main__":
    demo.queue(max_size=10).launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
