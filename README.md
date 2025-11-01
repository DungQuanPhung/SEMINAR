---
title: ABSA Pipeline - Aspect-Based Sentiment Analysis
emoji: 🔍
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
hardware: t4-medium
---

# ABSA Pipeline - Aspect-Based Sentiment Analysis

Unified system for **Aspect-Based Sentiment Analysis (ABSA)** with deployment on Hugging Face Spaces, Cloudflare Workers, and Cloudflare Pages.

## 🚀 Live Demo

Try the live demo on Hugging Face Spaces: [ABSA Pipeline](https://huggingface.co/spaces/YOUR_USERNAME/absa-pipeline)

## 🎯 Features

This system provides a complete ABSA pipeline with three deployment options:

### Pipeline Components
- **Clause Splitting**: Split sentences into meaningful clauses (Qwen LLM)
- **Term Extraction**: Extract aspect terms being discussed (Qwen LLM)
- **Opinion Extraction**: Extract opinion words (Qwen LLM)
- **Category Classification**: Classify into aspect categories (RoBERTa with LoRA)
- **Polarity Detection**: Detect sentiment polarity (DeBERTa)

### Deployment Architecture
1. **Hugging Face Spaces**: Backend ML models with GPU (T4 Medium)
2. **Cloudflare Workers**: API Gateway with caching and rate limiting
3. **Cloudflare Pages**: React frontend with global CDN

## 🏗️ Architecture

```
User Browser
    ↓
Cloudflare CDN (Pages) - Frontend React
    ↓
Cloudflare Workers - API Gateway + Caching
    ↓
Hugging Face Spaces - Backend ML Models (GPU)
```

## 📋 Requirements

### For Hugging Face Spaces Deployment

Install dependencies:

```bash
pip install -r requirements.txt
```

**Required files**:
- `app.py`: Main Gradio application
- `requirements.txt`: Python dependencies
- `roberta_lora_goal/`: Fine-tuned RoBERTa model directory
- All `*_lib.py` files: Pipeline worker functions

**Hardware**: T4 Medium GPU recommended for optimal performance

### For Local Development

```bash
pip install torch transformers peft datasets scikit-learn pandas tqdm gradio
```

**Note**: `bitsandbytes` is required for 4-bit quantization. On Windows:
```bash
pip install bitsandbytes-windows
```

## Quick Start

### 1. Split Sentences into Clauses + Extract Terms

**Single sentence:**
```bash
python app.py split-clause -s "The room was clean and the staff was helpful." -o clauses.json
```

**From text file (one sentence per line):**
```bash
python app.py split-clause -i sentences.txt -o clauses.jsonl --format jsonl
```

### 2. Extract Aspect Categories

Requires a fine-tuned RoBERTa model (see section 5 below).

```bash
python app.py extract-category -i clauses.json -o clauses_with_cat.json --model-path ./roberta_lora_goal
```

### 3. Detect Sentiment Polarity

```bash
python app.py extract-polarity -i clauses_with_cat.json -o clauses_with_pol.json
```

### 4. Extract Opinion Words

```bash
python app.py extract-opinion -i clauses_with_pol.json -o clauses_final.json
```

### 5. Fine-tune RoBERTa for Category Classification

Prepare two CSV files:
- `goal.csv`: Ground-truth data with columns `clause`, `category`
- `llm_predict.csv`: Test/validation data with same columns

```bash
python app.py fine-tune-roberta --goal goal.csv --llm llm_predict.csv --out ./roberta_lora_goal --epochs 80 --save
```

**Options:**
- `--epochs`: Number of training epochs (default: 80)
- `--bsz`: Batch size (default: 8)
- `--grad-accum`: Gradient accumulation steps (default: 4)
- `--lr`: Learning rate (default: 1.5e-4)
- `--save`: Save checkpoints per epoch (default: no)

### 6. Full ABSA Pipeline

Run all steps in one command:

```bash
python app.py pipeline -s "The hotel was great but the room was dirty." -o final_result.json --category-model ./roberta_lora_goal
```

**Options:**
- `-s` / `--sentence`: Single sentence
- `-i` / `--input-file`: Text file (one sentence per line)
- `--format`: Output format (`json`, `jsonl`, `csv`)
- `--qwen-model`: Qwen model ID (default: `Qwen/Qwen3-4B-Instruct-2507`)
- `--category-model`: Path to fine-tuned RoBERTa

## Output Schema

After running the full pipeline, each clause will have:

```json
{
  "sentence_original": "The hotel was great but the room was dirty.",
  "clause": "The hotel was great",
  "term": "hotel",
  "category": "Facility",
  "polarity": "Positive",
  "polarity_score": 0.9876,
  "opinion": "great"
}
```

## Module Structure

- `app.py`: Unified CLI entrypoint
- `split_clause.py`: Clause splitting + term extraction (Qwen)
- `Extract_Category.py`: Category classification (RoBERTa)
- `Extract_Polarity.py`: Polarity detection (DeBERTa)
- `Term_Opinion.py`: Opinion extraction (Qwen)
- `Fine_Tune_RoBertaBase.py`: RoBERTa fine-tuning with LoRA

## Troubleshooting

**Issue: `bitsandbytes` not found on Windows**
- Solution: `pip install bitsandbytes-windows` or use CPU-only mode (contact maintainer for CPU flag)

**Issue: Out of memory (GPU)**
- Reduce batch size: `--bsz 4` or `--bsz 2`
- Use smaller model: `--model-id Qwen/Qwen-1_8B-Chat`

**Issue: Slow on CPU**
- Model loading is heavy. Consider using Google Colab or a cloud GPU.

## Example Workflow

```bash
# Step 1: Prepare training data and fine-tune RoBERTa
python app.py fine-tune-roberta --goal goal.csv --llm test.csv --out ./my_model --epochs 50

# Step 2: Run full pipeline on new data
python app.py pipeline -i new_sentences.txt -o results.csv --format csv --category-model ./my_model
```

## 🚀 Deployment

### Deploy to Hugging Face Spaces

1. Create a new Space on [Hugging Face](https://huggingface.co/new-space)
2. Choose "Gradio" as the SDK
3. Select "T4 Medium" hardware
4. Clone this repository and push to your Space:

```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/absa-pipeline
cd absa-pipeline
git remote add github https://github.com/YOUR_REPO/SEMINAR.git
git pull github main
git push origin main
```

5. Your Space will automatically build and deploy!

### API Endpoint

Once deployed, your Space exposes an API endpoint:

```python
import requests

API_URL = "https://YOUR_USERNAME-absa-pipeline.hf.space/api/predict"

response = requests.post(API_URL, json={
    "data": ["The food was great and the staff was friendly."]
})

results = response.json()
print(results)
```

### Deploy Frontend to Cloudflare Pages

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed instructions on deploying the complete system with Cloudflare Workers and Pages.

## 📚 API Documentation

### Health Check

```
GET /
```

Returns health status of the models.

### Analyze Review

```
POST /api/predict
```

**Request Body**:
```json
{
  "data": ["Your review text here"]
}
```

**Response**:
```json
{
  "data": [
    {
      "clause": "The food was great",
      "term": "food",
      "opinion": "great",
      "category": "Amenity",
      "polarity": "Positive",
      "polarity_score": 0.9876,
      "sentence_original": "The food was great..."
    }
  ]
}
```

## 🔧 Troubleshooting

### Hugging Face Spaces

**Issue: Out of memory**
- Use T4 Medium or larger GPU
- Reduce batch processing in code

**Issue: Model files not found**
- Ensure `roberta_lora_goal/` directory is pushed to Space
- Check file sizes don't exceed limits

**Issue: Slow startup**
- First load takes 2-3 minutes to download models
- Subsequent requests are cached and fast

## 📄 License

MIT License

## 👥 Contributors

Built with ❤️ for ABSA research and NLP applications
