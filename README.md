# ABSA CLI Toolkit

Unified command-line interface for **Aspect-Based Sentiment Analysis (ABSA)** tasks.

## Features

- **split-clause**: Split sentences into clauses and extract aspect terms (Qwen LLM)
- **extract-category**: Classify clauses into aspect categories (RoBERTa)
- **extract-polarity**: Detect sentiment polarity (DeBERTa)
- **extract-opinion**: Extract opinion words from clauses (Qwen LLM)
- **fine-tune-roberta**: Fine-tune RoBERTa with LoRA for category classification
- **pipeline**: Run the full ABSA pipeline end-to-end

## Requirements

Install dependencies:

```bash
pip install torch transformers peft datasets scikit-learn pandas tqdm
```

**Note**: `bitsandbytes` is required for 4-bit quantization. On Windows, you may need:
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

## License

MIT License (adjust as needed)

## Contributors

- Your Name / Team
- Built with ❤️ for ABSA research
