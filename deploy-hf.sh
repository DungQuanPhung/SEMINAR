#!/bin/bash
# Deploy to Hugging Face Spaces

set -e

echo "🚀 Deploying to Hugging Face Spaces..."

# Configuration
HF_USERNAME="YOUR_USERNAME"
HF_SPACE_NAME="absa-pipeline"
HF_SPACE_URL="https://huggingface.co/spaces/$HF_USERNAME/$HF_SPACE_NAME"

# Check if HF_TOKEN is set
if [ -z "$HF_TOKEN" ]; then
  echo "⚠️  Warning: HF_TOKEN environment variable not set"
  echo "   You may need to authenticate manually"
fi

# Clone the Space repository if not exists
if [ ! -d "hf-space" ]; then
  echo "📥 Cloning Hugging Face Space..."
  git clone $HF_SPACE_URL hf-space
  cd hf-space
else
  echo "📂 Using existing hf-space directory..."
  cd hf-space
  git pull origin main
fi

# Copy necessary files
echo "📋 Copying files..."
cp ../app.py .
cp ../requirements.txt .
cp ../README.md .
cp ../*.py . 2>/dev/null || true

# Copy model directory if exists
if [ -d "../roberta_lora_goal" ]; then
  echo "📦 Copying model files..."
  cp -r ../roberta_lora_goal .
fi

# Check file sizes
echo "📊 Checking file sizes..."
find . -type f -size +100M -exec ls -lh {} \;

# Git operations
echo "📤 Pushing to Hugging Face..."
git add .
git commit -m "Update: $(date '+%Y-%m-%d %H:%M:%S')" || echo "No changes to commit"

# Push with authentication
if [ ! -z "$HF_TOKEN" ]; then
  git remote set-url origin https://$HF_USERNAME:$HF_TOKEN@huggingface.co/spaces/$HF_USERNAME/$HF_SPACE_NAME
fi

git push origin main

cd ..

echo "✅ Deployment complete!"
echo "🌐 Your Space will be available at: $HF_SPACE_URL"
echo "⏰ Note: First build may take 5-10 minutes"
