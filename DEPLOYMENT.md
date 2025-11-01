# 🚀 Complete Deployment Guide

This guide walks you through deploying the ABSA Pipeline application with a complete architecture using Hugging Face Spaces, Cloudflare Workers, and Cloudflare Pages.

## 📋 Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Prerequisites](#prerequisites)
3. [Deploy Backend (Hugging Face Spaces)](#deploy-backend-hugging-face-spaces)
4. [Deploy API Gateway (Cloudflare Workers)](#deploy-api-gateway-cloudflare-workers)
5. [Deploy Frontend (Cloudflare Pages)](#deploy-frontend-cloudflare-pages)
6. [Configuration](#configuration)
7. [Testing](#testing)
8. [Troubleshooting](#troubleshooting)
9. [Performance Optimization](#performance-optimization)

---

## 🏗️ Architecture Overview

```
┌─────────────────┐
│   User Browser  │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────┐
│  Cloudflare CDN & Pages     │  ← React Frontend
│  - Global Edge Network      │  ← Static Assets
│  - Cache & Minification     │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  Cloudflare Workers         │  ← API Gateway
│  - Request Routing          │  ← Caching Layer
│  - Rate Limiting            │  ← Security Headers
│  - CORS Handling            │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  Hugging Face Spaces        │  ← ML Models (GPU)
│  - Qwen LLM (4-bit)         │  ← Gradio Interface
│  - RoBERTa (Category)       │  ← Model Inference
│  - DeBERTa (Polarity)       │
└─────────────────────────────┘
```

## 📋 Prerequisites

### Required Accounts

1. **Hugging Face Account** (Free)
   - Sign up at: https://huggingface.co/join
   - Get your token: https://huggingface.co/settings/tokens
   - Create a new Space: https://huggingface.co/new-space

2. **Cloudflare Account** (Free tier available)
   - Sign up at: https://dash.cloudflare.com/sign-up
   - Get your API token: https://dash.cloudflare.com/profile/api-tokens

### Required Tools

```bash
# Node.js (v18+)
node --version

# Python (v3.9+)
python --version

# Wrangler CLI (Cloudflare)
npm install -g wrangler

# Git
git --version
```

---

## 🤗 Deploy Backend (Hugging Face Spaces)

### Step 1: Prepare Your Space

1. Create a new Space on Hugging Face:
   - Go to https://huggingface.co/new-space
   - Choose a name: `absa-pipeline`
   - SDK: **Gradio**
   - Hardware: **T4 Medium** (recommended for optimal performance)
   - Visibility: Public or Private

2. Clone your Space repository:

```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/absa-pipeline
cd absa-pipeline
```

### Step 2: Copy Required Files

```bash
# Copy main application files
cp /path/to/SEMINAR/app.py .
cp /path/to/SEMINAR/requirements.txt .
cp /path/to/SEMINAR/README.md .

# Copy library files
cp /path/to/SEMINAR/split_clause_lib.py .
cp /path/to/SEMINAR/Term_Opinion_lib.py .
cp /path/to/SEMINAR/Extract_Category.py .
cp /path/to/SEMINAR/Extract_Polarity_lib.py .

# Copy model directory (if you have fine-tuned model)
cp -r /path/to/SEMINAR/roberta_lora_goal .
```

### Step 3: Verify README.md Header

Ensure your `README.md` starts with this metadata:

```yaml
---
title: ABSA Pipeline
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
```

### Step 4: Push to Hugging Face

```bash
git add .
git commit -m "Initial deployment"
git push origin main
```

### Step 5: Monitor Build

1. Go to your Space URL: `https://huggingface.co/spaces/YOUR_USERNAME/absa-pipeline`
2. Wait for the build to complete (5-10 minutes for first build)
3. Check logs for any errors
4. Test the Gradio interface once it's ready

### Step 6: Get API Endpoint

Once deployed, your Space exposes an API at:
```
https://YOUR_USERNAME-absa-pipeline.hf.space/api/predict
```

Test it with:
```bash
curl -X POST https://YOUR_USERNAME-absa-pipeline.hf.space/api/predict \
  -H "Content-Type: application/json" \
  -d '{"data": ["The food was great and the staff was friendly."]}'
```

---

## ⚡ Deploy API Gateway (Cloudflare Workers)

### Step 1: Authenticate with Cloudflare

```bash
wrangler login
```

This will open a browser window for authentication.

### Step 2: Update Configuration

Edit `cloudflare-worker/wrangler.toml`:

```toml
# Replace with your values
name = "absa-pipeline-api"
account_id = "your-account-id"  # Found in Cloudflare dashboard

[vars]
HF_SPACE_URL = "https://YOUR_USERNAME-absa-pipeline.hf.space"
```

### Step 3: Update Worker Code

Edit `cloudflare-worker/worker.js`:

```javascript
// Update line 10
const HF_SPACE_URL = 'https://YOUR_USERNAME-absa-pipeline.hf.space';
```

### Step 4: Deploy Worker

```bash
cd cloudflare-worker
wrangler deploy
```

You'll get a URL like: `https://absa-pipeline-api.YOUR_SUBDOMAIN.workers.dev`

### Step 5: Test Worker

```bash
curl -X POST https://absa-pipeline-api.YOUR_SUBDOMAIN.workers.dev/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"data": ["The hotel room was clean and comfortable."]}'
```

### Optional: Setup Custom Domain

1. Go to Cloudflare Dashboard → Workers & Pages
2. Select your worker
3. Go to Settings → Triggers → Custom Domains
4. Add your custom domain (e.g., `api.yourdomain.com`)

---

## 🌐 Deploy Frontend (Cloudflare Pages)

### Step 1: Install Dependencies

```bash
cd frontend
npm install
```

### Step 2: Configure Environment Variables

Create `frontend/.env`:

```bash
VITE_API_ENDPOINT=https://absa-pipeline-api.YOUR_SUBDOMAIN.workers.dev/api/analyze
```

### Step 3: Build Frontend

```bash
npm run build
```

This creates an optimized production build in `dist/`.

### Step 4: Deploy to Cloudflare Pages

Option A: Using Wrangler CLI

```bash
npx wrangler pages deploy dist --project-name=absa-pipeline
```

Option B: Using Git Integration

1. Push your code to GitHub
2. Go to Cloudflare Dashboard → Pages
3. Create new project
4. Connect your GitHub repository
5. Configure build settings:
   - Build command: `cd frontend && npm install && npm run build`
   - Build output directory: `frontend/dist`
   - Environment variables: `VITE_API_ENDPOINT=...`

### Step 5: Configure Build Settings

In Cloudflare Pages dashboard:

**Build settings:**
- Framework preset: `Vite`
- Build command: `npm run build`
- Build output directory: `dist`
- Root directory: `frontend`

**Environment variables:**
```
VITE_API_ENDPOINT=https://absa-pipeline-api.YOUR_SUBDOMAIN.workers.dev/api/analyze
```

### Step 6: Access Your Site

Your site will be available at: `https://absa-pipeline.pages.dev`

### Optional: Custom Domain

1. Go to Pages project → Custom domains
2. Add your domain
3. Follow DNS configuration instructions

---

## ⚙️ Configuration

### Environment Variables

Create `.env` files for each component:

#### Backend (Hugging Face)
```bash
# .env (if needed for private models)
HF_TOKEN=your_huggingface_token
```

#### API Gateway (Cloudflare Worker)
```bash
# Set via: wrangler secret put SECRET_NAME
HF_TOKEN=your_huggingface_token  # Only if Space is private
API_KEY=your_api_key  # For authentication (optional)
```

#### Frontend (Cloudflare Pages)
```bash
VITE_API_ENDPOINT=https://your-worker-url/api/analyze
```

### Fine-tuning Configuration

Adjust these parameters in `app.py` as needed:

```python
# Model configuration
model_id = "Qwen/Qwen2.5-3B-Instruct"  # Change model
CACHE_TTL = 3600  # Cache duration in seconds
MAX_LENGTH = 128  # Token limit for classification
```

---

## 🧪 Testing

### 1. Test Backend (Hugging Face)

```bash
curl -X POST https://YOUR_USERNAME-absa-pipeline.hf.space/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "data": ["The room was clean but the service was slow."]
  }'
```

Expected response:
```json
{
  "data": [[
    {
      "clause": "The room was clean",
      "term": "room",
      "opinion": "clean",
      "category": "Facility",
      "polarity": "Positive",
      "polarity_score": 0.9823
    },
    {
      "clause": "the service was slow",
      "term": "service",
      "opinion": "slow",
      "category": "Service",
      "polarity": "Negative",
      "polarity_score": 0.9156
    }
  ]]
}
```

### 2. Test API Gateway (Cloudflare Worker)

```bash
# Test health endpoint
curl https://your-worker-url/health

# Test analysis endpoint
curl -X POST https://your-worker-url/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"data": ["Test review text"]}'
```

### 3. Test Frontend

1. Open your Pages URL in a browser
2. Enter a test review
3. Click "Analyze"
4. Verify results display correctly
5. Test example sentences
6. Test error handling (invalid input, network errors)

### 4. Performance Testing

```bash
# Load testing with Apache Bench
ab -n 100 -c 10 -p test.json -T application/json \
  https://your-worker-url/api/analyze

# Where test.json contains:
# {"data": ["The hotel was excellent."]}
```

---

## 🔧 Troubleshooting

### Hugging Face Spaces Issues

#### Build Fails
```
Error: Out of memory
```
**Solution:** 
- Use T4 Medium or larger hardware
- Optimize model loading in `app.py`
- Use smaller models or quantization

#### Models Not Loading
```
Error: Model files not found
```
**Solution:**
- Ensure `roberta_lora_goal/` directory is pushed
- Check file sizes (max 10GB per Space)
- Verify model files are committed with Git LFS

```bash
git lfs install
git lfs track "*.bin"
git add .gitattributes
```

#### Slow Inference
**Solution:**
- Upgrade to T4 Medium GPU
- Implement request queuing in Gradio
- Add caching for repeated queries

### Cloudflare Workers Issues

#### CORS Errors
```
Access to fetch blocked by CORS policy
```
**Solution:**
Update `worker.js` CORS headers:
```javascript
'Access-Control-Allow-Origin': '*',
'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
'Access-Control-Allow-Headers': 'Content-Type'
```

#### 502 Bad Gateway
```
Error: Backend unavailable
```
**Solution:**
- Check HF Space is running
- Verify HF_SPACE_URL in `wrangler.toml`
- Check Worker logs: `wrangler tail`

#### Rate Limiting
**Solution:**
Implement KV storage for rate limiting:
```bash
wrangler kv:namespace create RATE_LIMIT
# Update wrangler.toml with namespace ID
```

### Cloudflare Pages Issues

#### Build Failures
```
Error: Module not found
```
**Solution:**
- Verify `package.json` dependencies
- Check build command in Pages settings
- Review build logs in Cloudflare dashboard

#### Environment Variables Not Working
**Solution:**
- Redeploy after adding environment variables
- Use `VITE_` prefix for Vite variables
- Check `.env` files are not committed

#### Assets Not Loading
**Solution:**
- Verify build output directory is `dist`
- Check `vite.config.ts` base path
- Clear Cloudflare cache

---

## 🚀 Performance Optimization

### 1. Backend Optimization

```python
# In app.py
# Enable model caching
@st.cache_resource  # or similar caching mechanism

# Use smaller models if needed
model_id = "Qwen/Qwen2.5-1.5B-Instruct"  # Smaller, faster

# Batch processing
def batch_analyze(sentences: List[str], batch_size=4):
    # Process multiple sentences efficiently
    pass
```

### 2. Worker Optimization

```javascript
// Implement caching with KV
async function getCachedResult(key) {
  return await env.CACHE.get(key);
}

async function cacheResult(key, value, ttl = 3600) {
  await env.CACHE.put(key, value, { expirationTtl: ttl });
}
```

### 3. Frontend Optimization

```javascript
// Lazy loading
const AnalysisResults = lazy(() => import('./AnalysisResults'));

// Code splitting
// Reduce bundle size
// Optimize images
```

### 4. CDN Configuration

In Cloudflare Pages:
- Enable Auto Minify (JS, CSS, HTML)
- Enable Brotli compression
- Configure cache rules for static assets

---

## 📊 Monitoring

### Hugging Face Spaces

- View logs in Space settings
- Monitor GPU usage
- Track request counts

### Cloudflare Workers

```bash
# Real-time logs
wrangler tail

# Analytics in dashboard
# Workers → Analytics
```

### Cloudflare Pages

- Page views and bandwidth in Analytics
- Core Web Vitals
- Error tracking

---

## 🔒 Security Best Practices

1. **API Authentication** (Optional)
   ```javascript
   // In worker.js
   const API_KEY = env.API_KEY;
   if (request.headers.get('X-API-Key') !== API_KEY) {
     return new Response('Unauthorized', { status: 401 });
   }
   ```

2. **Rate Limiting**
   - Implement per-IP limits
   - Use Cloudflare Rate Limiting rules

3. **Input Validation**
   - Sanitize user inputs
   - Limit request sizes
   - Validate data types

4. **HTTPS Only**
   - Enforce HTTPS redirects
   - Set secure headers

---

## 🎯 Next Steps

1. **Custom Domain Setup**
   - Configure DNS
   - Add SSL certificates
   - Set up redirects

2. **Analytics Integration**
   - Google Analytics
   - Cloudflare Web Analytics
   - Custom event tracking

3. **CI/CD Pipeline**
   - GitHub Actions for auto-deployment
   - Automated testing
   - Version tagging

4. **Scaling**
   - Upgrade HF hardware as needed
   - Optimize caching strategies
   - Consider multiple Workers for load balancing

---

## 📞 Support

- **Hugging Face:** https://discuss.huggingface.co
- **Cloudflare:** https://community.cloudflare.com
- **Issues:** GitHub Issues on your repository

---

## 📝 License

MIT License - See LICENSE file for details
