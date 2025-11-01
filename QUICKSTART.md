# 🚀 Quick Start Guide

Get your ABSA Pipeline deployed in 3 steps!

## Prerequisites

- [Hugging Face Account](https://huggingface.co/join) (free)
- [Cloudflare Account](https://dash.cloudflare.com/sign-up) (free tier available)
- Node.js 18+ and Python 3.9+

## Step 1: Deploy Backend to Hugging Face (5 minutes)

### 1.1 Create a Space

1. Go to https://huggingface.co/new-space
2. Name: `absa-pipeline`
3. SDK: **Gradio**
4. Hardware: **T4 Medium** (free GPU)
5. Click "Create Space"

### 1.2 Clone and Setup

```bash
# Clone your new Space
git clone https://huggingface.co/spaces/YOUR_USERNAME/absa-pipeline
cd absa-pipeline

# Copy files from this repository
cp /path/to/SEMINAR/app.py .
cp /path/to/SEMINAR/requirements.txt .
cp /path/to/SEMINAR/README.md .
cp /path/to/SEMINAR/*_lib.py .
cp -r /path/to/SEMINAR/roberta_lora_goal .

# Push to Hugging Face
git add .
git commit -m "Initial deployment"
git push origin main
```

### 1.3 Wait for Build

- Your Space will build automatically (5-10 minutes)
- Watch progress at: `https://huggingface.co/spaces/YOUR_USERNAME/absa-pipeline`
- Once ready, test at: `https://YOUR_USERNAME-absa-pipeline.hf.space`

**✅ Checkpoint:** You should see a Gradio interface where you can test the analysis.

---

## Step 2: Deploy API Gateway to Cloudflare Workers (2 minutes)

### 2.1 Install Wrangler

```bash
npm install -g wrangler
wrangler login
```

### 2.2 Update Configuration

Edit `cloudflare-worker/wrangler.toml`:

```toml
name = "absa-pipeline-api"

[vars]
HF_SPACE_URL = "https://YOUR_USERNAME-absa-pipeline.hf.space"
```

Edit `cloudflare-worker/worker.js` line 10:

```javascript
const HF_SPACE_URL = 'https://YOUR_USERNAME-absa-pipeline.hf.space';
```

### 2.3 Deploy

```bash
cd cloudflare-worker
wrangler deploy
```

You'll get a URL like: `https://absa-pipeline-api.YOUR_SUBDOMAIN.workers.dev`

**✅ Checkpoint:** Test your Worker:

```bash
curl -X POST https://absa-pipeline-api.YOUR_SUBDOMAIN.workers.dev/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"data": ["The hotel was great"]}'
```

---

## Step 3: Deploy Frontend to Cloudflare Pages (3 minutes)

### 3.1 Update API Endpoint

Create `frontend/.env`:

```bash
VITE_API_ENDPOINT=https://absa-pipeline-api.YOUR_SUBDOMAIN.workers.dev/api/analyze
```

### 3.2 Build and Deploy

```bash
cd frontend

# Install dependencies
npm install

# Build
npm run build

# Deploy
npx wrangler pages deploy dist --project-name=absa-pipeline
```

Your site will be live at: `https://absa-pipeline.pages.dev`

**✅ Checkpoint:** Open the URL in your browser and test the analysis!

---

## 🎉 You're Done!

Your ABSA Pipeline is now live with:

- 🤗 **Backend**: https://YOUR_USERNAME-absa-pipeline.hf.space
- ⚡ **API Gateway**: https://absa-pipeline-api.YOUR_SUBDOMAIN.workers.dev
- 🌐 **Frontend**: https://absa-pipeline.pages.dev

## Next Steps

1. **Custom Domain** (Optional)
   - Add custom domain in Cloudflare Pages settings
   - Update DNS records

2. **Monitoring**
   - Check Cloudflare Analytics
   - Monitor Hugging Face Space logs

3. **Optimization**
   - Enable caching in Workers
   - Configure rate limiting
   - Setup CDN rules

## Common Issues

### Backend not responding?
- Check HF Space build logs
- Verify GPU is enabled (T4 Medium)
- Wait 5-10 minutes for models to load

### Worker returns 502?
- Verify HF_SPACE_URL is correct
- Check HF Space is running
- Look at Worker logs: `wrangler tail`

### Frontend shows errors?
- Verify VITE_API_ENDPOINT is set
- Check browser console for errors
- Ensure Worker CORS is enabled

## Need Help?

- 📖 [Full Documentation](DEPLOYMENT.md)
- 💬 [GitHub Issues](https://github.com/YOUR_REPO/issues)
- 🤗 [Hugging Face Discussions](https://discuss.huggingface.co)

---

## Alternative: Docker Deployment (Local Testing)

```bash
# Build Docker image
docker build -f Dockerfile.hf -t absa-pipeline .

# Run locally
docker run -p 7860:7860 absa-pipeline

# Access at http://localhost:7860
```

## Automated Deployment Scripts

We provide helper scripts:

```bash
# Deploy to Hugging Face
./deploy-hf.sh

# Deploy to Cloudflare
./deploy-cloudflare.sh
```

Remember to configure your credentials first!
