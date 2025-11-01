# 🏗️ System Architecture

## Overview

The ABSA Pipeline uses a modern 3-tier architecture with edge computing, API gateway, and GPU-accelerated ML inference.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER LAYER                              │
│                                                                 │
│  🌐 Web Browser                  📱 Mobile App                 │
│     │                                │                          │
│     └────────────────┬───────────────┘                          │
│                      │                                          │
└──────────────────────┼──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                    EDGE LAYER (Cloudflare)                       │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              Cloudflare CDN + Pages                      │  │
│  │  • Global Edge Network (200+ cities)                     │  │
│  │  • Static Asset Delivery                                 │  │
│  │  • React Frontend (SPA)                                  │  │
│  │  • Auto-minification & Compression                       │  │
│  │  • DDoS Protection                                       │  │
│  │  • SSL/TLS Termination                                   │  │
│  └──────────────────────────────────────────────────────────┘  │
│                            │                                    │
│                            ▼                                    │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │           Cloudflare Workers (API Gateway)               │  │
│  │  • Request Routing & Forwarding                          │  │
│  │  • Smart Caching (KV Storage)                            │  │
│  │  • Rate Limiting (per IP)                                │  │
│  │  • CORS Handling                                         │  │
│  │  • Security Headers                                      │  │
│  │  • Retry Logic with Exponential Backoff                  │  │
│  │  • Request/Response Transformation                       │  │
│  │  • Error Handling & Logging                              │  │
│  └──────────────────────────────────────────────────────────┘  │
│                            │                                    │
└────────────────────────────┼────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│              ML INFERENCE LAYER (Hugging Face)                  │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              Hugging Face Spaces (T4 GPU)                │  │
│  │                                                          │  │
│  │  ┌────────────────────────────────────────────────────┐ │  │
│  │  │          Gradio Application (app.py)               │ │  │
│  │  │  • API Endpoint: /api/predict                      │ │  │
│  │  │  • Health Check: /                                 │ │  │
│  │  │  • Request Queue Management                        │ │  │
│  │  └────────────────────────────────────────────────────┘ │  │
│  │                          │                              │  │
│  │  ┌───────────────────────┼───────────────────────────┐ │  │
│  │  │           ML Model Pipeline                       │ │  │
│  │  │                       │                           │ │  │
│  │  │  ┌────────────────────▼──────────────────────┐   │ │  │
│  │  │  │  1. Qwen LLM (4-bit quantized)          │   │ │  │
│  │  │  │     • Clause Splitting                    │   │ │  │
│  │  │  │     • Term Extraction                     │   │ │  │
│  │  │  │     • Opinion Extraction                  │   │ │  │
│  │  │  │     Model: Qwen2.5-3B-Instruct            │   │ │  │
│  │  │  └────────────────────┬──────────────────────┘   │ │  │
│  │  │                       │                           │ │  │
│  │  │  ┌────────────────────▼──────────────────────┐   │ │  │
│  │  │  │  2. RoBERTa (Fine-tuned with LoRA)       │   │ │  │
│  │  │  │     • Category Classification             │   │ │  │
│  │  │  │     Categories: Amenity, Branding,        │   │ │  │
│  │  │  │       Experience, Facility, Loyalty,      │   │ │  │
│  │  │  │       Service                             │   │ │  │
│  │  │  │     Model: roberta_lora_goal/             │   │ │  │
│  │  │  └────────────────────┬──────────────────────┘   │ │  │
│  │  │                       │                           │ │  │
│  │  │  ┌────────────────────▼──────────────────────┐   │ │  │
│  │  │  │  3. DeBERTa (Pre-trained ABSA)           │   │ │  │
│  │  │  │     • Sentiment Polarity Detection        │   │ │  │
│  │  │  │     Labels: Positive, Negative, Neutral   │   │ │  │
│  │  │  │     Model: deberta-v3-base-absa-v1.1      │   │ │  │
│  │  │  └───────────────────────────────────────────┘   │ │  │
│  │  │                                                   │ │  │
│  │  └───────────────────────────────────────────────────┘ │  │
│  │                                                        │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                               │
└───────────────────────────────────────────────────────────────┘
```

## Data Flow

### Request Flow

```
1. User Input
   │
   ├─ User enters review text in browser
   │
   ▼
2. Frontend Processing (React)
   │
   ├─ Validate input
   ├─ Show loading state
   ├─ Format request
   │
   ▼
3. Edge Layer (Cloudflare Pages → Workers)
   │
   ├─ Check cache for identical request
   │  └─ If cached: Return immediately
   │
   ├─ Apply rate limiting
   │  └─ If exceeded: Return 429 error
   │
   ├─ Add security headers
   ├─ Handle CORS
   │
   ▼
4. API Gateway (Cloudflare Worker)
   │
   ├─ Forward request to HF Space
   ├─ Implement retry logic (3 attempts)
   ├─ Exponential backoff on failure
   │
   ▼
5. ML Inference (Hugging Face Space)
   │
   ├─ Queue request (Gradio)
   ├─ Load models (cached after first load)
   │
   ├─ Step 1: Clause Splitting & Term Extraction
   │  └─ Qwen LLM generates clauses and terms
   │
   ├─ Step 2: Opinion Extraction
   │  └─ Qwen LLM extracts opinion words
   │
   ├─ Step 3: Category Classification
   │  └─ RoBERTa classifies aspect categories
   │
   ├─ Step 4: Polarity Detection
   │  └─ DeBERTa detects sentiment polarity
   │
   ▼
6. Response Flow
   │
   ├─ Format results as DataFrame
   ├─ Return JSON response
   │
   ▼
7. API Gateway Processing
   │
   ├─ Cache successful response (1 hour TTL)
   ├─ Add cache headers
   ├─ Transform response if needed
   │
   ▼
8. Frontend Display
   │
   ├─ Parse JSON response
   ├─ Render result cards
   ├─ Show color-coded sentiment badges
   └─ Display analysis metrics
```

## Component Details

### Frontend (Cloudflare Pages)

**Technology:**
- React 18+ with TypeScript
- Vite for building
- Modern CSS with gradients and animations

**Features:**
- Responsive design (mobile-first)
- Loading states with spinners
- Error boundaries and handling
- Example sentences
- Real-time results display
- Color-coded sentiment badges

**Performance:**
- Bundle size: < 500KB
- Tree-shaking and code splitting
- Lazy loading of components
- Service Worker for offline support (optional)

### API Gateway (Cloudflare Workers)

**Technology:**
- JavaScript (ES6+)
- Cloudflare Workers runtime
- KV storage for caching (optional)

**Features:**
- Request routing and forwarding
- Smart caching with SHA-256 hashing
- Rate limiting (100 req/min per IP)
- CORS handling with preflight
- Security headers (CSP, XSS, etc.)
- Retry logic with exponential backoff
- Health check endpoint
- Error handling and logging

**Performance:**
- Edge computing (0ms cold start)
- Global distribution
- Sub-10ms overhead
- Automatic scaling

### Backend (Hugging Face Spaces)

**Technology:**
- Python 3.10+
- Gradio 4.0+ for API
- PyTorch with CUDA support
- Transformers library
- PEFT for LoRA

**Models:**

1. **Qwen LLM (Qwen2.5-3B-Instruct)**
   - Size: 3B parameters
   - Quantization: 4-bit (NF4)
   - Memory: ~2GB VRAM
   - Purpose: Text generation tasks

2. **RoBERTa (Fine-tuned)**
   - Base: roberta-base
   - Fine-tuning: LoRA (r=64)
   - Classes: 6 categories
   - Purpose: Category classification

3. **DeBERTa (deberta-v3-base-absa-v1.1)**
   - Pre-trained for ABSA
   - Classes: Positive, Negative, Neutral
   - Purpose: Sentiment polarity

**Optimization:**
- 4-bit quantization for Qwen
- Model caching after first load
- Batch processing support
- GPU memory optimization
- Request queuing

## Infrastructure

### Cloudflare

**Advantages:**
- 200+ data centers globally
- DDoS protection included
- Free tier available
- Auto-scaling
- 99.99% uptime SLA

**Costs:**
- Workers: Free tier (100K requests/day)
- Pages: Free tier (unlimited requests)
- KV Storage: $0.50/GB/month

### Hugging Face Spaces

**Advantages:**
- Free GPU (T4)
- Gradio integration
- Git-based deployment
- Automatic builds
- Model hosting included

**Costs:**
- Free tier: T4 Small (limited hours)
- Paid: T4 Medium ($0.60/hour)
- GPU upgrades available

## Security

### Network Security
- HTTPS only (TLS 1.3)
- DDoS protection (Cloudflare)
- Rate limiting per IP
- CORS policy enforcement

### Application Security
- Input validation and sanitization
- XSS protection headers
- CSP headers
- No secrets in code
- Environment variables for config

### Data Privacy
- No data persistence by default
- Optional caching with TTL
- GDPR compliant (EU data centers available)
- No tracking or analytics by default

## Monitoring & Observability

### Metrics

**Frontend (Cloudflare Pages):**
- Page views
- Core Web Vitals
- Error rates
- Geographic distribution

**API Gateway (Workers):**
- Request count
- Response times
- Cache hit rate
- Error rates
- Rate limit hits

**Backend (HF Spaces):**
- Request queue length
- GPU utilization
- Model inference time
- Error logs

### Logging

```javascript
// Worker logs
wrangler tail

// HF Space logs
// Available in Space settings
```

## Scaling Considerations

### Horizontal Scaling
- **Frontend**: Automatic (Cloudflare CDN)
- **API Gateway**: Automatic (Workers)
- **Backend**: Upgrade GPU or add replicas

### Vertical Scaling
- **Backend GPU**: 
  - T4 Small → T4 Medium → A10G → A100

### Caching Strategy
1. Browser cache (static assets)
2. CDN cache (Cloudflare)
3. Worker cache (KV storage)
4. Model cache (GPU memory)

## Disaster Recovery

### Backup Strategy
- Git repository (source code)
- Model weights (Hugging Face Hub)
- Configuration in version control

### Failover
- Multiple Workers (automatic)
- HF Space restart (automatic)
- Health checks every 30s

## Future Enhancements

1. **Performance**
   - Model quantization (8-bit → 4-bit)
   - Batch processing optimization
   - Response streaming

2. **Features**
   - Multi-language support
   - Aspect extraction improvements
   - Fine-grained sentiment

3. **Infrastructure**
   - Multiple HF Space replicas
   - Load balancing
   - A/B testing framework

4. **Observability**
   - Distributed tracing
   - Custom metrics dashboard
   - Alerting system

## Cost Estimation

### Monthly Costs (Moderate Usage)

```
Cloudflare Workers:    $0 - $5/month    (< 10M requests)
Cloudflare Pages:      $0               (unlimited)
HF Spaces (T4 Medium): $432/month       (24/7 operation)
                       $0 - $50/month   (intermittent use)

Total: $0 - $487/month
```

### Optimization Tips
- Use HF Spaces only when needed (startup time: 2-3 min)
- Implement aggressive caching
- Use rate limiting to control costs
- Consider serverless alternatives for low traffic

---

## References

- [Cloudflare Workers Docs](https://developers.cloudflare.com/workers/)
- [Cloudflare Pages Docs](https://developers.cloudflare.com/pages/)
- [Hugging Face Spaces Docs](https://huggingface.co/docs/hub/spaces)
- [Gradio Documentation](https://gradio.app/docs/)
