#!/bin/bash
# Deploy to Cloudflare Workers and Pages

set -e

echo "🚀 Deploying to Cloudflare..."

# Check if wrangler is installed
if ! command -v wrangler &> /dev/null; then
    echo "⚠️  Wrangler CLI not found. Installing..."
    npm install -g wrangler
fi

# Check authentication
echo "🔐 Checking Cloudflare authentication..."
wrangler whoami || {
    echo "❌ Not authenticated. Please run: wrangler login"
    exit 1
}

# Deploy Worker (API Gateway)
echo ""
echo "📡 Deploying Cloudflare Worker (API Gateway)..."
cd cloudflare-worker
wrangler deploy
cd ..

echo "✅ Worker deployed successfully!"

# Deploy Pages (Frontend)
echo ""
echo "🌐 Deploying Cloudflare Pages (Frontend)..."
cd frontend

# Install dependencies
if [ ! -d "node_modules" ]; then
    echo "📦 Installing dependencies..."
    npm install
fi

# Build the project
echo "🔨 Building frontend..."
npm run build

# Deploy to Cloudflare Pages
echo "📤 Deploying to Cloudflare Pages..."
npx wrangler pages deploy dist --project-name=absa-pipeline

cd ..

echo ""
echo "✅ Deployment complete!"
echo ""
echo "🎉 Your application is now live:"
echo "   📡 API Gateway: https://absa-pipeline-api.YOUR_SUBDOMAIN.workers.dev"
echo "   🌐 Frontend: https://absa-pipeline.pages.dev"
echo ""
echo "📝 Next steps:"
echo "   1. Update frontend/.env with your Worker URL"
echo "   2. Update cloudflare-worker/wrangler.toml with your HF Space URL"
echo "   3. Configure custom domain (optional)"
