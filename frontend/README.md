# ABSA Pipeline Frontend

React + TypeScript frontend for the ABSA Pipeline application.

## 🚀 Quick Start

### Development

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Open http://localhost:3000
```

### Build for Production

```bash
# Build optimized bundle
npm run build

# Preview production build
npm run preview
```

### Lint and Type Check

```bash
# Lint code
npm run lint

# Type check
npm run type-check
```

## 📦 Project Structure

```
frontend/
├── src/
│   ├── App.tsx          # Main application component
│   ├── App.css          # Styles with gradients and animations
│   ├── main.tsx         # Entry point
│   └── vite-env.d.ts    # TypeScript declarations
├── public/
│   └── index.html       # HTML template
├── functions/
│   └── _middleware.ts   # Cloudflare Pages middleware
├── package.json         # Dependencies
├── tsconfig.json        # TypeScript config
├── vite.config.ts       # Vite config
└── README.md           # This file
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file:

```bash
VITE_API_ENDPOINT=https://your-worker-url/api/analyze
```

### Update API Endpoint

In `src/App.tsx`, update the API endpoint:

```typescript
const API_ENDPOINT = import.meta.env.VITE_API_ENDPOINT || 
  'https://absa-pipeline-api.YOUR_SUBDOMAIN.workers.dev/api/analyze';
```

## 🎨 Features

- **Modern UI**: Gradient backgrounds, animations, and hover effects
- **TypeScript**: Full type safety
- **Responsive**: Mobile-first design
- **Loading States**: Spinners and loading indicators
- **Error Handling**: User-friendly error messages
- **Examples**: Pre-filled example sentences
- **Real-time Results**: Displays analysis results with color-coded badges

## 📊 Performance

- Bundle size: < 500KB (optimized)
- Vite build with tree-shaking
- Code splitting for vendors
- Minified production build

## 🚀 Deployment to Cloudflare Pages

### Option 1: Wrangler CLI

```bash
npm run build
npx wrangler pages deploy dist --project-name=absa-pipeline
```

### Option 2: Git Integration

1. Push to GitHub
2. Connect repository in Cloudflare Pages
3. Configure build settings:
   - Build command: `npm run build`
   - Build output: `dist`
   - Root directory: `frontend`

## 🔒 Security

- CSP headers via middleware
- XSS protection
- Input sanitization
- HTTPS only

## 📄 License

MIT License
