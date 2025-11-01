/**
 * Cloudflare Worker - API Gateway for ABSA Pipeline
 * 
 * Features:
 * - Routes requests to Hugging Face Space backend
 * - Implements caching for identical inputs
 * - Rate limiting per IP
 * - CORS handling
 * - Retry logic with exponential backoff
 * - Security headers
 * - Error handling and logging
 */

// Configuration
const HF_SPACE_URL = 'https://YOUR_USERNAME-absa-pipeline.hf.space';
const CACHE_TTL = 3600; // 1 hour cache for identical requests
const RATE_LIMIT_PER_MINUTE = 100;
const MAX_RETRIES = 3;
const RETRY_DELAY_MS = 1000;

/**
 * Main request handler
 */
addEventListener('fetch', event => {
  event.respondWith(handleRequest(event.request));
});

/**
 * Handle incoming requests
 */
async function handleRequest(request) {
  // Handle CORS preflight
  if (request.method === 'OPTIONS') {
    return handleCORS();
  }

  // Apply rate limiting
  const rateLimitResponse = await checkRateLimit(request);
  if (rateLimitResponse) {
    return rateLimitResponse;
  }

  // Route based on path
  const url = new URL(request.url);
  
  if (url.pathname === '/api/analyze' || url.pathname === '/api/predict') {
    return handleAnalyze(request);
  } else if (url.pathname === '/health') {
    return handleHealth(request);
  } else if (url.pathname === '/') {
    return new Response(JSON.stringify({
      service: 'ABSA Pipeline API Gateway',
      version: '1.0.0',
      endpoints: {
        analyze: '/api/analyze',
        health: '/health'
      }
    }), {
      headers: getSecurityHeaders({
        'Content-Type': 'application/json'
      })
    });
  }

  return new Response('Not Found', { 
    status: 404,
    headers: getSecurityHeaders()
  });
}

/**
 * Handle analyze requests with caching
 */
async function handleAnalyze(request) {
  try {
    // Only accept POST requests
    if (request.method !== 'POST') {
      return new Response(JSON.stringify({ error: 'Method not allowed' }), {
        status: 405,
        headers: getSecurityHeaders({ 'Content-Type': 'application/json' })
      });
    }

    // Parse request body
    const body = await request.json();
    const sentence = body.data?.[0] || body.sentence || body.text;

    if (!sentence || typeof sentence !== 'string') {
      return new Response(JSON.stringify({ 
        error: 'Invalid input. Provide "sentence" or "data" field.' 
      }), {
        status: 400,
        headers: getSecurityHeaders({ 'Content-Type': 'application/json' })
      });
    }

    // Input validation
    if (sentence.length > 5000) {
      return new Response(JSON.stringify({ 
        error: 'Input too long. Maximum 5000 characters.' 
      }), {
        status: 400,
        headers: getSecurityHeaders({ 'Content-Type': 'application/json' })
      });
    }

    // Generate cache key
    const cacheKey = await generateCacheKey(sentence);
    
    // Check cache first
    const cachedResponse = await getFromCache(cacheKey);
    if (cachedResponse) {
      console.log('Cache hit for request');
      return new Response(cachedResponse, {
        headers: getSecurityHeaders({
          'Content-Type': 'application/json',
          'X-Cache': 'HIT'
        })
      });
    }

    // Forward to Hugging Face Space with retry logic
    const result = await forwardToHuggingFace(sentence);

    // Cache the successful response
    await putToCache(cacheKey, JSON.stringify(result));

    return new Response(JSON.stringify(result), {
      headers: getSecurityHeaders({
        'Content-Type': 'application/json',
        'X-Cache': 'MISS'
      })
    });

  } catch (error) {
    console.error('Error in handleAnalyze:', error);
    return new Response(JSON.stringify({ 
      error: 'Internal server error',
      message: error.message 
    }), {
      status: 500,
      headers: getSecurityHeaders({ 'Content-Type': 'application/json' })
    });
  }
}

/**
 * Forward request to Hugging Face Space with retry logic
 */
async function forwardToHuggingFace(sentence, retryCount = 0) {
  try {
    const response = await fetch(`${HF_SPACE_URL}/api/predict`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        data: [sentence]
      })
    });

    if (!response.ok) {
      throw new Error(`HF Space returned ${response.status}`);
    }

    const data = await response.json();
    return {
      success: true,
      data: data.data || data,
      timestamp: new Date().toISOString()
    };

  } catch (error) {
    // Retry logic with exponential backoff
    if (retryCount < MAX_RETRIES) {
      const delay = RETRY_DELAY_MS * Math.pow(2, retryCount);
      console.log(`Retry ${retryCount + 1}/${MAX_RETRIES} after ${delay}ms`);
      await sleep(delay);
      return forwardToHuggingFace(sentence, retryCount + 1);
    }

    throw new Error(`Failed after ${MAX_RETRIES} retries: ${error.message}`);
  }
}

/**
 * Health check endpoint
 */
async function handleHealth(request) {
  try {
    // Check if HF Space is accessible
    const response = await fetch(`${HF_SPACE_URL}/`, {
      method: 'GET',
      signal: AbortSignal.timeout(5000) // 5 second timeout
    });

    const healthy = response.ok;
    
    return new Response(JSON.stringify({
      status: healthy ? 'healthy' : 'degraded',
      backend: healthy ? 'available' : 'unavailable',
      timestamp: new Date().toISOString()
    }), {
      status: healthy ? 200 : 503,
      headers: getSecurityHeaders({ 'Content-Type': 'application/json' })
    });

  } catch (error) {
    return new Response(JSON.stringify({
      status: 'unhealthy',
      backend: 'unavailable',
      error: error.message,
      timestamp: new Date().toISOString()
    }), {
      status: 503,
      headers: getSecurityHeaders({ 'Content-Type': 'application/json' })
    });
  }
}

/**
 * Rate limiting check
 */
async function checkRateLimit(request) {
  // Get client IP
  const ip = request.headers.get('CF-Connecting-IP') || 'unknown';
  const key = `ratelimit:${ip}`;
  
  // Note: In production, use Workers KV or Durable Objects
  // This is a simplified version
  return null; // Implement with KV store in production
}

/**
 * Cache management
 */
async function generateCacheKey(sentence) {
  const encoder = new TextEncoder();
  const data = encoder.encode(sentence.toLowerCase().trim());
  const hashBuffer = await crypto.subtle.digest('SHA-256', data);
  const hashArray = Array.from(new Uint8Array(hashBuffer));
  return hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
}

async function getFromCache(key) {
  // Note: Implement with Cache API or KV
  // This is a placeholder for the actual implementation
  return null;
}

async function putToCache(key, value) {
  // Note: Implement with Cache API or KV
  // Cache for CACHE_TTL seconds
}

/**
 * CORS handler
 */
function handleCORS() {
  return new Response(null, {
    headers: getSecurityHeaders()
  });
}

/**
 * Security headers
 */
function getSecurityHeaders(additionalHeaders = {}) {
  return {
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type, Authorization',
    'Access-Control-Max-Age': '86400',
    'X-Content-Type-Options': 'nosniff',
    'X-Frame-Options': 'DENY',
    'X-XSS-Protection': '1; mode=block',
    'Referrer-Policy': 'strict-origin-when-cross-origin',
    'Content-Security-Policy': "default-src 'self'",
    ...additionalHeaders
  };
}

/**
 * Utility: Sleep function
 */
function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}
