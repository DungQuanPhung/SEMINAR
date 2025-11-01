import { useState } from 'react';
import './App.css';

// API endpoint - Must be set via environment variable
const API_ENDPOINT = import.meta.env.VITE_API_ENDPOINT;

if (!API_ENDPOINT) {
  console.error('VITE_API_ENDPOINT environment variable is not set');
}

interface AnalysisResult {
  clause: string;
  term: string;
  opinion: string;
  category: string;
  polarity: string;
  polarity_score: number;
  sentence_original: string;
}

interface ApiResponse {
  success: boolean;
  data: AnalysisResult[];
  timestamp?: string;
}

const EXAMPLE_SENTENCES = [
  "The food was great and the staff was very friendly, but the room was a bit small.",
  "The hotel was clean and modern, with excellent service and a beautiful view.",
  "Quy trình check-in rất suôn sẻ và nhân viên vô cùng nhiệt tình. Phòng ốc rộng rãi nhưng wifi chậm.",
  "Nhà hàng trong khách sạn có đồ ăn rất ngon, nhưng dịch vụ tại bàn lại chậm."
];

function App() {
  const [inputText, setInputText] = useState('');
  const [results, setResults] = useState<AnalysisResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleAnalyze = async () => {
    if (!inputText.trim()) {
      setError('Please enter some text to analyze');
      return;
    }

    setLoading(true);
    setError(null);
    setResults([]);

    try {
      const response = await fetch(API_ENDPOINT, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          data: [inputText]
        }),
      });

      if (!response.ok) {
        throw new Error(`API error: ${response.status} ${response.statusText}`);
      }

      const data: ApiResponse = await response.json();
      
      if (data.success && data.data) {
        // Handle both array and nested array responses
        const results = Array.isArray(data.data) && data.data.length > 0 && Array.isArray(data.data[0]) 
          ? data.data[0] 
          : data.data;
        setResults(Array.isArray(results) ? results : []);
      } else {
        throw new Error('Invalid response format from API');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred while analyzing');
      console.error('Analysis error:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleExampleClick = (example: string) => {
    setInputText(example);
    setResults([]);
    setError(null);
  };

  const getPolarityColor = (polarity: string): string => {
    switch (polarity.toLowerCase()) {
      case 'positive':
        return 'polarity-positive';
      case 'negative':
        return 'polarity-negative';
      default:
        return 'polarity-neutral';
    }
  };

  return (
    <div className="app">
      {/* Header */}
      <header className="header">
        <div className="header-content">
          <h1 className="header-title">
            🔍 ABSA Pipeline
          </h1>
          <p className="header-subtitle">
            Aspect-Based Sentiment Analysis
          </p>
        </div>
      </header>

      {/* Main Content */}
      <main className="main-content">
        <div className="container">
          {/* Input Section */}
          <div className="input-section card">
            <h2 className="section-title">Enter Your Review</h2>
            <textarea
              className="text-input"
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              placeholder="Enter a customer review to analyze aspects, opinions, and sentiment..."
              rows={6}
              disabled={loading}
            />
            
            <button
              className="analyze-button"
              onClick={handleAnalyze}
              disabled={loading || !inputText.trim()}
            >
              {loading ? (
                <>
                  <span className="spinner"></span>
                  Analyzing...
                </>
              ) : (
                <>
                  🚀 Analyze Review
                </>
              )}
            </button>

            {error && (
              <div className="error-message">
                ⚠️ {error}
              </div>
            )}
          </div>

          {/* Examples Section */}
          <div className="examples-section">
            <h3 className="examples-title">Try these examples:</h3>
            <div className="examples-grid">
              {EXAMPLE_SENTENCES.map((example, idx) => (
                <button
                  key={idx}
                  className="example-button"
                  onClick={() => handleExampleClick(example)}
                  disabled={loading}
                >
                  {example.substring(0, 80)}...
                </button>
              ))}
            </div>
          </div>

          {/* Results Section */}
          {results.length > 0 && (
            <div className="results-section">
              <h2 className="section-title">Analysis Results</h2>
              <div className="results-grid">
                {results.map((result, idx) => (
                  <div key={idx} className="result-card card">
                    <div className="result-header">
                      <span className="result-number">#{idx + 1}</span>
                      <span className={`polarity-badge ${getPolarityColor(result.polarity)}`}>
                        {result.polarity}
                      </span>
                    </div>
                    
                    <div className="result-content">
                      <div className="result-field">
                        <span className="field-label">Clause:</span>
                        <span className="field-value">{result.clause}</span>
                      </div>
                      
                      <div className="result-field">
                        <span className="field-label">Term:</span>
                        <span className="field-value highlight-term">{result.term || 'N/A'}</span>
                      </div>
                      
                      <div className="result-field">
                        <span className="field-label">Opinion:</span>
                        <span className="field-value highlight-opinion">{result.opinion || 'N/A'}</span>
                      </div>
                      
                      <div className="result-field">
                        <span className="field-label">Category:</span>
                        <span className="field-value category-badge">{result.category}</span>
                      </div>
                      
                      <div className="result-field">
                        <span className="field-label">Confidence:</span>
                        <span className="field-value">{(result.polarity_score * 100).toFixed(1)}%</span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Info Section */}
          <div className="info-section card">
            <h3 className="info-title">About the Pipeline</h3>
            <div className="info-grid">
              <div className="info-item">
                <div className="info-icon">🔤</div>
                <div className="info-text">
                  <strong>Clause Splitting</strong>
                  <p>Breaks reviews into meaningful clauses</p>
                </div>
              </div>
              
              <div className="info-item">
                <div className="info-icon">🎯</div>
                <div className="info-text">
                  <strong>Aspect Extraction</strong>
                  <p>Identifies terms and aspects discussed</p>
                </div>
              </div>
              
              <div className="info-item">
                <div className="info-icon">💭</div>
                <div className="info-text">
                  <strong>Opinion Mining</strong>
                  <p>Extracts opinion words and phrases</p>
                </div>
              </div>
              
              <div className="info-item">
                <div className="info-icon">📊</div>
                <div className="info-text">
                  <strong>Category Classification</strong>
                  <p>Classifies into aspect categories</p>
                </div>
              </div>
              
              <div className="info-item">
                <div className="info-icon">😊</div>
                <div className="info-text">
                  <strong>Sentiment Analysis</strong>
                  <p>Detects positive, negative, or neutral</p>
                </div>
              </div>
              
              <div className="info-item">
                <div className="info-icon">⚡</div>
                <div className="info-text">
                  <strong>Fast & Cached</strong>
                  <p>Edge computing with global CDN</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="footer">
        <p>
          Powered by <strong>Qwen LLM</strong>, <strong>RoBERTa</strong>, and <strong>DeBERTa</strong>
        </p>
        <p className="footer-tech">
          Deployed on Cloudflare Pages • API on Cloudflare Workers • Models on Hugging Face Spaces
        </p>
      </footer>
    </div>
  );
}

export default App;
