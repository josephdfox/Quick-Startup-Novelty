
import React, { useState, useEffect, useRef, useCallback } from 'react';
import Navbar from './components/Navbar';
import SimilarityMap from './components/SimilarityMap';
import { AnalysisResult, ScatterPoint } from './types';
import { toPng } from 'html-to-image';

// Database item structure
interface DBItem {
  pitch: string;
  embedding?: Float32Array;
  x?: number;
  y?: number;
}

// Inlined CSV data to prevent fetch errors on GitHub Pages subdirectories
const REGISTRY_CSV = `pitch
Uber for dog walking in urban areas
Airbnb for high-end photography studios
SaaS platform for automated tax filing for freelancers
Social network for vintage car collectors
Marketplace for sustainable building materials
AI-powered coding assistant for legacy systems
On-demand drone delivery for medical supplies in rural areas
Blockchain-based voting system for corporate governance
Subscription box for rare succulents and cacti
Direct-to-consumer sustainable furniture brand`;

const App: React.FC = () => {
  const [pitch, setPitch] = useState('');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [isCapturing, setIsCapturing] = useState(false);
  const [isReady, setIsReady] = useState(false);
  const [statusMessage, setStatusMessage] = useState('Initializing Engine...');
  const [modelProgress, setModelProgress] = useState(0);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [corpusPoints, setCorpusPoints] = useState<ScatterPoint[]>([]);
  
  const extractorRef = useRef<any>(null);
  const databaseRef = useRef<DBItem[]>([]);
  const mapContainerRef = useRef<HTMLDivElement>(null);

  // Helper to calculate cosine similarity
  const cosineSimilarity = (a: Float32Array, b: Float32Array) => {
    let dotProduct = 0;
    let mA = 0;
    let mB = 0;
    for (let i = 0; i < a.length; i++) {
      dotProduct += a[i] * b[i];
      mA += a[i] * a[i];
      mB += b[i] * b[i];
    }
    const mag = Math.sqrt(mA) * Math.sqrt(mB);
    return mag === 0 ? 0 : dotProduct / mag;
  };

  const getPositionFromVector = (vector: Float32Array): { x: number; y: number } => {
    const half = Math.floor(vector.length / 2);
    let xSum = 0;
    let ySum = 0;
    for (let i = 0; i < half; i++) xSum += vector[i];
    for (let i = half; i < vector.length; i++) ySum += vector[i];
    
    return {
      x: Math.max(-100, Math.min(100, (xSum / half) * 400)),
      y: Math.max(-100, Math.min(100, (ySum / half) * 400))
    };
  };

  const loadAndIndexData = useCallback(async () => {
    try {
      // 1. Load Model
      setStatusMessage('Loading Semantic Model...');
      const { pipeline, env } = await import('@xenova/transformers');
      
      env.allowLocalModels = false;
      env.useBrowserCache = true;
      env.remoteHost = 'https://huggingface.co';

      extractorRef.current = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2', {
        progress_callback: (data: any) => {
          if (data.status === 'progress') setModelProgress(data.progress);
        }
      });
      setModelProgress(100);

      // 2. Index Database from inlined string
      setStatusMessage(`Indexing Registry...`);
      const rows = REGISTRY_CSV.split('\n').slice(1).filter(r => r.trim().length > 5);
      const indexedItems: DBItem[] = [];
      const points: ScatterPoint[] = [];

      for (let i = 0; i < rows.length; i++) {
        const text = rows[i].replace(/^"|"$/g, '').trim();
        try {
          const output = await extractorRef.current(text, { pooling: 'mean', normalize: true });
          const vector = output.data as Float32Array;
          const pos = getPositionFromVector(vector);
          
          indexedItems.push({ pitch: text, embedding: vector, ...pos });
          points.push({ id: `item-${i}`, ...pos, isUser: false });
        } catch (e) {
          console.warn(`Skipping item ${i}`);
        }
        
        if (i % 2 === 0 || i === rows.length - 1) {
           setStatusMessage(`Indexing: ${Math.round(((i + 1) / rows.length) * 100)}%`);
        }
      }

      databaseRef.current = indexedItems;
      setCorpusPoints(points);
      setIsReady(true);
      setStatusMessage('System Ready');
    } catch (err) {
      console.error('Initialization failed:', err);
      setStatusMessage(`Error: ${err instanceof Error ? err.message : 'Environment Setup Failed'}`);
    }
  }, []);

  useEffect(() => {
    loadAndIndexData();
  }, [loadAndIndexData]);

  const handleAnalyze = async () => {
    if (!pitch || pitch.length < 15) {
      alert("Please provide a more descriptive pitch (min 15 chars).");
      return;
    }
    
    setIsAnalyzing(true);
    setResult(null);

    try {
      const output = await extractorRef.current(pitch, { pooling: 'mean', normalize: true });
      const userVector = output.data as Float32Array;
      const userPos = getPositionFromVector(userVector);

      let maxSim = 0;
      databaseRef.current.forEach(item => {
        if (item.embedding) {
          const sim = cosineSimilarity(userVector, item.embedding);
          if (sim > maxSim) maxSim = sim;
        }
      });

      const novelty = Math.max(1, Math.min(10, Math.round((1 - maxSim) * 10)));
      
      let title = "Moderate Novelty";
      let desc = "Your pitch shows some unique angles but overlaps with established semantic patterns.";

      if (maxSim > 0.8) {
        title = "Highly Saturated";
        desc = "Extremely high similarity detected. This concept is very similar to existing market models.";
      } else if (maxSim < 0.4) {
        title = "High Innovation Area";
        desc = "Your idea is semantically distinct from our registry, suggesting a strong unique value prop.";
      }

      setResult({
        x: userPos.x,
        y: userPos.y,
        noveltyScore: novelty,
        localSimilarityScore: maxSim,
        assessmentTitle: title,
        assessmentDescription: desc
      });
    } catch (error) {
      console.error("Analysis failed:", error);
      alert("Analysis error.");
    } finally {
      setIsAnalyzing(false);
    }
  };

  const saveScreenshot = async () => {
    if (!mapContainerRef.current) return;
    setIsCapturing(true);
    try {
      const dataUrl = await toPng(mapContainerRef.current, {
        cacheBust: true,
        backgroundColor: '#182d34',
        pixelRatio: 2,
      });
      const link = document.createElement('a');
      link.download = `novelty-plot-${Date.now()}.png`;
      link.href = dataUrl;
      link.click();
    } catch (err) {
      console.error('Screenshot failed:', err);
    } finally {
      setIsCapturing(false);
    }
  };

  const userPoint: ScatterPoint | null = result ? {
    id: 'user-idea',
    x: result.x,
    y: result.y,
    isUser: true,
  } : null;

  return (
    <div className="flex min-h-screen w-full flex-col bg-background-dark text-slate-100 selection:bg-primary/30">
      <Navbar />

      {!isReady && (
        <div className="fixed inset-0 z-[100] bg-background-dark/95 backdrop-blur-sm flex flex-col items-center justify-center p-8 text-center">
          <div className="w-full max-w-md flex flex-col gap-8">
            <div className="relative size-24 mx-auto bg-primary/10 rounded-3xl flex items-center justify-center text-primary">
               <span className="material-symbols-outlined text-5xl">memory</span>
            </div>
            <div className="space-y-3">
              <h2 className="text-3xl font-black text-white tracking-tight">{statusMessage}</h2>
              <p className="text-text-dim text-sm leading-relaxed">Preparing local vector engine. All processing happens in your browser.</p>
            </div>
            <div className="w-full h-1.5 bg-surface-dark rounded-full overflow-hidden border border-border-dark">
               <div className="h-full bg-primary transition-all duration-300 shadow-[0_0_10px_#0db9f2]" style={{ width: `${modelProgress}%` }} />
            </div>
          </div>
        </div>
      )}

      <main className="flex-1 flex flex-col items-center w-full px-4 py-12 md:px-8 lg:px-40">
        <div className="flex flex-col w-full max-w-[1100px] gap-12">
          
          <section className="flex flex-col gap-6">
            <h1 className="text-white text-5xl md:text-7xl font-black leading-[1.1] tracking-tighter">
              Novelty <span className="text-primary italic">Index</span>.
            </h1>
            <p className="text-text-dim text-xl max-w-3xl">
              Compare your vision against <span className="text-white font-medium">{corpusPoints.length} existing market patterns</span>.
            </p>
          </section>

          <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
            <div className="lg:col-span-4 flex flex-col gap-6 p-8 rounded-[2rem] bg-surface-dark border border-border-dark shadow-2xl relative">
               <textarea 
                  value={pitch}
                  onChange={(e) => setPitch(e.target.value)}
                  className="form-input w-full resize-none rounded-2xl text-white placeholder:text-text-dim/20 bg-background-dark/50 border border-border-dark focus:border-primary focus:ring-4 focus:ring-primary/10 transition-all p-5 text-sm min-h-[200px]" 
                  placeholder="Describe your startup idea here..."
                />

                <button 
                  onClick={handleAnalyze}
                  disabled={isAnalyzing}
                  className="w-full flex items-center justify-center gap-3 rounded-2xl h-16 bg-primary hover:bg-white text-background-dark font-black text-lg transition-all shadow-xl shadow-primary/20 disabled:opacity-50"
                >
                  {isAnalyzing ? "Analyzing..." : "Compare Idea"}
                </button>
            </div>

            <div className="lg:col-span-8 flex flex-col gap-8">
              <div className="relative">
                {isAnalyzing && <div className="scanning-line"></div>}
                <SimilarityMap ref={mapContainerRef} userPoint={userPoint} corpusPoints={corpusPoints} />
                
                {result && !isAnalyzing && (
                   <div className="absolute top-6 right-6 bg-background-dark/90 backdrop-blur-xl p-6 rounded-[1.5rem] border border-primary/20 shadow-2xl max-w-[280px]">
                      <h4 className="text-white text-lg font-black leading-tight mb-2 tracking-tight">{result.assessmentTitle}</h4>
                      <p className="text-text-dim text-sm">{result.assessmentDescription}</p>
                      <div className="mt-6 flex flex-col gap-2">
                        <div className="flex justify-between text-[10px] text-text-dim font-black uppercase tracking-wider">
                          <span>Novelty Score</span>
                          <span className="text-white">{result.noveltyScore} / 10</span>
                        </div>
                        <div className="w-full h-2 bg-border-dark rounded-full overflow-hidden">
                          <div className="h-full bg-primary rounded-full" style={{ width: `${result.noveltyScore * 10}%` }}></div>
                        </div>
                      </div>
                   </div>
                )}
              </div>

              {result && (
                <div className="p-8 rounded-[2rem] bg-surface-dark border border-border-dark flex flex-col gap-6 shadow-xl">
                  <div className="flex flex-wrap gap-4">
                    <button 
                      onClick={() => { setPitch(''); setResult(null); }} 
                      className="px-6 py-3 rounded-xl bg-background-dark border border-border-dark text-text-dim hover:text-white transition-all text-xs font-black uppercase tracking-widest"
                    >
                      New Plot
                    </button>
                    <button 
                      onClick={saveScreenshot} 
                      disabled={isCapturing}
                      className="px-6 py-3 rounded-xl bg-white text-background-dark hover:bg-primary transition-all text-xs font-black uppercase tracking-widest flex items-center gap-2"
                    >
                      Export Results
                    </button>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </main>
    </div>
  );
};

export default App;
