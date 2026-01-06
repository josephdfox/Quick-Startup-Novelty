
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
    return dotProduct / (Math.sqrt(mA) * Math.sqrt(mB));
  };

  const getPositionFromVector = (vector: Float32Array): { x: number; y: number } => {
    const half = Math.floor(vector.length / 2);
    let xSum = 0;
    let ySum = 0;
    for (let i = 0; i < half; i++) xSum += vector[i];
    for (let i = half; i < vector.length; i++) ySum += vector[i];
    return {
      x: Math.max(-100, Math.min(100, xSum * 70)),
      y: Math.max(-100, Math.min(100, ySum * 70))
    };
  };

  const loadAndIndexData = useCallback(async () => {
    try {
      // 1. Load Model
      setStatusMessage('Loading Semantic Model (25MB)...');
      const { pipeline } = await import('@xenova/transformers');
      extractorRef.current = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2', {
        progress_callback: (data: any) => {
          if (data.status === 'progress') setModelProgress(data.progress);
        }
      });
      setModelProgress(100);

      // 2. Fetch "Hidden" CSV
      // Tip: Name this file something like 'db_cache_01.csv' in your actual repo
      setStatusMessage('Fetching Idea Database...');
      const response = await fetch('./data.csv');
      const csvText = await response.text();
      
      // Simple CSV Parse (Assumes first row is header 'pitch')
      const rows = csvText.split('\n').slice(1).filter(r => r.trim().length > 5);
      
      // 3. Index Database
      setStatusMessage(`Indexing ${rows.length} reference ideas...`);
      const indexedItems: DBItem[] = [];
      const points: ScatterPoint[] = [];

      for (let i = 0; i < rows.length; i++) {
        const text = rows[i].replace(/^"|"$/g, '').trim();
        const output = await extractorRef.current(text, { pooling: 'mean', normalize: true });
        const vector = output.data as Float32Array;
        const pos = getPositionFromVector(vector);
        
        indexedItems.push({ pitch: text, embedding: vector, ...pos });
        points.push({ id: `item-${i}`, ...pos, isUser: false });
        
        // Progress update every few items
        if (i % 5 === 0) setStatusMessage(`Indexing: ${Math.round((i/rows.length)*100)}%`);
      }

      databaseRef.current = indexedItems;
      setCorpusPoints(points);
      setIsReady(true);
      setStatusMessage('System Ready');
    } catch (err) {
      console.error('Initialization failed:', err);
      setStatusMessage('Error: Failed to load local database.');
    }
  }, []);

  useEffect(() => {
    loadAndIndexData();
  }, [loadAndIndexData]);

  const handleAnalyze = async () => {
    if (!pitch || pitch.length < 15) {
      alert("Please provide a bit more detail about your idea.");
      return;
    }
    
    setIsAnalyzing(true);
    setResult(null);

    try {
      // Generate user embedding
      const output = await extractorRef.current(pitch, { pooling: 'mean', normalize: true });
      const userVector = output.data as Float32Array;
      const userPos = getPositionFromVector(userVector);

      // Find highest similarity against the database
      let maxSim = 0;
      databaseRef.current.forEach(item => {
        if (item.embedding) {
          const sim = cosineSimilarity(userVector, item.embedding);
          if (sim > maxSim) maxSim = sim;
        }
      });

      // Interpretation
      const novelty = Math.round((1 - maxSim) * 10);
      let title = "Moderate Novelty";
      let desc = "Your pitch shows some unique angles but overlaps with established semantic patterns in our database.";

      if (maxSim > 0.8) {
        title = "Strong Semantic Match";
        desc = "High similarity detected with existing concepts. Consider differentiating your core value proposition further.";
      } else if (maxSim < 0.5) {
        title = "Significant Innovation";
        desc = "Low similarity found. Your idea occupies a relatively vacant area in the current startup semantic landscape.";
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
      console.error("Local analysis failed:", error);
      alert("Analysis failed. Please refresh and try again.");
    } finally {
      setIsAnalyzing(false);
    }
  };

  const saveScreenshot = async () => {
    if (!mapContainerRef.current) return;
    setIsCapturing(true);
    try {
      await new Promise(r => setTimeout(r, 100));
      const dataUrl = await toPng(mapContainerRef.current, {
        cacheBust: true,
        backgroundColor: '#182d34',
      });
      const link = document.createElement('a');
      link.download = `novelty-index-${Date.now()}.png`;
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
    <div className="flex min-h-screen w-full flex-col bg-background-dark text-slate-100">
      <Navbar />

      {!isReady && (
        <div className="fixed inset-0 z-[100] bg-background-dark flex flex-col items-center justify-center p-8 text-center">
          <div className="w-full max-w-md flex flex-col gap-6">
            <div className="size-16 mx-auto rounded-2xl bg-primary/20 flex items-center justify-center text-primary animate-pulse">
              <span className="material-symbols-outlined text-4xl">database</span>
            </div>
            <div className="space-y-2">
              <h2 className="text-2xl font-black text-white">{statusMessage}</h2>
              <p className="text-text-dim text-sm">Indexing database locally for zero-latency privacy-first comparison.</p>
            </div>
            <div className="w-full h-2 bg-surface-dark rounded-full overflow-hidden">
               <div className="h-full bg-primary transition-all duration-300" style={{ width: `${modelProgress}%` }} />
            </div>
          </div>
        </div>
      )}

      <main className="flex-1 flex flex-col items-center w-full px-4 py-8 md:px-8 lg:px-40">
        <div className="flex flex-col w-full max-w-[1100px] gap-8">
          
          <section className="flex flex-col gap-4">
            <div className="flex items-center gap-2 text-primary">
              <span className="material-symbols-outlined text-sm">verified</span>
              <span className="text-xs font-bold uppercase tracking-widest">Connected to Local Idea Registry</span>
            </div>
            <h1 className="text-white text-4xl md:text-6xl font-black leading-tight tracking-tighter">
              Novelty <span className="text-primary">Benchmark</span>.
            </h1>
            <p className="text-text-dim text-lg font-normal leading-relaxed max-w-3xl">
              Mapping your concept against our <span className="text-white font-medium">internal database of {corpusPoints.length} startups</span>. 
              The analysis is purely mathematical and runs entirely in your browser.
            </p>
          </section>

          <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
            {/* Input */}
            <div className="lg:col-span-4 flex flex-col gap-6 p-6 rounded-2xl bg-surface-dark border border-border-dark shadow-xl">
               <div className="flex flex-col gap-1">
                  <h3 className="text-white font-bold text-lg">Your Idea</h3>
                  <p className="text-text-dim text-xs">Vector comparison against registry.</p>
               </div>
               
               <textarea 
                  value={pitch}
                  onChange={(e) => setPitch(e.target.value)}
                  className="form-input w-full resize-none rounded-xl text-white placeholder:text-text-dim/30 bg-background-dark border border-border-dark focus:border-primary focus:ring-1 focus:ring-primary/50 transition-all p-4 text-sm font-normal leading-relaxed min-h-[200px]" 
                  placeholder="Summarize your startup..."
                />

                <button 
                  onClick={handleAnalyze}
                  disabled={isAnalyzing}
                  className="w-full flex items-center justify-center gap-2 rounded-xl h-14 bg-primary hover:bg-primary/90 text-background-dark font-black text-lg transition-all shadow-lg shadow-primary/20 active:scale-95 disabled:opacity-50"
                >
                  {isAnalyzing ? (
                    <div className="size-6 border-4 border-background-dark/30 border-t-background-dark rounded-full animate-spin"></div>
                  ) : (
                    <>
                      <span className="material-symbols-outlined">radar</span>
                      <span>Run Comparison</span>
                    </>
                  )}
                </button>

                <div className="p-3 rounded-lg bg-background-dark/50 border border-border-dark text-[10px] text-text-dim uppercase font-bold text-center tracking-widest">
                  Registry Integrity: Verified
                </div>
            </div>

            {/* Viz */}
            <div className="lg:col-span-8 flex flex-col gap-6">
              <div className="relative group">
                {isAnalyzing && <div className="scanning-line"></div>}
                <SimilarityMap ref={mapContainerRef} userPoint={userPoint} corpusPoints={corpusPoints} />
                
                {result && !isAnalyzing && (
                   <div className="absolute top-4 right-4 bg-background-dark/90 backdrop-blur-md p-4 rounded-xl border border-primary/30 shadow-2xl max-w-[240px] animate-in fade-in zoom-in duration-300">
                      <div className="flex justify-between items-center mb-2">
                        <span className="text-[10px] font-bold text-primary uppercase tracking-widest">Match Strength</span>
                        <div className="bg-primary text-background-dark text-xs font-black px-2 py-0.5 rounded">
                          {(result.localSimilarityScore * 100).toFixed(0)}%
                        </div>
                      </div>
                      <h4 className="text-white font-bold leading-tight">{result.assessmentTitle}</h4>
                      <div className="mt-4 flex flex-col gap-1">
                        <div className="flex justify-between text-[10px] text-text-dim font-mono">
                          <span>Novelty Rank</span>
                          <span>{result.noveltyScore}/10</span>
                        </div>
                        <div className="w-full h-1 bg-border-dark rounded-full overflow-hidden">
                          <div className="h-full bg-primary" style={{ width: `${result.noveltyScore * 10}%` }}></div>
                        </div>
                      </div>
                   </div>
                )}
              </div>

              {result && (
                <div className="p-6 rounded-2xl bg-surface-dark border border-border-dark flex flex-col gap-4 animate-in slide-in-from-bottom-4 duration-500">
                  <div className="flex items-center gap-4">
                    <div className="size-12 rounded-full bg-primary/10 flex items-center justify-center text-primary">
                      <span className="material-symbols-outlined text-3xl">account_tree</span>
                    </div>
                    <div>
                      <h2 className="text-xl font-black text-white">Registry Analysis</h2>
                      <p className="text-text-dim text-sm">Positioned via semantic cluster analysis.</p>
                    </div>
                  </div>
                  <p className="text-slate-300 leading-relaxed italic border-l-2 border-primary/50 pl-4 py-1">
                    "{result.assessmentDescription}"
                  </p>
                  <div className="flex gap-4 pt-2">
                    <button onClick={() => { setPitch(''); setResult(null); }} className="px-4 py-2 rounded-lg bg-background-dark border border-border-dark text-text-dim hover:text-white transition-colors text-sm font-bold uppercase tracking-wider">
                      Clear
                    </button>
                    <button 
                      onClick={saveScreenshot} 
                      disabled={isCapturing}
                      className="px-4 py-2 rounded-lg bg-border-dark text-white hover:bg-slate-700 transition-colors text-sm font-bold uppercase tracking-wider flex items-center gap-2"
                    >
                      <span className="material-symbols-outlined text-sm">download</span>
                      {isCapturing ? 'Capturing...' : 'Download Plot'}
                    </button>
                  </div>
                </div>
              )}
            </div>
          </div>

          <section className="grid grid-cols-1 md:grid-cols-4 gap-6 py-12 border-t border-border-dark/50">
            {[
              { label: 'Data Registry', val: 'Loaded Local CSV' },
              { label: 'Privacy', val: 'Client-Side Only' },
              { label: 'Comparison', val: 'Cosine Similarity' },
              { label: 'Registry Size', val: `${corpusPoints.length} Entries` }
            ].map((stat, i) => (
              <div key={i} className="flex flex-col gap-1">
                <span className="text-xs uppercase font-bold text-text-dim/60 tracking-tighter">{stat.label}</span>
                <span className="text-sm font-bold text-white">{stat.val}</span>
              </div>
            ))}
          </section>
        </div>
      </main>
    </div>
  );
};

export default App;
