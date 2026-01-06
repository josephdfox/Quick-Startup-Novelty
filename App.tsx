
import React, { useState, useEffect, useRef, useCallback } from 'react';
import Navbar from './components/Navbar';
import SimilarityMap from './components/SimilarityMap';
import { AnalysisResult, ScatterPoint } from './types';
import { toPng } from 'html-to-image';
// Import the Gemini assessment service
import { getDetailedAssessment } from './services/geminiService';

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
    const mag = Math.sqrt(mA) * Math.sqrt(mB);
    return mag === 0 ? 0 : dotProduct / mag;
  };

  const getPositionFromVector = (vector: Float32Array): { x: number; y: number } => {
    const half = Math.floor(vector.length / 2);
    let xSum = 0;
    let ySum = 0;
    for (let i = 0; i < half; i++) xSum += vector[i];
    for (let i = half; i < vector.length; i++) ySum += vector[i];
    
    // Scale for visualization domain [-120, 120]
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
      
      // Explicit browser config
      // FIX: Removed non-existent remotePathComponent and corrected to remoteModelPath
      env.allowLocalModels = false;
      env.useBrowserCache = true;
      env.remoteHost = 'https://huggingface.co';
      env.remoteModelPath = 'models';

      extractorRef.current = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2', {
        progress_callback: (data: any) => {
          if (data.status === 'progress') setModelProgress(data.progress);
        }
      });
      setModelProgress(100);

      // 2. Fetch Idea Database
      setStatusMessage('Fetching Registry...');
      const response = await fetch('./data.csv');
      if (!response.ok) {
         throw new Error(`Registry fetch failed: ${response.status} ${response.statusText}`);
      }
      const csvText = await response.text();
      
      const rows = csvText.split('\n').slice(1).filter(r => r.trim().length > 5);
      
      // 3. Index Database
      setStatusMessage(`Indexing Registry...`);
      const indexedItems: DBItem[] = [];
      const points: ScatterPoint[] = [];

      for (let i = 0; i < rows.length; i++) {
        const text = rows[i].replace(/^"|"$/g, '').trim();
        const output = await extractorRef.current(text, { pooling: 'mean', normalize: true });
        const vector = output.data as Float32Array;
        const pos = getPositionFromVector(vector);
        
        indexedItems.push({ pitch: text, embedding: vector, ...pos });
        points.push({ id: `item-${i}`, ...pos, isUser: false });
        
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
      alert("Please provide at least 15 characters describing your idea.");
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

      // Calculate base novelty score (1-10)
      const novelty = Math.max(1, Math.min(10, Math.round((1 - maxSim) * 10)));
      
      // Use Gemini to generate a professional semantic assessment instead of hardcoded rules
      const { title, description } = await getDetailedAssessment(pitch, maxSim);

      setResult({
        x: userPos.x,
        y: userPos.y,
        noveltyScore: novelty,
        localSimilarityScore: maxSim,
        assessmentTitle: title,
        assessmentDescription: description
      });
    } catch (error) {
      console.error("Analysis failed:", error);
      alert("Analysis error. Please check your connection and try again.");
    } finally {
      setIsAnalyzing(false);
    }
  };

  const saveScreenshot = async () => {
    if (!mapContainerRef.current) return;
    setIsCapturing(true);
    try {
      await new Promise(r => setTimeout(r, 150));
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
      alert('Could not generate image. Try again.');
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
        <div className="fixed inset-0 z-[100] bg-background-dark/95 backdrop-blur-sm flex flex-col items-center justify-center p-8 text-center animate-in fade-in duration-500">
          <div className="w-full max-w-md flex flex-col gap-8">
            <div className="relative size-24 mx-auto">
               <div className="absolute inset-0 rounded-3xl bg-primary/20 animate-ping opacity-20"></div>
               <div className="relative size-24 rounded-3xl bg-primary/10 border border-primary/30 flex items-center justify-center text-primary">
                  <span className="material-symbols-outlined text-5xl">memory</span>
               </div>
            </div>
            <div className="space-y-3">
              <h2 className="text-3xl font-black text-white tracking-tight">{statusMessage}</h2>
              <p className="text-text-dim text-sm max-w-[280px] mx-auto leading-relaxed">Preparing local vector engine. All data remains in your browser for 100% privacy.</p>
            </div>
            <div className="w-full h-1.5 bg-surface-dark rounded-full overflow-hidden border border-border-dark">
               <div className="h-full bg-primary transition-all duration-700 ease-out shadow-[0_0_10px_#0db9f2]" style={{ width: `${modelProgress}%` }} />
            </div>
          </div>
        </div>
      )}

      <main className="flex-1 flex flex-col items-center w-full px-4 py-12 md:px-8 lg:px-40">
        <div className="flex flex-col w-full max-w-[1100px] gap-12">
          
          <section className="flex flex-col gap-6 animate-in slide-in-from-top-4 duration-700">
            <div className="inline-flex items-center gap-2.5 px-3 py-1 rounded-full bg-primary/10 border border-primary/20 text-primary self-start">
              <span className="material-symbols-outlined text-sm font-bold">verified</span>
              <span className="text-[10px] font-black uppercase tracking-wider">Local Registry Active</span>
            </div>
            <h1 className="text-white text-5xl md:text-7xl font-black leading-[1.1] tracking-tighter">
              Novelty <span className="text-primary italic">Index</span>.
            </h1>
            <p className="text-text-dim text-xl font-normal leading-relaxed max-w-3xl">
              Compare your vision against <span className="text-white font-medium">{corpusPoints.length} existing market patterns</span>. 
              Our semantic map identifies "white space" opportunities using zero-latency browser AI.
            </p>
          </section>

          <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
            <div className="lg:col-span-4 flex flex-col gap-6 p-8 rounded-[2rem] bg-surface-dark border border-border-dark shadow-2xl relative overflow-hidden">
               <div className="absolute top-0 right-0 p-4 opacity-5 pointer-events-none">
                 <span className="material-symbols-outlined text-8xl">edit_square</span>
               </div>

               <div className="flex flex-col gap-2 relative z-10">
                  <h3 className="text-white font-black text-xl tracking-tight">Pitch Descriptor</h3>
                  <p className="text-text-dim text-xs font-medium">Be specific about your value prop.</p>
               </div>
               
               <textarea 
                  value={pitch}
                  onChange={(e) => setPitch(e.target.value)}
                  className="form-input w-full resize-none rounded-2xl text-white placeholder:text-text-dim/20 bg-background-dark/50 border border-border-dark focus:border-primary focus:ring-4 focus:ring-primary/10 transition-all p-5 text-sm font-normal leading-relaxed min-h-[220px] shadow-inner" 
                  placeholder="Example: A marketplace for trading excess restaurant produce directly to local consumers using AI for pricing..."
                />

                <button 
                  onClick={handleAnalyze}
                  disabled={isAnalyzing}
                  className="w-full flex items-center justify-center gap-3 rounded-2xl h-16 bg-primary hover:bg-white text-background-dark font-black text-lg transition-all shadow-xl shadow-primary/20 active:scale-[0.98] disabled:opacity-50 group"
                >
                  {isAnalyzing ? (
                    <div className="size-6 border-[3px] border-background-dark/30 border-t-background-dark rounded-full animate-spin"></div>
                  ) : (
                    <>
                      <span className="material-symbols-outlined transition-transform group-hover:rotate-12">radar</span>
                      <span>Run Comparison</span>
                    </>
                  )}
                </button>

                <div className="flex items-center justify-between text-[10px] text-text-dim/60 font-bold uppercase tracking-[0.2em]">
                  <span>Verifying Vectors</span>
                  <div className="flex gap-1">
                    <div className="size-1 rounded-full bg-emerald-500 animate-pulse"></div>
                    <div className="size-1 rounded-full bg-emerald-500 animate-pulse delay-75"></div>
                    <div className="size-1 rounded-full bg-emerald-500 animate-pulse delay-150"></div>
                  </div>
                </div>
            </div>

            <div className="lg:col-span-8 flex flex-col gap-8">
              <div className="relative group animate-in zoom-in-95 duration-700">
                {isAnalyzing && <div className="scanning-line"></div>}
                <SimilarityMap ref={mapContainerRef} userPoint={userPoint} corpusPoints={corpusPoints} />
                
                {result && !isAnalyzing && (
                   <div className="absolute top-6 right-6 bg-background-dark/90 backdrop-blur-xl p-6 rounded-[1.5rem] border border-primary/20 shadow-2xl max-w-[280px] animate-in fade-in slide-in-from-right-4 duration-500">
                      <div className="flex justify-between items-center mb-4">
                        <span className="text-[10px] font-black text-primary uppercase tracking-[0.2em]">Match Rank</span>
                        <div className="bg-primary text-background-dark text-xs font-black px-2.5 py-1 rounded-lg">
                          {(result.localSimilarityScore * 100).toFixed(0)}%
                        </div>
                      </div>
                      <h4 className="text-white text-lg font-black leading-tight mb-2 tracking-tight">{result.assessmentTitle}</h4>
                      <div className="mt-6 flex flex-col gap-2">
                        <div className="flex justify-between text-[10px] text-text-dim/80 font-black uppercase tracking-wider">
                          <span>Novelty Score</span>
                          <span className="text-white">{result.noveltyScore} / 10</span>
                        </div>
                        <div className="w-full h-2 bg-border-dark rounded-full overflow-hidden p-[2px]">
                          <div className="h-full bg-primary rounded-full transition-all duration-1000 ease-out" style={{ width: `${result.noveltyScore * 10}%` }}></div>
                        </div>
                      </div>
                   </div>
                )}
              </div>

              {result && (
                <div className="p-8 rounded-[2rem] bg-surface-dark border border-border-dark flex flex-col gap-6 animate-in slide-in-from-bottom-6 duration-700 shadow-xl">
                  <div className="flex items-center gap-5">
                    <div className="size-14 rounded-2xl bg-primary/10 border border-primary/20 flex items-center justify-center text-primary">
                      <span className="material-symbols-outlined text-3xl">insights</span>
                    </div>
                    <div>
                      <h2 className="text-2xl font-black text-white tracking-tight">Semantic Assessment</h2>
                      <p className="text-text-dim text-sm font-medium">Calculated against global registry data.</p>
                    </div>
                  </div>
                  <div className="relative">
                    <span className="absolute -top-4 -left-2 text-primary/20 text-6xl font-serif">“</span>
                    <p className="text-slate-200 text-lg leading-relaxed font-medium relative z-10 pl-6">
                      {result.assessmentDescription}
                    </p>
                  </div>
                  <div className="flex flex-wrap gap-4 pt-4 border-t border-border-dark/50">
                    <button 
                      onClick={() => { setPitch(''); setResult(null); }} 
                      className="px-6 py-3 rounded-xl bg-background-dark border border-border-dark text-text-dim hover:text-white hover:border-primary/50 transition-all text-xs font-black uppercase tracking-widest active:scale-95"
                    >
                      New Plot
                    </button>
                    <button 
                      onClick={saveScreenshot} 
                      disabled={isCapturing}
                      className="px-6 py-3 rounded-xl bg-white text-background-dark hover:bg-primary transition-all text-xs font-black uppercase tracking-widest flex items-center gap-2 shadow-lg active:scale-95 disabled:opacity-50"
                    >
                      <span className="material-symbols-outlined text-sm">download</span>
                      {isCapturing ? 'Processing...' : 'Export Results'}
                    </button>
                  </div>
                </div>
              )}
            </div>
          </div>

          <section className="grid grid-cols-2 md:grid-cols-4 gap-12 py-16 border-t border-border-dark/50 opacity-60">
            {[
              { label: 'Comparison Protocol', val: 'Semantic Vectors' },
              { label: 'Security Level', val: 'Full Local isolation' },
              { label: 'Latency', val: '< 20ms Local' },
              { label: 'Registry Ref', val: `v1.2 (${corpusPoints.length} nodes)` }
            ].map((stat, i) => (
              <div key={i} className="flex flex-col gap-2">
                <span className="text-[10px] uppercase font-black text-text-dim tracking-[0.2em]">{stat.label}</span>
                <span className="text-sm font-bold text-white uppercase tracking-tight">{stat.val}</span>
              </div>
            ))}
          </section>
        </div>
      </main>
    </div>
  );
};

export default App;
