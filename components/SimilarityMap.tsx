
import React, { useMemo, forwardRef } from 'react';
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  ZAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from 'recharts';
import { ScatterPoint } from '../types';

interface SimilarityMapProps {
  userPoint: ScatterPoint | null;
  corpusPoints: ScatterPoint[];
}

const CustomTooltip = ({ active, payload }: any) => {
  if (active && payload && payload.length) {
    const data = payload[0].payload as ScatterPoint;
    return (
      <div className="bg-slate-800 text-white text-xs rounded-lg p-3 shadow-xl border border-border-dark min-w-[140px]">
        <p className="font-bold text-primary mb-1">
          {data.isUser ? "Your Idea" : "Existing Startup"}
        </p>
        <p className="text-slate-300">Semantic Dist: {Math.sqrt(data.x ** 2 + data.y ** 2).toFixed(1)}</p>
        {data.isUser && (
          <p className="text-slate-400 mt-1 italic">"Unique positioning area"</p>
        )}
      </div>
    );
  }
  return null;
};

const SimilarityMap = forwardRef<HTMLDivElement, SimilarityMapProps>(({ userPoint, corpusPoints }, ref) => {
  const data = useMemo(() => {
    return userPoint ? [...corpusPoints, userPoint] : corpusPoints;
  }, [userPoint, corpusPoints]);

  return (
    <div ref={ref} className="relative w-full aspect-[16/9] md:aspect-[21/9] bg-surface-dark rounded-xl border border-border-dark overflow-hidden">
      {/* Decorative Grid handled by Recharts mostly, but we can add a subtle overlay if needed */}
      <div className="absolute inset-0 opacity-5 pointer-events-none" style={{ backgroundImage: 'linear-gradient(#315a68 1px, transparent 1px), linear-gradient(90deg, #315a68 1px, transparent 1px)', backgroundSize: '40px 40px' }} />
      
      <ResponsiveContainer width="100%" height="100%">
        <ScatterChart
          margin={{ top: 40, right: 40, bottom: 40, left: 40 }}
        >
          <CartesianGrid strokeDasharray="3 3" stroke="#223f49" />
          <XAxis 
            type="number" 
            dataKey="x" 
            domain={[-120, 120]} 
            hide 
          />
          <YAxis 
            type="number" 
            dataKey="y" 
            domain={[-120, 120]} 
            hide 
          />
          <ZAxis type="number" range={[50, 400]} />
          <Tooltip content={<CustomTooltip />} cursor={{ strokeDasharray: '3 3' }} />
          <Scatter name="Startups" data={data}>
            {data.map((entry, index) => (
              <Cell 
                key={`cell-${index}`} 
                fill={entry.isUser ? '#0db9f2' : '#475569'} 
                fillOpacity={entry.isUser ? 1 : 0.4}
                className={entry.isUser ? 'user-dot' : ''}
              />
            ))}
          </Scatter>
        </ScatterChart>
      </ResponsiveContainer>

      {userPoint && (
        <div 
          className="absolute pointer-events-none" 
          style={{ 
            left: `${((userPoint.x + 120) / 240) * 100}%`, 
            top: `${(1 - (userPoint.y + 120) / 240) * 100}%` 
          }}
        >
           <div className="absolute -inset-6 rounded-full border border-primary/40 user-dot-ring"></div>
        </div>
      )}

      {/* Axis Labels */}
      <div className="absolute bottom-4 left-1/2 -translate-x-1/2 text-[10px] font-mono text-text-dim/40 uppercase tracking-widest pointer-events-none">
        Semantic Dimension X
      </div>
      <div className="absolute top-1/2 left-4 -rotate-90 origin-left text-[10px] font-mono text-text-dim/40 uppercase tracking-widest pointer-events-none">
        Semantic Dimension Y
      </div>
    </div>
  );
});

SimilarityMap.displayName = 'SimilarityMap';

export default SimilarityMap;
