
export interface AnalysisResult {
  x: number;
  y: number;
  noveltyScore: number;
  localSimilarityScore: number;
  assessmentTitle: string;
  assessmentDescription: string;
}

export interface ScatterPoint {
  x: number;
  y: number;
  id: string;
  isUser?: boolean;
  score?: number;
}
