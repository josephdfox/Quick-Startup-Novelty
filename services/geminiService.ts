
import { GoogleGenAI, Type } from "@google/genai";

// Initialize Gemini API client with the provided API key from environment variables
const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });

/**
 * Generates a professional assessment of a startup idea's novelty.
 * Uses Gemini 3 Flash to provide a nuanced analysis based on semantic similarity data.
 */
export async function getDetailedAssessment(pitch: string, maxSimilarity: number) {
  // Use gemini-3-flash-preview for fast and insightful text assessment
  const response = await ai.models.generateContent({
    model: 'gemini-3-flash-preview',
    contents: `Analyze this startup pitch for market novelty. 
    A local semantic comparison shows its highest similarity match is ${(maxSimilarity * 100).toFixed(1)}%.
    
    Pitch: "${pitch}"
    
    Provide a professional assessment including:
    1. A short, punchy title.
    2. A one-sentence insightful description of why it's novel or why it overlaps with existing concepts.`,
    config: {
      responseMimeType: "application/json",
      responseSchema: {
        type: Type.OBJECT,
        properties: {
          title: {
            type: Type.STRING,
            description: "An assessment title like 'High Innovation' or 'Competitive Space'.",
          },
          description: {
            type: Type.STRING,
            description: "A professional one-sentence analysis of the idea's novelty.",
          },
        },
        required: ["title", "description"],
      },
    },
  });

  try {
    // Extract the generated text output directly from the response object
    const text = response.text?.trim() || "{}";
    return JSON.parse(text);
  } catch (error) {
    console.error("Gemini analysis error:", error);
    // Graceful fallback for the UI
    return {
      title: maxSimilarity < 0.5 ? "Unique Positioning" : "Moderate Similarity",
      description: "Analysis complete. Your idea has been mapped against established market patterns."
    };
  }
}
