import { useState, useCallback, useEffect } from 'react';
const ort = await import('onnxruntime-web');
import { imageDataToTensor, runModel, softmax } from '../utils/utils'; 

export const useOnnx = () => {
  const [results, setResults] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [classes, setClasses] = useState<string[]>([]);
  
  // Read the class labels
  useEffect(() => {
    async function loadClasses() {
      try {
        const response = await fetch('./classes.txt');
        if (!response.ok) throw new Error('Failed to load class labels');
        const text = await response.text();
        setClasses(text.split('\n').map(line => line.trim()).filter(line => line.length > 0));
      } catch (err) {
        setError((err as Error).message);
      }
    }

    loadClasses();
  }, []);

  const runInference = useCallback(async (imagePath: File) => {
    setLoading(true);
    setError(null);
    setResults([]);

    try {
      // Read the image as a buffer
      const imageArrayBuffer = await imagePath.arrayBuffer();
      const rawImage = new Uint8Array(imageArrayBuffer);
      // Preprocess the image using the function from utils.tsx
      const tensorData = imageDataToTensor(rawImage, [1, 3, 224, 224]);

      // Ensure the model path is correct and accessible
      const session = await ort.InferenceSession.create('./model/resnet50v2.onnx', {
        executionProviders: ["cpu"],
        graphOptimizationLevel: 'all',
      })

      // Run the model with the preprocessed image tensor
      const [output, inferenceTime] = await runModel(session, tensorData);
      // Apply softmax to the output to get probabilities
      console.log('Model output:', output.data);
      const scores = softmax(Array.from(output.data as Float32Array));
      console.log('Scores:', scores);

      const topPred = scores
        .map((score, idx) => ({ label: classes[idx], score }))
        .reduce((max, item) => (item.score > max.score ? item : max));

      setResults(topPred.label ? [topPred.label] : []);
      console.log(`Inference time: ${inferenceTime} ms`);
    } catch (err: any) {
      setError(err.message || 'Error running inference');
    } finally {
      setLoading(false);
    }
  }, []);

  return { results, loading, error, runInference };
};
