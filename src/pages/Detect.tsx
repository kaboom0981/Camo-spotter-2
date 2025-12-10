import React from "react";


import { useState } from "react";
import { Button } from "@/components/ui/button";
import { useToast } from "@/hooks/use-toast";
import Navigation from "@/components/Navigation";
import ImageUpload from "@/components/ImageUpload";
import RadarScanner from "@/components/RadarScanner";
import AnalysisResults from "@/components/AnalysisResults";


const Detect = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string>("");
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisData, setAnalysisData] = useState<any>(null);
  const { toast } = useToast();

  const handleImageSelect = (file: File, preview: string) => {
    setSelectedFile(file);
    setImagePreview(preview);
    setAnalysisData(null);
  };

  const analyzeImage = async () => {
    if (!selectedFile) return;

    setIsAnalyzing(true);

    try {
      const formData = new FormData();
      formData.append("file", selectedFile);

      // 🔥 CALL TO FASTAPI BACKEND
      const response = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Backend returned an error");
      }

      const result = await response.json();
      setAnalysisData(result);

      toast({
        title: "Analysis Complete",
        description: "Detection finished successfully!",
      });
    } catch (err) {
      console.error("Analysis error:", err);
      toast({
        title: "Analysis Failed",
        description: "Something went wrong. Try again.",
        variant: "destructive",
      });
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <div className="min-h-screen bg-background">
      <Navigation />
      <RadarScanner isScanning={isAnalyzing} />

      <div className="pt-32 pb-20">
        <div className="container mx-auto px-6">
          <div className="max-w-6xl mx-auto space-y-12">
            {/* Header */}
            <div className="text-center space-y-4">
              <h1 className="text-5xl font-bold">
                <span className="bg-gradient-to-r from-primary-glow to-secondary bg-clip-text text-transparent">
                  Camouflage Detection
                </span>
              </h1>
              <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
                Upload an image and let our AI reveal hidden wildlife through advanced pattern recognition.
              </p>
            </div>

            {/* Upload Section */}
            <ImageUpload
              onImageSelect={handleImageSelect}
              isAnalyzing={isAnalyzing}
            />

            {/* Analyze Button */}
            {imagePreview && !analysisData && (
              <div className="flex justify-center">
                <Button
                  size="lg"
                  onClick={analyzeImage}
                  disabled={isAnalyzing}
                  className="bg-gradient-hero hover:shadow-glow-primary transition-all duration-300 hover:scale-105 text-lg px-12 py-6"
                >
                  {isAnalyzing ? "Analyzing..." : "Analyze Image"}
                </Button>
              </div>
            )}

            {/* Results */}
            {analysisData && (
              <AnalysisResults data={analysisData} imagePreview={imagePreview} />
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default Detect;
