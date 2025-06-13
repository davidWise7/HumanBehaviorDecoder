import React, { useState } from 'react';
import { Upload, Mic, FileText, BarChart3, Brain, AlertTriangle, CheckCircle, Clock, Activity, Zap } from 'lucide-react';

const BehaviorDecoderApp = () => {
  const [activeTab, setActiveTab] = useState('upload');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResults, setAnalysisResults] = useState(null);
  const [textInput, setTextInput] = useState('');
  const [audioFile, setAudioFile] = useState(null);

  // Mock analysis function
  const analyzeInput = async (type, input) => {
    setIsAnalyzing(true);
    
    // Simulate API call delay
    await new Promise(resolve => setTimeout(resolve, 3000));
    
    // Mock results
    const mockResults = {
      analysis_id: `${type}_${Date.now()}`,
      timestamp: new Date().toISOString(),
      input_type: type,
      results: {
        emotions: {
          happy: 0.65,
          confident: 0.72,
          calm: 0.58,
          anxious: 0.23,
          sad: 0.15
        },
        sentiment: {
          label: 'positive',
          confidence: 0.78
        },
        stress_indicators: {
          level: 'low',
          confidence: 0.68,
          overall_score: 0.25
        },
        confidence_markers: {
          level: 'high',
          score: 0.71
        },
        deception_signals: {
          risk_level: 'low',
          risk_score: 0.15
        }
      },
      confidence_scores: {
        overall_confidence: 0.75,
        emotion_confidence: 0.72,
        sentiment_confidence: 0.78
      },
      summary: type === 'text' 
        ? 'Analysis shows confident emotion with positive sentiment. High confidence markers detected.'
        : 'Voice analysis reveals confident emotion with low stress level. Clear and stable vocal patterns.'
    };
    
    setAnalysisResults(mockResults);
    setIsAnalyzing(false);
  };

  const handleTextAnalysis = () => {
    if (textInput.trim()) {
      analyzeInput('text', textInput);
    }
  };

  const handleAudioUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      setAudioFile(file);
      analyzeInput('voice', file);
    }
  };

  const EmotionChart = ({ emotions }) => (
    <div className="space-y-3">
      {Object.entries(emotions).map(([emotion, score]) => (
        <div key={emotion} className="flex items-center justify-between">
          <span className="capitalize text-sm font-medium">{emotion}</span>
          <div className="flex items-center space-x-2">
            <div className="w-32 bg-gray-200 rounded-full h-2">
              <div 
                className="bg-blue-500 h-2 rounded-full transition-all duration-500"
                style={{ width: `${score * 100}%` }}
              />
            </div>
            <span className="text-sm text-gray-600 w-12 text-right">
              {(score * 100).toFixed(0)}%
            </span>
          </div>
        </div>
      ))}
    </div>
  );

  const ResultsPanel = () => {
    if (!analysisResults) return null;

    const { results, confidence_scores, summary } = analysisResults;

    return (
      <div className="mt-8 p-6 bg-white rounded-lg shadow-lg border">
        <h3 className="text-xl font-bold mb-4 flex items-center">
          <BarChart3 className="mr-2" />
          Analysis Results
        </h3>
        
        {/* Summary */}
        <div className="mb-6 p-4 bg-blue-50 rounded-lg">
          <p className="text-blue-800">{summary}</p>
        </div>

        <div className="grid md:grid-cols-2 gap-6">
          {/* Emotions */}
          <div className="space-y-4">
            <h4 className="font-semibold text-lg">Emotional Analysis</h4>
            <EmotionChart emotions={results.emotions} />
          </div>

          {/* Key Metrics */}
          <div className="space-y-4">
            <h4 className="font-semibold text-lg">Key Indicators</h4>
            
            <div className="space-y-3">
              {/* Sentiment */}
              <div className="flex justify-between items-center p-3 bg-gray-50 rounded">
                <span className="font-medium">Sentiment</span>
                <div className="flex items-center space-x-2">
                  <span className={`px-2 py-1 rounded text-sm ${
                    results.sentiment.label === 'positive' ? 'bg-green-100 text-green-800' :
                    results.sentiment.label === 'negative' ? 'bg-red-100 text-red-800' :
                    'bg-gray-100 text-gray-800'
                  }`}>
                    {results.sentiment.label}
                  </span>
                  <span className="text-sm text-gray-600">
                    {(results.sentiment.confidence * 100).toFixed(0)}%
                  </span>
                </div>
              </div>

              {/* Stress Level */}
              <div className="flex justify-between items-center p-3 bg-gray-50 rounded">
                <span className="font-medium">Stress Level</span>
                <div className="flex items-center space-x-2">
                  <span className={`px-2 py-1 rounded text-sm ${
                    results.stress_indicators.level === 'low' ? 'bg-green-100 text-green-800' :
                    results.stress_indicators.level === 'medium' ? 'bg-yellow-100 text-yellow-800' :
                    'bg-red-100 text-red-800'
                  }`}>
                    {results.stress_indicators.level}
                  </span>
                  <span className="text-sm text-gray-600">
                    {(results.stress_indicators.confidence * 100).toFixed(0)}%
                  </span>
                </div>
              </div>

              {/* Confidence */}
              <div className="flex justify-between items-center p-3 bg-gray-50 rounded">
                <span className="font-medium">Confidence</span>
                <div className="flex items-center space-x-2">
                  <span className={`px-2 py-1 rounded text-sm ${
                    results.confidence_markers.level === 'high' ? 'bg-green-100 text-green-800' :
                    results.confidence_markers.level === 'medium' ? 'bg-yellow-100 text-yellow-800' :
                    'bg-red-100 text-red-800'
                  }`}>
                    {results.confidence_markers.level}
                  </span>
                  <span className="text-sm text-gray-600">
                    {(Math.abs(results.confidence_markers.score) * 100).toFixed(0)}%
                  </span>
                </div>
              </div>

              {/* Deception Risk */}
              <div className="flex justify-between items-center p-3 bg-gray-50 rounded">
                <span className="font-medium">Deception Risk</span>
                <div className="flex items-center space-x-2">
                  <span className={`px-2 py-1 rounded text-sm ${
                    results.deception_signals.risk_level === 'low' ? 'bg-green-100 text-green-800' :
                    results.deception_signals.risk_level === 'medium' ? 'bg-yellow-100 text-yellow-800' :
                    'bg-red-100 text-red-800'
                  }`}>
                    {results.deception_signals.risk_level}
                  </span>
                  <span className="text-sm text-gray-600">
                    {(results.deception_signals.risk_score * 100).toFixed(0)}%
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Overall Confidence */}
        <div className="mt-6 p-4 bg-gray-50 rounded-lg">
          <h4 className="font-semibold mb-2">Analysis Confidence</h4>
          <div className="w-full bg-gray-200 rounded-full h-3">
            <div 
              className="bg-blue-500 h-3 rounded-full transition-all duration-500"
              style={{ width: `${confidence_scores.overall_confidence * 100}%` }}
            />
          </div>
          <p className="text-sm text-gray-600 mt-1">
            Overall confidence: {(confidence_scores.overall_confidence * 100).toFixed(0)}%
          </p>
        </div>

        {/* Disclaimer */}
        <div className="mt-4 p-3 bg-yellow-50 border border-yellow-200 rounded">
          <div className="flex items-start">
            <AlertTriangle className="w-5 h-5 text-yellow-600 mr-2 mt-0.5" />
            <div>
              <p className="text-sm text-yellow-800">
                <strong>Disclaimer:</strong> This analysis is experimental and for research purposes only. 
                Results should not be used for legal, employment, or other critical decisions.
              </p>
            </div>
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-gray-100">
      {/* Header */}
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-6xl mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <Brain className="w-8 h-8 text-blue-600" />
              <div>
                <h1 className="text-2xl font-bold text-gray-900">Human Behavior Decoder</h1>
                <p className="text-sm text-gray-600">AI-powered voice and text analysis</p>
              </div>
            </div>
            <div className="flex items-center space-x-2">
              <Activity className="w-5 h-5 text-green-500" />
              <span className="text-sm text-green-600">System Online</span>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-6xl mx-auto px-4 py-8">
        {/* Tabs */}
        <div className="flex space-x-1 mb-8">
          <button
            onClick={() => setActiveTab('upload')}
            className={`px-4 py-2 rounded-lg font-medium ${
              activeTab === 'upload'
                ? 'bg-blue-600 text-white'
                : 'bg-white text-gray-600 hover:bg-gray-50'
            }`}
          >
            <Upload className="w-4 h-4 inline mr-2" />
            Upload & Analyze
          </button>
          <button
            onClick={() => setActiveTab('demo')}
            className={`px-4 py-2 rounded-lg font-medium ${
              activeTab === 'demo'
                ? 'bg-blue-600 text-white'
                : 'bg-white text-gray-600 hover:bg-gray-50'
            }`}
          >
            <Zap className="w-4 h-4 inline mr-2" />
            Demo
          </button>
        </div>

        {/* Upload Tab */}
        {activeTab === 'upload' && (
          <div className="grid lg:grid-cols-2 gap-8">
            {/* Text Analysis */}
            <div className="bg-white rounded-lg shadow-lg p-6">
              <h2 className="text-xl font-bold mb-4 flex items-center">
                <FileText className="mr-2" />
                Text Analysis
              </h2>
              <textarea
                value={textInput}
                onChange={(e) => setTextInput(e.target.value)}
                placeholder="Enter text to analyze emotions, sentiment, stress, and confidence..."
                className="w-full h-32 p-3 border rounded-lg resize-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                disabled={isAnalyzing}
              />
              <button
                onClick={handleTextAnalysis}
                disabled={!textInput.trim() || isAnalyzing}
                className="mt-4 w-full bg-blue-600 text-white py-2 px-4 rounded-lg hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed flex items-center justify-center"
              >
                {isAnalyzing ? (
                  <>
                    <Clock className="w-4 h-4 mr-2 animate-spin" />
                    Analyzing...
                  </>
                ) : (
                  'Analyze Text'
                )}
              </button>
            </div>

            {/* Voice Analysis */}
            <div className="bg-white rounded-lg shadow-lg p-6">
              <h2 className="text-xl font-bold mb-4 flex items-center">
                <Mic className="mr-2" />
                Voice Analysis
              </h2>
              <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center">
                <input
                  type="file"
                  accept="audio/*"
                  onChange={handleAudioUpload}
                  className="hidden"
                  id="audio-upload"
                  disabled={isAnalyzing}
                />
                <label
                  htmlFor="audio-upload"
                  className={`cursor-pointer ${isAnalyzing ? 'cursor-not-allowed opacity-50' : ''}`}
                >
                  <Upload className="w-12 h-12 mx-auto text-gray-400 mb-4" />
                  <p className="text-gray-600 mb-2">
                    {audioFile ? audioFile.name : 'Upload audio file'}
                  </p>
                  <p className="text-sm text-gray-500">
                    Supports WAV, MP3, M4A files (max 50MB)
                  </p>
                </label>
              </div>
              {isAnalyzing && (
                <div className="mt-4 flex items-center justify-center text-blue-600">
                  <Clock className="w-4 h-4 mr-2 animate-spin" />
                  Processing audio...
                </div>
              )}
            </div>
          </div>
        )}

        {/* Demo Tab */}
        {activeTab === 'demo' && (
          <div className="bg-white rounded-lg shadow-lg p-6">
            <h2 className="text-xl font-bold mb-4">Quick Demo</h2>
            <div className="grid md:grid-cols-2 gap-4 mb-6">
              <button
                onClick={() => {
                  setTextInput("I'm feeling really confident about this presentation! I know it's going to go well.");
                  analyzeInput('text', "I'm feeling really confident about this presentation! I know it's going to go well.");
                }}
                className="p-4 text-left border rounded-lg hover:bg-gray-50"
                disabled={isAnalyzing}
              >
                <h3 className="font-medium text-green-600">Confident Text</h3>
                <p className="text-sm text-gray-600">Analyze confident, positive message</p>
              </button>
              <button
                onClick={() => {
                  setTextInput("I'm not sure about this... maybe it will work, but I have some doubts.");
                  analyzeInput('text', "I'm not sure about this... maybe it will work, but I have some doubts.");
                }}
                className="p-4 text-left border rounded-lg hover:bg-gray-50"
                disabled={isAnalyzing}
              >
                <h3 className="font-medium text-yellow-600">Uncertain Text</h3>
                <p className="text-sm text-gray-600">Analyze hesitant, uncertain message</p>
              </button>
            </div>
            <div className="p-4 bg-blue-50 rounded-lg">
              <h3 className="font-medium text-blue-800 mb-2">Try the API directly:</h3>
              <code className="text-sm text-blue-700 block">
                POST http://localhost:8000/analyze/text
              </code>
              <code className="text-sm text-blue-700 block mt-1">
                {"{ \"text\": \"Your text here\", \"analysis_type\": \"full\" }"}
              </code>
            </div>
          </div>
        )}

        {/* Results */}
        <ResultsPanel />
      </main>

      {/* Footer */}
      <footer className="bg-white border-t mt-12">
        <div className="max-w-6xl mx-auto px-4 py-6">
          <div className="text-center text-gray-600">
            <p className="mb-2">Human Behavior Decoder MVP - Built for analyzing human behavior through AI</p>
            <p className="text-sm">
              <strong>Market Size:</strong> $75.5B by 2030 | 
              <strong> Applications:</strong> Healthcare, Customer Service, Marketing, HR
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default BehaviorDecoderApp;