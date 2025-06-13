# backend/main.py
"""
Human Behavior Decoder - Main FastAPI Application
Central API server for voice and text behavior analysis
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import uvicorn
import os
import logging
from datetime import datetime

# Internal imports - using correct folder structure
from voice_analysis.emotion_detector import VoiceEmotionDetector
from text_analysis.sentiment_analyzer import TextSentimentAnalyzer
from stress_detector.stress_analyzer import StressAnalyzer
from confidence_scorer.confidence_evaluator import ConfidenceEvaluator
from deception_analyzer.lie_detector import DeceptionAnalyzer
from utils.file_handler import AudioFileHandler, TextFileHandler
from config.settings import Settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Human Behavior Decoder API",
    description="AI-powered voice and text behavior analysis platform",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load settings
settings = Settings()

# Initialize analyzers (import BehaviorProcessor here to avoid circular imports)
from emotion_engine.behavior_processor import BehaviorProcessor

voice_detector = VoiceEmotionDetector()
text_analyzer = TextSentimentAnalyzer()
behavior_processor = BehaviorProcessor()
stress_analyzer = StressAnalyzer()
confidence_evaluator = ConfidenceEvaluator()
deception_analyzer = DeceptionAnalyzer()

# Initialize file handlers
audio_handler = AudioFileHandler()
text_handler = TextFileHandler()

# Pydantic models for request/response
class TextAnalysisRequest(BaseModel):
    text: str
    analysis_type: str = "full"  # Options: emotion, sentiment, stress, confidence, deception, full

class AnalysisResponse(BaseModel):
    analysis_id: str
    timestamp: str
    input_type: str
    results: Dict[str, Any]
    confidence_scores: Dict[str, float]
    summary: str

class VoiceAnalysisRequest(BaseModel):
    analysis_type: str = "full"

# Health check endpoint
@app.get("/")
async def root():
    return {
        "message": "Human Behavior Decoder API",
        "status": "active",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    try:
        # Test core components
        voice_status = voice_detector.health_check()
        text_status = text_analyzer.health_check()
        
        return {
            "status": "healthy",
            "components": {
                "voice_analyzer": voice_status,
                "text_analyzer": text_status,
                "api_server": "operational"
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail="Service unhealthy")

# Text analysis endpoint
@app.post("/analyze/text", response_model=AnalysisResponse)
async def analyze_text(request: TextAnalysisRequest):
    """
    Analyze text for emotions, sentiment, stress, confidence, and deception indicators
    """
    try:
        logger.info(f"Processing text analysis request: {request.analysis_type}")
        
        if len(request.text) > settings.MAX_TEXT_LENGTH:
            raise HTTPException(
                status_code=400, 
                detail=f"Text too long. Maximum {settings.MAX_TEXT_LENGTH} characters allowed."
            )
        
        # Process text through behavior processor
        analysis_results = await behavior_processor.process_text(
            text=request.text,
            analysis_type=request.analysis_type
        )
        
        # Generate analysis ID
        analysis_id = f"txt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Compile results
        results = {
            "input_text": request.text[:100] + "..." if len(request.text) > 100 else request.text,
            "analysis_type": request.analysis_type,
            "emotions": analysis_results.get("emotions", {}),
            "sentiment": analysis_results.get("sentiment", {}),
            "stress_indicators": analysis_results.get("stress", {}),
            "confidence_markers": analysis_results.get("confidence", {}),
            "deception_signals": analysis_results.get("deception", {}),
            "metadata": {
                "text_length": len(request.text),
                "processing_time_ms": analysis_results.get("processing_time", 0)
            }
        }
        
        # Calculate confidence scores
        confidence_scores = {
            "overall_confidence": analysis_results.get("overall_confidence", 0.0),
            "emotion_confidence": analysis_results.get("emotion_confidence", 0.0),
            "sentiment_confidence": analysis_results.get("sentiment_confidence", 0.0),
            "stress_confidence": analysis_results.get("stress_confidence", 0.0)
        }
        
        # Generate summary
        summary = generate_analysis_summary(results, "text")
        
        return AnalysisResponse(
            analysis_id=analysis_id,
            timestamp=datetime.now().isoformat(),
            input_type="text",
            results=results,
            confidence_scores=confidence_scores,
            summary=summary
        )
        
    except Exception as e:
        logger.error(f"Text analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

# Voice analysis endpoint
@app.post("/analyze/voice", response_model=AnalysisResponse)
async def analyze_voice(
    file: UploadFile = File(...),
    analysis_type: str = "full"
):
    """
    Analyze voice file for emotions, stress, confidence, and deception indicators
    """
    try:
        logger.info(f"Processing voice analysis request: {file.filename}")
        
        # Validate file
        if not audio_handler.validate_audio_file(file):
            raise HTTPException(status_code=400, detail="Invalid audio file format")
        
        # Save and process audio file
        file_path = await audio_handler.save_upload(file)
        
        try:
            # Process audio through behavior processor
            analysis_results = await behavior_processor.process_voice(
                file_path=file_path,
                analysis_type=analysis_type
            )
            
            # Generate analysis ID
            analysis_id = f"voi_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Compile results
            results = {
                "input_file": file.filename,
                "analysis_type": analysis_type,
                "emotions": analysis_results.get("emotions", {}),
                "stress_indicators": analysis_results.get("stress", {}),
                "confidence_markers": analysis_results.get("confidence", {}),
                "deception_signals": analysis_results.get("deception", {}),
                "voice_characteristics": analysis_results.get("voice_features", {}),
                "metadata": {
                    "file_duration": analysis_results.get("duration", 0),
                    "sample_rate": analysis_results.get("sample_rate", 0),
                    "processing_time_ms": analysis_results.get("processing_time", 0)
                }
            }
            
            # Calculate confidence scores
            confidence_scores = {
                "overall_confidence": analysis_results.get("overall_confidence", 0.0),
                "emotion_confidence": analysis_results.get("emotion_confidence", 0.0),
                "stress_confidence": analysis_results.get("stress_confidence", 0.0),
                "voice_quality": analysis_results.get("voice_quality", 0.0)
            }
            
            # Generate summary
            summary = generate_analysis_summary(results, "voice")
            
            return AnalysisResponse(
                analysis_id=analysis_id,
                timestamp=datetime.now().isoformat(),
                input_type="voice",
                results=results,
                confidence_scores=confidence_scores,
                summary=summary
            )
            
        finally:
            # Clean up uploaded file
            audio_handler.cleanup_file(file_path)
            
    except Exception as e:
        logger.error(f"Voice analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

# Real-time analysis endpoint
@app.post("/analyze/realtime")
async def start_realtime_analysis():
    """
    Start real-time voice analysis (WebSocket connection needed)
    """
    return {"message": "Real-time analysis endpoint - WebSocket implementation needed"}

# Get analysis history
@app.get("/analyses")
async def get_analysis_history(limit: int = 10):
    """
    Retrieve analysis history
    """
    try:
        # This would connect to database in real implementation
        return {
            "analyses": [],
            "total": 0,
            "limit": limit,
            "message": "Database implementation needed"
        }
    except Exception as e:
        logger.error(f"Failed to retrieve history: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve history")

# Model information endpoints
@app.get("/models/info")
async def get_model_info():
    """
    Get information about loaded models
    """
    return {
        "voice_models": voice_detector.get_model_info(),
        "text_models": text_analyzer.get_model_info(),
        "status": "Models loaded and ready"
    }

@app.get("/models/performance")
async def get_model_performance():
    """
    Get model performance metrics
    """
    return {
        "voice_performance": voice_detector.get_performance_metrics(),
        "text_performance": text_analyzer.get_performance_metrics(),
        "system_stats": behavior_processor.get_system_stats()
    }

# Utility functions
def generate_analysis_summary(results: Dict[str, Any], input_type: str) -> str:
    """
    Generate a human-readable summary of analysis results
    """
    try:
        if input_type == "text":
            emotions = results.get("emotions", {})
            sentiment = results.get("sentiment", {})
            dominant_emotion = max(emotions.items(), key=lambda x: x[1])[0] if emotions else "neutral"
            sentiment_label = sentiment.get("label", "neutral")
            
            return f"Analysis shows {dominant_emotion} emotion with {sentiment_label} sentiment. " \
                   f"Text length: {results['metadata']['text_length']} characters."
        
        elif input_type == "voice":
            emotions = results.get("emotions", {})
            stress = results.get("stress_indicators", {})
            dominant_emotion = max(emotions.items(), key=lambda x: x[1])[0] if emotions else "neutral"
            stress_level = stress.get("level", "unknown")
            
            return f"Voice analysis reveals {dominant_emotion} emotion with {stress_level} stress level. " \
                   f"Duration: {results['metadata']['file_duration']:.1f}s."
        
        return "Analysis completed successfully."
        
    except Exception as e:
        logger.warning(f"Failed to generate summary: {str(e)}")
        return "Analysis completed - summary generation failed."

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"message": "Endpoint not found", "path": str(request.url)}
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error(f"Internal server error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"message": "Internal server error", "detail": str(exc)}
    )

# Startup event
@app.on_event("startup")
async def startup_event():
    """
    Initialize services on startup
    """
    logger.info("Starting Human Behavior Decoder API...")
    
    # Initialize models
    await voice_detector.initialize()
    await text_analyzer.initialize()
    await behavior_processor.initialize()
    
    logger.info("API server started successfully!")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """
    Cleanup on shutdown
    """
    logger.info("Shutting down Human Behavior Decoder API...")
    
    # Cleanup resources
    await voice_detector.cleanup()
    await text_analyzer.cleanup()
    await behavior_processor.cleanup()
    
    logger.info("API server shut down successfully!")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )