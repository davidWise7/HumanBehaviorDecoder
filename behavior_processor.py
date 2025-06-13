# backend/emotion_engine/behavior_processor.py
"""
Central Behavior Processing Engine
Coordinates voice and text analysis to provide comprehensive behavior insights
"""

import time
import logging
from typing import Dict, List, Any, Optional
import numpy as np
from datetime import datetime
import asyncio

# Import analysis modules - using correct folder structure
from voice_analysis.emotion_detector import VoiceEmotionDetector
from text_analysis.sentiment_analyzer import TextSentimentAnalyzer
from stress_detector.stress_analyzer import StressAnalyzer
from confidence_scorer.confidence_evaluator import ConfidenceEvaluator
from deception_analyzer.lie_detector import DeceptionAnalyzer

logger = logging.getLogger(__name__)

class BehaviorProcessor:
    """
    Central engine that processes voice and text inputs through multiple analysis models
    and provides comprehensive behavioral insights
    """
    
    def __init__(self):
        # Initialize analyzers
        self.voice_detector = VoiceEmotionDetector()
        self.text_analyzer = TextSentimentAnalyzer()
        self.stress_analyzer = StressAnalyzer()
        self.confidence_evaluator = ConfidenceEvaluator()
        self.deception_analyzer = DeceptionAnalyzer()
        
        # Analysis weights for ensemble results
        self.analysis_weights = {
            "emotion": 0.3,
            "sentiment": 0.2,
            "stress": 0.2,
            "confidence": 0.15,
            "deception": 0.15
        }
        
        # Performance tracking
        self.system_stats = {
            "total_analyses": 0,
            "avg_processing_time": 0.0,
            "success_rate": 0.0,
            "last_analysis": None
        }
        
        self.initialized = False

    async def initialize(self):
        """
        Initialize all analysis components
        """
        try:
            logger.info("Initializing Behavior Processor...")
            
            # Initialize all analyzers
            await self.voice_detector.initialize()
            await self.text_analyzer.initialize()
            await self.stress_analyzer.initialize()
            await self.confidence_evaluator.initialize()
            await self.deception_analyzer.initialize()
            
            self.initialized = True
            logger.info("Behavior Processor initialized successfully!")
            
        except Exception as e:
            logger.error(f"Failed to initialize Behavior Processor: {str(e)}")
            raise

    async def process_text(self, text: str, analysis_type: str = "full") -> Dict[str, Any]:
        """
        Process text input through comprehensive analysis pipeline
        """
        try:
            if not self.initialized:
                raise ValueError("Behavior Processor not initialized")
            
            start_time = time.time()
            logger.info(f"Processing text analysis: {analysis_type}")
            
            # Initialize results dictionary
            results = {
                "input_type": "text",
                "input_text": text[:100] + "..." if len(text) > 100 else text,
                "analysis_type": analysis_type,
                "timestamp": datetime.now().isoformat()
            }
            
            # Run analyses based on type
            if analysis_type in ["emotion", "sentiment", "full"]:
                # Text sentiment and emotion analysis
                text_results = await self.text_analyzer.analyze_text(text, analysis_type)
                results.update(text_results)
            
            if analysis_type in ["stress", "full"]:
                # Stress analysis
                stress_results = await self.stress_analyzer.analyze_text_stress(text)
                results["stress"] = stress_results
            
            if analysis_type in ["confidence", "full"]:
                # Confidence analysis
                confidence_results = await self.confidence_evaluator.analyze_text_confidence(text)
                results["confidence"] = confidence_results
            
            if analysis_type in ["deception", "full"]:
                # Deception analysis
                deception_results = await self.deception_analyzer.analyze_text_deception(text)
                results["deception"] = deception_results
            
            # Calculate ensemble metrics
            if analysis_type == "full":
                results["ensemble_analysis"] = self._calculate_ensemble_results(results, "text")
            
            # Performance metrics
            processing_time = (time.time() - start_time) * 1000
            results["processing_time"] = processing_time
            
            # Update system stats
            self._update_system_stats(processing_time, True)
            
            return results
            
        except Exception as e:
            logger.error(f"Text processing failed: {str(e)}")
            self._update_system_stats(0, False)
            raise

    async def process_voice(self, file_path: str, analysis_type: str = "full") -> Dict[str, Any]:
        """
        Process voice input through comprehensive analysis pipeline
        """
        try:
            if not self.initialized:
                raise ValueError("Behavior Processor not initialized")
            
            start_time = time.time()
            logger.info(f"Processing voice analysis: {analysis_type}")
            
            # Initialize results dictionary
            results = {
                "input_type": "voice",
                "input_file": file_path.split("/")[-1],
                "analysis_type": analysis_type,
                "timestamp": datetime.now().isoformat()
            }
            
            # Voice emotion analysis (primary)
            if analysis_type in ["emotion", "full"]:
                voice_results = await self.voice_detector.analyze_emotion(file_path)
                results["emotions"] = voice_results["emotions"]
                results["dominant_emotion"] = voice_results["dominant_emotion"]
                results["voice_features"] = voice_results["audio_features"]
                results["emotion_confidence"] = voice_results["confidence_score"]
            
            # Voice stress analysis
            if analysis_type in ["stress", "full"]:
                stress_results = await self.stress_analyzer.analyze_voice_stress(file_path)
                results["stress"] = stress_results
            
            # Voice confidence analysis
            if analysis_type in ["confidence", "full"]:
                confidence_results = await self.confidence_evaluator.analyze_voice_confidence(file_path)
                results["confidence"] = confidence_results
            
            # Voice deception analysis
            if analysis_type in ["deception", "full"]:
                deception_results = await self.deception_analyzer.analyze_voice_deception(file_path)
                results["deception"] = deception_results
            
            # Get audio metadata
            audio_metadata = await self._extract_audio_metadata(file_path)
            results.update(audio_metadata)
            
            # Calculate ensemble metrics
            if analysis_type == "full":
                results["ensemble_analysis"] = self._calculate_ensemble_results(results, "voice")
            
            # Performance metrics
            processing_time = (time.time() - start_time) * 1000
            results["processing_time"] = processing_time
            
            # Update system stats
            self._update_system_stats(processing_time, True)
            
            return results
            
        except Exception as e:
            logger.error(f"Voice processing failed: {str(e)}")
            self._update_system_stats(0, False)
            raise

    async def process_multimodal(self, voice_path: str, text: str) -> Dict[str, Any]:
        """
        Process both voice and text inputs for comprehensive multimodal analysis
        """
        try:
            logger.info("Processing multimodal analysis (voice + text)")
            start_time = time.time()
            
            # Process voice and text in parallel
            voice_task = self.process_voice(voice_path, "full")
            text_task = self.process_text(text, "full")
            
            voice_results, text_results = await asyncio.gather(voice_task, text_task)
            
            # Combine results
            multimodal_results = {
                "input_type": "multimodal",
                "timestamp": datetime.now().isoformat(),
                "voice_analysis": voice_results,
                "text_analysis": text_results,
                "fusion_analysis": self._fuse_multimodal_results(voice_results, text_results)
            }
            
            processing_time = (time.time() - start_time) * 1000
            multimodal_results["total_processing_time"] = processing_time
            
            return multimodal_results
            
        except Exception as e:
            logger.error(f"Multimodal processing failed: {str(e)}")
            raise

    def _calculate_ensemble_results(self, results: Dict[str, Any], input_type: str) -> Dict[str, Any]:
        """
        Calculate ensemble results by combining different analysis outputs
        """
        try:
            ensemble = {
                "overall_emotion": "neutral",
                "overall_sentiment": "neutral",
                "overall_stress_level": "medium",
                "overall_confidence_level": "medium",
                "overall_deception_risk": "low",
                "combined_score": 0.0,
                "reliability": 0.0
            }
            
            scores = []
            
            # Emotion analysis
            if "emotions" in results:
                emotions = results["emotions"]
                dominant_emotion = max(emotions.items(), key=lambda x: x[1])
                ensemble["overall_emotion"] = dominant_emotion[0]
                scores.append(dominant_emotion[1])
            
            # Sentiment analysis
            if "sentiment" in results:
                sentiment = results["sentiment"]
                ensemble["overall_sentiment"] = sentiment.get("label", "neutral")
                scores.append(sentiment.get("confidence", 0.0))
            
            # Stress analysis
            if "stress" in results:
                stress = results["stress"]
                ensemble["overall_stress_level"] = stress.get("level", "medium")
                scores.append(stress.get("confidence", 0.0))
            
            # Confidence analysis
            if "confidence" in results:
                confidence = results["confidence"]
                ensemble["overall_confidence_level"] = confidence.get("level", "medium")
                scores.append(abs(confidence.get("score", 0.0)))
            
            # Deception analysis
            if "deception" in results:
                deception = results["deception"]
                ensemble["overall_deception_risk"] = deception.get("risk_level", "low")
                scores.append(1.0 - (deception.get("risk_score", 0.0) / 100.0))
            
            # Calculate combined score and reliability
            if scores:
                ensemble["combined_score"] = np.mean(scores)
                ensemble["reliability"] = 1.0 - np.std(scores) if len(scores) > 1 else scores[0]
            
            return ensemble
            
        except Exception as e:
            logger.error(f"Ensemble calculation failed: {str(e)}")
            return ensemble

    def _fuse_multimodal_results(self, voice_results: Dict[str, Any], text_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fuse voice and text analysis results for enhanced accuracy
        """
        try:
            fusion = {
                "confidence_agreement": 0.0,
                "emotion_agreement": 0.0,
                "stress_agreement": 0.0,
                "overall_consistency": 0.0,
                "combined_prediction": {}
            }
            
            # Compare emotion predictions
            if "emotions" in voice_results and "emotions" in text_results:
                voice_emotions = voice_results["emotions"]
                text_emotions = text_results["emotions"]
                
                # Calculate emotion agreement
                emotion_correlation = self._calculate_correlation(voice_emotions, text_emotions)
                fusion["emotion_agreement"] = emotion_correlation
            
            # Compare confidence levels
            voice_conf = voice_results.get("confidence", {}).get("score", 0.0)
            text_conf = text_results.get("confidence", {}).get("score", 0.0)
            
            conf_diff = abs(voice_conf - text_conf)
            fusion["confidence_agreement"] = 1.0 - min(conf_diff, 1.0)
            
            # Compare stress levels
            voice_stress = voice_results.get("stress", {}).get("overall_score", 0.0)
            text_stress = text_results.get("stress", {}).get("overall_score", 0.0)
            
            stress_diff = abs(voice_stress - text_stress) / max(abs(voice_stress), abs(text_stress), 1.0)
            fusion["stress_agreement"] = 1.0 - min(stress_diff, 1.0)
            
            # Overall consistency
            agreements = [
                fusion["emotion_agreement"],
                fusion["confidence_agreement"],
                fusion["stress_agreement"]
            ]
            fusion["overall_consistency"] = np.mean([a for a in agreements if a > 0])
            
            return fusion
            
        except Exception as e:
            logger.error(f"Multimodal fusion failed: {str(e)}")
            return {"overall_consistency": 0.0}

    def _calculate_correlation(self, dict1: Dict[str, float], dict2: Dict[str, float]) -> float:
        """
        Calculate correlation between two dictionaries of scores
        """
        try:
            common_keys = set(dict1.keys()) & set(dict2.keys())
            if len(common_keys) < 2:
                return 0.0
            
            values1 = [dict1[key] for key in common_keys]
            values2 = [dict2[key] for key in common_keys]
            
            correlation = np.corrcoef(values1, values2)[0, 1]
            return max(0.0, correlation) if not np.isnan(correlation) else 0.0
            
        except Exception:
            return 0.0

    async def _extract_audio_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        Extract metadata from audio file
        """
        try:
            import librosa
            
            # Load audio to get basic info
            y, sr = librosa.load(file_path, sr=None)
            duration = len(y) / sr
            
            return {
                "duration": duration,
                "sample_rate": sr,
                "audio_length": len(y),
                "file_size_mb": 0.0  # Would need os.path.getsize(file_path) / (1024*1024)
            }
            
        except Exception as e:
            logger.warning(f"Could not extract audio metadata: {str(e)}")
            return {"duration": 0.0, "sample_rate": 0}

    def _update_system_stats(self, processing_time: float, success: bool):
        """
        Update system performance statistics
        """
        try:
            self.system_stats["total_analyses"] += 1
            
            if success:
                # Update average processing time
                current_avg = self.system_stats["avg_processing_time"]
                total = self.system_stats["total_analyses"]
                new_avg = (current_avg * (total - 1) + processing_time) / total
                self.system_stats["avg_processing_time"] = new_avg
            
            # Update success rate
            # Note: This is a simplified calculation - in production, you'd want a sliding window
            self.system_stats["success_rate"] = (
                self.system_stats.get("successful_analyses", 0) + (1 if success else 0)
            ) / self.system_stats["total_analyses"]
            
            if success:
                self.system_stats["successful_analyses"] = self.system_stats.get("successful_analyses", 0) + 1
            
            self.system_stats["last_analysis"] = datetime.now().isoformat()
            
        except Exception as e:
            logger.warning(f"Stats update failed: {str(e)}")

    def get_system_stats(self) -> Dict[str, Any]:
        """
        Get system performance statistics
        """
        return self.system_stats.copy()

    def health_check(self) -> Dict[str, str]:
        """
        Check health of all analysis components
        """
        return {
            "behavior_processor": "operational" if self.initialized else "not_initialized",
            "voice_detector": self.voice_detector.health_check(),
            "text_analyzer": self.text_analyzer.health_check(),
            "stress_analyzer": self.stress_analyzer.health_check(),
            "confidence_evaluator": self.confidence_evaluator.health_check(),
            "deception_analyzer": self.deception_analyzer.health_check()
        }

    async def cleanup(self):
        """
        Cleanup all analysis components
        """
        try:
            logger.info("Cleaning up Behavior Processor...")
            
            await self.voice_detector.cleanup()
            await self.text_analyzer.cleanup()
            await self.stress_analyzer.cleanup()
            await self.confidence_evaluator.cleanup()
            await self.deception_analyzer.cleanup()
            
            self.initialized = False
            logger.info("Behavior Processor cleaned up successfully")
            
        except Exception as e:
            logger.error(f"Cleanup failed: {str(e)}")

    def get_analysis_capabilities(self) -> Dict[str, List[str]]:
        """
        Get available analysis capabilities
        """
        return {
            "voice_analysis": [
                "emotion_detection",
                "stress_analysis", 
                "confidence_evaluation",
                "deception_indicators",
                "voice_characteristics"
            ],
            "text_analysis": [
                "sentiment_analysis",
                "emotion_detection",
                "stress_indicators",
                "confidence_markers", 
                "deception_patterns",
                "linguistic_features"
            ],
            "multimodal": [
                "cross_modal_validation",
                "consistency_analysis",
                "ensemble_predictions"
            ]
        }