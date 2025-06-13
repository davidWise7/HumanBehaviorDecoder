# backend/stress_detector/stress_analyzer.py
"""
Stress Detection Analyzer
Analyzes both voice and text for stress level indicators
"""

import numpy as np
import librosa
import logging
from typing import Dict, List, Any, Optional
import time
import re
from sklearn.preprocessing import StandardScaler
import joblib
import os

logger = logging.getLogger(__name__)

class StressAnalyzer:
    """
    Analyzes stress levels from voice and text inputs using multiple indicators
    """
    
    def __init__(self, model_path: str = "./ai-models/stress_detection/"):
        self.model_path = model_path
        
        # Voice stress indicators
        self.voice_stress_features = {
            'pitch_variance': {'low_stress': (0.1, 0.3), 'high_stress': (0.7, 1.0)},
            'speaking_rate': {'low_stress': (2.0, 4.0), 'high_stress': (5.5, 8.0)},
            'pause_frequency': {'low_stress': (0.1, 0.3), 'high_stress': (0.6, 1.0)},
            'energy_variance': {'low_stress': (0.1, 0.4), 'high_stress': (0.7, 1.0)}
        }
        
        # Text stress keywords (expanded from text analyzer)
        self.stress_patterns = {
            'extreme_stress': [
                'overwhelmed', 'breaking down', 'can\'t handle', 'losing it',
                'mental breakdown', 'falling apart', 'drowning', 'suffocating'
            ],
            'high_stress': [
                'stressed', 'anxious', 'panic', 'worried', 'pressure',
                'deadline', 'urgent', 'crisis', 'disaster', 'terrible'
            ],
            'medium_stress': [
                'busy', 'hectic', 'challenging', 'demanding', 'difficult',
                'concerned', 'uncertain', 'frustrated', 'tired', 'overwhelm'
            ],
            'low_stress': [
                'calm', 'relaxed', 'peaceful', 'comfortable', 'easy',
                'manageable', 'clear', 'organized', 'confident', 'stable'
            ]
        }
        
        # Physiological indicators in text
        self.physical_stress_indicators = [
            'headache', 'tired', 'exhausted', 'sleepless', 'insomnia',
            'heart racing', 'sweating', 'shaking', 'trembling', 'nauseous'
        ]
        
        self.scaler = None
        self.stress_model = None

    async def initialize(self):
        """
        Initialize stress analysis models
        """
        try:
            logger.info("Initializing Stress Analyzer...")
            
            os.makedirs(self.model_path, exist_ok=True)
            
            # Try to load existing models
            await self._load_models()
            
            # Create basic models if none exist
            if not self._models_loaded():
                await self._create_basic_models()
            
            logger.info("Stress Analyzer initialized successfully!")
            
        except Exception as e:
            logger.error(f"Failed to initialize Stress Analyzer: {str(e)}")
            raise

    async def _load_models(self):
        """
        Load pre-trained stress detection models
        """
        try:
            stress_model_path = os.path.join(self.model_path, "stress_classifier.pkl")
            scaler_path = os.path.join(self.model_path, "stress_scaler.pkl")
            
            if os.path.exists(stress_model_path):
                self.stress_model = joblib.load(stress_model_path)
                logger.info("Stress classification model loaded")
            
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
                logger.info("Stress feature scaler loaded")
                
        except Exception as e:
            logger.warning(f"Could not load stress models: {str(e)}")

    async def _create_basic_models(self):
        """
        Create basic stress detection models
        """
        try:
            from sklearn.ensemble import RandomForestClassifier
            
            # Create a basic model
            self.stress_model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.scaler = StandardScaler()
            
            # Save models (even though they're not trained on real data yet)
            await self._save_models()
            
            logger.info("Basic stress models created")
            
        except Exception as e:
            logger.error(f"Error creating basic stress models: {str(e)}")

    async def analyze_voice_stress(self, audio_path: str) -> Dict[str, Any]:
        """
        Analyze stress levels from voice audio
        """
        try:
            start_time = time.time()
            
            # Extract voice features for stress analysis
            stress_features = self._extract_voice_stress_features(audio_path)
            
            # Calculate stress score based on voice characteristics
            stress_score = self._calculate_voice_stress_score(stress_features)
            
            # Determine stress level
            stress_level = self._categorize_stress_level(stress_score)
            
            # Get detailed analysis
            detailed_analysis = self._analyze_voice_patterns(stress_features)
            
            processing_time = (time.time() - start_time) * 1000
            
            return {
                "level": stress_level,
                "overall_score": stress_score,
                "confidence": min(abs(stress_score) * 2, 1.0),  # Scale confidence
                "voice_indicators": {
                    "pitch_variance": stress_features.get("pitch_variance", 0.0),
                    "speaking_rate": stress_features.get("speaking_rate", 0.0),
                    "pause_frequency": stress_features.get("pause_frequency", 0.0),
                    "energy_variance": stress_features.get("energy_variance", 0.0),
                    "fundamental_frequency": stress_features.get("f0_mean", 0.0)
                },
                "detailed_analysis": detailed_analysis,
                "processing_time": processing_time
            }
            
        except Exception as e:
            logger.error(f"Voice stress analysis failed: {str(e)}")
            return self._default_stress_result()

    def _extract_voice_stress_features(self, audio_path: str) -> Dict[str, float]:
        """
        Extract voice features related to stress
        """
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=16000)
            
            features = {}
            
            # Pitch analysis
            f0 = librosa.piptrack(y=y, sr=sr, threshold=0.1)
            f0_values = []
            for frame in range(f0.shape[1]):
                freqs = f0[0][:, frame]
                mags = f0[1][:, frame]
                if len(mags[mags > 0]) > 0:
                    f0_values.append(freqs[np.argmax(mags)])
            
            if f0_values:
                features["f0_mean"] = np.mean(f0_values)
                features["f0_std"] = np.std(f0_values)
                features["pitch_variance"] = features["f0_std"] / (features["f0_mean"] + 1e-8)
            else:
                features["f0_mean"] = 0.0
                features["f0_std"] = 0.0
                features["pitch_variance"] = 0.0
            
            # Speaking rate (estimate from zero crossing rate and energy)
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            energy = librosa.feature.rms(y=y)[0]
            
            # Estimate syllables per second (rough approximation)
            syllable_rate = np.mean(zcr) * sr / 1000  # Simplified calculation
            features["speaking_rate"] = syllable_rate
            
            # Pause detection (low energy regions)
            energy_threshold = np.mean(energy) * 0.1
            pause_frames = np.sum(energy < energy_threshold)
            features["pause_frequency"] = pause_frames / len(energy)
            
            # Energy variance
            features["energy_mean"] = np.mean(energy)
            features["energy_std"] = np.std(energy)
            features["energy_variance"] = features["energy_std"] / (features["energy_mean"] + 1e-8)
            
            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            features["spectral_centroid_mean"] = np.mean(spectral_centroids)
            features["spectral_centroid_std"] = np.std(spectral_centroids)
            
            # MFCC variance (voice quality indicator)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            features["mfcc_variance"] = np.mean(np.var(mfcc, axis=1))
            
            return features
            
        except Exception as e:
            logger.error(f"Voice feature extraction failed: {str(e)}")
            return {}

    def _calculate_voice_stress_score(self, features: Dict[str, float]) -> float:
        """
        Calculate stress score from voice features
        """
        try:
            if not features:
                return 0.0
            
            stress_score = 0.0
            feature_count = 0
            
            # Pitch variance contribution
            if "pitch_variance" in features:
                pitch_var = features["pitch_variance"]
                if pitch_var > 0.6:  # High variance indicates stress
                    stress_score += min(pitch_var, 1.0) * 0.3
                elif pitch_var < 0.2:  # Very low variance can also indicate stress (monotone)
                    stress_score += (0.2 - pitch_var) * 0.2
                feature_count += 1
            
            # Speaking rate contribution
            if "speaking_rate" in features:
                rate = features["speaking_rate"]
                if rate > 4.0:  # Fast speaking
                    stress_score += min((rate - 4.0) / 4.0, 1.0) * 0.25
                elif rate < 1.5:  # Very slow speaking
                    stress_score += min((1.5 - rate) / 1.5, 1.0) * 0.15
                feature_count += 1
            
            # Pause frequency contribution
            if "pause_frequency" in features:
                pause_freq = features["pause_frequency"]
                if pause_freq > 0.4:  # Many pauses
                    stress_score += min(pause_freq, 1.0) * 0.2
                feature_count += 1
            
            # Energy variance contribution
            if "energy_variance" in features:
                energy_var = features["energy_variance"]
                if energy_var > 0.5:  # High energy variation
                    stress_score += min(energy_var, 1.0) * 0.25
                feature_count += 1
            
            # Normalize by number of features
            if feature_count > 0:
                stress_score = stress_score / feature_count * 4  # Scale back up
            
            return min(stress_score, 1.0)  # Cap at 1.0
            
        except Exception as e:
            logger.error(f"Stress score calculation failed: {str(e)}")
            return 0.0

    def _analyze_voice_patterns(self, features: Dict[str, float]) -> Dict[str, str]:
        """
        Provide detailed analysis of voice stress patterns
        """
        try:
            analysis = {}
            
            # Pitch analysis
            if "pitch_variance" in features:
                pitch_var = features["pitch_variance"]
                if pitch_var > 0.7:
                    analysis["pitch"] = "High pitch variability suggests emotional stress"
                elif pitch_var < 0.1:
                    analysis["pitch"] = "Very low pitch variation may indicate suppressed emotion"
                else:
                    analysis["pitch"] = "Normal pitch variation"
            
            # Speaking rate analysis
            if "speaking_rate" in features:
                rate = features["speaking_rate"]
                if rate > 5.0:
                    analysis["rate"] = "Rapid speech often indicates anxiety or stress"
                elif rate < 1.0:
                    analysis["rate"] = "Very slow speech may indicate depression or fatigue"
                else:
                    analysis["rate"] = "Normal speaking rate"
            
            # Energy analysis
            if "energy_variance" in features:
                energy_var = features["energy_variance"]
                if energy_var > 0.6:
                    analysis["energy"] = "High energy variation suggests emotional instability"
                else:
                    analysis["energy"] = "Stable energy levels"
            
            return analysis
            
        except Exception as e:
            logger.error(f"Voice pattern analysis failed: {str(e)}")
            return {}

    async def analyze_text_stress(self, text: str) -> Dict[str, Any]:
        """
        Analyze stress levels from text content
        """
        try:
            start_time = time.time()
            
            # Keyword-based stress analysis
            stress_scores = self._analyze_stress_keywords(text)
            
            # Linguistic stress indicators
            linguistic_indicators = self._analyze_linguistic_stress(text)
            
            # Physical stress mentions
            physical_indicators = self._detect_physical_stress_indicators(text)
            
            # Calculate overall stress score
            overall_score = self._calculate_text_stress_score(
                stress_scores, linguistic_indicators, physical_indicators
            )
            
            # Determine stress level
            stress_level = self._categorize_stress_level(overall_score)
            
            processing_time = (time.time() - start_time) * 1000
            
            return {
                "level": stress_level,
                "overall_score": overall_score,
                "confidence": self._calculate_text_confidence(stress_scores),
                "keyword_analysis": stress_scores,
                "linguistic_indicators": linguistic_indicators,
                "physical_indicators": physical_indicators,
                "processing_time": processing_time
            }
            
        except Exception as e:
            logger.error(f"Text stress analysis failed: {str(e)}")
            return self._default_stress_result()

    def _analyze_stress_keywords(self, text: str) -> Dict[str, Any]:
        """
        Analyze stress-related keywords in text
        """
        try:
            text_lower = text.lower()
            word_count = len(text.split())
            
            keyword_counts = {}
            total_stress_words = 0
            
            for level, keywords in self.stress_patterns.items():
                count = 0
                found_keywords = []
                
                for keyword in keywords:
                    keyword_count = text_lower.count(keyword.lower())
                    count += keyword_count
                    if keyword_count > 0:
                        found_keywords.append(keyword)
                
                keyword_counts[level] = {
                    "count": count,
                    "keywords_found": found_keywords,
                    "ratio": count / word_count if word_count > 0 else 0
                }
                
                if level != 'low_stress':  # Don't count low stress as stress
                    total_stress_words += count
            
            # Calculate stress intensity
            stress_intensity = total_stress_words / word_count if word_count > 0 else 0
            
            return {
                "keyword_counts": keyword_counts,
                "total_stress_words": total_stress_words,
                "stress_intensity": stress_intensity,
                "word_count": word_count
            }
            
        except Exception as e:
            logger.error(f"Keyword stress analysis failed: {str(e)}")
            return {}

    def _analyze_linguistic_stress(self, text: str) -> Dict[str, Any]:
        """
        Analyze linguistic patterns that indicate stress
        """
        try:
            indicators = {}
            
            # Exclamation marks (excitement/stress)
            exclamation_count = text.count('!')
            indicators["exclamation_frequency"] = exclamation_count / len(text.split()) if text else 0
            
            # All caps words (shouting/stress)
            words = text.split()
            caps_words = [w for w in words if w.isupper() and len(w) > 1]
            indicators["caps_ratio"] = len(caps_words) / len(words) if words else 0
            
            # Repetitive punctuation
            repeat_punct = len(re.findall(r'[!?]{2,}', text))
            indicators["repeat_punctuation"] = repeat_punct
            
            # Negative emotion words
            negative_patterns = [
                r'\b(hate|awful|terrible|horrible|disgusting|furious)\b',
                r'\b(never|nothing|nobody|nowhere)\b',  # Absolute negatives
                r'\b(why me|unfair|injustice)\b'
            ]
            
            negative_count = 0
            for pattern in negative_patterns:
                negative_count += len(re.findall(pattern, text.lower()))
            
            indicators["negative_intensity"] = negative_count / len(words) if words else 0
            
            # Short, fragmented sentences (can indicate stress)
            sentences = re.split(r'[.!?]+', text)
            short_sentences = [s for s in sentences if len(s.split()) < 5 and len(s.strip()) > 0]
            indicators["fragmentation"] = len(short_sentences) / len(sentences) if sentences else 0
            
            return indicators
            
        except Exception as e:
            logger.error(f"Linguistic stress analysis failed: {str(e)}")
            return {}

    def _detect_physical_stress_indicators(self, text: str) -> Dict[str, Any]:
        """
        Detect mentions of physical stress symptoms
        """
        try:
            text_lower = text.lower()
            found_indicators = []
            
            for indicator in self.physical_stress_indicators:
                if indicator in text_lower:
                    found_indicators.append(indicator)
            
            return {
                "physical_symptoms_mentioned": found_indicators,
                "physical_symptom_count": len(found_indicators),
                "has_physical_indicators": len(found_indicators) > 0
            }
            
        except Exception as e:
            logger.error(f"Physical stress detection failed: {str(e)}")
            return {"physical_symptoms_mentioned": [], "physical_symptom_count": 0}

    def _calculate_text_stress_score(self, stress_scores: Dict, linguistic: Dict, physical: Dict) -> float:
        """
        Calculate overall text stress score
        """
        try:
            score = 0.0
            
            # Keyword-based score (0.5 weight)
            if stress_scores and "keyword_counts" in stress_scores:
                keyword_counts = stress_scores["keyword_counts"]
                
                extreme_ratio = keyword_counts.get("extreme_stress", {}).get("ratio", 0)
                high_ratio = keyword_counts.get("high_stress", {}).get("ratio", 0)
                medium_ratio = keyword_counts.get("medium_stress", {}).get("ratio", 0)
                low_ratio = keyword_counts.get("low_stress", {}).get("ratio", 0)
                
                keyword_score = (extreme_ratio * 4 + high_ratio * 3 + medium_ratio * 2 - low_ratio * 1)
                score += keyword_score * 0.5
            
            # Linguistic indicators (0.3 weight)
            if linguistic:
                ling_score = (
                    linguistic.get("exclamation_frequency", 0) * 2 +
                    linguistic.get("caps_ratio", 0) * 3 +
                    linguistic.get("negative_intensity", 0) * 2 +
                    linguistic.get("fragmentation", 0) * 1
                )
                score += ling_score * 0.3
            
            # Physical indicators (0.2 weight)
            if physical:
                phys_score = min(physical.get("physical_symptom_count", 0) * 0.2, 1.0)
                score += phys_score * 0.2
            
            return min(score, 1.0)  # Cap at 1.0
            
        except Exception as e:
            logger.error(f"Text stress score calculation failed: {str(e)}")
            return 0.0

    def _calculate_text_confidence(self, stress_scores: Dict) -> float:
        """
        Calculate confidence in text stress analysis
        """
        try:
            if not stress_scores or "total_stress_words" not in stress_scores:
                return 0.0
            
            total_words = stress_scores.get("total_stress_words", 0)
            word_count = stress_scores.get("word_count", 1)
            
            # Confidence based on number of stress indicators found
            if total_words >= 3:
                return 0.9
            elif total_words >= 1:
                return 0.6
            elif word_count > 20:  # Long text with no stress words
                return 0.7
            else:
                return 0.3
                
        except Exception as e:
            logger.error(f"Confidence calculation failed: {str(e)}")
            return 0.0

    def _categorize_stress_level(self, score: float) -> str:
        """
        Categorize stress score into levels
        """
        if score >= 0.7:
            return "high"
        elif score >= 0.4:
            return "medium"
        elif score >= 0.1:
            return "low"
        else:
            return "minimal"

    def _default_stress_result(self) -> Dict[str, Any]:
        """
        Return default stress analysis result
        """
        return {
            "level": "unknown",
            "overall_score": 0.0,
            "confidence": 0.0,
            "processing_time": 0.0
        }

    def _models_loaded(self) -> bool:
        """
        Check if models are loaded
        """
        return self.stress_model is not None

    async def _save_models(self):
        """
        Save stress analysis models
        """
        try:
            if self.stress_model:
                joblib.dump(self.stress_model, os.path.join(self.model_path, "stress_classifier.pkl"))
            
            if self.scaler:
                joblib.dump(self.scaler, os.path.join(self.model_path, "stress_scaler.pkl"))
                
        except Exception as e:
            logger.error(f"Error saving stress models: {str(e)}")

    def health_check(self) -> str:
        """
        Health check for stress analyzer
        """
        try:
            return "operational"
        except Exception:
            return "error"

    async def cleanup(self):
        """
        Cleanup stress analyzer resources
        """
        try:
            self.stress_model = None
            self.scaler = None
            logger.info("Stress Analyzer cleaned up successfully")
        except Exception as e:
            logger.error(f"Stress analyzer cleanup failed: {str(e)}")