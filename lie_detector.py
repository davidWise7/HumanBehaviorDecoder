# backend/deception_analyzer/lie_detector.py
"""
Deception Detection Analyzer
Analyzes voice and text for potential deception indicators
Note: This is for research/educational purposes and should not be used
for legal or critical decision-making without proper validation
"""

import numpy as np
import librosa
import logging
import re
import time
from typing import Dict, List, Any, Optional
from collections import Counter

logger = logging.getLogger(__name__)

class DeceptionAnalyzer:
    """
    Analyzes potential deception indicators in voice and text
    WARNING: This is experimental and should not be used for legal decisions
    """
    
    def __init__(self):
        # Linguistic deception indicators
        self.deception_patterns = {
            # Distancing language
            'distancing': {
                'patterns': [
                    r'\bthat person\b', r'\bhe/she\b', r'\bthey\b(?!\s+are)',
                    r'\bsomeone\b', r'\banyone\b', r'\bpeople\b',
                    r'\bthe individual\b'
                ],
                'description': 'Using third person to distance from events'
            },
            
            # Hedging and qualification
            'hedging': {
                'patterns': [
                    r'\bas far as I know\b', r'\bto the best of my knowledge\b',
                    r'\bI believe\b', r'\bI think\b', r'\bas I recall\b',
                    r'\bif I remember correctly\b', r'\bI suppose\b'
                ],
                'description': 'Qualifying statements to avoid commitment'
            },
            
            # Unnecessary elaboration
            'elaboration': {
                'patterns': [
                    r'\bactually\b', r'\bhonestly\b', r'\bto tell the truth\b',
                    r'\bto be honest\b', r'\bbasically\b', r'\breally\b',
                    r'\btruthfully\b', r'\bfrankly\b'
                ],
                'description': 'Unnecessary emphasis on truthfulness'
            },
            
            # Evasive language
            'evasion': {
                'patterns': [
                    r'\bI don\'t remember\b', r'\bI can\'t recall\b',
                    r'\bI\'m not sure\b', r'\bI don\'t know\b',
                    r'\bI have no idea\b', r'\bthat\'s hard to say\b'
                ],
                'description': 'Avoiding direct answers'
            },
            
            # Temporal inconsistencies
            'temporal': {
                'patterns': [
                    r'\bthen\b.*\bthen\b.*\bthen\b',  # Excessive then usage
                    r'\bafter that\b.*\bafter that\b',
                    r'\bnext\b.*\bnext\b.*\bnext\b'
                ],
                'description': 'Unusual temporal sequencing'
            }
        }
        
        # Voice stress indicators that may correlate with deception
        self.voice_deception_indicators = {
            'pitch_changes': 'Significant pitch elevation under stress',
            'speech_rate_changes': 'Slower or faster than baseline speech',
            'pause_patterns': 'Unusual pauses or hesitations',
            'voice_quality': 'Changes in voice quality or tremor'
        }
        
        # Disclaimer about accuracy
        self.accuracy_disclaimer = """
        IMPORTANT: Deception detection is highly experimental and unreliable.
        These indicators should NEVER be used for legal, employment, or 
        other critical decisions. Many factors can cause false positives.
        """

    async def initialize(self):
        """
        Initialize deception analyzer
        """
        try:
            logger.info("Initializing Deception Analyzer...")
            logger.warning("Deception detection is experimental - use with extreme caution")
            logger.info("Deception Analyzer initialized successfully!")
        except Exception as e:
            logger.error(f"Failed to initialize Deception Analyzer: {str(e)}")
            raise

    async def analyze_text_deception(self, text: str) -> Dict[str, Any]:
        """
        Analyze text for potential deception indicators
        """
        try:
            start_time = time.time()
            
            # Pattern analysis
            pattern_results = self._analyze_deception_patterns(text)
            
            # Linguistic complexity analysis
            complexity_analysis = self._analyze_linguistic_complexity(text)
            
            # Pronoun usage analysis
            pronoun_analysis = self._analyze_pronoun_usage(text)
            
            # Emotional language analysis
            emotion_analysis = self._analyze_emotional_language(text)
            
            # Calculate overall risk score
            risk_score = self._calculate_text_deception_score(
                pattern_results, complexity_analysis, 
                pronoun_analysis, emotion_analysis
            )
            
            # Determine risk level
            risk_level = self._categorize_deception_risk(risk_score)
            
            processing_time = (time.time() - start_time) * 1000
            
            return {
                "risk_level": risk_level,
                "risk_score": risk_score,
                "confidence": self._calculate_analysis_confidence(pattern_results),
                "pattern_analysis": pattern_results,
                "complexity_analysis": complexity_analysis,
                "pronoun_analysis": pronoun_analysis,
                "emotion_analysis": emotion_analysis,
                "disclaimer": "Experimental analysis - not suitable for critical decisions",
                "processing_time": processing_time
            }
            
        except Exception as e:
            logger.error(f"Text deception analysis failed: {str(e)}")
            return self._default_deception_result()

    async def analyze_voice_deception(self, audio_path: str) -> Dict[str, Any]:
        """
        Analyze voice for potential deception indicators
        """
        try:
            start_time = time.time()
            
            # Extract voice stress features
            voice_features = self._extract_voice_stress_features(audio_path)
            
            # Analyze voice patterns
            pattern_analysis = self._analyze_voice_deception_patterns(voice_features)
            
            # Calculate voice stress score
            stress_score = self._calculate_voice_stress_score(voice_features)
            
            # Determine risk level based on stress indicators
            risk_level = self._categorize_voice_deception_risk(stress_score)
            
            processing_time = (time.time() - start_time) * 1000
            
            return {
                "risk_level": risk_level,
                "stress_score": stress_score,
                "confidence": 0.3,  # Low confidence for voice-only deception detection
                "voice_features": voice_features,
                "pattern_analysis": pattern_analysis,
                "disclaimer": "Voice stress analysis has very low reliability for deception detection",
                "processing_time": processing_time
            }
            
        except Exception as e:
            logger.error(f"Voice deception analysis failed: {str(e)}")
            return self._default_deception_result()

    def _analyze_deception_patterns(self, text: str) -> Dict[str, Any]:
        """
        Analyze text for linguistic deception patterns
        """
        try:
            results = {}
            total_matches = 0
            word_count = len(text.split())
            
            for category, pattern_info in self.deception_patterns.items():
                matches = []
                match_count = 0
                
                for pattern in pattern_info['patterns']:
                    found_matches = re.findall(pattern, text, re.IGNORECASE)
                    matches.extend(found_matches)
                    match_count += len(found_matches)
                
                results[category] = {
                    'count': match_count,
                    'matches': matches[:5],  # Limit to first 5 matches
                    'ratio': match_count / word_count if word_count > 0 else 0,
                    'description': pattern_info['description']
                }
                
                total_matches += match_count
            
            results['total_indicators'] = total_matches
            results['overall_ratio'] = total_matches / word_count if word_count > 0 else 0
            
            return results
            
        except Exception as e:
            logger.error(f"Pattern analysis failed: {str(e)}")
            return {}

    def _analyze_linguistic_complexity(self, text: str) -> Dict[str, Any]:
        """
        Analyze linguistic complexity which may indicate deception
        """
        try:
            words = text.split()
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            if not words or not sentences:
                return {}
            
            # Average word length
            avg_word_length = np.mean([len(word) for word in words])
            
            # Average sentence length
            avg_sentence_length = len(words) / len(sentences)
            
            # Sentence length variance
            sentence_lengths = [len(s.split()) for s in sentences]
            sentence_variance = np.var(sentence_lengths) if len(sentence_lengths) > 1 else 0
            
            # Complex words (>6 characters)
            complex_words = [w for w in words if len(w) > 6]
            complex_word_ratio = len(complex_words) / len(words)
            
            # Unusual complexity patterns
            complexity_flags = []
            if avg_word_length > 6:
                complexity_flags.append("Unusually long words")
            if avg_sentence_length > 25:
                complexity_flags.append("Very long sentences")
            if sentence_variance < 5:
                complexity_flags.append("Uniform sentence structure")
            
            return {
                "avg_word_length": avg_word_length,
                "avg_sentence_length": avg_sentence_length,
                "sentence_variance": sentence_variance,
                "complex_word_ratio": complex_word_ratio,
                "complexity_flags": complexity_flags
            }
            
        except Exception as e:
            logger.error(f"Complexity analysis failed: {str(e)}")
            return {}

    def _analyze_pronoun_usage(self, text: str) -> Dict[str, Any]:
        """
        Analyze pronoun usage patterns
        """
        try:
            text_lower = text.lower()
            words = text_lower.split()
            word_count = len(words)
            
            if word_count == 0:
                return {}
            
            # Count different pronoun types
            first_person = ['i', 'me', 'my', 'mine', 'myself']
            second_person = ['you', 'your', 'yours', 'yourself']
            third_person = ['he', 'she', 'it', 'they', 'them', 'their', 'him', 'her']
            
            first_count = sum(words.count(pronoun) for pronoun in first_person)
            second_count = sum(words.count(pronoun) for pronoun in second_person)
            third_count = sum(words.count(pronoun) for pronoun in third_person)
            
            total_pronouns = first_count + second_count + third_count
            
            # Calculate ratios
            first_ratio = first_count / word_count
            second_ratio = second_count / word_count
            third_ratio = third_count / word_count
            pronoun_ratio = total_pronouns / word_count
            
            # Deception indicators
            deception_flags = []
            if first_ratio < 0.02:  # Very low first-person usage
                deception_flags.append("Unusually low first-person pronoun usage")
            if third_ratio > 0.08:  # High third-person usage
                deception_flags.append("High third-person pronoun usage (distancing)")
            
            return {
                "first_person_ratio": first_ratio,
                "second_person_ratio": second_ratio,
                "third_person_ratio": third_ratio,
                "total_pronoun_ratio": pronoun_ratio,
                "deception_flags": deception_flags
            }
            
        except Exception as e:
            logger.error(f"Pronoun analysis failed: {str(e)}")
            return {}

    def _analyze_emotional_language(self, text: str) -> Dict[str, Any]:
        """
        Analyze emotional language patterns
        """
        try:
            text_lower = text.lower()
            
            # Emotional intensity words
            high_emotion = ['amazing', 'terrible', 'incredible', 'awful', 'fantastic', 'horrible']
            emotion_count = sum(text_lower.count(word) for word in high_emotion)
            
            # Negative emotions (can indicate stress/deception)
            negative_emotions = ['angry', 'sad', 'frustrated', 'upset', 'worried', 'nervous']
            negative_count = sum(text_lower.count(word) for word in negative_emotions)
            
            # Excessive certainty (overcompensation)
            certainty_words = ['absolutely', 'definitely', 'certainly', 'completely', 'totally']
            certainty_count = sum(text_lower.count(word) for word in certainty_words)
            
            word_count = len(text.split())
            
            return {
                "emotion_intensity": emotion_count / word_count if word_count > 0 else 0,
                "negative_emotion_ratio": negative_count / word_count if word_count > 0 else 0,
                "excessive_certainty": certainty_count / word_count if word_count > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Emotional language analysis failed: {str(e)}")
            return {}

    def _extract_voice_stress_features(self, audio_path: str) -> Dict[str, float]:
        """
        Extract voice features that may indicate stress/deception
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
                features["pitch_mean"] = np.mean(f0_values)
                features["pitch_std"] = np.std(f0_values)
                features["pitch_range"] = np.max(f0_values) - np.min(f0_values)
            
            # Voice quality measures
            rms = librosa.feature.rms(y=y)[0]
            features["energy_mean"] = np.mean(rms)
            features["energy_std"] = np.std(rms)
            
            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            features["spectral_centroid"] = np.mean(spectral_centroids)
            
            # Zero crossing rate (voice quality)
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            features["zcr_mean"] = np.mean(zcr)
            features["zcr_std"] = np.std(zcr)
            
            return features
            
        except Exception as e:
            logger.error(f"Voice feature extraction failed: {str(e)}")
            return {}

    def _analyze_voice_deception_patterns(self, features: Dict[str, float]) -> Dict[str, str]:
        """
        Analyze voice patterns for potential deception indicators
        """
        try:
            analysis = {}
            
            if "pitch_std" in features and "pitch_mean" in features:
                pitch_cv = features["pitch_std"] / (features["pitch_mean"] + 1e-8)
                if pitch_cv > 0.3:
                    analysis["pitch"] = "High pitch variation may indicate stress"
                else:
                    analysis["pitch"] = "Normal pitch variation"
            
            if "energy_std" in features and "energy_mean" in features:
                energy_cv = features["energy_std"] / (features["energy_mean"] + 1e-8)
                if energy_cv > 0.5:
                    analysis["energy"] = "Variable energy levels detected"
                else:
                    analysis["energy"] = "Stable energy levels"
            
            return analysis
            
        except Exception as e:
            logger.error(f"Voice pattern analysis failed: {str(e)}")
            return {}

    def _calculate_text_deception_score(self, patterns: Dict, complexity: Dict, 
                                      pronouns: Dict, emotions: Dict) -> float:
        """
        Calculate overall text deception risk score
        """
        try:
            score = 0.0
            
            # Pattern-based score (40% weight)
            if patterns and "overall_ratio" in patterns:
                pattern_score = min(patterns["overall_ratio"] * 10, 1.0)
                score += pattern_score * 0.4
            
            # Complexity score (20% weight)
            if complexity:
                complexity_flags = len(complexity.get("complexity_flags", []))
                complexity_score = min(complexity_flags / 3, 1.0)
                score += complexity_score * 0.2
            
            # Pronoun score (25% weight)
            if pronouns:
                pronoun_flags = len(pronouns.get("deception_flags", []))
                pronoun_score = min(pronoun_flags / 2, 1.0)
                score += pronoun_score * 0.25
            
            # Emotional score (15% weight)
            if emotions:
                emotion_score = emotions.get("excessive_certainty", 0) * 2
                emotion_score += emotions.get("negative_emotion_ratio", 0)
                score += min(emotion_score, 1.0) * 0.15
            
            return min(score, 1.0)
            
        except Exception as e:
            logger.error(f"Text deception score calculation failed: {str(e)}")
            return 0.0

    def _calculate_voice_stress_score(self, features: Dict[str, float]) -> float:
        """
        Calculate voice stress score (poor indicator of deception)
        """
        try:
            if not features:
                return 0.0
            
            stress_indicators = 0
            total_indicators = 0
            
            # High pitch variation
            if "pitch_std" in features and "pitch_mean" in features:
                pitch_cv = features["pitch_std"] / (features["pitch_mean"] + 1e-8)
                if pitch_cv > 0.25:
                    stress_indicators += 1
                total_indicators += 1
            
            # High energy variation
            if "energy_std" in features and "energy_mean" in features:
                energy_cv = features["energy_std"] / (features["energy_mean"] + 1e-8)
                if energy_cv > 0.4:
                    stress_indicators += 1
                total_indicators += 1
            
            return stress_indicators / total_indicators if total_indicators > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Voice stress score calculation failed: {str(e)}")
            return 0.0

    def _calculate_analysis_confidence(self, patterns: Dict) -> float:
        """
        Calculate confidence in the deception analysis
        """
        try:
            if not patterns or "total_indicators" not in patterns:
                return 0.1  # Very low confidence
            
            indicators = patterns["total_indicators"]
            if indicators >= 5:
                return 0.4  # Still low confidence
            elif indicators >= 2:
                return 0.3
            else:
                return 0.2
                
        except Exception:
            return 0.1

    def _categorize_deception_risk(self, score: float) -> str:
        """
        Categorize deception risk level
        """
        if score >= 0.7:
            return "high"
        elif score >= 0.4:
            return "medium"
        elif score >= 0.2:
            return "low"
        else:
            return "minimal"

    def _categorize_voice_deception_risk(self, score: float) -> str:
        """
        Categorize voice-based deception risk (always low confidence)
        """
        if score >= 0.6:
            return "elevated_stress"  # Don't call it deception
        elif score >= 0.4:
            return "moderate_stress"
        elif score >= 0.2:
            return "mild_stress"
        else:
            return "normal"

    def _default_deception_result(self) -> Dict[str, Any]:
        """
        Return default deception analysis result
        """
        return {
            "risk_level": "unknown",
            "risk_score": 0.0,
            "confidence": 0.0,
            "disclaimer": "Analysis failed - results unreliable",
            "processing_time": 0.0
        }

    def health_check(self) -> str:
        """
        Health check for deception analyzer
        """
        return "operational"

    async def cleanup(self):
        """
        Cleanup deception analyzer resources
        """
        try:
            logger.info("Deception Analyzer cleaned up successfully")
        except Exception as e:
            logger.error(f"Deception analyzer cleanup failed: {str(e)}")

    def get_accuracy_disclaimer(self) -> str:
        """
        Get important disclaimer about deception detection accuracy
        """
        return self.accuracy_disclaimer