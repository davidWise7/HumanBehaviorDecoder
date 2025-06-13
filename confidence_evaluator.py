# backend/confidence_scorer/confidence_evaluator.py
"""
Confidence Level Evaluator
Analyzes confidence levels from voice and text using linguistic and acoustic cues
"""

import numpy as np
import librosa
import logging
import re
import time
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class ConfidenceEvaluator:
    """
    Evaluates confidence levels from voice and text inputs
    """
    
    def __init__(self):
        # High confidence linguistic markers
        self.high_confidence_markers = [
            # Certainty words
            'definitely', 'certainly', 'absolutely', 'sure', 'confident',
            'positive', 'know', 'believe', 'convinced', 'guarantee',
            'without doubt', 'clearly', 'obviously', 'undoubtedly',
            'precisely', 'exactly', 'indeed', 'surely', 'unquestionably',
            
            # Strong assertions
            'will', 'must', 'always', 'never', 'completely', 'totally',
            'fully', 'entirely', 'perfectly', 'strongly',
            
            # Decisive language
            'decided', 'determined', 'resolved', 'committed', 'established'
        ]
        
        # Low confidence linguistic markers
        self.low_confidence_markers = [
            # Uncertainty words
            'maybe', 'perhaps', 'possibly', 'might', 'could', 'would',
            'uncertain', 'unsure', 'doubt', 'guess', 'think', 'believe',
            'probably', 'seem', 'appear', 'suppose', 'assume',
            
            # Hedging phrases
            'kind of', 'sort of', 'i think', 'i guess', 'i suppose',
            'it seems', 'it appears', 'may be', 'could be', 'might be',
            
            # Qualifiers
            'somewhat', 'rather', 'quite', 'fairly', 'relatively',
            'pretty much', 'more or less', 'to some extent'
        ]
        
        # Voice confidence indicators (acoustic features)
        self.voice_confidence_features = {
            'pitch_stability': {'confident': (0.8, 1.0), 'unconfident': (0.0, 0.4)},
            'volume_consistency': {'confident': (0.7, 1.0), 'unconfident': (0.0, 0.5)},
            'speech_rate': {'confident': (3.0, 5.0), 'unconfident': (1.0, 2.5)},
            'pause_patterns': {'confident': (0.1, 0.3), 'unconfident': (0.5, 1.0)}
        }

    async def initialize(self):
        """
        Initialize confidence evaluator
        """
        try:
            logger.info("Initializing Confidence Evaluator...")
            logger.info("Confidence Evaluator initialized successfully!")
        except Exception as e:
            logger.error(f"Failed to initialize Confidence Evaluator: {str(e)}")
            raise

    async def analyze_text_confidence(self, text: str) -> Dict[str, Any]:
        """
        Analyze confidence level from text content
        """
        try:
            start_time = time.time()
            
            # Basic text analysis
            word_count = len(text.split())
            text_lower = text.lower()
            
            # Count confidence markers
            high_conf_count = self._count_markers(text_lower, self.high_confidence_markers)
            low_conf_count = self._count_markers(text_lower, self.low_confidence_markers)
            
            # Analyze linguistic patterns
            linguistic_analysis = self._analyze_linguistic_confidence(text)
            
            # Analyze sentence structure
            structure_analysis = self._analyze_sentence_structure(text)
            
            # Calculate overall confidence score
            confidence_score = self._calculate_text_confidence_score(
                high_conf_count, low_conf_count, word_count,
                linguistic_analysis, structure_analysis
            )
            
            # Determine confidence level
            confidence_level = self._categorize_confidence_level(confidence_score)
            
            processing_time = (time.time() - start_time) * 1000
            
            return {
                "level": confidence_level,
                "score": confidence_score,
                "confidence": abs(confidence_score),
                "markers": {
                    "high_confidence_count": high_conf_count,
                    "low_confidence_count": low_conf_count,
                    "high_confidence_ratio": high_conf_count / word_count if word_count > 0 else 0,
                    "low_confidence_ratio": low_conf_count / word_count if word_count > 0 else 0
                },
                "linguistic_analysis": linguistic_analysis,
                "structure_analysis": structure_analysis,
                "processing_time": processing_time
            }
            
        except Exception as e:
            logger.error(f"Text confidence analysis failed: {str(e)}")
            return self._default_confidence_result()

    async def analyze_voice_confidence(self, audio_path: str) -> Dict[str, Any]:
        """
        Analyze confidence level from voice audio
        """
        try:
            start_time = time.time()
            
            # Extract voice features
            voice_features = self._extract_voice_confidence_features(audio_path)
            
            # Calculate confidence score from voice
            confidence_score = self._calculate_voice_confidence_score(voice_features)
            
            # Analyze voice patterns
            pattern_analysis = self._analyze_voice_confidence_patterns(voice_features)
            
            # Determine confidence level
            confidence_level = self._categorize_confidence_level(confidence_score)
            
            processing_time = (time.time() - start_time) * 1000
            
            return {
                "level": confidence_level,
                "score": confidence_score,
                "confidence": abs(confidence_score),
                "voice_features": voice_features,
                "pattern_analysis": pattern_analysis,
                "processing_time": processing_time
            }
            
        except Exception as e:
            logger.error(f"Voice confidence analysis failed: {str(e)}")
            return self._default_confidence_result()

    def _count_markers(self, text: str, markers: List[str]) -> int:
        """
        Count confidence markers in text
        """
        count = 0
        for marker in markers:
            # Use word boundaries to avoid partial matches
            pattern = r'\b' + re.escape(marker) + r'\b'
            count += len(re.findall(pattern, text, re.IGNORECASE))
        return count

    def _analyze_linguistic_confidence(self, text: str) -> Dict[str, Any]:
        """
        Analyze linguistic patterns that indicate confidence
        """
        try:
            analysis = {}
            
            # Question marks (uncertainty)
            question_count = text.count('?')
            analysis["question_frequency"] = question_count / len(text.split()) if text else 0
            
            # Exclamation marks (emphasis/confidence)
            exclamation_count = text.count('!')
            analysis["exclamation_frequency"] = exclamation_count / len(text.split()) if text else 0
            
            # Modal verbs analysis
            modal_verbs = {
                'strong': ['will', 'must', 'shall'],
                'weak': ['might', 'could', 'would', 'should', 'may']
            }
            
            text_lower = text.lower()
            strong_modal_count = sum(text_lower.count(verb) for verb in modal_verbs['strong'])
            weak_modal_count = sum(text_lower.count(verb) for verb in modal_verbs['weak'])
            
            analysis["strong_modal_ratio"] = strong_modal_count / len(text.split()) if text else 0
            analysis["weak_modal_ratio"] = weak_modal_count / len(text.split()) if text else 0
            
            # Filler words (hesitation)
            filler_words = ['um', 'uh', 'er', 'ah', 'like', 'you know', 'well']
            filler_count = sum(text_lower.count(filler) for filler in filler_words)
            analysis["filler_frequency"] = filler_count / len(text.split()) if text else 0
            
            # Intensifiers (confidence boosters)
            intensifiers = ['very', 'extremely', 'highly', 'really', 'truly', 'absolutely']
            intensifier_count = sum(text_lower.count(word) for word in intensifiers)
            analysis["intensifier_frequency"] = intensifier_count / len(text.split()) if text else 0
            
            return analysis
            
        except Exception as e:
            logger.error(f"Linguistic confidence analysis failed: {str(e)}")
            return {}

    def _analyze_sentence_structure(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentence structure for confidence indicators
        """
        try:
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            if not sentences:
                return {"avg_sentence_length": 0, "short_sentence_ratio": 0}
            
            # Sentence length analysis
            sentence_lengths = [len(s.split()) for s in sentences]
            avg_length = np.mean(sentence_lengths)
            
            # Short sentences can indicate hesitation or emphasis
            short_sentences = [s for s in sentence_lengths if s < 5]
            short_ratio = len(short_sentences) / len(sentences)
            
            # Very long sentences can indicate rambling (low confidence)
            long_sentences = [s for s in sentence_lengths if s > 20]
            long_ratio = len(long_sentences) / len(sentences)
            
            # Sentence variety (confident speakers vary sentence length)
            length_variance = np.var(sentence_lengths) if len(sentence_lengths) > 1 else 0
            
            return {
                "avg_sentence_length": avg_length,
                "short_sentence_ratio": short_ratio,
                "long_sentence_ratio": long_ratio,
                "length_variance": length_variance,
                "sentence_count": len(sentences)
            }
            
        except Exception as e:
            logger.error(f"Sentence structure analysis failed: {str(e)}")
            return {}

    def _extract_voice_confidence_features(self, audio_path: str) -> Dict[str, float]:
        """
        Extract voice features related to confidence
        """
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=16000)
            
            features = {}
            
            # Pitch stability (confident speakers have stable pitch)
            f0 = librosa.piptrack(y=y, sr=sr, threshold=0.1)
            f0_values = []
            for frame in range(f0.shape[1]):
                freqs = f0[0][:, frame]
                mags = f0[1][:, frame]
                if len(mags[mags > 0]) > 0:
                    f0_values.append(freqs[np.argmax(mags)])
            
            if f0_values:
                f0_mean = np.mean(f0_values)
                f0_std = np.std(f0_values)
                features["pitch_stability"] = 1.0 - min(f0_std / (f0_mean + 1e-8), 1.0)
                features["pitch_mean"] = f0_mean
            else:
                features["pitch_stability"] = 0.5
                features["pitch_mean"] = 0.0
            
            # Volume consistency (confident speakers maintain consistent volume)
            rms = librosa.feature.rms(y=y)[0]
            rms_mean = np.mean(rms)
            rms_std = np.std(rms)
            features["volume_consistency"] = 1.0 - min(rms_std / (rms_mean + 1e-8), 1.0)
            features["volume_mean"] = rms_mean
            
            # Speech rate estimation
            # Use zero crossing rate and energy to estimate speech rate
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            energy = librosa.feature.rms(y=y)[0]
            
            # Rough speech rate estimation
            speech_frames = np.sum(energy > np.mean(energy) * 0.1)
            speech_duration = speech_frames * 512 / sr  # hop_length = 512
            word_estimate = speech_duration * 2.5  # rough words per second
            features["speech_rate"] = word_estimate
            
            # Pause detection and analysis
            silence_threshold = np.mean(energy) * 0.1
            pause_frames = np.sum(energy < silence_threshold)
            total_frames = len(energy)
            features["pause_ratio"] = pause_frames / total_frames
            
            # Spectral confidence features
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            features["spectral_centroid_mean"] = np.mean(spectral_centroids)
            features["spectral_centroid_std"] = np.std(spectral_centroids)
            
            # Voice tremor detection (nervousness indicator)
            # High frequency modulation in pitch can indicate nervousness
            if f0_values and len(f0_values) > 10:
                f0_diff = np.diff(f0_values)
                tremor_measure = np.std(f0_diff)
                features["voice_tremor"] = min(tremor_measure / 100, 1.0)  # Normalize
            else:
                features["voice_tremor"] = 0.0
            
            return features
            
        except Exception as e:
            logger.error(f"Voice feature extraction failed: {str(e)}")
            return {}

    def _calculate_voice_confidence_score(self, features: Dict[str, float]) -> float:
        """
        Calculate confidence score from voice features
        """
        try:
            if not features:
                return 0.0
            
            confidence_score = 0.0
            feature_count = 0
            
            # Pitch stability (25% weight)
            if "pitch_stability" in features:
                pitch_stability = features["pitch_stability"]
                confidence_score += pitch_stability * 0.25
                feature_count += 1
            
            # Volume consistency (20% weight)
            if "volume_consistency" in features:
                volume_consistency = features["volume_consistency"]
                confidence_score += volume_consistency * 0.20
                feature_count += 1
            
            # Speech rate (20% weight)
            if "speech_rate" in features:
                speech_rate = features["speech_rate"]
                # Optimal speech rate is around 3-5 words per second
                if 3.0 <= speech_rate <= 5.0:
                    rate_score = 1.0
                elif speech_rate < 3.0:
                    rate_score = speech_rate / 3.0
                else:  # speech_rate > 5.0
                    rate_score = max(0.0, 1.0 - (speech_rate - 5.0) / 3.0)
                
                confidence_score += rate_score * 0.20
                feature_count += 1
            
            # Pause patterns (15% weight)
            if "pause_ratio" in features:
                pause_ratio = features["pause_ratio"]
                # Moderate pauses are good, too many or too few indicate issues
                if 0.1 <= pause_ratio <= 0.3:
                    pause_score = 1.0
                elif pause_ratio < 0.1:
                    pause_score = pause_ratio / 0.1
                else:  # pause_ratio > 0.3
                    pause_score = max(0.0, 1.0 - (pause_ratio - 0.3) / 0.4)
                
                confidence_score += pause_score * 0.15
                feature_count += 1
            
            # Voice tremor (20% weight, inverse)
            if "voice_tremor" in features:
                tremor = features["voice_tremor"]
                tremor_score = 1.0 - tremor  # Less tremor = more confidence
                confidence_score += tremor_score * 0.20
                feature_count += 1
            
            # Normalize by number of features and convert to -1 to 1 scale
            if feature_count > 0:
                confidence_score = confidence_score / feature_count
                # Convert from 0-1 to -1-1 scale (0.5 becomes 0)
                confidence_score = (confidence_score - 0.5) * 2
            
            return max(-1.0, min(1.0, confidence_score))
            
        except Exception as e:
            logger.error(f"Voice confidence score calculation failed: {str(e)}")
            return 0.0

    def _analyze_voice_confidence_patterns(self, features: Dict[str, float]) -> Dict[str, str]:
        """
        Provide detailed analysis of voice confidence patterns
        """
        try:
            analysis = {}
            
            # Pitch analysis
            if "pitch_stability" in features:
                stability = features["pitch_stability"]
                if stability > 0.8:
                    analysis["pitch"] = "Very stable pitch indicates high confidence"
                elif stability > 0.6:
                    analysis["pitch"] = "Moderately stable pitch suggests reasonable confidence"
                elif stability > 0.4:
                    analysis["pitch"] = "Some pitch variation may indicate mild uncertainty"
                else:
                    analysis["pitch"] = "High pitch variation suggests nervousness or stress"
            
            # Volume analysis
            if "volume_consistency" in features:
                consistency = features["volume_consistency"]
                if consistency > 0.7:
                    analysis["volume"] = "Consistent volume suggests confident delivery"
                elif consistency > 0.5:
                    analysis["volume"] = "Moderate volume consistency"
                else:
                    analysis["volume"] = "Inconsistent volume may indicate hesitation"
            
            # Speech rate analysis
            if "speech_rate" in features:
                rate = features["speech_rate"]
                if 3.0 <= rate <= 5.0:
                    analysis["rate"] = "Optimal speech rate indicates confidence"
                elif rate < 2.0:
                    analysis["rate"] = "Very slow speech may indicate uncertainty or depression"
                elif rate > 6.0:
                    analysis["rate"] = "Rapid speech may indicate anxiety or nervousness"
                else:
                    analysis["rate"] = "Speech rate slightly outside optimal range"
            
            # Tremor analysis
            if "voice_tremor" in features:
                tremor = features["voice_tremor"]
                if tremor > 0.5:
                    analysis["tremor"] = "Noticeable voice tremor suggests nervousness"
                elif tremor > 0.3:
                    analysis["tremor"] = "Slight voice tremor detected"
                else:
                    analysis["tremor"] = "Steady voice indicates calm confidence"
            
            return analysis
            
        except Exception as e:
            logger.error(f"Voice pattern analysis failed: {str(e)}")
            return {}

    def _calculate_text_confidence_score(self, high_count: int, low_count: int, 
                                       word_count: int, linguistic: Dict, structure: Dict) -> float:
        """
        Calculate overall text confidence score
        """
        try:
            if word_count == 0:
                return 0.0
            
            # Base score from confidence markers
            high_ratio = high_count / word_count
            low_ratio = low_count / word_count
            marker_score = (high_ratio - low_ratio) * 5  # Amplify the difference
            
            # Linguistic patterns contribution
            ling_score = 0.0
            if linguistic:
                # Positive indicators
                ling_score += linguistic.get("strong_modal_ratio", 0) * 3
                ling_score += linguistic.get("intensifier_frequency", 0) * 2
                ling_score += linguistic.get("exclamation_frequency", 0) * 1
                
                # Negative indicators
                ling_score -= linguistic.get("weak_modal_ratio", 0) * 2
                ling_score -= linguistic.get("question_frequency", 0) * 2
                ling_score -= linguistic.get("filler_frequency", 0) * 3
            
            # Sentence structure contribution
            struct_score = 0.0
            if structure:
                avg_length = structure.get("avg_sentence_length", 0)
                short_ratio = structure.get("short_sentence_ratio", 0)
                long_ratio = structure.get("long_sentence_ratio", 0)
                
                # Optimal sentence length is around 10-15 words
                if 10 <= avg_length <= 15:
                    struct_score += 0.2
                elif avg_length < 5:
                    struct_score -= 0.2  # Very short sentences can indicate hesitation
                elif avg_length > 25:
                    struct_score -= 0.1  # Very long sentences can indicate rambling
                
                # Too many short sentences can indicate uncertainty
                if short_ratio > 0.5:
                    struct_score -= short_ratio * 0.3
                
                # Too many long sentences can indicate overcompensation
                if long_ratio > 0.3:
                    struct_score -= long_ratio * 0.2
            
            # Combine scores
            total_score = marker_score * 0.6 + ling_score * 0.3 + struct_score * 0.1
            
            # Normalize to -1 to 1 range
            return max(-1.0, min(1.0, total_score))
            
        except Exception as e:
            logger.error(f"Text confidence score calculation failed: {str(e)}")
            return 0.0

    def _categorize_confidence_level(self, score: float) -> str:
        """
        Categorize confidence score into levels
        """
        if score >= 0.5:
            return "high"
        elif score >= 0.1:
            return "medium"
        elif score >= -0.1:
            return "neutral"
        elif score >= -0.5:
            return "low"
        else:
            return "very_low"

    def _default_confidence_result(self) -> Dict[str, Any]:
        """
        Return default confidence analysis result
        """
        return {
            "level": "unknown",
            "score": 0.0,
            "confidence": 0.0,
            "processing_time": 0.0
        }

    def health_check(self) -> str:
        """
        Health check for confidence evaluator
        """
        return "operational"

    async def cleanup(self):
        """
        Cleanup confidence evaluator resources
        """
        try:
            logger.info("Confidence Evaluator cleaned up successfully")
        except Exception as e:
            logger.error(f"Confidence evaluator cleanup failed: {str(e)}")