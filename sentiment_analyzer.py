# backend/text_analysis/sentiment_analyzer.py
"""
Text Sentiment and Emotion Analysis Module
Analyzes text for emotions, sentiment, stress indicators, and behavioral patterns
"""

import re
import time
import logging
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
from textblob import TextBlob
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib
import os

logger = logging.getLogger(__name__)

class TextSentimentAnalyzer:
    """
    Advanced text analysis for sentiment, emotions, and behavioral indicators
    """
    
    def __init__(self, model_path: str = "./ai-models/text_sentiment/"):
        self.model_path = model_path
        
        # Emotion categories
        self.emotions = [
            'joy', 'sadness', 'anger', 'fear', 'surprise', 
            'disgust', 'trust', 'anticipation', 'neutral'
        ]
        
        # Stress indicators
        self.stress_keywords = {
            'high_stress': [
                'overwhelmed', 'stressed', 'anxious', 'panic', 'worried',
                'exhausted', 'burnt out', 'pressure', 'deadline', 'crisis',
                'urgent', 'emergency', 'difficult', 'impossible', 'failing'
            ],
            'medium_stress': [
                'busy', 'hectic', 'challenging', 'demanding', 'tight',
                'concerned', 'uncertain', 'confused', 'frustrated', 'tired'
            ],
            'low_stress': [
                'calm', 'relaxed', 'peaceful', 'confident', 'comfortable',
                'easy', 'manageable', 'clear', 'organized', 'prepared'
            ]
        }
        
        # Confidence markers
        self.confidence_indicators = {
            'high_confidence': [
                'definitely', 'certainly', 'absolutely', 'sure', 'confident',
                'positive', 'know', 'believe', 'convinced', 'guarantee',
                'without doubt', 'clearly', 'obviously', 'undoubtedly'
            ],
            'low_confidence': [
                'maybe', 'perhaps', 'possibly', 'might', 'could',
                'uncertain', 'unsure', 'doubt', 'guess', 'think',
                'probably', 'seem', 'appear', 'suppose', 'assume'
            ]
        }
        
        # Deception indicators (linguistic patterns)
        self.deception_patterns = {
            'distancing': [
                r'\bthat person\b', r'\bhe/she\b', r'\bthey\b(?!\s+are)',
                r'\bsomeone\b', r'\banyone\b'
            ],
            'hedging': [
                r'\bas far as I know\b', r'\bto the best of my knowledge\b',
                r'\bI believe\b', r'\bI think\b', r'\bas I recall\b'
            ],
            'unnecessary_details': [
                r'\bactually\b', r'\bhonestly\b', r'\bto tell the truth\b',
                r'\bto be honest\b', r'\bbasically\b'
            ]
        }
        
        # Initialize models and tools
        self.sentiment_analyzer = None
        self.nlp = None
        self.emotion_model = None
        self.tfidf_vectorizer = None
        self.lemmatizer = None
        
        # Performance tracking
        self.performance_metrics = {
            "accuracy": 0.0,
            "processing_speed": 0.0,
            "confidence_threshold": 0.7
        }

    async def initialize(self):
        """
        Initialize NLP models and download required data
        """
        try:
            logger.info("Initializing Text Sentiment Analyzer...")
            
            # Download NLTK data
            await self._download_nltk_data()
            
            # Initialize NLTK components
            self.sentiment_analyzer = SentimentIntensityAnalyzer()
            self.lemmatizer = WordNetLemmatizer()
            
            # Load spaCy model
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                logger.warning("spaCy model not found. Using basic text processing.")
                self.nlp = None
            
            # Initialize TF-IDF vectorizer
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=10000,
                stop_words='english',
                ngram_range=(1, 2)
            )
            
            # Load or create emotion classification model
            await self._load_or_create_models()
            
            logger.info("Text Sentiment Analyzer initialized successfully!")
            
        except Exception as e:
            logger.error(f"Failed to initialize Text Sentiment Analyzer: {str(e)}")
            raise

    async def _download_nltk_data(self):
        """
        Download required NLTK data
        """
        try:
            nltk.download('vader_lexicon', quiet=True)
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            nltk.download('omw-1.4', quiet=True)
        except Exception as e:
            logger.warning(f"Some NLTK data could not be downloaded: {str(e)}")

    async def _load_or_create_models(self):
        """
        Load existing models or create basic ones
        """
        try:
            os.makedirs(self.model_path, exist_ok=True)
            
            # Try to load existing emotion model
            emotion_model_path = os.path.join(self.model_path, "emotion_classifier.pkl")
            tfidf_path = os.path.join(self.model_path, "tfidf_vectorizer.pkl")
            
            if os.path.exists(emotion_model_path) and os.path.exists(tfidf_path):
                self.emotion_model = joblib.load(emotion_model_path)
                self.tfidf_vectorizer = joblib.load(tfidf_path)
                logger.info("Loaded existing emotion classification model")
            else:
                # Create basic model with sample data
                await self._create_basic_emotion_model()
                logger.info("Created basic emotion classification model")
                
        except Exception as e:
            logger.error(f"Error loading/creating models: {str(e)}")

    async def _create_basic_emotion_model(self):
        """
        Create a basic emotion classification model
        """
        try:
            # Sample training data for basic model
            sample_texts = [
                "I am so happy and excited about this news!",
                "This makes me really sad and disappointed.",
                "I am furious about what happened today.",
                "I'm scared about the upcoming presentation.",
                "What a surprising turn of events!",
                "This is disgusting and unacceptable.",
                "I trust that everything will work out fine.",
                "I'm looking forward to tomorrow's meeting.",
                "This is a normal day with nothing special."
            ]
            
            sample_emotions = [
                'joy', 'sadness', 'anger', 'fear', 'surprise',
                'disgust', 'trust', 'anticipation', 'neutral'
            ]
            
            # Create TF-IDF features
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(sample_texts)
            
            # Train a simple Naive Bayes classifier
            self.emotion_model = MultinomialNB()
            self.emotion_model.fit(tfidf_matrix, sample_emotions)
            
            # Save models
            joblib.dump(self.emotion_model, os.path.join(self.model_path, "emotion_classifier.pkl"))
            joblib.dump(self.tfidf_vectorizer, os.path.join(self.model_path, "tfidf_vectorizer.pkl"))
            
        except Exception as e:
            logger.error(f"Error creating basic emotion model: {str(e)}")

    async def analyze_text(self, text: str, analysis_type: str = "full") -> Dict[str, Any]:
        """
        Comprehensive text analysis
        """
        try:
            start_time = time.time()
            
            # Preprocess text
            processed_text = self._preprocess_text(text)
            
            results = {}
            
            # Basic sentiment analysis
            if analysis_type in ["sentiment", "full"]:
                results["sentiment"] = self._analyze_sentiment(text)
            
            # Emotion detection
            if analysis_type in ["emotion", "full"]:
                results["emotions"] = self._detect_emotions(processed_text)
            
            # Stress indicators
            if analysis_type in ["stress", "full"]:
                results["stress"] = self._analyze_stress_indicators(text)
            
            # Confidence markers
            if analysis_type in ["confidence", "full"]:
                results["confidence"] = self._analyze_confidence_markers(text)
            
            # Deception indicators
            if analysis_type in ["deception", "full"]:
                results["deception"] = self._analyze_deception_indicators(text)
            
            # Linguistic features
            if analysis_type == "full":
                results["linguistic_features"] = self._extract_linguistic_features(text)
            
            # Calculate overall metrics
            results["overall_confidence"] = self._calculate_overall_confidence(results)
            results["processing_time"] = (time.time() - start_time) * 1000
            
            return results
            
        except Exception as e:
            logger.error(f"Text analysis failed: {str(e)}")
            raise

    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text for analysis
        """
        try:
            # Convert to lowercase
            text = text.lower()
            
            # Remove URLs
            text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
            
            # Remove email addresses
            text = re.sub(r'\S+@\S+', '', text)
            
            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            
            return text
            
        except Exception as e:
            logger.error(f"Text preprocessing failed: {str(e)}")
            return text

    def _analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment using multiple approaches
        """
        try:
            results = {}
            
            # VADER sentiment analysis
            if self.sentiment_analyzer:
                vader_scores = self.sentiment_analyzer.polarity_scores(text)
                results["vader"] = vader_scores
                
                # Determine overall sentiment
                if vader_scores['compound'] >= 0.05:
                    sentiment_label = "positive"
                elif vader_scores['compound'] <= -0.05:
                    sentiment_label = "negative"
                else:
                    sentiment_label = "neutral"
                
                results["label"] = sentiment_label
                results["confidence"] = abs(vader_scores['compound'])
            
            # TextBlob sentiment
            blob = TextBlob(text)
            results["textblob"] = {
                "polarity": blob.sentiment.polarity,
                "subjectivity": blob.sentiment.subjectivity
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {str(e)}")
            return {"label": "neutral", "confidence": 0.0}

    def _detect_emotions(self, text: str) -> Dict[str, float]:
        """
        Detect emotions in text
        """
        try:
            if not self.emotion_model or not self.tfidf_vectorizer:
                return {emotion: 0.1 for emotion in self.emotions}
            
            # Transform text using TF-IDF
            text_tfidf = self.tfidf_vectorizer.transform([text])
            
            # Get emotion probabilities
            emotion_probs = self.emotion_model.predict_proba(text_tfidf)[0]
            
            # Create emotion dictionary
            emotion_scores = dict(zip(self.emotions, emotion_probs))
            
            # Normalize scores
            total_score = sum(emotion_scores.values())
            if total_score > 0:
                emotion_scores = {k: v/total_score for k, v in emotion_scores.items()}
            
            return emotion_scores
            
        except Exception as e:
            logger.error(f"Emotion detection failed: {str(e)}")
            return {emotion: 1.0/len(self.emotions) for emotion in self.emotions}

    def _analyze_stress_indicators(self, text: str) -> Dict[str, Any]:
        """
        Analyze stress level indicators in text
        """
        try:
            text_lower = text.lower()
            word_count = len(text.split())
            
            stress_scores = {
                'high_stress': 0,
                'medium_stress': 0,
                'low_stress': 0
            }
            
            # Count stress-related keywords
            for level, keywords in self.stress_keywords.items():
                for keyword in keywords:
                    stress_scores[level] += text_lower.count(keyword.lower())
            
            # Normalize by word count
            if word_count > 0:
                normalized_scores = {k: v/word_count for k, v in stress_scores.items()}
            else:
                normalized_scores = stress_scores
            
            # Determine overall stress level
            max_stress = max(normalized_scores.items(), key=lambda x: x[1])
            
            # Additional stress indicators
            exclamation_count = text.count('!')
            caps_ratio = sum(1 for c in text if c.isupper()) / len(text) if text else 0
            
            return {
                "level": max_stress[0].replace('_stress', '') if max_stress[1] > 0 else "neutral",
                "confidence": max_stress[1],
                "indicators": normalized_scores,
                "exclamation_count": exclamation_count,
                "caps_ratio": caps_ratio,
                "overall_score": normalized_scores['high_stress'] * 3 + 
                               normalized_scores['medium_stress'] * 2 - 
                               normalized_scores['low_stress']
            }
            
        except Exception as e:
            logger.error(f"Stress analysis failed: {str(e)}")
            return {"level": "unknown", "confidence": 0.0}

    def _analyze_confidence_markers(self, text: str) -> Dict[str, Any]:
        """
        Analyze confidence level from text
        """
        try:
            text_lower = text.lower()
            word_count = len(text.split())
            
            confidence_counts = {
                'high_confidence': 0,
                'low_confidence': 0
            }
            
            # Count confidence markers
            for level, markers in self.confidence_indicators.items():
                for marker in markers:
                    confidence_counts[level] += text_lower.count(marker.lower())
            
            # Calculate confidence score
            if word_count > 0:
                high_conf_ratio = confidence_counts['high_confidence'] / word_count
                low_conf_ratio = confidence_counts['low_confidence'] / word_count
            else:
                high_conf_ratio = low_conf_ratio = 0
            
            # Overall confidence score (-1 to 1)
            confidence_score = high_conf_ratio - low_conf_ratio
            
            # Determine confidence level
            if confidence_score > 0.02:
                level = "high"
            elif confidence_score < -0.02:
                level = "low"
            else:
                level = "medium"
            
            return {
                "level": level,
                "score": confidence_score,
                "high_confidence_ratio": high_conf_ratio,
                "low_confidence_ratio": low_conf_ratio,
                "indicators": confidence_counts
            }
            
        except Exception as e:
            logger.error(f"Confidence analysis failed: {str(e)}")
            return {"level": "medium", "score": 0.0}

    def _analyze_deception_indicators(self, text: str) -> Dict[str, Any]:
        """
        Analyze potential deception indicators in text
        """
        try:
            deception_scores = {}
            total_indicators = 0
            
            # Check for deception patterns
            for category, patterns in self.deception_patterns.items():
                count = 0
                for pattern in patterns:
                    matches = len(re.findall(pattern, text, re.IGNORECASE))
                    count += matches
                
                deception_scores[category] = count
                total_indicators += count
            
            # Additional linguistic indicators
            word_count = len(text.split())
            sentence_count = len(sent_tokenize(text))
            
            # Calculate ratios
            avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
            first_person_count = len(re.findall(r'\b(I|me|my|mine)\b', text, re.IGNORECASE))
            first_person_ratio = first_person_count / word_count if word_count > 0 else 0
            
            # Overall deception risk score
            risk_score = (
                total_indicators / word_count if word_count > 0 else 0
            ) * 100  # Convert to percentage
            
            # Determine risk level
            if risk_score > 5:
                risk_level = "high"
            elif risk_score > 2:
                risk_level = "medium"
            else:
                risk_level = "low"
            
            return {
                "risk_level": risk_level,
                "risk_score": risk_score,
                "indicators": deception_scores,
                "linguistic_markers": {
                    "avg_sentence_length": avg_sentence_length,
                    "first_person_ratio": first_person_ratio,
                    "total_indicators": total_indicators
                }
            }
            
        except Exception as e:
            logger.error(f"Deception analysis failed: {str(e)}")
            return {"risk_level": "unknown", "risk_score": 0.0}

    def _extract_linguistic_features(self, text: str) -> Dict[str, Any]:
        """
        Extract detailed linguistic features
        """
        try:
            # Basic text statistics
            word_count = len(text.split())
            char_count = len(text)
            sentence_count = len(sent_tokenize(text))
            
            # Advanced features using spaCy (if available)
            linguistic_features = {
                "word_count": word_count,
                "char_count": char_count,
                "sentence_count": sentence_count,
                "avg_word_length": sum(len(word) for word in text.split()) / word_count if word_count > 0 else 0,
                "avg_sentence_length": word_count / sentence_count if sentence_count > 0 else 0
            }
            
            if self.nlp:
                doc = self.nlp(text)
                
                # POS tag distribution
                pos_counts = {}
                for token in doc:
                    pos = token.pos_
                    pos_counts[pos] = pos_counts.get(pos, 0) + 1
                
                linguistic_features["pos_distribution"] = pos_counts
                
                # Named entities
                entities = [(ent.text, ent.label_) for ent in doc.ents]
                linguistic_features["named_entities"] = entities
                
                # Dependency parsing features
                linguistic_features["dependency_depth"] = max(
                    len(list(token.ancestors)) for token in doc
                ) if len(doc) > 0 else 0
            
            return linguistic_features
            
        except Exception as e:
            logger.error(f"Linguistic feature extraction failed: {str(e)}")
            return {"word_count": len(text.split())}

    def _calculate_overall_confidence(self, results: Dict[str, Any]) -> float:
        """
        Calculate overall confidence in the analysis
        """
        try:
            confidences = []
            
            # Sentiment confidence
            if "sentiment" in results:
                confidences.append(results["sentiment"].get("confidence", 0.0))
            
            # Emotion confidence (highest emotion score)
            if "emotions" in results:
                max_emotion_score = max(results["emotions"].values())
                confidences.append(max_emotion_score)
            
            # Stress confidence
            if "stress" in results:
                confidences.append(results["stress"].get("confidence", 0.0))
            
            # Average confidence across all analyses
            if confidences:
                return sum(confidences) / len(confidences)
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Overall confidence calculation failed: {str(e)}")
            return 0.0

    def health_check(self) -> str:
        """
        Check if the text analyzer is working properly
        """
        try:
            if self.sentiment_analyzer and self.emotion_model:
                return "operational"
            elif self.sentiment_analyzer:
                return "partial"
            else:
                return "error"
        except Exception:
            return "error"

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about loaded models and capabilities
        """
        return {
            "available_emotions": self.emotions,
            "models_loaded": {
                "sentiment_analyzer": self.sentiment_analyzer is not None,
                "emotion_model": self.emotion_model is not None,
                "spacy_nlp": self.nlp is not None,
                "tfidf_vectorizer": self.tfidf_vectorizer is not None
            },
            "analysis_capabilities": [
                "sentiment_analysis",
                "emotion_detection", 
                "stress_indicators",
                "confidence_markers",
                "deception_indicators",
                "linguistic_features"
            ]
        }

    def get_performance_metrics(self) -> Dict[str, float]:
        """
        Get performance metrics
        """
        return self.performance_metrics

    async def cleanup(self):
        """
        Cleanup resources
        """
        try:
            # Clear models from memory
            self.emotion_model = None
            self.tfidf_vectorizer = None
            self.nlp = None
            
            logger.info("Text Sentiment Analyzer cleaned up successfully")
            
        except Exception as e:
            logger.error(f"Cleanup failed: {str(e)}")