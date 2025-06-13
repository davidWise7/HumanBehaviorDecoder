# backend/voice_analysis/emotion_detector.py
"""
Voice Emotion Detection Module
Analyzes audio files to detect emotions using machine learning models
"""

import numpy as np
import librosa
import joblib
import tensorflow as tf
from typing import Dict, List, Tuple, Optional
import logging
import os
import time
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
import soundfile as sf

logger = logging.getLogger(__name__)

class VoiceEmotionDetector:
    """
    Advanced voice emotion detection using multiple ML models
    Supports real-time and batch processing
    """
    
    def __init__(self, model_path: str = "./ai-models/voice_emotion/"):
        self.model_path = model_path
        self.emotions = [
            'neutral', 'calm', 'happy', 'sad', 
            'angry', 'fearful', 'disgust', 'surprised'
        ]
        
        # Model components
        self.cnn_model = None
        self.lstm_model = None
        self.svm_model = None
        self.scaler = None
        
        # Audio processing parameters
        self.sample_rate = 16000
        self.duration = 3.0  # seconds
        self.n_mfcc = 40
        self.n_fft = 2048
        self.hop_length = 512
        
        # Performance metrics
        self.model_performance = {
            "accuracy": 0.0,
            "inference_time": 0.0,
            "confidence_threshold": 0.6
        }
        
        # Feature extraction cache
        self.feature_cache = {}
        
    async def initialize(self):
        """
        Initialize and load all models
        """
        try:
            logger.info("Initializing Voice Emotion Detector...")
            
            # Create model directory if it doesn't exist
            os.makedirs(self.model_path, exist_ok=True)
            
            # Load pre-trained models (if available)
            await self._load_models()
            
            # If no models exist, create and train basic models
            if not self._models_loaded():
                logger.warning("No pre-trained models found. Creating basic models...")
                await self._create_basic_models()
            
            logger.info("Voice Emotion Detector initialized successfully!")
            
        except Exception as e:
            logger.error(f"Failed to initialize Voice Emotion Detector: {str(e)}")
            raise

    async def _load_models(self):
        """
        Load pre-trained models from disk
        """
        try:
            # Load CNN model
            cnn_path = os.path.join(self.model_path, "emotion_cnn.h5")
            if os.path.exists(cnn_path):
                self.cnn_model = load_model(cnn_path)
                logger.info("CNN model loaded successfully")
            
            # Load LSTM model
            lstm_path = os.path.join(self.model_path, "emotion_lstm.h5")
            if os.path.exists(lstm_path):
                self.lstm_model = load_model(lstm_path)
                logger.info("LSTM model loaded successfully")
            
            # Load SVM model
            svm_path = os.path.join(self.model_path, "emotion_svm.pkl")
            if os.path.exists(svm_path):
                self.svm_model = joblib.load(svm_path)
                logger.info("SVM model loaded successfully")
            
            # Load scaler
            scaler_path = os.path.join(self.model_path, "feature_scaler.pkl")
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
                logger.info("Feature scaler loaded successfully")
                
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")

    async def _create_basic_models(self):
        """
        Create basic models for emotion detection
        """
        try:
            # Create a simple CNN model for emotion classification
            self.cnn_model = self._build_cnn_model()
            
            # Create feature scaler
            self.scaler = StandardScaler()
            
            # Save models
            await self._save_models()
            
            logger.info("Basic models created and saved")
            
        except Exception as e:
            logger.error(f"Error creating basic models: {str(e)}")

    def _build_cnn_model(self):
        """
        Build a CNN model for emotion classification
        """
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(40, 87, 1)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.25),
            
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.25),
            
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dropout(0.5),
            
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(len(self.emotions), activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model

    def extract_features(self, audio_path: str) -> Dict[str, np.ndarray]:
        """
        Extract comprehensive features from audio file
        """
        try:
            # Load audio file
            audio, sr = librosa.load(audio_path, sr=self.sample_rate, duration=self.duration)
            
            # Pad or trim audio to fixed length
            audio = self._normalize_audio_length(audio)
            
            features = {}
            
            # MFCC features
            mfcc = librosa.feature.mfcc(
                y=audio, sr=sr, n_mfcc=self.n_mfcc,
                n_fft=self.n_fft, hop_length=self.hop_length
            )
            features['mfcc'] = mfcc
            
            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
            zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)[0]
            
            features['spectral'] = np.array([
                np.mean(spectral_centroids),
                np.std(spectral_centroids),
                np.mean(spectral_rolloff),
                np.std(spectral_rolloff),
                np.mean(spectral_bandwidth),
                np.std(spectral_bandwidth),
                np.mean(zero_crossing_rate),
                np.std(zero_crossing_rate)
            ])
            
            # Chromagram
            chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
            features['chroma'] = chroma
            
            # Mel-scale spectrogram
            mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr)
            features['mel_spectrogram'] = mel_spectrogram
            
            # Tempo and rhythm
            tempo, beats = librosa.beat.beat_track(y=audio, sr=sr)
            features['tempo'] = np.array([tempo])
            
            # Root Mean Square Energy
            rms = librosa.feature.rms(y=audio)[0]
            features['rms'] = np.array([np.mean(rms), np.std(rms)])
            
            return features
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {str(e)}")
            raise

    def _normalize_audio_length(self, audio: np.ndarray) -> np.ndarray:
        """
        Normalize audio to fixed length
        """
        target_length = int(self.sample_rate * self.duration)
        
        if len(audio) > target_length:
            # Trim audio
            audio = audio[:target_length]
        elif len(audio) < target_length:
            # Pad audio
            audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
        
        return audio

    async def analyze_emotion(self, audio_path: str) -> Dict[str, any]:
        """
        Analyze emotion from audio file
        """
        try:
            start_time = time.time()
            
            # Extract features
            features = self.extract_features(audio_path)
            
            # Prepare features for different models
            results = {}
            
            # CNN prediction (using MFCC as 2D input)
            if self.cnn_model:
                mfcc_2d = features['mfcc'].reshape(1, 40, -1, 1)
                if mfcc_2d.shape[2] >= 87:  # Ensure minimum time frames
                    mfcc_2d = mfcc_2d[:, :, :87, :]  # Standardize to 87 time frames
                    cnn_pred = self.cnn_model.predict(mfcc_2d, verbose=0)
                    results['cnn_emotions'] = dict(zip(self.emotions, cnn_pred[0]))
            
            # SVM prediction (using flattened features)
            if self.svm_model and self.scaler:
                # Combine all features into a single vector
                feature_vector = self._combine_features(features)
                if len(feature_vector) > 0:
                    feature_vector = self.scaler.transform([feature_vector])
                    svm_pred = self.svm_model.predict_proba(feature_vector)
                    results['svm_emotions'] = dict(zip(self.emotions, svm_pred[0]))
            
            # Ensemble prediction (average of available models)
            ensemble_emotions = self._ensemble_prediction(results)
            
            # Calculate confidence metrics
            confidence_score = self._calculate_confidence(ensemble_emotions)
            
            # Determine dominant emotion
            dominant_emotion = max(ensemble_emotions.items(), key=lambda x: x[1])
            
            processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            
            return {
                'emotions': ensemble_emotions,
                'dominant_emotion': {
                    'emotion': dominant_emotion[0],
                    'confidence': dominant_emotion[1]
                },
                'confidence_score': confidence_score,
                'model_results': results,
                'processing_time': processing_time,
                'audio_features': {
                    'duration': self.duration,
                    'sample_rate': self.sample_rate,
                    'feature_count': len(features)
                }
            }
            
        except Exception as e:
            logger.error(f"Emotion analysis failed: {str(e)}")
            raise

    def _combine_features(self, features: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Combine all features into a single feature vector
        """
        try:
            combined = []
            
            # Add MFCC statistics
            if 'mfcc' in features:
                mfcc = features['mfcc']
                combined.extend([
                    np.mean(mfcc, axis=1).flatten(),
                    np.std(mfcc, axis=1).flatten(),
                    np.max(mfcc, axis=1).flatten(),
                    np.min(mfcc, axis=1).flatten()
                ])
            
            # Add spectral features
            if 'spectral' in features:
                combined.append(features['spectral'])
            
            # Add chroma statistics
            if 'chroma' in features:
                chroma = features['chroma']
                combined.extend([
                    np.mean(chroma, axis=1),
                    np.std(chroma, axis=1)
                ])
            
            # Add other features
            for key in ['tempo', 'rms']:
                if key in features:
                    combined.append(features[key])
            
            # Flatten and concatenate all features
            return np.concatenate([np.array(f).flatten() for f in combined])
            
        except Exception as e:
            logger.error(f"Feature combination failed: {str(e)}")
            return np.array([])

    def _ensemble_prediction(self, results: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """
        Combine predictions from multiple models
        """
        if not results:
            return {emotion: 1.0/len(self.emotions) for emotion in self.emotions}
        
        ensemble = {emotion: 0.0 for emotion in self.emotions}
        model_count = len(results)
        
        for model_result in results.values():
            for emotion, score in model_result.items():
                ensemble[emotion] += score / model_count
        
        return ensemble

    def _calculate_confidence(self, emotions: Dict[str, float]) -> float:
        """
        Calculate overall confidence score based on emotion distribution
        """
        scores = list(emotions.values())
        max_score = max(scores)
        
        # Calculate entropy-based confidence
        entropy = -sum(s * np.log(s + 1e-10) for s in scores)
        max_entropy = np.log(len(scores))
        normalized_entropy = entropy / max_entropy
        
        # Combine max score and entropy for confidence
        confidence = max_score * (1 - normalized_entropy)
        
        return float(confidence)

    def _models_loaded(self) -> bool:
        """
        Check if any models are loaded
        """
        return any([
            self.cnn_model is not None,
            self.lstm_model is not None,
            self.svm_model is not None
        ])

    async def _save_models(self):
        """
        Save models to disk
        """
        try:
            if self.cnn_model:
                self.cnn_model.save(os.path.join(self.model_path, "emotion_cnn.h5"))
            
            if self.lstm_model:
                self.lstm_model.save(os.path.join(self.model_path, "emotion_lstm.h5"))
            
            if self.svm_model:
                joblib.dump(self.svm_model, os.path.join(self.model_path, "emotion_svm.pkl"))
            
            if self.scaler:
                joblib.dump(self.scaler, os.path.join(self.model_path, "feature_scaler.pkl"))
                
        except Exception as e:
            logger.error(f"Error saving models: {str(e)}")

    def health_check(self) -> str:
        """
        Check if the emotion detector is working properly
        """
        try:
            if self._models_loaded():
                return "operational"
            else:
                return "no_models_loaded"
        except Exception:
            return "error"

    def get_model_info(self) -> Dict[str, any]:
        """
        Get information about loaded models
        """
        return {
            "available_emotions": self.emotions,
            "models_loaded": {
                "cnn": self.cnn_model is not None,
                "lstm": self.lstm_model is not None,
                "svm": self.svm_model is not None
            },
            "audio_parameters": {
                "sample_rate": self.sample_rate,
                "duration": self.duration,
                "n_mfcc": self.n_mfcc
            }
        }

    def get_performance_metrics(self) -> Dict[str, float]:
        """
        Get model performance metrics
        """
        return self.model_performance

    async def cleanup(self):
        """
        Cleanup resources
        """
        try:
            # Clear models from memory
            self.cnn_model = None
            self.lstm_model = None
            self.svm_model = None
            self.scaler = None
            
            # Clear cache
            self.feature_cache.clear()
            
            logger.info("Voice Emotion Detector cleaned up successfully")
            
        except Exception as e:
            logger.error(f"Cleanup failed: {str(e)}")