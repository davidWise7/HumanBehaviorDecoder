# backend/config/settings.py
"""
Application Settings Configuration
Manages configuration settings for the Human Behavior Decoder
"""

import os
from typing import Optional, List, Dict, Any
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Settings(BaseSettings):
    """
    Application settings with environment variable support
    """
    
    # API Configuration
    API_TITLE: str = "Human Behavior Decoder API"
    API_VERSION: str = "1.0.0"
    API_DESCRIPTION: str = "AI-powered voice and text behavior analysis platform"
    
    # Server Configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False
    RELOAD: bool = False
    
    # Security
    SECRET_KEY: str = "your-secret-key-change-in-production"
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]
    
    # Database Configuration
    DATABASE_URL: str = "sqlite:///./behavior_decoder.db"
    
    # File Upload Settings
    MAX_AUDIO_SIZE_MB: int = 50
    MAX_TEXT_LENGTH: int = 100000
    UPLOAD_DIR: str = "./data/user_uploads/"
    
    # Audio Processing Settings
    AUDIO_SAMPLE_RATE: int = 16000
    AUDIO_MAX_DURATION: int = 300  # 5 minutes
    AUDIO_SUPPORTED_FORMATS: List[str] = [".wav", ".mp3", ".m4a", ".flac", ".ogg"]
    
    # Model Configuration
    MODEL_BASE_PATH: str = "./ai-models/"
    VOICE_MODEL_PATH: str = "./ai-models/voice_emotion/"
    TEXT_MODEL_PATH: str = "./ai-models/text_sentiment/"
    STRESS_MODEL_PATH: str = "./ai-models/stress_detection/"
    
    # Analysis Settings
    CONFIDENCE_THRESHOLD: float = 0.6
    ENABLE_VOICE_ANALYSIS: bool = True
    ENABLE_TEXT_ANALYSIS: bool = True
    ENABLE_STRESS_DETECTION: bool = True
    ENABLE_DECEPTION_ANALYSIS: bool = True
    
    # Performance Settings
    MAX_CONCURRENT_ANALYSES: int = 5
    ANALYSIS_TIMEOUT_SECONDS: int = 120
    CACHE_RESULTS: bool = True
    CACHE_TTL_MINUTES: int = 60
    
    # Logging Configuration
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_FILE: Optional[str] = None
    
    # External API Keys (if needed)
    OPENAI_API_KEY: Optional[str] = None
    HUGGINGFACE_API_KEY: Optional[str] = None
    GOOGLE_CLOUD_API_KEY: Optional[str] = None
    
    # Feature Flags
    ENABLE_REAL_TIME_ANALYSIS: bool = False
    ENABLE_BATCH_PROCESSING: bool = True
    ENABLE_ANALYTICS: bool = True
    ENABLE_USER_ACCOUNTS: bool = False
    
    # Business Settings
    RATE_LIMIT_PER_HOUR: int = 100
    FREE_TIER_ANALYSES_PER_DAY: int = 10
    PREMIUM_TIER_ANALYSES_PER_DAY: int = 1000
    
    # Data Retention
    CLEANUP_OLD_FILES_HOURS: int = 24
    KEEP_ANALYSIS_HISTORY_DAYS: int = 30
    
    class Config:
        env_file = ".env"
        case_sensitive = True

class DevelopmentSettings(Settings):
    """
    Development environment settings
    """
    DEBUG: bool = True
    RELOAD: bool = True
    LOG_LEVEL: str = "DEBUG"
    
    # Development-specific overrides
    RATE_LIMIT_PER_HOUR: int = 1000
    CLEANUP_OLD_FILES_HOURS: int = 1  # Clean up more frequently in dev

class ProductionSettings(Settings):
    """
    Production environment settings
    """
    DEBUG: bool = False
    RELOAD: bool = False
    LOG_LEVEL: str = "INFO"
    
    # Production security
    SECRET_KEY: str = os.getenv("SECRET_KEY", "CHANGE-THIS-IN-PRODUCTION")
    CORS_ORIGINS: List[str] = []  # Should be set via environment
    
    # Production performance
    MAX_CONCURRENT_ANALYSES: int = 10
    CACHE_TTL_MINUTES: int = 120

class TestSettings(Settings):
    """
    Test environment settings
    """
    DATABASE_URL: str = "sqlite:///./test_behavior_decoder.db"
    UPLOAD_DIR: str = "./test_data/uploads/"
    MODEL_BASE_PATH: str = "./test_models/"
    
    # Faster settings for testing
    ANALYSIS_TIMEOUT_SECONDS: int = 30
    CLEANUP_OLD_FILES_HOURS: int = 1

def get_settings() -> Settings:
    """
    Get settings based on environment
    """
    environment = os.getenv("ENVIRONMENT", "development").lower()
    
    if environment == "production":
        return ProductionSettings()
    elif environment == "test":
        return TestSettings()
    else:
        return DevelopmentSettings()

# Global settings instance
settings = get_settings()

# Validation functions
def validate_settings():
    """
    Validate critical settings
    """
    errors = []
    
    # Check required paths exist
    required_dirs = [
        settings.UPLOAD_DIR,
        settings.MODEL_BASE_PATH,
        settings.VOICE_MODEL_PATH,
        settings.TEXT_MODEL_PATH,
        settings.STRESS_MODEL_PATH
    ]
    
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            try:
                os.makedirs(dir_path, exist_ok=True)
            except Exception as e:
                errors.append(f"Cannot create directory {dir_path}: {e}")
    
    # Check file size limits
    if settings.MAX_AUDIO_SIZE_MB > 500:  # 500MB seems excessive
        errors.append("MAX_AUDIO_SIZE_MB is very large")
    
    if settings.MAX_TEXT_LENGTH > 1000000:  # 1M characters seems excessive
        errors.append("MAX_TEXT_LENGTH is very large")
    
    # Check timeout settings
    if settings.ANALYSIS_TIMEOUT_SECONDS < 10:
        errors.append("ANALYSIS_TIMEOUT_SECONDS too low")
    
    # Warn about default secret key in production
    if (os.getenv("ENVIRONMENT") == "production" and 
        settings.SECRET_KEY == "your-secret-key-change-in-production"):
        errors.append("Using default SECRET_KEY in production!")
    
    return errors

# Analysis configuration
class AnalysisConfig:
    """
    Configuration for different types of analysis
    """
    
    EMOTION_CATEGORIES = [
        'neutral', 'calm', 'happy', 'sad', 
        'angry', 'fearful', 'disgust', 'surprised'
    ]
    
    STRESS_LEVELS = ['minimal', 'low', 'medium', 'high']
    
    CONFIDENCE_LEVELS = ['very_low', 'low', 'neutral', 'medium', 'high']
    
    DECEPTION_RISK_LEVELS = ['minimal', 'low', 'medium', 'high']
    
    # Feature weights for ensemble models
    ENSEMBLE_WEIGHTS = {
        'emotion': 0.30,
        'sentiment': 0.20,
        'stress': 0.20,
        'confidence': 0.15,
        'deception': 0.15
    }
    
    # Quality thresholds
    QUALITY_THRESHOLDS = {
        'minimum_audio_duration': 1.0,  # seconds
        'minimum_text_length': 10,      # characters
        'minimum_confidence': 0.3       # analysis confidence
    }

def get_analysis_config() -> AnalysisConfig:
    """
    Get analysis configuration
    """
    return AnalysisConfig()

# Model configuration
class ModelConfig:
    """
    Configuration for ML models
    """
    
    # Voice emotion model settings
    VOICE_SAMPLE_RATE = 16000
    VOICE_DURATION = 3.0
    VOICE_N_MFCC = 40
    VOICE_N_FFT = 2048
    VOICE_HOP_LENGTH = 512
    
    # Text analysis settings
    TEXT_MAX_FEATURES = 10000
    TEXT_NGRAM_RANGE = (1, 2)
    TEXT_MIN_DF = 2
    TEXT_MAX_DF = 0.95
    
    # Model performance targets
    TARGET_ACCURACY = 0.75
    TARGET_INFERENCE_TIME_MS = 1000
    
    # Training settings (for future model updates)
    BATCH_SIZE = 32
    EPOCHS = 100
    LEARNING_RATE = 0.001
    VALIDATION_SPLIT = 0.2

def get_model_config() -> ModelConfig:
    """
    Get model configuration
    """
    return ModelConfig()

# Application metadata
APP_METADATA = {
    "name": "Human Behavior Decoder",
    "version": "1.0.0",
    "description": "AI-powered voice and text behavior analysis platform",
    "author": "Your Development Team",
    "license": "MIT",
    "repository": "https://github.com/your-org/behavior-decoder",
    "documentation": "https://docs.behavior-decoder.com",
    "support_email": "support@behavior-decoder.com"
}

# Export commonly used settings
__all__ = [
    'Settings',
    'DevelopmentSettings', 
    'ProductionSettings',
    'TestSettings',
    'get_settings',
    'validate_settings',
    'AnalysisConfig',
    'get_analysis_config',
    'ModelConfig', 
    'get_model_config',
    'APP_METADATA',
    'settings'
]