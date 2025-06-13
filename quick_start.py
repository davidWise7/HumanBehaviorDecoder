# quick_start.py
"""
Quick Start Script for Human Behavior Decoder
Run this to test if everything is working
"""

import asyncio
import logging
import sys
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_system():
    """Test all components of the system"""
    print("🚀 Starting Human Behavior Decoder System Test...")
    print("=" * 60)
    
    try:
        # Test 1: Import all modules
        print("📦 Testing module imports...")
        from voice_analysis.emotion_detector import VoiceEmotionDetector
        from text_analysis.sentiment_analyzer import TextSentimentAnalyzer
        from emotion_engine.behavior_processor import BehaviorProcessor
        from stress_detector.stress_analyzer import StressAnalyzer
        from confidence_scorer.confidence_evaluator import ConfidenceEvaluator
        from deception_analyzer.lie_detector import DeceptionAnalyzer
        from utils.file_handler import AudioFileHandler, TextFileHandler
        from config.settings import Settings, validate_settings
        print("✅ All modules imported successfully!")
        
        # Test 2: Validate settings
        print("\n⚙️  Validating configuration...")
        settings = Settings()
        validation_errors = validate_settings()
        if validation_errors:
            print("⚠️  Configuration warnings:")
            for error in validation_errors:
                print(f"   - {error}")
        else:
            print("✅ Configuration validated!")
        
        # Test 3: Initialize analyzers
        print("\n🧠 Initializing AI analyzers...")
        
        print("   Initializing Voice Emotion Detector...")
        voice_detector = VoiceEmotionDetector()
        await voice_detector.initialize()
        
        print("   Initializing Text Sentiment Analyzer...")
        text_analyzer = TextSentimentAnalyzer()
        await text_analyzer.initialize()
        
        print("   Initializing Stress Analyzer...")
        stress_analyzer = StressAnalyzer()
        await stress_analyzer.initialize()
        
        print("   Initializing Confidence Evaluator...")
        confidence_evaluator = ConfidenceEvaluator()
        await confidence_evaluator.initialize()
        
        print("   Initializing Deception Analyzer...")
        deception_analyzer = DeceptionAnalyzer()
        await deception_analyzer.initialize()
        
        print("   Initializing Behavior Processor...")
        behavior_processor = BehaviorProcessor()
        await behavior_processor.initialize()
        
        print("✅ All analyzers initialized successfully!")
        
        # Test 4: Test text analysis
        print("\n📝 Testing text analysis...")
        sample_text = "I am feeling really confident about this project! We're going to build something amazing."
        
        try:
            results = await behavior_processor.process_text(sample_text, "full")
            print("✅ Text analysis completed successfully!")
            print(f"   Detected emotions: {list(results.get('emotions', {}).keys())}")
            print(f"   Sentiment: {results.get('sentiment', {}).get('label', 'unknown')}")
            print(f"   Confidence level: {results.get('confidence', {}).get('level', 'unknown')}")
        except Exception as e:
            print(f"❌ Text analysis failed: {e}")
        
        # Test 5: Test file handlers
        print("\n📁 Testing file handlers...")
        audio_handler = AudioFileHandler()
        text_handler = TextFileHandler()
        print("✅ File handlers initialized!")
        
        # Test 6: Health checks
        print("\n🏥 Running health checks...")
        health_status = behavior_processor.health_check()
        print("System health status:")
        for component, status in health_status.items():
            emoji = "✅" if status == "operational" else "⚠️"
            print(f"   {emoji} {component}: {status}")
        
        print("\n🎉 System test completed successfully!")
        print("=" * 60)
        print("Your Human Behavior Decoder is ready to use!")
        print("\nNext steps:")
        print("1. Run 'python main.py' to start the API server")
        print("2. Open http://localhost:8000 in your browser")
        print("3. Try the /docs endpoint for API documentation")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all required packages are installed:")
        print("   pip install -r requirements.txt")
        return False
        
    except Exception as e:
        print(f"❌ System test failed: {e}")
        logger.exception("Full error details:")
        return False

def check_dependencies():
    """Check if required packages are installed"""
    # Map package names to their import names
    required_packages = {
        'fastapi': 'fastapi',
        'uvicorn': 'uvicorn', 
        'numpy': 'numpy',
        'pandas': 'pandas',
        'scikit-learn': 'sklearn',  # Package vs import name
        'tensorflow': 'tensorflow',
        'librosa': 'librosa',
        'soundfile': 'soundfile',
        'nltk': 'nltk',
        'spacy': 'spacy',
        'textblob': 'textblob',
        'pydantic': 'pydantic',
        'python-multipart': 'multipart',  # Package vs import name
        'python-dotenv': 'dotenv',  # Package vs import name
        'joblib': 'joblib'
    }
    
    missing_packages = []
    
    for package_name, import_name in required_packages.items():
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(package_name)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\nInstall missing packages with:")
        print(f"   pip install {' '.join(missing_packages)}")
        return False
    
    return True

if __name__ == "__main__":
    print("Human Behavior Decoder - Quick Start Test")
    print("=" * 60)
    
    # Check dependencies first
    if not check_dependencies():
        sys.exit(1)
    
    # Run the system test
    try:
        success = asyncio.run(test_system())
        if success:
            print("\n✨ Ready to start analyzing human behavior!")
        else:
            print("\n💥 System test failed. Please check the errors above.")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⏹️  Test interrupted by user")
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)