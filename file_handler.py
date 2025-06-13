# backend/utils/file_handler.py
"""
File Handling Utilities
Handles audio and text file uploads, validation, and processing
"""

import os
import uuid
import logging
import mimetypes
import tempfile
from typing import Optional, List, Dict, Any
from fastapi import UploadFile, HTTPException
import librosa
import soundfile as sf
from pathlib import Path

logger = logging.getLogger(__name__)

class AudioFileHandler:
    """
    Handles audio file uploads and processing
    """
    
    def __init__(self, upload_dir: str = "./data/user_uploads/"):
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        
        # Supported audio formats
        self.supported_formats = {
            '.wav', '.mp3', '.m4a', '.flac', '.ogg', '.aac'
        }
        
        # MIME types for audio files
        self.supported_mimes = {
            'audio/wav', 'audio/wave', 'audio/x-wav',
            'audio/mpeg', 'audio/mp3',
            'audio/mp4', 'audio/m4a',
            'audio/flac', 'audio/x-flac',
            'audio/ogg', 'audio/vorbis',
            'audio/aac', 'audio/x-aac'
        }
        
        # File size limits (in bytes)
        self.max_file_size = 50 * 1024 * 1024  # 50MB
        self.max_duration = 300  # 5 minutes
        
    def validate_audio_file(self, file: UploadFile) -> bool:
        """
        Validate uploaded audio file
        """
        try:
            # Check file extension
            if file.filename:
                file_ext = Path(file.filename).suffix.lower()
                if file_ext not in self.supported_formats:
                    logger.warning(f"Unsupported file extension: {file_ext}")
                    return False
            
            # Check MIME type
            if file.content_type and file.content_type not in self.supported_mimes:
                logger.warning(f"Unsupported MIME type: {file.content_type}")
                return False
            
            # Check file size (basic check - more detailed check after saving)
            if hasattr(file.file, 'seek') and hasattr(file.file, 'tell'):
                current_pos = file.file.tell()
                file.file.seek(0, 2)  # Seek to end
                file_size = file.file.tell()
                file.file.seek(current_pos)  # Reset position
                
                if file_size > self.max_file_size:
                    logger.warning(f"File too large: {file_size} bytes")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"File validation failed: {str(e)}")
            return False

    async def save_upload(self, file: UploadFile) -> str:
        """
        Save uploaded audio file and return path
        """
        try:
            # Generate unique filename
            file_id = str(uuid.uuid4())
            original_ext = Path(file.filename).suffix.lower() if file.filename else '.wav'
            filename = f"{file_id}{original_ext}"
            file_path = self.upload_dir / filename
            
            # Save file
            contents = await file.read()
            with open(file_path, 'wb') as f:
                f.write(contents)
            
            # Validate audio file properties
            await self._validate_audio_properties(str(file_path))
            
            # Convert to standard format if needed
            standardized_path = await self._standardize_audio(str(file_path))
            
            logger.info(f"Audio file saved: {standardized_path}")
            return standardized_path
            
        except Exception as e:
            logger.error(f"Failed to save audio file: {str(e)}")
            # Cleanup on failure
            if 'file_path' in locals() and file_path.exists():
                file_path.unlink()
            raise

    async def _validate_audio_properties(self, file_path: str):
        """
        Validate audio file properties using librosa
        """
        try:
            # Load audio to check properties
            y, sr = librosa.load(file_path, sr=None)
            duration = len(y) / sr
            
            # Check duration
            if duration > self.max_duration:
                raise ValueError(f"Audio too long: {duration:.1f}s (max {self.max_duration}s)")
            
            # Check if audio data is valid
            if len(y) == 0:
                raise ValueError("Audio file contains no data")
            
            # Check sample rate
            if sr < 8000:
                logger.warning(f"Low sample rate detected: {sr}Hz")
            
            logger.info(f"Audio validated: {duration:.1f}s, {sr}Hz, {len(y)} samples")
            
        except Exception as e:
            logger.error(f"Audio validation failed: {str(e)}")
            raise

    async def _standardize_audio(self, file_path: str) -> str:
        """
        Convert audio to standard format (16kHz WAV) for processing
        """
        try:
            # Load and resample to 16kHz
            y, sr = librosa.load(file_path, sr=16000)
            
            # Create standardized filename
            path_obj = Path(file_path)
            standardized_path = path_obj.parent / f"{path_obj.stem}_std.wav"
            
            # Save as WAV
            sf.write(str(standardized_path), y, 16000)
            
            # Remove original if different
            if str(standardized_path) != file_path:
                os.remove(file_path)
            
            return str(standardized_path)
            
        except Exception as e:
            logger.error(f"Audio standardization failed: {str(e)}")
            return file_path  # Return original if standardization fails

    def cleanup_file(self, file_path: str):
        """
        Clean up temporary file
        """
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Cleaned up file: {file_path}")
        except Exception as e:
            logger.warning(f"Failed to cleanup file {file_path}: {str(e)}")

    def get_audio_info(self, file_path: str) -> Dict[str, Any]:
        """
        Get audio file information
        """
        try:
            y, sr = librosa.load(file_path, sr=None)
            duration = len(y) / sr
            file_size = os.path.getsize(file_path)
            
            return {
                "duration": duration,
                "sample_rate": sr,
                "channels": 1,  # librosa loads as mono by default
                "file_size": file_size,
                "format": Path(file_path).suffix.lower()
            }
            
        except Exception as e:
            logger.error(f"Failed to get audio info: {str(e)}")
            return {}

class TextFileHandler:
    """
    Handles text file uploads and processing
    """
    
    def __init__(self, upload_dir: str = "./data/user_uploads/"):
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        
        # Supported text formats
        self.supported_formats = {
            '.txt', '.md', '.rtf'
        }
        
        # MIME types for text files
        self.supported_mimes = {
            'text/plain', 'text/markdown', 'text/rtf',
            'application/rtf'
        }
        
        # File size limits
        self.max_file_size = 10 * 1024 * 1024  # 10MB
        self.max_text_length = 100000  # 100k characters

    def validate_text_file(self, file: UploadFile) -> bool:
        """
        Validate uploaded text file
        """
        try:
            # Check file extension
            if file.filename:
                file_ext = Path(file.filename).suffix.lower()
                if file_ext not in self.supported_formats:
                    return False
            
            # Check MIME type
            if file.content_type and file.content_type not in self.supported_mimes:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Text file validation failed: {str(e)}")
            return False

    async def save_text_upload(self, file: UploadFile) -> str:
        """
        Save uploaded text file and return content
        """
        try:
            contents = await file.read()
            
            # Try different encodings
            text_content = None
            for encoding in ['utf-8', 'utf-16', 'latin-1', 'cp1252']:
                try:
                    text_content = contents.decode(encoding)
                    break
                except UnicodeDecodeError:
                    continue
            
            if text_content is None:
                raise ValueError("Could not decode text file")
            
            # Validate text length
            if len(text_content) > self.max_text_length:
                raise ValueError(f"Text too long: {len(text_content)} characters")
            
            # Basic text cleaning
            cleaned_text = self._clean_text(text_content)
            
            return cleaned_text
            
        except Exception as e:
            logger.error(f"Failed to process text file: {str(e)}")
            raise

    def _clean_text(self, text: str) -> str:
        """
        Basic text cleaning
        """
        try:
            # Remove null characters
            text = text.replace('\x00', '')
            
            # Normalize line endings
            text = text.replace('\r\n', '\n').replace('\r', '\n')
            
            # Remove excessive whitespace
            lines = text.split('\n')
            cleaned_lines = []
            for line in lines:
                cleaned_line = ' '.join(line.split())
                if cleaned_line:  # Skip empty lines
                    cleaned_lines.append(cleaned_line)
            
            return '\n'.join(cleaned_lines)
            
        except Exception as e:
            logger.warning(f"Text cleaning failed: {str(e)}")
            return text

    def extract_text_from_file(self, file_path: str) -> str:
        """
        Extract text from various file formats
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            return self._clean_text(content)
            
        except Exception as e:
            logger.error(f"Failed to extract text from {file_path}: {str(e)}")
            raise

class FileManager:
    """
    General file management utilities
    """
    
    def __init__(self, base_dir: str = "./data/"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize subdirectories
        self.audio_handler = AudioFileHandler(str(self.base_dir / "audio_uploads"))
        self.text_handler = TextFileHandler(str(self.base_dir / "text_uploads"))

    def cleanup_old_files(self, max_age_hours: int = 24):
        """
        Clean up old uploaded files
        """
        try:
            import time
            current_time = time.time()
            cutoff_time = current_time - (max_age_hours * 3600)
            
            cleanup_count = 0
            
            # Clean audio files
            for file_path in self.audio_handler.upload_dir.glob("*"):
                if file_path.is_file() and file_path.stat().st_mtime < cutoff_time:
                    file_path.unlink()
                    cleanup_count += 1
            
            # Clean text files
            for file_path in self.text_handler.upload_dir.glob("*"):
                if file_path.is_file() and file_path.stat().st_mtime < cutoff_time:
                    file_path.unlink()
                    cleanup_count += 1
            
            logger.info(f"Cleaned up {cleanup_count} old files")
            return cleanup_count
            
        except Exception as e:
            logger.error(f"File cleanup failed: {str(e)}")
            return 0

    def get_storage_stats(self) -> Dict[str, Any]:
        """
        Get storage usage statistics
        """
        try:
            stats = {
                "audio_files": 0,
                "text_files": 0,
                "total_size_mb": 0.0
            }
            
            total_size = 0
            
            # Count audio files
            for file_path in self.audio_handler.upload_dir.glob("*"):
                if file_path.is_file():
                    stats["audio_files"] += 1
                    total_size += file_path.stat().st_size
            
            # Count text files  
            for file_path in self.text_handler.upload_dir.glob("*"):
                if file_path.is_file():
                    stats["text_files"] += 1
                    total_size += file_path.stat().st_size
            
            stats["total_size_mb"] = total_size / (1024 * 1024)
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get storage stats: {str(e)}")
            return {"error": str(e)}