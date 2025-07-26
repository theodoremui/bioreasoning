"""
Audio file management utilities for the bioagents package.

This module provides robust audio file handling with OOP best practices,
including modularity, extensibility, and error handling.
"""

import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Union
import logging

logger = logging.getLogger(__name__)


class AudioFileError(Exception):
    """Base exception for audio file operations."""
    pass


class AudioFileNotFoundError(AudioFileError):
    """Raised when an audio file is not found."""
    pass


class AudioFileSaveError(AudioFileError):
    """Raised when saving an audio file fails."""
    pass


class AudioFileManager:
    """
    Manages audio file operations with robust error handling and extensibility.
    
    This class provides a clean interface for saving, organizing, and managing
    audio files generated from document processing.
    """
    
    def __init__(self, base_directory: Union[str, Path] = "data/uploads"):
        """
        Initialize the audio file manager.
        
        Args:
            base_directory: Base directory for storing audio files
        """
        self.base_directory = Path(base_directory)
        self.audio_directory = self.base_directory / "audio"
        self._ensure_directories()
    
    def _ensure_directories(self) -> None:
        """Ensure all required directories exist."""
        try:
            self.base_directory.mkdir(parents=True, exist_ok=True)
            self.audio_directory.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Audio directories ensured: {self.audio_directory}")
        except Exception as e:
            logger.error(f"Failed to create audio directories: {e}")
            raise AudioFileError(f"Failed to create audio directories: {e}") from e
    
    def _generate_audio_filename(self, original_filename: str, audio_format: str = "mp3") -> str:
        """
        Generate an audio filename based on the original document filename.
        
        Args:
            original_filename: The original document filename
            audio_format: The audio format (default: mp3)
            
        Returns:
            str: Generated audio filename
        """
        # Remove any existing extension
        base_name = Path(original_filename).stem
        
        # Add audio extension
        audio_filename = f"{base_name}.{audio_format}"
        
        return audio_filename
    
    def _generate_unique_audio_filename(self, original_filename: str, audio_format: str = "mp3") -> str:
        """
        Generate a unique audio filename to avoid conflicts.
        
        Args:
            original_filename: The original document filename
            audio_format: The audio format (default: mp3)
            
        Returns:
            str: Unique audio filename
        """
        base_filename = self._generate_audio_filename(original_filename, audio_format)
        audio_path = self.audio_directory / base_filename
        
        if not audio_path.exists():
            return base_filename
        
        # Generate unique name with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = Path(base_filename).stem
        unique_filename = f"{base_name}_{timestamp}.{audio_format}"
        
        return unique_filename
    
    def save_audio_file(
        self, 
        temp_audio_path: Union[str, Path], 
        original_document_name: str,
        audio_format: str = "mp3",
        handle_duplicates: str = "rename"
    ) -> Dict[str, Any]:
        """
        Save a temporary audio file to the audio directory.
        
        Args:
            temp_audio_path: Path to the temporary audio file
            original_document_name: Name of the original document
            audio_format: Audio format (default: mp3)
            handle_duplicates: How to handle duplicates ("rename", "overwrite", "skip")
            
        Returns:
            Dict containing audio file information
            
        Raises:
            AudioFileNotFoundError: If temp file doesn't exist
            AudioFileSaveError: If saving fails
        """
        temp_path = Path(temp_audio_path)
        
        # Validate temp file exists
        if not temp_path.exists():
            raise AudioFileNotFoundError(f"Temporary audio file not found: {temp_path}")
        
        # Generate target filename
        if handle_duplicates == "rename":
            target_filename = self._generate_unique_audio_filename(original_document_name, audio_format)
        elif handle_duplicates == "overwrite":
            target_filename = self._generate_audio_filename(original_document_name, audio_format)
        elif handle_duplicates == "skip":
            base_filename = self._generate_audio_filename(original_document_name, audio_format)
            if (self.audio_directory / base_filename).exists():
                return {
                    "error": True,
                    "message": f"Audio file '{base_filename}' already exists. Skipped.",
                    "filename": base_filename
                }
            target_filename = base_filename
        else:
            target_filename = self._generate_unique_audio_filename(original_document_name, audio_format)
        
        target_path = self.audio_directory / target_filename
        
        try:
            # Copy the file to the target location
            shutil.copy2(temp_path, target_path)
            
            # Get file stats
            stat = target_path.stat()
            
            # Create file info
            file_info = {
                "filename": target_filename,
                "path": str(target_path),
                "size_bytes": stat.st_size,
                "created_time": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                "modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "original_document": original_document_name,
                "audio_format": audio_format,
                "was_renamed": target_filename != self._generate_audio_filename(original_document_name, audio_format)
            }
            
            logger.info(f"Audio file saved successfully: {target_path}")
            return file_info
            
        except Exception as e:
            logger.error(f"Failed to save audio file: {e}")
            raise AudioFileSaveError(f"Failed to save audio file: {e}") from e
    
    def get_audio_file_path(self, filename: str) -> Optional[Path]:
        """
        Get the full path to an audio file.
        
        Args:
            filename: The audio filename
            
        Returns:
            Path to the audio file or None if not found
        """
        audio_path = self.audio_directory / filename
        return audio_path if audio_path.exists() else None
    
    def list_audio_files(self) -> list:
        """
        List all audio files in the audio directory.
        
        Returns:
            List of audio file information dictionaries
        """
        audio_files = []
        
        try:
            for audio_path in self.audio_directory.glob("*.mp3"):
                stat = audio_path.stat()
                audio_files.append({
                    "filename": audio_path.name,
                    "path": str(audio_path),
                    "size_bytes": stat.st_size,
                    "created_time": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                    "modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat()
                })
        except Exception as e:
            logger.error(f"Failed to list audio files: {e}")
        
        return audio_files
    
    def delete_audio_file(self, filename: str) -> bool:
        """
        Delete an audio file.
        
        Args:
            filename: The audio filename to delete
            
        Returns:
            bool: True if deletion was successful
        """
        try:
            audio_path = self.audio_directory / filename
            if audio_path.exists():
                audio_path.unlink()
                logger.info(f"Audio file deleted: {audio_path}")
                return True
            else:
                logger.warning(f"Audio file not found for deletion: {audio_path}")
                return False
        except Exception as e:
            logger.error(f"Failed to delete audio file {filename}: {e}")
            return False
    
    def get_audio_file_info(self, filename: str) -> Optional[Dict[str, Any]]:
        """
        Get information about an audio file.
        
        Args:
            filename: The audio filename
            
        Returns:
            Dict with file information or None if not found
        """
        try:
            audio_path = self.audio_directory / filename
            if audio_path.exists():
                stat = audio_path.stat()
                return {
                    "filename": filename,
                    "path": str(audio_path),
                    "size_bytes": stat.st_size,
                    "created_time": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                    "modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat()
                }
        except Exception as e:
            logger.error(f"Failed to get audio file info for {filename}: {e}")
        
        return None
    
    def cleanup_temp_files(self, temp_files: list) -> None:
        """
        Clean up temporary audio files.
        
        Args:
            temp_files: List of temporary file paths to clean up
        """
        for temp_file in temp_files:
            try:
                temp_path = Path(temp_file)
                if temp_path.exists():
                    temp_path.unlink()
                    logger.debug(f"Cleaned up temporary file: {temp_path}")
            except Exception as e:
                logger.warning(f"Failed to clean up temporary file {temp_file}: {e}")


class AudioFileProcessor:
    """
    High-level processor for audio file operations.
    
    This class provides a simplified interface for common audio file operations
    and can be extended for different audio processing workflows.
    """
    
    def __init__(self, audio_manager: Optional[AudioFileManager] = None):
        """
        Initialize the audio file processor.
        
        Args:
            audio_manager: AudioFileManager instance (creates default if None)
        """
        self.audio_manager = audio_manager or AudioFileManager()
    
    def process_podcast_audio(
        self, 
        temp_audio_path: Union[str, Path], 
        original_document_name: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process and save a podcast audio file.
        
        Args:
            temp_audio_path: Path to temporary audio file
            original_document_name: Name of the original document
            **kwargs: Additional arguments for save_audio_file
            
        Returns:
            Dict containing processing results
        """
        try:
            result = self.audio_manager.save_audio_file(
                temp_audio_path, 
                original_document_name,
                **kwargs
            )
            
            if result.get("error"):
                return result
            
            # Add processing metadata
            result["processed_time"] = datetime.now().isoformat()
            result["processor"] = self.__class__.__name__
            
            return result
            
        except AudioFileError as e:
            logger.error(f"Audio processing failed: {e}")
            return {
                "error": True,
                "message": str(e),
                "original_document": original_document_name
            }
        except Exception as e:
            logger.error(f"Unexpected error in audio processing: {e}")
            return {
                "error": True,
                "message": f"Unexpected error: {str(e)}",
                "original_document": original_document_name
            } 