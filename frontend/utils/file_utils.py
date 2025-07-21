"""
File utilities for the BioReasoning application.

This module provides common file operations and utilities following the DRY principle.

Author: Theodore Mui
Date: 2025-04-26
"""

import os
import hashlib
import datetime
from pathlib import Path
from typing import Dict, Any, Optional


class FileUtils:
    """
    Utility class for common file operations.
    
    This class provides static methods for file handling operations that can be
    reused across different parts of the application.
    """
    
    @staticmethod
    def ensure_directory_exists(directory_path: str) -> Path:
        """
        Ensure a directory exists, creating it if necessary.
        
        Args:
            directory_path: Path to the directory
            
        Returns:
            Path: Path object for the directory
        """
        path = Path(directory_path)
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    @staticmethod
    def generate_safe_filename(original_filename: str, include_timestamp: bool = True) -> str:
        """
        Generate a safe filename to avoid conflicts.
        
        Args:
            original_filename: The original filename
            include_timestamp: Whether to include timestamp in the filename
            
        Returns:
            str: Safe filename
        """
        # Get file extension
        file_path = Path(original_filename)
        name = file_path.stem
        extension = file_path.suffix
        
        # Generate timestamp if requested
        timestamp = ""
        if include_timestamp:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + "_"
        
        # Generate hash for uniqueness
        file_hash = hashlib.md5(original_filename.encode()).hexdigest()[:8]
        
        return f"{timestamp}{file_hash}_{name}{extension}"
    
    @staticmethod
    def format_file_size(size_bytes: int) -> str:
        """
        Format file size in human readable format.
        
        Args:
            size_bytes: Size in bytes
            
        Returns:
            str: Formatted size string
        """
        if size_bytes == 0:
            return "0 B"
        
        size_names = ["B", "KB", "MB", "GB", "TB"]
        i = 0
        while size_bytes >= 1024 and i < len(size_names) - 1:
            size_bytes /= 1024.0
            i += 1
        
        return f"{size_bytes:.1f} {size_names[i]}"
    
    @staticmethod
    def get_file_extension(filename: str) -> str:
        """
        Get the file extension from a filename.
        
        Args:
            filename: The filename
            
        Returns:
            str: File extension (without the dot)
        """
        return Path(filename).suffix.lstrip('.')
    
    @staticmethod
    def is_supported_file_type(filename: str, supported_types: list) -> bool:
        """
        Check if a file type is supported.
        
        Args:
            filename: The filename to check
            supported_types: List of supported file extensions
            
        Returns:
            bool: True if file type is supported
        """
        extension = FileUtils.get_file_extension(filename).lower()
        return extension in [ext.lower() for ext in supported_types]
    
    @staticmethod
    def calculate_file_hash(file_content: bytes) -> str:
        """
        Calculate MD5 hash of file content.
        
        Args:
            file_content: The file content as bytes
            
        Returns:
            str: MD5 hash string
        """
        return hashlib.md5(file_content).hexdigest()
    
    @staticmethod
    def format_timestamp(timestamp_str: str, format_str: str = "%Y-%m-%d %H:%M") -> str:
        """
        Format a timestamp string for display.
        
        Args:
            timestamp_str: ISO format timestamp string
            format_str: Target format string
            
        Returns:
            str: Formatted timestamp or "Unknown" if parsing fails
        """
        try:
            dt = datetime.datetime.fromisoformat(timestamp_str)
            return dt.strftime(format_str)
        except:
            return "Unknown"
    
    @staticmethod
    def get_file_info(file_path: str) -> Optional[Dict[str, Any]]:
        """
        Get comprehensive information about a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dict with file information or None if file doesn't exist
        """
        path = Path(file_path)
        if not path.exists():
            return None
        
        stat = path.stat()
        return {
            "name": path.name,
            "path": str(path),
            "size": stat.st_size,
            "size_formatted": FileUtils.format_file_size(stat.st_size),
            "extension": FileUtils.get_file_extension(path.name),
            "created_time": datetime.datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "modified_time": datetime.datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "is_file": path.is_file(),
            "is_directory": path.is_dir()
        } 