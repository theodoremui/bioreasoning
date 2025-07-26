# Audio File Management System

## Overview

The audio file management system provides robust, object-oriented handling of generated audio files with a focus on modularity, extensibility, and best practices. This system automatically saves generated podcast audio files to the upload directory with the same filename as the original document (with `.mp3` extension).

## Architecture

### Core Classes

#### `AudioFileManager`
The primary class responsible for managing audio file operations:

- **File Organization**: Automatically creates `data/uploads/audio/` directory structure
- **Filename Generation**: Converts document names to audio filenames (e.g., `document.pdf` → `document.mp3`)
- **Duplicate Handling**: Supports rename, overwrite, and skip strategies
- **File Operations**: Save, delete, list, and retrieve file information
- **Error Handling**: Comprehensive exception handling with specific error types

#### `AudioFileProcessor`
High-level processor that provides a simplified interface for common audio operations:

- **Workflow Integration**: Designed to work seamlessly with podcast generation workflows
- **Error Recovery**: Graceful error handling with detailed error messages
- **Metadata Management**: Automatically adds processing metadata to saved files
- **Extensibility**: Can be extended for different audio processing workflows

### Exception Hierarchy

```python
AudioFileError (base)
├── AudioFileNotFoundError
└── AudioFileSaveError
```

## Key Features

### 1. Automatic File Organization
- Creates organized directory structure: `data/uploads/audio/`
- Maintains separation between uploaded documents and generated audio
- Ensures directories exist before operations

### 2. Smart Filename Generation
```python
# Examples of filename conversion
"research_paper.pdf" → "research_paper.mp3"
"presentation.pptx" → "presentation.mp3"
"notes.txt" → "notes.mp3"
```

### 3. Duplicate Handling Strategies
- **Rename**: Automatically adds timestamp to avoid conflicts
- **Overwrite**: Replaces existing files
- **Skip**: Preserves existing files and reports conflict

### 4. Comprehensive File Information
Each saved audio file includes metadata:
- Original document name
- File size and timestamps
- Audio format information
- Processing metadata

## Usage Examples

### Basic Usage in Documents Page

```python
from bioagents.utils.audio_manager import AudioFileProcessor

class DocumentsPage:
    def __init__(self):
        self.audio_processor = AudioFileProcessor()
    
    def generate_podcast(self, content, original_document_name):
        # Generate temporary audio file
        temp_audio_file = sync_create_podcast(content)
        
        # Save with automatic naming
        result = self.audio_processor.process_podcast_audio(
            temp_audio_file,
            original_document_name,
            handle_duplicates="rename"
        )
        
        if not result.get("error"):
            return result["path"]  # Path to saved audio file
```

### Advanced Usage

```python
from bioagents.utils.audio_manager import AudioFileManager

# Custom audio manager
manager = AudioFileManager(base_directory="custom/path")

# Save audio file with custom options
result = manager.save_audio_file(
    temp_audio_path="path/to/temp.mp3",
    original_document_name="document.pdf",
    audio_format="wav",
    handle_duplicates="overwrite"
)

# List all audio files
audio_files = manager.list_audio_files()

# Get file information
file_info = manager.get_audio_file_info("document.mp3")

# Delete audio file
success = manager.delete_audio_file("document.mp3")
```

## OOP Design Principles

### 1. Single Responsibility Principle
- `AudioFileManager`: Handles file system operations
- `AudioFileProcessor`: Manages workflow integration
- Each class has a clear, focused purpose

### 2. Open/Closed Principle
- Base classes are open for extension
- New audio formats can be added without modifying existing code
- Custom processors can inherit from `AudioFileProcessor`

### 3. Dependency Inversion
- `AudioFileProcessor` depends on `AudioFileManager` abstraction
- Easy to inject mock managers for testing
- Loose coupling between components

### 4. Interface Segregation
- Clean, focused interfaces for each responsibility
- No unnecessary dependencies between classes

## Extensibility

### Adding New Audio Formats
```python
class CustomAudioManager(AudioFileManager):
    def _generate_audio_filename(self, original_filename: str, audio_format: str = "wav") -> str:
        # Custom logic for different audio formats
        base_name = Path(original_filename).stem
        return f"{base_name}.{audio_format}"
```

### Custom Processing Workflows
```python
class PodcastAudioProcessor(AudioFileProcessor):
    def process_podcast_audio(self, temp_audio_path, original_document_name, **kwargs):
        # Custom podcast-specific processing
        result = super().process_podcast_audio(temp_audio_path, original_document_name, **kwargs)
        
        # Add podcast-specific metadata
        if not result.get("error"):
            result["podcast_metadata"] = self._extract_podcast_metadata(temp_audio_path)
        
        return result
```

## Error Handling

### Robust Error Recovery
- Validates file existence before operations
- Handles file system errors gracefully
- Provides detailed error messages for debugging
- Maintains system stability even with file system issues

### Exception Types
- `AudioFileNotFoundError`: When source files don't exist
- `AudioFileSaveError`: When saving operations fail
- `AudioFileError`: Base exception for all audio operations

## Testing

Comprehensive test suite covering:
- File operations (save, delete, list)
- Error conditions and edge cases
- Duplicate handling strategies
- Integration with workflow systems
- Mock and real file system testing

## Integration with BioReasoning

### Automatic Integration
The audio management system is automatically integrated into the Documents page:

1. **Workflow Results**: Original document name is stored in workflow results
2. **Podcast Generation**: Audio files are automatically saved with proper naming
3. **User Feedback**: Clear information about saved files and locations
4. **Error Handling**: Graceful error messages for users

### File Organization
```
data/uploads/
├── document1.pdf
├── document2.docx
└── audio/
    ├── document1.mp3
    ├── document2.mp3
    └── document1_20250126_143022.mp3  # Renamed duplicate
```

## Benefits

1. **User Experience**: Audio files are automatically organized and easily accessible
2. **Developer Experience**: Clean, extensible API for audio operations
3. **Maintainability**: Well-structured code with clear separation of concerns
4. **Reliability**: Comprehensive error handling and validation
5. **Scalability**: Easy to extend for new audio formats and workflows 