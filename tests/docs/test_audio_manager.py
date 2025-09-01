"""
Tests for audio file management utilities.
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add the bioagents directory to the path for imports
import sys

sys.path.append("../bioagents")

from bioagents.utils.audio_manager import (
    AudioFileManager,
    AudioFileProcessor,
    AudioFileError,
    AudioFileNotFoundError,
    AudioFileSaveError,
)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def audio_manager(temp_dir):
    """Create an AudioFileManager instance for testing."""
    return AudioFileManager(base_directory=temp_dir)


@pytest.fixture
def audio_processor(audio_manager):
    """Create an AudioFileProcessor instance for testing."""
    return AudioFileProcessor(audio_manager=audio_manager)


@pytest.fixture
def sample_temp_audio_file():
    """Create a sample temporary audio file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        f.write(b"fake audio content")
        temp_path = f.name

    yield temp_path

    # Cleanup
    try:
        os.unlink(temp_path)
    except OSError:
        pass


def test_audio_manager_initialization(temp_dir):
    """Test AudioFileManager initialization."""
    manager = AudioFileManager(base_directory=temp_dir)

    assert manager.base_directory == Path(temp_dir)
    assert manager.audio_directory == Path(temp_dir) / "audio"
    assert manager.audio_directory.exists()


def test_generate_audio_filename(audio_manager):
    """Test audio filename generation."""
    # Test with different file extensions
    test_cases = [
        ("document.pdf", "document.mp3"),
        ("research_paper.docx", "research_paper.mp3"),
        ("notes.txt", "notes.mp3"),
        ("presentation.pptx", "presentation.mp3"),
        ("file_without_extension", "file_without_extension.mp3"),
    ]

    for original_name, expected_audio_name in test_cases:
        result = audio_manager._generate_audio_filename(original_name)
        assert result == expected_audio_name


def test_generate_unique_audio_filename(audio_manager):
    """Test unique audio filename generation."""
    original_name = "test_document.pdf"

    # First call should return the base name
    first_filename = audio_manager._generate_unique_audio_filename(original_name)
    assert first_filename == "test_document.mp3"

    # Create a file with that name
    audio_file_path = audio_manager.audio_directory / first_filename
    audio_file_path.touch()

    # Second call should return a unique name with timestamp
    second_filename = audio_manager._generate_unique_audio_filename(original_name)
    assert second_filename != first_filename
    assert second_filename.startswith("test_document_")
    assert second_filename.endswith(".mp3")


def test_save_audio_file_success(audio_manager, sample_temp_audio_file):
    """Test successful audio file saving."""
    original_document_name = "test_document.pdf"

    result = audio_manager.save_audio_file(
        sample_temp_audio_file, original_document_name
    )

    assert not result.get("error")
    assert result["filename"] == "test_document.mp3"
    assert result["original_document"] == original_document_name
    assert result["audio_format"] == "mp3"
    assert result["size_bytes"] > 0
    assert "created_time" in result
    assert "modified_time" in result

    # Verify file was actually saved
    saved_path = Path(result["path"])
    assert saved_path.exists()
    assert saved_path.stat().st_size > 0


def test_save_audio_file_not_found(audio_manager):
    """Test saving non-existent audio file."""
    with pytest.raises(AudioFileNotFoundError):
        audio_manager.save_audio_file("non_existent_file.mp3", "test_document.pdf")


def test_save_audio_file_duplicate_handling(audio_manager, sample_temp_audio_file):
    """Test duplicate handling options."""
    original_document_name = "test_document.pdf"

    # Save first file
    first_result = audio_manager.save_audio_file(
        sample_temp_audio_file, original_document_name, handle_duplicates="rename"
    )

    # Create another temp file for the duplicate
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        f.write(b"different audio content")
        second_temp_file = f.name

    try:
        # Test rename handling
        second_result = audio_manager.save_audio_file(
            second_temp_file, original_document_name, handle_duplicates="rename"
        )

        assert not second_result.get("error")
        assert second_result["filename"] != first_result["filename"]
        assert second_result["was_renamed"] == True

        # Test skip handling
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(b"third audio content")
            third_temp_file = f.name

        try:
            third_result = audio_manager.save_audio_file(
                third_temp_file, original_document_name, handle_duplicates="skip"
            )

            assert third_result.get("error") == True
            assert "already exists" in third_result["message"]
        finally:
            try:
                os.unlink(third_temp_file)
            except OSError:
                pass
    finally:
        try:
            os.unlink(second_temp_file)
        except OSError:
            pass


def test_audio_processor_process_podcast_audio(audio_processor, sample_temp_audio_file):
    """Test AudioFileProcessor podcast audio processing."""
    original_document_name = "research_paper.pdf"

    result = audio_processor.process_podcast_audio(
        sample_temp_audio_file, original_document_name
    )

    assert not result.get("error")
    assert result["filename"] == "research_paper.mp3"
    assert result["original_document"] == original_document_name
    assert "processed_time" in result
    assert result["processor"] == "AudioFileProcessor"


def test_audio_processor_error_handling(audio_processor):
    """Test AudioFileProcessor error handling."""
    result = audio_processor.process_podcast_audio(
        "non_existent_file.mp3", "test_document.pdf"
    )

    assert result.get("error") == True
    assert "not found" in result["message"]


def test_list_audio_files(audio_manager, sample_temp_audio_file):
    """Test listing audio files."""
    # Save a few audio files
    documents = ["doc1.pdf", "doc2.pdf", "doc3.pdf"]

    for doc_name in documents:
        audio_manager.save_audio_file(sample_temp_audio_file, doc_name)

    # List audio files
    audio_files = audio_manager.list_audio_files()

    assert len(audio_files) == 3
    for audio_file in audio_files:
        assert "filename" in audio_file
        assert "path" in audio_file
        assert "size_bytes" in audio_file
        assert "created_time" in audio_file
        assert "modified_time" in audio_file


def test_delete_audio_file(audio_manager, sample_temp_audio_file):
    """Test deleting audio files."""
    original_document_name = "test_document.pdf"

    # Save an audio file
    result = audio_manager.save_audio_file(
        sample_temp_audio_file, original_document_name
    )

    filename = result["filename"]
    file_path = Path(result["path"])

    # Verify file exists
    assert file_path.exists()

    # Delete the file
    success = audio_manager.delete_audio_file(filename)
    assert success == True

    # Verify file is gone
    assert not file_path.exists()

    # Try to delete non-existent file
    success = audio_manager.delete_audio_file("non_existent.mp3")
    assert success == False


def test_get_audio_file_info(audio_manager, sample_temp_audio_file):
    """Test getting audio file information."""
    original_document_name = "test_document.pdf"

    # Save an audio file
    result = audio_manager.save_audio_file(
        sample_temp_audio_file, original_document_name
    )

    filename = result["filename"]

    # Get file info
    file_info = audio_manager.get_audio_file_info(filename)

    assert file_info is not None
    assert file_info["filename"] == filename
    assert file_info["size_bytes"] > 0
    assert "created_time" in file_info
    assert "modified_time" in file_info

    # Test with non-existent file
    file_info = audio_manager.get_audio_file_info("non_existent.mp3")
    assert file_info is None


def test_cleanup_temp_files(audio_manager):
    """Test temporary file cleanup."""
    # Create some temporary files
    temp_files = []
    for i in range(3):
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(f"temp content {i}".encode())
            temp_files.append(f.name)

    # Verify files exist
    for temp_file in temp_files:
        assert Path(temp_file).exists()

    # Clean up
    audio_manager.cleanup_temp_files(temp_files)

    # Verify files are gone
    for temp_file in temp_files:
        assert not Path(temp_file).exists()


if __name__ == "__main__":
    pytest.main([__file__])
