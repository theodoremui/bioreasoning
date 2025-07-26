"""
Tests for async utilities used in Streamlit applications.
"""

import pytest
import asyncio
import sys
from unittest.mock import patch, MagicMock

# Add the bioagents directory to the path for imports
sys.path.append('../bioagents')

from bioagents.utils.async_utils import run_async_in_streamlit, create_async_wrapper


@pytest.fixture
def sample_async_function():
    """Create a sample async function for testing."""
    async def sample_func(value: int) -> int:
        await asyncio.sleep(0.01)  # Small delay to simulate async work
        return value * 2
    
    return sample_func


def test_run_async_in_streamlit_basic(sample_async_function):
    """Test basic async execution."""
    result = run_async_in_streamlit(sample_async_function(5))
    assert result == 10


def test_run_async_in_streamlit_with_existing_loop(sample_async_function):
    """Test async execution when an event loop already exists but is not running."""
    # Create a new event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        result = run_async_in_streamlit(sample_async_function(3))
        assert result == 6
    finally:
        loop.close()


def test_run_async_in_streamlit_with_running_loop(sample_async_function):
    """Test async execution when an event loop is already running."""
    # Mock the event loop to simulate it being already running
    mock_loop = MagicMock()
    mock_loop.is_running.return_value = True
    
    with patch('asyncio.get_event_loop', return_value=mock_loop):
        with patch('concurrent.futures.ThreadPoolExecutor') as mock_executor:
            # Mock the executor context manager
            mock_context = MagicMock()
            mock_executor.return_value.__enter__.return_value = mock_context
            
            # Mock the future result
            mock_future = MagicMock()
            mock_future.result.return_value = 8
            mock_context.submit.return_value = mock_future
            
            result = run_async_in_streamlit(sample_async_function(4))
            assert result == 8


def test_run_async_in_streamlit_with_runtime_error(sample_async_function):
    """Test async execution when no event loop exists."""
    # Mock get_event_loop to raise RuntimeError
    with patch('asyncio.get_event_loop', side_effect=RuntimeError("No event loop")):
        result = run_async_in_streamlit(sample_async_function(6))
        assert result == 12


def test_create_async_wrapper(sample_async_function):
    """Test the async wrapper creation."""
    sync_func = create_async_wrapper(sample_async_function)
    
    # Test that the wrapper works
    result = sync_func(7)
    assert result == 14


def test_run_async_in_streamlit_with_exception():
    """Test async execution that raises an exception."""
    async def failing_func():
        await asyncio.sleep(0.01)
        raise ValueError("Test error")
    
    with pytest.raises(ValueError, match="Test error"):
        run_async_in_streamlit(failing_func())


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-specific test")
def test_run_async_in_streamlit_windows():
    """Test async execution on Windows with proper event loop policy."""
    async def sample_func():
        await asyncio.sleep(0.01)
        return "windows_test"
    
    result = run_async_in_streamlit(sample_func())
    assert result == "windows_test"


if __name__ == "__main__":
    pytest.main([__file__]) 