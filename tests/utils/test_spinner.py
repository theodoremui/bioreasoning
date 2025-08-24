"""
Comprehensive test suite for the spinner utilities.

This module contains thorough tests for all spinner components including:
- OutputWriter implementations
- Timer functionality
- SpinnerAnimation behavior
- ProgressSpinner core functionality
- Context manager behavior
- Thread safety
- Error handling

Author: Theodore Mui
Date: 2025-08-24
"""

import pytest
import time
import threading
from unittest.mock import Mock, patch, call
from io import StringIO

from bioagents.utils.spinner import (
    OutputWriter,
    ConsoleOutputWriter,
    Timer,
    SpinnerAnimation,
    ProgressSpinner,
    Spinner,
    spinner_context,
)


class MockOutputWriter(OutputWriter):
    """Mock output writer for testing."""
    
    def __init__(self):
        self.written_text = []
        self.flush_count = 0
        self.clear_calls = []
    
    def write(self, text: str) -> None:
        self.written_text.append(text)
    
    def flush(self) -> None:
        self.flush_count += 1
    
    def clear_line(self, length: int) -> None:
        self.clear_calls.append(length)
    
    def get_all_text(self) -> str:
        return "".join(self.written_text)


class TestOutputWriter:
    """Test OutputWriter interface and implementations."""
    
    def test_console_output_writer_initialization(self):
        """Test ConsoleOutputWriter initialization."""
        # Default initialization
        writer = ConsoleOutputWriter()
        assert writer._stream is not None
        
        # Custom stream
        custom_stream = StringIO()
        writer = ConsoleOutputWriter(custom_stream)
        assert writer._stream is custom_stream
    
    def test_console_output_writer_write(self):
        """Test ConsoleOutputWriter write functionality."""
        stream = StringIO()
        writer = ConsoleOutputWriter(stream)
        
        writer.write("Hello, World!")
        assert stream.getvalue() == "Hello, World!"
        
        writer.write(" More text")
        assert stream.getvalue() == "Hello, World! More text"
    
    def test_console_output_writer_flush(self):
        """Test ConsoleOutputWriter flush functionality."""
        stream = StringIO()
        writer = ConsoleOutputWriter(stream)
        
        # Write some text
        writer.write("Test")
        writer.flush()  # Should not raise any exception
        
        # Verify text is still there
        assert stream.getvalue() == "Test"
    
    def test_console_output_writer_clear_line(self):
        """Test ConsoleOutputWriter clear_line functionality."""
        stream = StringIO()
        writer = ConsoleOutputWriter(stream)
        
        writer.clear_line(10)
        expected = "\r" + " " * 10 + "\r"
        assert stream.getvalue() == expected


class TestTimer:
    """Test Timer functionality."""
    
    def test_timer_initialization(self):
        """Test Timer initialization."""
        timer = Timer()
        assert timer._start_time is None
        assert timer.get_elapsed_seconds() == 0.0
    
    def test_timer_start_and_elapsed(self):
        """Test timer start and elapsed time calculation."""
        timer = Timer()
        
        # Start timer
        timer.start()
        assert timer._start_time is not None
        
        # Wait a bit and check elapsed time
        time.sleep(0.1)
        elapsed = timer.get_elapsed_seconds()
        assert 0.05 < elapsed < 0.2  # Allow for timing variations
    
    def test_timer_reset(self):
        """Test timer reset functionality."""
        timer = Timer()
        
        # Start and reset
        timer.start()
        time.sleep(0.05)
        timer.reset()
        
        assert timer._start_time is None
        assert timer.get_elapsed_seconds() == 0.0
    
    def test_timer_thread_safety(self):
        """Test timer thread safety."""
        timer = Timer()
        results = []
        
        def worker():
            timer.start()
            time.sleep(0.05)
            results.append(timer.get_elapsed_seconds())
        
        # Start multiple threads
        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # All should have recorded some elapsed time
        assert len(results) == 5
        assert all(r > 0 for r in results)
    
    @pytest.mark.parametrize("seconds,format_type,expected", [
        (0, "mm:ss", "00:00"),
        (30, "mm:ss", "00:30"),
        (90, "mm:ss", "01:30"),
        (3661, "mm:ss", "61:01"),
        (3661, "hh:mm:ss", "01:01:01"),
        (30, "compact", "30s"),
        (90, "compact", "1m30s"),
        (3661, "compact", "1h01m01s"),
    ])
    def test_format_duration(self, seconds, format_type, expected):
        """Test duration formatting."""
        result = Timer.format_duration(seconds, format_type)
        assert result == expected
    
    def test_format_duration_negative(self):
        """Test duration formatting with negative values."""
        result = Timer.format_duration(-10, "mm:ss")
        assert result == "00:00"
    
    def test_format_duration_invalid_format(self):
        """Test duration formatting with invalid format."""
        with pytest.raises(ValueError, match="Unsupported format_type"):
            Timer.format_duration(60, "invalid")


class TestSpinnerAnimation:
    """Test SpinnerAnimation functionality."""
    
    def test_animation_initialization_with_style(self):
        """Test animation initialization with built-in style."""
        animation = SpinnerAnimation("classic")
        assert animation._chars == "|/-\\"
        assert animation._index == 0
    
    def test_animation_initialization_with_custom_chars(self):
        """Test animation initialization with custom characters."""
        custom_chars = "ABC"
        animation = SpinnerAnimation(custom_chars)
        assert animation._chars == custom_chars
        assert animation._index == 0
    
    def test_animation_empty_chars_raises_error(self):
        """Test that empty animation characters raise an error."""
        with pytest.raises(ValueError, match="Animation characters cannot be empty"):
            SpinnerAnimation("")
    
    def test_animation_next_char_sequence(self):
        """Test animation character sequence."""
        animation = SpinnerAnimation("ABC")
        
        assert animation.next_char() == "A"
        assert animation.next_char() == "B"
        assert animation.next_char() == "C"
        assert animation.next_char() == "A"  # Should wrap around
    
    def test_animation_reset(self):
        """Test animation reset functionality."""
        animation = SpinnerAnimation("ABC")
        
        # Advance animation
        animation.next_char()
        animation.next_char()
        
        # Reset and verify
        animation.reset()
        assert animation.next_char() == "A"
    
    def test_animation_thread_safety(self):
        """Test animation thread safety."""
        animation = SpinnerAnimation("ABCDEFGHIJ")
        results = []
        
        def worker():
            for _ in range(10):
                results.append(animation.next_char())
        
        # Start multiple threads
        threads = [threading.Thread(target=worker) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # Should have 30 results total
        assert len(results) == 30
        # All results should be valid characters
        assert all(c in "ABCDEFGHIJ" for c in results)
    
    def test_built_in_styles(self):
        """Test all built-in animation styles."""
        for style_name in SpinnerAnimation.STYLES:
            animation = SpinnerAnimation(style_name)
            assert len(animation._chars) > 0
            # Should be able to get next character
            char = animation.next_char()
            assert char in animation._chars


class TestProgressSpinner:
    """Test ProgressSpinner core functionality."""
    
    def test_spinner_initialization_defaults(self):
        """Test spinner initialization with default values."""
        mock_output = MockOutputWriter()
        spinner = ProgressSpinner(output_writer=mock_output)
        
        assert spinner._message == "Processing"
        assert spinner._update_interval == 0.1
        assert spinner._show_timer is True
        assert spinner._timer_format == "mm:ss"
        assert not spinner.is_running()
    
    def test_spinner_initialization_custom_values(self):
        """Test spinner initialization with custom values."""
        mock_output = MockOutputWriter()
        spinner = ProgressSpinner(
            message="Custom Task",
            animation_chars="dots",
            update_interval=0.05,
            show_timer=False,
            timer_format="compact",
            output_writer=mock_output
        )
        
        assert spinner._message == "Custom Task"
        assert spinner._update_interval == 0.05
        assert spinner._show_timer is False
        assert spinner._timer_format == "compact"
    
    def test_spinner_invalid_initialization(self):
        """Test spinner initialization with invalid values."""
        mock_output = MockOutputWriter()
        
        # Empty message
        with pytest.raises(ValueError, match="Message cannot be empty"):
            ProgressSpinner("", output_writer=mock_output)
        
        # Invalid update interval
        with pytest.raises(ValueError, match="Update interval must be positive"):
            ProgressSpinner("Test", update_interval=0, output_writer=mock_output)
        
        # Invalid timer format
        with pytest.raises(ValueError, match="Invalid timer format"):
            ProgressSpinner("Test", timer_format="invalid", output_writer=mock_output)
    
    def test_spinner_start_stop(self):
        """Test basic spinner start and stop functionality."""
        mock_output = MockOutputWriter()
        spinner = ProgressSpinner("Test", output_writer=mock_output)
        
        # Initially not running
        assert not spinner.is_running()
        
        # Start spinner
        spinner.start()
        assert spinner.is_running()
        
        # Wait a bit for animation
        time.sleep(0.2)
        
        # Stop spinner
        spinner.stop()
        assert not spinner.is_running()
        
        # Should have written some output
        assert len(mock_output.written_text) > 0
        assert mock_output.flush_count > 0
    
    def test_spinner_double_start_raises_error(self):
        """Test that starting an already running spinner raises an error."""
        mock_output = MockOutputWriter()
        spinner = ProgressSpinner("Test", output_writer=mock_output)
        
        spinner.start()
        with pytest.raises(RuntimeError, match="Spinner is already running"):
            spinner.start()
        
        spinner.stop()
    
    def test_spinner_stop_when_not_running(self):
        """Test that stopping a non-running spinner is safe."""
        mock_output = MockOutputWriter()
        spinner = ProgressSpinner("Test", output_writer=mock_output)
        
        # Should not raise any exception
        spinner.stop()
        spinner.stop("Final message")
    
    def test_spinner_update_message(self):
        """Test updating spinner message."""
        mock_output = MockOutputWriter()
        spinner = ProgressSpinner("Initial", output_writer=mock_output)
        
        # Update message
        spinner.update_message("Updated")
        assert spinner._message == "Updated"
        
        # Empty message should raise error
        with pytest.raises(ValueError, match="Message cannot be empty"):
            spinner.update_message("")
    
    def test_spinner_elapsed_time(self):
        """Test elapsed time tracking."""
        mock_output = MockOutputWriter()
        spinner = ProgressSpinner("Test", output_writer=mock_output)
        
        # Initially zero
        assert spinner.get_elapsed_time() == 0.0
        
        # Start and check elapsed time
        spinner.start()
        time.sleep(0.1)
        elapsed = spinner.get_elapsed_time()
        assert 0.05 < elapsed < 0.2
        
        spinner.stop()
    
    def test_spinner_with_timer_disabled(self):
        """Test spinner with timer disabled."""
        mock_output = MockOutputWriter()
        spinner = ProgressSpinner("Test", show_timer=False, output_writer=mock_output)
        
        spinner.start()
        time.sleep(0.15)
        spinner.stop("Done")
        
        # Check that timer is not in output
        all_text = mock_output.get_all_text()
        assert "[" not in all_text or "]" not in all_text  # No timer brackets
    
    def test_spinner_final_message(self):
        """Test spinner final message display."""
        mock_output = MockOutputWriter()
        spinner = ProgressSpinner("Test", output_writer=mock_output)
        
        spinner.start()
        time.sleep(0.1)
        spinner.stop("Task completed")
        
        # Should have final message in output
        all_text = mock_output.get_all_text()
        assert "Task completed" in all_text
    
    def test_spinner_repr(self):
        """Test spinner string representation."""
        mock_output = MockOutputWriter()
        spinner = ProgressSpinner("Test Task", output_writer=mock_output)
        
        repr_str = repr(spinner)
        assert "ProgressSpinner" in repr_str
        assert "Test Task" in repr_str
        assert "stopped" in repr_str
        
        spinner.start()
        time.sleep(0.05)
        repr_str = repr(spinner)
        assert "running" in repr_str
        
        spinner.stop()


class TestProgressSpinnerContextManager:
    """Test ProgressSpinner context manager functionality."""
    
    def test_context_manager_success(self):
        """Test context manager with successful operation."""
        mock_output = MockOutputWriter()
        
        with ProgressSpinner("Test", output_writer=mock_output) as spinner:
            assert spinner.is_running()
            time.sleep(0.1)
        
        # Should have stopped and shown success message
        assert not spinner.is_running()
        all_text = mock_output.get_all_text()
        assert "✓" in all_text
        assert "completed" in all_text
    
    def test_context_manager_exception(self):
        """Test context manager with exception."""
        mock_output = MockOutputWriter()
        
        with pytest.raises(ValueError):
            with ProgressSpinner("Test", output_writer=mock_output) as spinner:
                assert spinner.is_running()
                raise ValueError("Test exception")
        
        # Should have stopped and shown failure message
        assert not spinner.is_running()
        all_text = mock_output.get_all_text()
        assert "✗" in all_text
        assert "failed" in all_text
    
    def test_spinner_context_function(self):
        """Test spinner_context convenience function."""
        mock_output = MockOutputWriter()
        
        with patch('bioagents.utils.spinner.ProgressSpinner') as MockSpinner:
            mock_instance = Mock()
            MockSpinner.return_value = mock_instance
            
            with spinner_context("Test", "dots", output_writer=mock_output):
                pass
            
            # Should have created spinner with correct args
            MockSpinner.assert_called_once_with("Test", "dots", output_writer=mock_output)
            mock_instance.start.assert_called_once()
            mock_instance.stop.assert_called_once()


class TestSpinnerAlias:
    """Test Spinner alias for backward compatibility."""
    
    def test_spinner_alias(self):
        """Test that Spinner is an alias for ProgressSpinner."""
        assert Spinner is ProgressSpinner
    
    def test_spinner_alias_functionality(self):
        """Test that Spinner alias works correctly."""
        mock_output = MockOutputWriter()
        spinner = Spinner("Test", output_writer=mock_output)
        
        assert isinstance(spinner, ProgressSpinner)
        spinner.start()
        time.sleep(0.05)
        spinner.stop()
        
        assert len(mock_output.written_text) > 0


class TestSpinnerIntegration:
    """Integration tests for spinner components."""
    
    def test_full_spinner_workflow(self):
        """Test complete spinner workflow."""
        mock_output = MockOutputWriter()
        spinner = ProgressSpinner(
            message="Processing data",
            animation_chars="classic",
            update_interval=0.05,
            show_timer=True,
            timer_format="mm:ss",
            output_writer=mock_output
        )
        
        # Start spinner
        spinner.start()
        assert spinner.is_running()
        
        # Let it run for a bit
        time.sleep(0.2)
        
        # Update message
        spinner.update_message("Finalizing")
        
        # Wait a bit more
        time.sleep(0.1)
        
        # Stop with final message
        spinner.stop("Processing complete")
        
        # Verify output
        all_text = mock_output.get_all_text()
        assert "Processing data" in all_text
        assert "Finalizing" in all_text
        assert "Processing complete" in all_text
        assert mock_output.flush_count > 0
        assert len(mock_output.clear_calls) > 0
    
    def test_multiple_spinners_concurrent(self):
        """Test multiple spinners running concurrently."""
        mock_output1 = MockOutputWriter()
        mock_output2 = MockOutputWriter()
        
        spinner1 = ProgressSpinner("Task 1", output_writer=mock_output1)
        spinner2 = ProgressSpinner("Task 2", output_writer=mock_output2)
        
        # Start both spinners
        spinner1.start()
        spinner2.start()
        
        # Let them run
        time.sleep(0.2)
        
        # Stop both
        spinner1.stop("Task 1 done")
        spinner2.stop("Task 2 done")
        
        # Both should have output
        assert len(mock_output1.written_text) > 0
        assert len(mock_output2.written_text) > 0
        assert "Task 1" in mock_output1.get_all_text()
        assert "Task 2" in mock_output2.get_all_text()
    
    def test_spinner_stress_test(self):
        """Stress test spinner with rapid start/stop cycles."""
        mock_output = MockOutputWriter()
        
        for i in range(10):
            spinner = ProgressSpinner(f"Task {i}", output_writer=mock_output)
            spinner.start()
            time.sleep(0.02)  # Very short duration
            spinner.stop()
        
        # Should have handled all cycles without errors
        assert len(mock_output.written_text) > 0


class TestSpinnerErrorHandling:
    """Test spinner error handling and edge cases."""
    
    def test_spinner_with_failing_output_writer(self):
        """Test spinner behavior with failing output writer."""
        class FailingOutputWriter(OutputWriter):
            def write(self, text: str) -> None:
                raise IOError("Write failed")
            
            def flush(self) -> None:
                raise IOError("Flush failed")
            
            def clear_line(self, length: int) -> None:
                raise IOError("Clear failed")
        
        failing_output = FailingOutputWriter()
        spinner = ProgressSpinner("Test", output_writer=failing_output)
        
        # Should handle errors gracefully
        spinner.start()
        time.sleep(0.1)
        spinner.stop()  # Should not raise exception
    
    def test_spinner_thread_cleanup(self):
        """Test that spinner threads are properly cleaned up."""
        mock_output = MockOutputWriter()
        spinner = ProgressSpinner("Test", output_writer=mock_output)
        
        spinner.start()
        original_thread = spinner._thread
        assert original_thread is not None
        assert original_thread.is_alive()
        
        spinner.stop()
        
        # Thread should be cleaned up
        time.sleep(0.1)  # Give thread time to finish
        assert not original_thread.is_alive()
    
    def test_spinner_timeout_on_stop(self):
        """Test spinner stop timeout behavior."""
        mock_output = MockOutputWriter()
        
        # Create a spinner that might hang
        with patch('threading.Thread.join') as mock_join:
            mock_join.return_value = None  # Simulate timeout
            
            spinner = ProgressSpinner("Test", output_writer=mock_output)
            spinner.start()
            
            # Stop should handle timeout gracefully
            spinner.stop()  # Should not hang indefinitely


if __name__ == "__main__":
    pytest.main([__file__])
