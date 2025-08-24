"""
Progress Spinner Utilities

This module provides a robust, thread-safe progress spinner implementation
following SOLID principles and best practices for user interface feedback.

Classes:
    OutputWriter: Abstract interface for output operations
    ConsoleOutputWriter: Console-based output implementation
    Timer: Handles time tracking and formatting
    SpinnerAnimation: Manages animation sequences
    ProgressSpinner: Main spinner class with context manager support

Example:
    Basic usage:
        from bioagents.utils.spinner import ProgressSpinner

        with ProgressSpinner("Loading data"):
            # Long running operation
            time.sleep(5)

    Advanced usage:
        spinner = ProgressSpinner(
            message="Processing",
            animation_chars="⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏",
            update_interval=0.08
        )
        spinner.start()
        # ... work ...
        spinner.stop("Completed successfully")

Author: Theodore Mui
Date: 2025-08-24
"""

import sys
import threading
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Optional, TextIO, Union


class OutputWriter(ABC):
    """Abstract interface for output operations.

    This interface allows for dependency inversion, making the spinner
    testable by injecting mock output writers.
    """

    @abstractmethod
    def write(self, text: str) -> None:
        """Write text to output."""
        pass

    @abstractmethod
    def flush(self) -> None:
        """Flush output buffer."""
        pass

    @abstractmethod
    def clear_line(self, length: int) -> None:
        """Clear current line with specified length."""
        pass


class ConsoleOutputWriter(OutputWriter):
    """Console-based output writer implementation.

    Handles writing to stdout with proper line clearing and flushing.

    Args:
        output_stream: Output stream to write to (default: sys.stdout)
    """

    def __init__(self, output_stream: TextIO = sys.stdout):
        self._stream = output_stream

    def write(self, text: str) -> None:
        """Write text to console."""
        self._stream.write(text)

    def flush(self) -> None:
        """Flush console output buffer."""
        self._stream.flush()

    def clear_line(self, length: int) -> None:
        """Clear current console line."""
        self._stream.write("\r" + " " * length + "\r")


class Timer:
    """Handles time tracking and formatting.

    Provides precise timing functionality with various formatting options.
    Thread-safe implementation for concurrent access.
    """

    def __init__(self):
        self._start_time: Optional[float] = None
        self._lock = threading.Lock()

    def start(self) -> None:
        """Start the timer."""
        with self._lock:
            self._start_time = time.time()

    def reset(self) -> None:
        """Reset the timer."""
        with self._lock:
            self._start_time = None

    def get_elapsed_seconds(self) -> float:
        """Get elapsed time in seconds.

        Returns:
            Elapsed seconds since start, or 0.0 if not started
        """
        with self._lock:
            if self._start_time is None:
                return 0.0
            return time.time() - self._start_time

    @staticmethod
    def format_duration(seconds: float, format_type: str = "mm:ss") -> str:
        """Format duration in various formats.

        Args:
            seconds: Duration in seconds
            format_type: Format type ("mm:ss", "hh:mm:ss", "compact")

        Returns:
            Formatted duration string

        Raises:
            ValueError: If format_type is not supported
        """
        if seconds < 0:
            seconds = 0

        total_seconds = int(seconds)
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        secs = total_seconds % 60

        if format_type == "mm:ss":
            # For mm:ss format, show total minutes (can exceed 59)
            total_minutes = total_seconds // 60
            return f"{total_minutes:02d}:{secs:02d}"
        elif format_type == "hh:mm:ss":
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        elif format_type == "compact":
            if hours > 0:
                return f"{hours}h{minutes:02d}m{secs:02d}s"
            elif minutes > 0:
                return f"{minutes}m{secs:02d}s"
            else:
                return f"{secs}s"
        else:
            raise ValueError(f"Unsupported format_type: {format_type}")


class SpinnerAnimation:
    """Manages spinner animation sequences.

    Provides various built-in animation styles and supports custom animations.
    Thread-safe implementation for concurrent access.
    """

    # Built-in animation styles
    STYLES = {
        "classic": "|/-\\",
        "dots": "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏",
        "arrows": "←↖↑↗→↘↓↙",
        "bounce": "⠁⠂⠄⠁⠂⠄",
        "pulse": "●○●○",
        "blocks": "▁▃▄▅▆▇█▇▆▅▄▃",
        "simple": "◐◓◑◒",
    }

    def __init__(self, animation_chars: str = "classic"):
        """Initialize spinner animation.

        Args:
            animation_chars: Animation characters or style name
        """
        if animation_chars in self.STYLES:
            self._chars = self.STYLES[animation_chars]
        else:
            self._chars = animation_chars

        if not self._chars:
            raise ValueError("Animation characters cannot be empty")

        self._index = 0
        self._lock = threading.Lock()

    def next_char(self) -> str:
        """Get next animation character.

        Returns:
            Next character in the animation sequence
        """
        with self._lock:
            char = self._chars[self._index]
            self._index = (self._index + 1) % len(self._chars)
            return char

    def reset(self) -> None:
        """Reset animation to beginning."""
        with self._lock:
            self._index = 0


class ProgressSpinner:
    """Main progress spinner class with context manager support.

    A robust, thread-safe progress spinner that provides visual feedback
    for long-running operations. Supports customizable animations, timing,
    and output formatting.

    Features:
        - Thread-safe operation
        - Customizable animation styles
        - Elapsed time tracking
        - Context manager support
        - Dependency injection for testability
        - Graceful error handling

    Args:
        message: Display message for the spinner
        animation_chars: Animation characters or style name
        update_interval: Animation update interval in seconds
        show_timer: Whether to show elapsed time
        timer_format: Timer format ("mm:ss", "hh:mm:ss", "compact")
        output_writer: Output writer implementation (for testing)

    Example:
        with ProgressSpinner("Loading data", animation_chars="dots"):
            time.sleep(5)
    """

    def __init__(
        self,
        message: str = "Processing",
        animation_chars: str = "classic",
        update_interval: float = 0.1,
        show_timer: bool = True,
        timer_format: str = "mm:ss",
        output_writer: Optional[OutputWriter] = None,
    ):
        # Validate inputs
        if not message.strip():
            raise ValueError("Message cannot be empty")
        if update_interval <= 0:
            raise ValueError("Update interval must be positive")
        if timer_format not in ["mm:ss", "hh:mm:ss", "compact"]:
            raise ValueError(f"Invalid timer format: {timer_format}")

        self._message = message.strip()
        self._update_interval = update_interval
        self._show_timer = show_timer
        self._timer_format = timer_format

        # Initialize components
        self._timer = Timer()
        self._animation = SpinnerAnimation(animation_chars)
        self._output = output_writer or ConsoleOutputWriter()

        # Thread management
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # State tracking
        self._last_line_length = 0

    def start(self) -> None:
        """Start the spinner animation.

        Raises:
            RuntimeError: If spinner is already running
        """
        with self._lock:
            if self._running:
                raise RuntimeError("Spinner is already running")

            self._running = True
            self._timer.start()
            self._animation.reset()

            self._thread = threading.Thread(target=self._animate, daemon=True)
            self._thread.start()

    def stop(self, final_message: Optional[str] = None) -> None:
        """Stop the spinner animation.

        Args:
            final_message: Optional final message to display
        """
        with self._lock:
            if not self._running:
                return

            self._running = False

        # Wait for thread to finish
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)  # Prevent hanging

        # Clear spinner line
        self._output.clear_line(self._last_line_length)

        # Show final message
        if final_message:
            elapsed = self._timer.get_elapsed_seconds()
            if self._show_timer:
                formatted_time = Timer.format_duration(elapsed, self._timer_format)
                self._output.write(f"{final_message} [{formatted_time}]\n")
            else:
                self._output.write(f"{final_message}\n")
            self._output.flush()

    def update_message(self, new_message: str) -> None:
        """Update the spinner message while running.

        Args:
            new_message: New message to display

        Raises:
            ValueError: If message is empty
        """
        if not new_message.strip():
            raise ValueError("Message cannot be empty")

        with self._lock:
            self._message = new_message.strip()

    def is_running(self) -> bool:
        """Check if spinner is currently running.

        Returns:
            True if spinner is running, False otherwise
        """
        with self._lock:
            return self._running

    def get_elapsed_time(self) -> float:
        """Get elapsed time since spinner started.

        Returns:
            Elapsed time in seconds
        """
        return self._timer.get_elapsed_seconds()

    def _animate(self) -> None:
        """Internal animation loop (runs in separate thread)."""
        try:
            while self._running:
                # Get current animation frame
                char = self._animation.next_char()

                # Build display line
                line_parts = [f"\r{self._message} {char}"]

                if self._show_timer:
                    elapsed = self._timer.get_elapsed_seconds()
                    formatted_time = Timer.format_duration(elapsed, self._timer_format)
                    line_parts.append(f" [{formatted_time}]")

                line = "".join(line_parts)

                # Update display
                self._output.write(line)
                self._output.flush()

                # Track line length for clearing
                self._last_line_length = len(line) - 1  # Subtract \r

                # Sleep until next update
                time.sleep(self._update_interval)

        except Exception:
            # Gracefully handle any animation errors
            with self._lock:
                self._running = False

    def __enter__(self) -> "ProgressSpinner":
        """Context manager entry.

        Returns:
            Self for method chaining
        """
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Context manager exit.

        Args:
            exc_type: Exception type (if any)
            exc_val: Exception value (if any)
            exc_tb: Exception traceback (if any)

        Returns:
            False to propagate exceptions
        """
        if exc_type is None:
            self.stop(f"✓ {self._message}: completed")
        else:
            self.stop(f"✗ {self._message}: failed")
        return False

    def __repr__(self) -> str:
        """String representation of the spinner."""
        status = "running" if self.is_running() else "stopped"
        elapsed = self.get_elapsed_time()
        return f"ProgressSpinner(message='{self._message}', status='{status}', elapsed={elapsed:.1f}s)"


# Convenience aliases for backward compatibility
Spinner = ProgressSpinner


@contextmanager
def spinner_context(
    message: str = "Processing", animation_chars: str = "classic", **kwargs
):
    """Convenience context manager for spinner usage.

    Args:
        message: Display message
        animation_chars: Animation style or characters
        **kwargs: Additional arguments for ProgressSpinner

    Yields:
        ProgressSpinner instance

    Example:
        with spinner_context("Loading", "dots") as s:
            time.sleep(2)
            s.update_message("Almost done")
            time.sleep(1)
    """
    spinner = ProgressSpinner(message, animation_chars, **kwargs)
    try:
        spinner.start()
        yield spinner
    finally:
        spinner.stop()
