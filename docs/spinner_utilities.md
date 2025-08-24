# Spinner Utilities Documentation

## Overview

The `bioagents.utils.spinner` module provides a robust, thread-safe progress spinner implementation for providing visual feedback during long-running operations. The module follows SOLID principles and includes comprehensive testing and documentation.

## Features

- **Thread-safe operation**: Safe for use in multi-threaded environments
- **Customizable animations**: Multiple built-in styles and support for custom animations
- **Elapsed time tracking**: Optional timer display with multiple formats
- **Context manager support**: Easy integration with `with` statements
- **Dependency injection**: Testable design with pluggable output writers
- **Graceful error handling**: Robust operation even with I/O failures
- **SOLID principles**: Clean, maintainable, and extensible architecture

## Architecture

The spinner module is designed with separation of concerns:

```
ProgressSpinner (Main class)
├── Timer (Time tracking)
├── SpinnerAnimation (Animation sequences)
└── OutputWriter (Output abstraction)
    └── ConsoleOutputWriter (Console implementation)
```

### Key Components

1. **OutputWriter**: Abstract interface for output operations (dependency inversion)
2. **Timer**: Thread-safe time tracking with multiple formatting options
3. **SpinnerAnimation**: Animation sequence management with built-in styles
4. **ProgressSpinner**: Main spinner class with full functionality

## Quick Start

### Basic Usage

```python
from bioagents.utils.spinner import ProgressSpinner

# Simple spinner with context manager
with ProgressSpinner("Loading data"):
    # Your long-running operation here
    time.sleep(5)
```

### Advanced Usage

```python
from bioagents.utils.spinner import ProgressSpinner

# Customized spinner
spinner = ProgressSpinner(
    message="Processing files",
    animation_chars="dots",  # Built-in style
    update_interval=0.08,
    timer_format="hh:mm:ss"
)

spinner.start()
try:
    # Your work here
    time.sleep(2)
    spinner.update_message("Finalizing")
    time.sleep(1)
finally:
    spinner.stop("Processing complete")
```

### Convenience Function

```python
from bioagents.utils.spinner import spinner_context

with spinner_context("Loading", "pulse") as s:
    time.sleep(2)
    s.update_message("Almost done")
    time.sleep(1)
```

## API Reference

### ProgressSpinner

Main spinner class providing visual progress feedback.

#### Constructor

```python
ProgressSpinner(
    message: str = "Processing",
    animation_chars: str = "classic",
    update_interval: float = 0.1,
    show_timer: bool = True,
    timer_format: str = "mm:ss",
    output_writer: Optional[OutputWriter] = None
)
```

**Parameters:**
- `message`: Display message for the spinner
- `animation_chars`: Animation characters or built-in style name
- `update_interval`: Animation update interval in seconds (must be > 0)
- `show_timer`: Whether to display elapsed time
- `timer_format`: Timer format ("mm:ss", "hh:mm:ss", "compact")
- `output_writer`: Custom output writer (for testing)

**Raises:**
- `ValueError`: If message is empty, update_interval ≤ 0, or invalid timer_format

#### Methods

##### start()
Start the spinner animation.

**Raises:**
- `RuntimeError`: If spinner is already running

##### stop(final_message: Optional[str] = None)
Stop the spinner animation.

**Parameters:**
- `final_message`: Optional message to display after stopping

##### update_message(new_message: str)
Update the spinner message while running.

**Parameters:**
- `new_message`: New message to display

**Raises:**
- `ValueError`: If message is empty

##### is_running() -> bool
Check if spinner is currently running.

**Returns:**
- `True` if running, `False` otherwise

##### get_elapsed_time() -> float
Get elapsed time since spinner started.

**Returns:**
- Elapsed time in seconds

#### Context Manager

The spinner supports context manager protocol:

```python
with ProgressSpinner("Working") as spinner:
    # spinner.start() called automatically
    do_work()
    # spinner.stop() called automatically with success/failure message
```

### Timer

Thread-safe timer for elapsed time tracking.

#### Methods

##### start()
Start the timer.

##### reset()
Reset the timer to initial state.

##### get_elapsed_seconds() -> float
Get elapsed time in seconds.

##### Timer.format_duration(seconds: float, format_type: str = "mm:ss") -> str
Static method to format duration.

**Parameters:**
- `seconds`: Duration in seconds
- `format_type`: Format type ("mm:ss", "hh:mm:ss", "compact")

**Returns:**
- Formatted duration string

**Raises:**
- `ValueError`: If format_type is not supported

### SpinnerAnimation

Manages animation sequences with built-in styles.

#### Built-in Styles

- `"classic"`: `|/-\` (default)
- `"dots"`: `⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏`
- `"arrows"`: `←↖↑↗→↘↓↙`
- `"bounce"`: `⠁⠂⠄⠁⠂⠄`
- `"pulse"`: `●○●○`
- `"blocks"`: `▁▃▄▅▆▇█▇▆▅▄▃`
- `"simple"`: `◐◓◑◒`

#### Constructor

```python
SpinnerAnimation(animation_chars: str = "classic")
```

**Parameters:**
- `animation_chars`: Animation characters or built-in style name

**Raises:**
- `ValueError`: If animation_chars is empty

#### Methods

##### next_char() -> str
Get next character in animation sequence.

##### reset()
Reset animation to beginning.

### OutputWriter

Abstract interface for output operations (for dependency injection).

#### Methods

##### write(text: str)
Write text to output (abstract).

##### flush()
Flush output buffer (abstract).

##### clear_line(length: int)
Clear current line (abstract).

### ConsoleOutputWriter

Console implementation of OutputWriter.

#### Constructor

```python
ConsoleOutputWriter(output_stream: TextIO = sys.stdout)
```

**Parameters:**
- `output_stream`: Output stream to write to

## Examples

### Basic Progress Indication

```python
from bioagents.utils.spinner import ProgressSpinner
import time

def process_files():
    with ProgressSpinner("Processing files"):
        for i in range(10):
            time.sleep(0.5)  # Simulate work

process_files()
```

### Multi-step Process

```python
from bioagents.utils.spinner import ProgressSpinner

def complex_operation():
    spinner = ProgressSpinner("Initializing", animation_chars="dots")
    spinner.start()
    
    try:
        # Step 1
        time.sleep(2)
        
        # Step 2
        spinner.update_message("Processing data")
        time.sleep(3)
        
        # Step 3
        spinner.update_message("Finalizing")
        time.sleep(1)
        
        spinner.stop("✓ Operation completed successfully")
        
    except Exception as e:
        spinner.stop(f"✗ Operation failed: {e}")
        raise

complex_operation()
```

### Custom Animation

```python
from bioagents.utils.spinner import ProgressSpinner

# Custom animation with emojis
custom_spinner = ProgressSpinner(
    "Loading",
    animation_chars="🌍🌎🌏",
    update_interval=0.2
)

with custom_spinner:
    time.sleep(3)
```

### Testing with Mock Output

```python
from bioagents.utils.spinner import ProgressSpinner, OutputWriter

class MockOutputWriter(OutputWriter):
    def __init__(self):
        self.output = []
    
    def write(self, text: str):
        self.output.append(text)
    
    def flush(self):
        pass
    
    def clear_line(self, length: int):
        self.output.append(f"CLEAR:{length}")

# Use in tests
mock_output = MockOutputWriter()
spinner = ProgressSpinner("Test", output_writer=mock_output)
spinner.start()
time.sleep(0.1)
spinner.stop()

print(mock_output.output)  # See captured output
```

### Concurrent Spinners

```python
import threading
from bioagents.utils.spinner import ProgressSpinner

def worker(name, duration):
    with ProgressSpinner(f"Task {name}", animation_chars="pulse"):
        time.sleep(duration)

# Run multiple tasks concurrently
threads = [
    threading.Thread(target=worker, args=(i, 2 + i * 0.5))
    for i in range(3)
]

for t in threads:
    t.start()

for t in threads:
    t.join()
```

## Best Practices

### 1. Use Context Managers

Always prefer context managers for automatic cleanup:

```python
# Good
with ProgressSpinner("Working"):
    do_work()

# Avoid
spinner = ProgressSpinner("Working")
spinner.start()
try:
    do_work()
finally:
    spinner.stop()
```

### 2. Choose Appropriate Update Intervals

- **Fast operations** (< 1 second): Use 0.05-0.08 seconds
- **Normal operations** (1-10 seconds): Use 0.1 seconds (default)
- **Slow operations** (> 10 seconds): Use 0.2-0.5 seconds

### 3. Provide Meaningful Messages

```python
# Good
with ProgressSpinner("Downloading dataset (500MB)"):
    download_data()

# Less helpful
with ProgressSpinner("Please wait"):
    download_data()
```

### 4. Update Messages for Long Operations

```python
spinner = ProgressSpinner("Starting analysis")
spinner.start()

try:
    spinner.update_message("Loading data")
    load_data()
    
    spinner.update_message("Processing results")
    process_results()
    
    spinner.update_message("Generating report")
    generate_report()
    
    spinner.stop("✓ Analysis complete")
except Exception as e:
    spinner.stop(f"✗ Analysis failed: {e}")
    raise
```

### 5. Choose Appropriate Animation Styles

- **Professional/corporate**: `"classic"`, `"simple"`
- **Modern/technical**: `"dots"`, `"blocks"`
- **Playful/creative**: `"pulse"`, `"bounce"`
- **Custom branding**: Use custom characters

## Thread Safety

The spinner is designed to be thread-safe:

- All shared state is protected by locks
- Animation runs in a separate daemon thread
- Safe to start/stop from different threads
- Multiple spinners can run concurrently

## Error Handling

The spinner handles various error conditions gracefully:

- **I/O errors**: Continues operation even if output fails
- **Thread errors**: Proper cleanup of animation threads
- **Invalid parameters**: Clear error messages with validation
- **Concurrent access**: Thread-safe operations

## Testing

The module includes comprehensive tests covering:

- All public APIs
- Thread safety
- Error conditions
- Edge cases
- Integration scenarios

Run tests with:
```bash
pytest tests/utils/test_spinner.py -v
```

## Migration from Old Spinner

If migrating from the old `Spinner` class:

```python
# Old usage
from some_module import Spinner
with Spinner("Processing"):
    do_work()

# New usage (backward compatible)
from bioagents.utils.spinner import Spinner  # or ProgressSpinner
with Spinner("Processing"):
    do_work()
```

The new implementation is backward compatible but provides many additional features and improvements.

## Performance Considerations

- **Memory usage**: Minimal overhead, single animation thread
- **CPU usage**: Very low, configurable update intervals
- **I/O impact**: Minimal console output, efficient line clearing
- **Thread overhead**: Single daemon thread per spinner

## Troubleshooting

### Common Issues

1. **Spinner not visible**: Check if output is being redirected
2. **Performance impact**: Increase update_interval for slower systems
3. **Unicode issues**: Use "classic" style for better compatibility
4. **Thread hanging**: Ensure proper cleanup with context managers

### Debug Mode

For debugging, use a mock output writer:

```python
from bioagents.utils.spinner import ProgressSpinner

class DebugOutputWriter:
    def write(self, text): print(f"WRITE: {repr(text)}")
    def flush(self): print("FLUSH")
    def clear_line(self, length): print(f"CLEAR: {length}")

spinner = ProgressSpinner("Debug", output_writer=DebugOutputWriter())
```

## Contributing

When contributing to the spinner module:

1. Follow SOLID principles
2. Maintain thread safety
3. Add comprehensive tests
4. Update documentation
5. Consider backward compatibility

## License

This module is part of the BioAgents project and follows the same license terms.
