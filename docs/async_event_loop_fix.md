# Async Event Loop Fix for Streamlit Applications

## Problem

The "Event loop is closed" exception occurs when trying to use `asyncio.run()` in Streamlit applications. This happens because:

1. **Streamlit already has an event loop running** - Streamlit runs its own event loop for handling the web interface
2. **`asyncio.run()` creates a new event loop** - This conflicts with the existing running loop
3. **Multiple async operations** - When multiple async functions are called, they can interfere with each other

## Root Cause

The issue was in several places in the codebase:

```python
# ❌ Problematic code
agent_response = asyncio.run(reasoner.achat(prompt))
audio_file = asyncio.run(create_podcast(file_content, config))
```

This pattern was used in:
- `frontend/pages/1_Chat.py` (line 141)
- `frontend/pages/2_Documents.py` (line 198)
- `bioreasoning.py` (line 88)

## Solution

### 1. Created Async Utility Module

Created `bioagents/utils/async_utils.py` with a robust async execution function:

```python
def run_async_in_streamlit(coro: Coroutine[Any, Any, Any]) -> Any:
    """
    Run an async coroutine in a Streamlit-compatible way.
    
    This function handles the event loop properly to avoid "Event loop is closed"
    exceptions that commonly occur in Streamlit applications.
    """
    
    def _run_in_new_loop():
        """Run the coroutine in a new event loop."""
        if sys.platform == "win32":
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()
    
    try:
        # Try to use existing event loop
        loop = asyncio.get_running_loop()
        # If we get here, there's a running loop, so use ThreadPoolExecutor
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(_run_in_new_loop)
            return future.result()
    except RuntimeError:
        # No event loop exists or it's not running, create one
        return _run_in_new_loop()
```

### 2. Updated All Async Calls

Replaced all `asyncio.run()` calls with the new utility:

```python
# ✅ Fixed code
from bioagents.utils.async_utils import run_async_in_streamlit

agent_response = run_async_in_streamlit(reasoner.achat(prompt))
audio_file = run_async_in_streamlit(create_podcast(file_content, config))
```

### 3. Added Comprehensive Tests

Created `tests/test_async_utils.py` to verify the solution works in various scenarios:

- Basic async execution
- Execution with existing event loop
- Execution with running event loop
- Error handling
- Windows compatibility

## How It Works

1. **Check for running loop**: Uses `asyncio.get_running_loop()` to detect if there's already an event loop running
2. **Thread-based execution**: If a loop is running, executes the coroutine in a separate thread with a new event loop
3. **Direct execution**: If no loop is running, creates a new loop and executes directly
4. **Proper cleanup**: Ensures event loops are properly closed to prevent resource leaks

## Benefits

- ✅ **Fixes the "Event loop is closed" exception**
- ✅ **Works in all Streamlit scenarios** (single page, multi-page, with/without existing loops)
- ✅ **Cross-platform compatibility** (Windows, macOS, Linux)
- ✅ **Proper resource management** (event loops are cleaned up)
- ✅ **Backward compatible** (works with existing async code)
- ✅ **Well tested** (comprehensive test suite)

## Usage

### For New Async Functions

```python
from bioagents.utils.async_utils import run_async_in_streamlit

# Instead of asyncio.run()
result = run_async_in_streamlit(your_async_function())
```

### For Creating Wrappers

```python
from bioagents.utils.async_utils import create_async_wrapper

# Create a sync wrapper for an async function
sync_func = create_async_wrapper(your_async_function)
result = sync_func()  # No need to handle async manually
```

## Testing

Run the tests to verify the fix:

```bash
cd tests
python -m pytest test_async_utils.py -v
```

## Related Issues

This fix also resolves related issues:
- **Podcast generation failures** - Now works properly in Streamlit
- **Chat agent responses** - Async calls work reliably
- **Document processing** - Workflow execution is stable
- **Multi-page navigation** - No event loop conflicts between pages

## Troubleshooting

### Environment Variable Issues
- **"export: not a valid identifier" errors**: Run `./scripts/check-env.sh` to diagnose .env file issues
- **Multi-line values**: The script now properly handles multi-line environment variables
- **Special characters**: URLs and special characters in values are now handled correctly
- **Missing API keys**: Use `./scripts/setup-env.sh` to configure required API keys

### Common Error Messages
- **"Event loop is closed"**: Fixed by the async utilities refactoring
- **"Podcast generation is not available"**: Check that both `OPENAI_API_KEY` and `ELEVENLABS_API_KEY` are set
- **"Authentication failed"**: Verify your API keys are valid and have sufficient credits 