#------------------------------------------------------------------------------
# __init__.py
# 
# Utils package for bioagents.
# 
# Author: Theodore Mui
# Date: 2025-01-20
#------------------------------------------------------------------------------

"""
BioAgents Utils Package

This package contains utility modules for the BioAgents system, including
retry logic, error handling, and other common utilities.
"""

from .async_utils import run_async_in_streamlit, create_async_wrapper


# Always import retry utilities (core functionality)
from .retry import (
    RetryConfig,
    RetryStrategy,
    ExponentialBackoffStrategy,
    RateLimitAwareStrategy,
    RetryContext,
    retry_with_backoff,
    retry_with_rate_limit_awareness,
    RetryStrategies,
    retry_api_call,
    is_retryable_error,
    is_rate_limit_error,
)

# Try to import API clients (may fail if dependencies not installed)
try:
    from .api_clients import (
        BaseAPIClient,
        RobustOpenAIClient,
        RobustElevenLabsClient,
        APICallResult,
        OpenAIAPIError,
        ElevenLabsAPIError,
        RateLimitError,
        AuthenticationError,
        ValidationError,
        QuotaExceededError,
        create_openai_client,
        create_elevenlabs_client,
        check_api_connection,
        get_client_status,
        get_elevenlabs_credit_info,
    )
    API_CLIENTS_AVAILABLE = True
except ImportError as e:
    # Set these to None if dependencies are not available
    BaseAPIClient = None
    RobustOpenAIClient = None
    RobustElevenLabsClient = None
    APICallResult = None
    OpenAIAPIError = None
    ElevenLabsAPIError = None
    RateLimitError = None
    AuthenticationError = None
    ValidationError = None
    QuotaExceededError = None
    create_openai_client = None
    create_elevenlabs_client = None
    check_api_connection = None
    get_client_status = None
    get_elevenlabs_credit_info = None
    API_CLIENTS_AVAILABLE = False

# Define what's always available
__all__ = [
    # Retry utilities (always available)
    "RetryConfig",
    "RetryStrategy", 
    "ExponentialBackoffStrategy",
    "RateLimitAwareStrategy",
    "RetryContext",
    "retry_with_backoff",
    "retry_with_rate_limit_awareness",
    "RetryStrategies",
    "retry_api_call",
    "is_retryable_error",
    "is_rate_limit_error",
]

# Add API clients if available
if API_CLIENTS_AVAILABLE:
    __all__.extend([
        "BaseAPIClient",
        "RobustOpenAIClient",
        "RobustElevenLabsClient",
        "APICallResult",
        "OpenAIAPIError",
        "ElevenLabsAPIError",
        "RateLimitError",
        "AuthenticationError",
        "ValidationError",
        "QuotaExceededError",
        "create_openai_client",
        "create_elevenlabs_client",
        "check_api_connection",
        "get_client_status",
        "get_elevenlabs_credit_info",
    ]) 


# Add audio management utilities
from .audio_manager import AudioFileManager, AudioFileProcessor, AudioFileError

__all__ = [
    # Async utilities
    "run_async_in_streamlit", 
    "create_async_wrapper",
    # Audio management
    "AudioFileManager",
    "AudioFileProcessor", 
    "AudioFileError"
] 