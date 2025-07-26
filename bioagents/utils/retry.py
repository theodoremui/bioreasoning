#------------------------------------------------------------------------------
# retry.py
# 
# Robust retry logic using tenacity for API calls.
# Implements SOLID principles with clear separation of concerns.
# 
# Author: Theodore Mui
# Date: 2025-07-21
#------------------------------------------------------------------------------

"""
Retry Utilities for API Calls

This module provides robust retry logic for API calls using the tenacity library.
It implements SOLID principles with clear separation of concerns and follows
Occam's Razor by keeping the implementation simple yet effective.

Features:
- Configurable retry strategies for different API providers
- Exponential backoff with jitter
- Rate limiting awareness
- Comprehensive error handling
- Logging and monitoring
- Type-safe decorators

Usage:
    @retry_with_backoff(max_attempts=3, base_delay=1.0)
    async def api_call():
        # Your API call here
        pass

    # Or use the context manager
    async with RetryContext(max_attempts=3) as retry:
        result = await retry.call(api_function, *args, **kwargs)
"""

import asyncio
import logging
import random
import time
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union
from dataclasses import dataclass, field

from tenacity import (
    AsyncRetrying,
    RetryCallState,
    RetryError,
    before_sleep_log,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    wait_random_exponential,
    retry_if_result,
    before_log,
    after_log,
    retry_if_exception,
)
from tenacity.wait import wait_base

logger = logging.getLogger(__name__)

# Type variables for generic functions
T = TypeVar('T')
F = TypeVar('F', bound=Callable[..., Any])

# Exception types that should trigger retries
RETRYABLE_EXCEPTIONS = (
    Exception,  # Base exception - customize as needed
)

# Rate limit specific exceptions
RATE_LIMIT_EXCEPTIONS = (
    Exception,  # Replace with actual rate limit exception types
)

# Try to import ValidationError from our custom exceptions
try:
    from bioagents.utils.api_clients import ValidationError
    VALIDATION_ERROR_AVAILABLE = True
except ImportError:
    ValidationError = None
    VALIDATION_ERROR_AVAILABLE = False

# Exceptions that should NOT be retried
NON_RETRYABLE_EXCEPTIONS = (
    ValueError,
    TypeError,
    AttributeError,
    KeyError,
    IndexError,
    ImportError,
    SyntaxError,
    NameError,
) + ((ValidationError,) if VALIDATION_ERROR_AVAILABLE else ())

@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    retryable_exceptions: tuple = field(default_factory=lambda: RETRYABLE_EXCEPTIONS)
    rate_limit_exceptions: tuple = field(default_factory=lambda: RATE_LIMIT_EXCEPTIONS)
    log_retries: bool = True
    log_success: bool = False
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if self.base_delay <= 0:
            raise ValueError("base_delay must be positive")
        if self.max_delay <= self.base_delay:
            raise ValueError("max_delay must be greater than base_delay")


class RetryStrategy(ABC):
    """Abstract base class for retry strategies."""
    
    @abstractmethod
    def get_retrying(self, config: RetryConfig) -> AsyncRetrying:
        """Get the retrying strategy for the given configuration."""
        pass


class ExponentialBackoffStrategy(RetryStrategy):
    """Exponential backoff with jitter retry strategy."""
    
    def get_retrying(self, config: RetryConfig) -> AsyncRetrying:
        """Get exponential backoff retrying strategy."""
        wait_strategy: wait_base
        if config.jitter:
            wait_strategy = wait_random_exponential(
                multiplier=config.base_delay,
                max=config.max_delay,
                exp_base=config.exponential_base
            )
        else:
            wait_strategy = wait_exponential(
                multiplier=config.base_delay,
                max=config.max_delay,
                exp_base=config.exponential_base
            )
        
        # Create retry condition that excludes non-retryable exceptions
        def should_retry(exception):
            if isinstance(exception, NON_RETRYABLE_EXCEPTIONS):
                return False
            return isinstance(exception, config.retryable_exceptions)
        
        return AsyncRetrying(
            stop=stop_after_attempt(config.max_attempts),
            wait=wait_strategy,
            retry=retry_if_exception(should_retry),
            before_sleep=before_sleep_log(logger, logging.WARNING) if config.log_retries else None,
            before=before_log(logger, logging.DEBUG) if config.log_retries else None,
            after=after_log(logger, logging.DEBUG) if config.log_success else None,
        )


class RateLimitAwareStrategy(RetryStrategy):
    """Rate limit aware retry strategy with longer delays for rate limit errors."""
    
    def get_retrying(self, config: RetryConfig) -> AsyncRetrying:
        """Get rate limit aware retrying strategy."""
        def wait_for_rate_limit(retry_state: RetryCallState) -> float:
            """Custom wait function that handles rate limits differently."""
            exception = retry_state.outcome.exception()
            
            # Longer delay for rate limit exceptions
            if exception and any(isinstance(exception, exc_type) for exc_type in config.rate_limit_exceptions):
                # Wait longer for rate limits (5-15 seconds)
                return random.uniform(5.0, 15.0)
            
            # Standard exponential backoff for other exceptions
            attempt = retry_state.attempt_number
            delay = min(
                config.base_delay * (config.exponential_base ** (attempt - 1)),
                config.max_delay
            )
            if config.jitter:
                delay *= random.uniform(0.5, 1.5)
            return delay
        
        return AsyncRetrying(
            stop=stop_after_attempt(config.max_attempts),
            wait=wait_for_rate_limit,
            retry=retry_if_exception_type(config.retryable_exceptions),
            before_sleep=before_sleep_log(logger, logging.WARNING) if config.log_retries else None,
            before=before_log(logger, logging.DEBUG) if config.log_retries else None,
            after=after_log(logger, logging.DEBUG) if config.log_success else None,
        )


class RetryContext:
    """Context manager for retry operations."""
    
    def __init__(self, config: Optional[RetryConfig] = None, strategy: Optional[RetryStrategy] = None):
        """
        Initialize retry context.
        
        Args:
            config: Retry configuration
            strategy: Retry strategy to use
        """
        self.config = config or RetryConfig()
        self.strategy = strategy or ExponentialBackoffStrategy()
        self.retrying = self.strategy.get_retrying(self.config)
    
    async def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """
        Call a function with retry logic.
        
        Args:
            func: Function to call
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Result of the function call
            
        Raises:
            RetryError: If all retry attempts fail
        """
        async for attempt in self.retrying:
            with attempt:
                return await func(*args, **kwargs)
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        pass


def retry_with_backoff(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    retryable_exceptions: tuple = RETRYABLE_EXCEPTIONS,
    log_retries: bool = True,
    log_success: bool = False,
) -> Callable[[F], F]:
    """
    Decorator for adding retry logic to async functions.
    
    Args:
        max_attempts: Maximum number of retry attempts
        base_delay: Base delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        exponential_base: Base for exponential backoff
        jitter: Whether to add jitter to delays
        retryable_exceptions: Tuple of exception types to retry on
        log_retries: Whether to log retry attempts
        log_success: Whether to log successful attempts
        
    Returns:
        Decorated function with retry logic
    """
    def decorator(func: F) -> F:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            config = RetryConfig(
                max_attempts=max_attempts,
                base_delay=base_delay,
                max_delay=max_delay,
                exponential_base=exponential_base,
                jitter=jitter,
                retryable_exceptions=retryable_exceptions,
                log_retries=log_retries,
                log_success=log_success,
            )
            
            async with RetryContext(config) as retry:
                return await retry.call(func, *args, **kwargs)
        
        return wrapper
    return decorator


def retry_with_rate_limit_awareness(
    max_attempts: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 120.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    retryable_exceptions: tuple = RETRYABLE_EXCEPTIONS,
    rate_limit_exceptions: tuple = RATE_LIMIT_EXCEPTIONS,
    log_retries: bool = True,
    log_success: bool = False,
) -> Callable[[F], F]:
    """
    Decorator for adding rate limit aware retry logic to async functions.
    
    Args:
        max_attempts: Maximum number of retry attempts
        base_delay: Base delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        exponential_base: Base for exponential backoff
        jitter: Whether to add jitter to delays
        retryable_exceptions: Tuple of exception types to retry on
        rate_limit_exceptions: Tuple of rate limit exception types
        log_retries: Whether to log retry attempts
        log_success: Whether to log successful attempts
        
    Returns:
        Decorated function with rate limit aware retry logic
    """
    def decorator(func: F) -> F:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            config = RetryConfig(
                max_attempts=max_attempts,
                base_delay=base_delay,
                max_delay=max_delay,
                exponential_base=exponential_base,
                jitter=jitter,
                retryable_exceptions=retryable_exceptions,
                rate_limit_exceptions=rate_limit_exceptions,
                log_retries=log_retries,
                log_success=log_success,
            )
            
            async with RetryContext(config, RateLimitAwareStrategy()) as retry:
                return await retry.call(func, *args, **kwargs)
        
        return wrapper
    return decorator


# Pre-configured retry strategies for common use cases
class RetryStrategies:
    """Pre-configured retry strategies for common use cases."""
    
    @staticmethod
    def openai() -> RetryConfig:
        """Retry configuration optimized for OpenAI API calls."""
        return RetryConfig(
            max_attempts=3,
            base_delay=1.0,
            max_delay=30.0,
            exponential_base=2.0,
            jitter=True,
            log_retries=True,
            log_success=False,
        )
    
    @staticmethod
    def elevenlabs() -> RetryConfig:
        """Retry configuration optimized for ElevenLabs API calls."""
        return RetryConfig(
            max_attempts=5,
            base_delay=2.0,
            max_delay=60.0,
            exponential_base=2.0,
            jitter=True,
            log_retries=True,
            log_success=False,
        )
    
    @staticmethod
    def llamacloud() -> RetryConfig:
        """Retry configuration optimized for LlamaCloud API calls."""
        return RetryConfig(
            max_attempts=3,
            base_delay=1.0,
            max_delay=30.0,
            exponential_base=2.0,
            jitter=True,
            log_retries=True,
            log_success=False,
        )


# Utility functions for common retry patterns
async def retry_api_call(
    func: Callable[..., T],
    *args,
    config: Optional[RetryConfig] = None,
    strategy: Optional[RetryStrategy] = None,
    **kwargs
) -> T:
    """
    Utility function for retrying API calls.
    
    Args:
        func: Function to call
        *args: Positional arguments
        config: Retry configuration
        strategy: Retry strategy
        **kwargs: Keyword arguments
        
    Returns:
        Result of the function call
    """
    async with RetryContext(config, strategy) as retry:
        return await retry.call(func, *args, **kwargs)


def is_retryable_error(exception: Exception) -> bool:
    """
    Check if an exception is retryable.
    
    Args:
        exception: The exception to check
        
    Returns:
        True if the exception is retryable, False otherwise
    """
    # Add specific logic for determining if an exception is retryable
    # This is a simple implementation - customize based on your needs
    return isinstance(exception, RETRYABLE_EXCEPTIONS)


def is_rate_limit_error(exception: Exception) -> bool:
    """
    Check if an exception is a rate limit error.
    
    Args:
        exception: The exception to check
        
    Returns:
        True if the exception is a rate limit error, False otherwise
    """
    # Add specific logic for determining if an exception is a rate limit error
    # This is a simple implementation - customize based on your needs
    return isinstance(exception, RATE_LIMIT_EXCEPTIONS) 