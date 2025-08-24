# ------------------------------------------------------------------------------
# api_clients.py
#
# Robust API client wrappers with retry logic for OpenAI and ElevenLabs.
# Implements SOLID principles with clear separation of concerns.
#
# Author: Theodore Mui
# Date: 2025-01-20
# ------------------------------------------------------------------------------

"""
Robust API Client Wrappers

This module provides robust API client wrappers for OpenAI and ElevenLabs
that include retry logic, error handling, and monitoring. It follows SOLID
principles and implements the Strategy pattern for different retry behaviors.

Features:
- Retry logic with exponential backoff
- Rate limit awareness
- Comprehensive error handling
- Logging and monitoring
- Type safety
- Clean separation of concerns

Usage:
    # OpenAI client
    openai_client = RobustOpenAIClient(api_key="your-key")
    response = await openai_client.chat_completion(messages=[...])

    # ElevenLabs client
    elevenlabs_client = RobustElevenLabsClient(api_key="your-key")
    audio = await elevenlabs_client.text_to_speech(text="Hello", voice_id="...")
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, List, Optional, Union

from elevenlabs import AsyncElevenLabs
from openai import AsyncOpenAI, OpenAI
from tenacity import RetryError

from .retry import (
    RateLimitAwareStrategy,
    RetryConfig,
    RetryContext,
    RetryStrategies,
    retry_with_rate_limit_awareness,
)

logger = logging.getLogger(__name__)


# Exception types for different API providers
class OpenAIAPIError(Exception):
    """Base exception for OpenAI API errors."""

    pass


class ElevenLabsAPIError(Exception):
    """Base exception for ElevenLabs API errors."""

    pass


class RateLimitError(Exception):
    """Exception raised when API rate limit is exceeded."""

    pass


class AuthenticationError(Exception):
    """Exception raised when API authentication fails."""

    pass


class ValidationError(Exception):
    """Raised when input validation fails."""

    pass


class QuotaExceededError(Exception):
    """Raised when API quota is exceeded."""

    pass


def parse_elevenlabs_error(response_data: Any) -> Exception:
    """
    Parse ElevenLabs API error response and return appropriate exception.

    Args:
        response_data: The error response data from ElevenLabs API

    Returns:
        Appropriate exception based on error type
    """
    if isinstance(response_data, dict):
        # Check for quota exceeded error
        if response_data.get("status_code") == 401:
            detail = response_data.get("body", {})
            if isinstance(detail, dict):
                error_detail = detail.get("detail", {})
                if (
                    isinstance(error_detail, dict)
                    and error_detail.get("status") == "quota_exceeded"
                ):
                    message = error_detail.get("message", "Quota exceeded")
                    return QuotaExceededError(f"ElevenLabs quota exceeded: {message}")

        # Check for rate limit error
        if response_data.get("status_code") == 429:
            return RateLimitError("ElevenLabs rate limit exceeded")

        # Check for authentication error
        if response_data.get("status_code") == 401:
            return AuthenticationError("ElevenLabs authentication failed")

    # Default to generic API error
    return ElevenLabsAPIError(f"ElevenLabs API error: {response_data}")


@dataclass
class APICallResult:
    """Result of an API call with metadata."""

    success: bool
    data: Any
    error: Optional[Exception] = None
    attempt_count: int = 1
    total_time: float = 0.0
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BaseAPIClient(ABC):
    """Abstract base class for API clients."""

    def __init__(self, api_key: str, config: Optional[RetryConfig] = None):
        """
        Initialize the API client.

        Args:
            api_key: API key for authentication
            config: Retry configuration
        """
        self.api_key = api_key
        self.config = config or RetryConfig()
        self._validate_api_key()

    @abstractmethod
    def _validate_api_key(self) -> None:
        """Validate the API key format."""
        pass

    @abstractmethod
    async def _make_request(self, *args, **kwargs) -> Any:
        """Make the actual API request."""
        pass

    async def call_with_retry(self, *args, **kwargs) -> APICallResult:
        """
        Make an API call with retry logic.

        Args:
            *args: Positional arguments for the API call
            **kwargs: Keyword arguments for the API call

        Returns:
            APICallResult with the result and metadata
        """
        start_time = asyncio.get_event_loop().time()
        attempt_count = 0

        try:
            async with RetryContext(self.config, RateLimitAwareStrategy()) as retry:
                attempt_count = 1
                result = await retry.call(self._make_request, *args, **kwargs)

                total_time = asyncio.get_event_loop().time() - start_time
                return APICallResult(
                    success=True,
                    data=result,
                    attempt_count=attempt_count,
                    total_time=total_time,
                    metadata={"provider": self.__class__.__name__},
                )

        except RetryError as e:
            total_time = asyncio.get_event_loop().time() - start_time
            last_exception = e.last_attempt.exception()

            logger.error(
                f"API call failed after {attempt_count} attempts: {last_exception}"
            )

            return APICallResult(
                success=False,
                data=None,
                error=last_exception,
                attempt_count=attempt_count,
                total_time=total_time,
                metadata={"provider": self.__class__.__name__},
            )
        except Exception as e:
            total_time = asyncio.get_event_loop().time() - start_time
            logger.error(f"Unexpected error in API call: {e}")

            return APICallResult(
                success=False,
                data=None,
                error=e,
                attempt_count=attempt_count,
                total_time=total_time,
                metadata={"provider": self.__class__.__name__},
            )


class RobustOpenAIClient(BaseAPIClient):
    """Robust OpenAI API client with retry logic."""

    def __init__(self, api_key: str, config: Optional[RetryConfig] = None):
        """
        Initialize the OpenAI client.

        Args:
            api_key: OpenAI API key
            config: Retry configuration (uses OpenAI-optimized defaults if None)
        """
        self.config = config or RetryStrategies.openai()
        super().__init__(api_key, self.config)
        self._client = AsyncOpenAI(api_key=api_key)

    def _validate_api_key(self) -> None:
        """Validate OpenAI API key format."""
        if not self.api_key or not self.api_key.startswith("sk-"):
            raise ValidationError(
                "Invalid OpenAI API key format. Must start with 'sk-'"
            )

    async def _make_request(self, *args, **kwargs) -> Any:
        """Make OpenAI API request."""
        # Check if this is a TTS request by looking for voice parameter
        if "voice" in kwargs:
            # This is a text-to-speech request
            # Extract text and rename to 'input' for OpenAI API
            text = kwargs.pop("text")
            kwargs["input"] = text
            return await self._client.audio.speech.create(**kwargs)
        else:
            # This is a chat completion request
            return await self._client.chat.completions.create(*args, **kwargs)

    async def chat_completion(
        self, messages: List[Dict[str, str]], model: str = "gpt-4.1-mini", **kwargs
    ) -> str:
        """
        Make a chat completion request with retry logic.

        Args:
            messages: List of message dictionaries
            model: Model to use for completion
            **kwargs: Additional parameters

        Returns:
            Response content as string

        Raises:
            OpenAIAPIError: If the request fails after all retries
            ValueError: If the query string is empty
        """
        # Validate input before retry logic
        if not messages or not isinstance(messages, list):
            raise ValueError("Messages must be a non-empty list")

        if not messages[0].get("content", "").strip():
            raise ValueError("Query string cannot be empty")

        # Call the retry function
        return await self._chat_completion_with_retry(messages, model, **kwargs)

    @retry_with_rate_limit_awareness(
        max_attempts=3, base_delay=1.0, max_delay=30.0, log_retries=True
    )
    async def _chat_completion_with_retry(
        self, messages: List[Dict[str, str]], model: str = "gpt-4.1-mini", **kwargs
    ) -> str:
        """Internal method with retry logic for chat completion."""
        try:
            result = await self.call_with_retry(
                messages=messages, model=model, **kwargs
            )

            if not result.success:
                raise OpenAIAPIError(f"OpenAI API call failed: {result.error}")

            # Extract content from response
            response = result.data
            if response.choices and response.choices[0].message.content:
                return response.choices[0].message.content
            else:
                return ""

        except OpenAIAPIError:
            # Re-raise OpenAI API errors without wrapping
            raise
        except Exception as e:
            logger.error(f"Error in chat completion: {e}")
            raise OpenAIAPIError(f"Chat completion failed: {e}") from e

    @retry_with_rate_limit_awareness(
        max_attempts=3, base_delay=1.0, max_delay=30.0, log_retries=True
    )
    async def embeddings(
        self,
        input_text: Union[str, List[str]],
        model: str = "text-embedding-3-small",
        **kwargs,
    ) -> List[float]:
        """
        Generate embeddings with retry logic.

        Args:
            input_text: Text or list of texts to embed
            model: Embedding model to use
            **kwargs: Additional parameters

        Returns:
            List of embeddings

        Raises:
            OpenAIAPIError: If the request fails after all retries
        """
        try:
            result = await self.call_with_retry(input=input_text, model=model, **kwargs)

            if not result.success:
                raise OpenAIAPIError(f"OpenAI embeddings call failed: {result.error}")

            # Extract embeddings from response
            response = result.data
            if hasattr(response, "data") and response.data:
                return [item.embedding for item in response.data]
            else:
                return []

        except OpenAIAPIError:
            # Re-raise OpenAI API errors without wrapping
            raise
        except Exception as e:
            logger.error(f"Error in embeddings: {e}")
            raise OpenAIAPIError(f"Embeddings failed: {e}") from e

    def _validate_tts_input(
        self, text: str, voice: str, speed: float, response_format: str
    ) -> None:
        """
        Validate text-to-speech input parameters.

        Args:
            text: Text to convert to speech
            voice: Voice to use
            speed: Speed of speech
            response_format: Audio format

        Raises:
            ValidationError: If validation fails
        """
        # Validate text
        if not text.strip():
            raise ValidationError("Text cannot be empty")

        if len(text) > 4096:
            logger.warning(
                f"Text too long ({len(text)} chars), truncating to 4096 chars"
            )
            text = text[:4096]

        # Validate voice options
        valid_voices = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
        if voice not in valid_voices:
            raise ValidationError(
                f"Invalid voice '{voice}'. Must be one of: {', '.join(valid_voices)}"
            )

        # Validate speed
        if not 0.25 <= speed <= 4.0:
            raise ValidationError(f"Speed must be between 0.25 and 4.0, got {speed}")

        # Validate response format
        valid_formats = ["mp3", "opus", "aac", "flac"]
        if response_format not in valid_formats:
            raise ValidationError(
                f"Invalid response_format '{response_format}'. Must be one of: {', '.join(valid_formats)}"
            )

    async def text_to_speech(
        self,
        text: str,
        voice: str = "alloy",
        model: str = "tts-1",
        response_format: str = "mp3",
        speed: float = 1.0,
        **kwargs,
    ) -> bytes:
        """
        Convert text to speech using OpenAI's TTS-1 model with retry logic.

        This method uses OpenAI's tts-1 model which provides high-quality,
        natural-sounding speech synthesis. The default voice "alloy" is
        optimized for natural conversation and clarity.

        Args:
            text: Text to convert to speech (max 4096 characters)
            voice: Voice to use. Options: "alloy", "echo", "fable", "onyx", "nova", "shimmer"
                  Default "alloy" provides natural, conversational tone
            model: TTS model to use (default: "tts-1")
            response_format: Audio format. Options: "mp3", "opus", "aac", "flac"
                           Default "mp3" for broad compatibility
            speed: Speed of speech (0.25 to 4.0). Default 1.0 for natural pace
            **kwargs: Additional parameters

        Returns:
            Audio data as bytes

        Raises:
            OpenAIAPIError: If the request fails after all retries
            ValidationError: If input validation fails
        """
        # Validate input before retry logic
        self._validate_tts_input(text, voice, speed, response_format)

        # Truncate text if needed
        if len(text) > 4096:
            text = text[:4096]

        # Call the retry function
        return await self._text_to_speech_with_retry(
            text, voice, model, response_format, speed, **kwargs
        )

    @retry_with_rate_limit_awareness(
        max_attempts=3, base_delay=1.0, max_delay=30.0, log_retries=True
    )
    async def _text_to_speech_with_retry(
        self,
        text: str,
        voice: str,
        model: str,
        response_format: str,
        speed: float,
        **kwargs,
    ) -> bytes:
        """Internal method with retry logic for text to speech."""
        try:
            result = await self.call_with_retry(
                text=text,
                voice=voice,
                model=model,
                response_format=response_format,
                speed=speed,
                **kwargs,
            )

            if not result.success:
                raise OpenAIAPIError(f"OpenAI TTS call failed: {result.error}")

            # Return the audio data directly
            return result.data.content

        except OpenAIAPIError:
            # Re-raise OpenAI API errors without wrapping
            raise
        except Exception as e:
            logger.error(f"Error in text to speech: {e}")
            raise OpenAIAPIError(f"Text to speech failed: {e}") from e


class RobustElevenLabsClient(BaseAPIClient):
    """Robust ElevenLabs API client with retry logic."""

    def __init__(self, api_key: str, config: Optional[RetryConfig] = None):
        """
        Initialize the ElevenLabs client.

        Args:
            api_key: ElevenLabs API key
            config: Retry configuration (uses ElevenLabs-optimized defaults if None)
        """
        self.config = config or RetryStrategies.elevenlabs()
        super().__init__(api_key, self.config)
        self._client = AsyncElevenLabs(api_key=api_key)

    def _validate_api_key(self) -> None:
        """Validate ElevenLabs API key format."""
        if not self.api_key or not self.api_key.startswith("sk_"):
            raise ValidationError(
                "Invalid ElevenLabs API key format. Must start with 'sk_'"
            )

    async def _make_request(self, *args, **kwargs) -> Any:
        """Make ElevenLabs API request."""
        # For text_to_speech, call the convert method
        return self._client.text_to_speech.convert(*args, **kwargs)

    def estimate_credits(self, text: str, model_id: str = "eleven_turbo_v2_5") -> int:
        """
        Estimate the number of credits required for text-to-speech conversion.

        Args:
            text: Text to convert to speech
            model_id: Model ID to use

        Returns:
            Estimated number of credits required
        """
        # ElevenLabs credit calculation:
        # - 1 credit per 1,000 characters for most models
        # - Some models may have different rates
        char_count = len(text)

        if model_id == "eleven_turbo_v2_5":
            # Turbo model: 1 credit per 1,000 characters
            return max(1, (char_count + 999) // 1000)
        elif model_id == "eleven_multilingual_v2":
            # Multilingual model: 1 credit per 1,000 characters
            return max(1, (char_count + 999) // 1000)
        else:
            # Default: 1 credit per 1,000 characters
            return max(1, (char_count + 999) // 1000)

    async def check_credits_available(self, required_credits: int) -> bool:
        """
        Check if the user has enough credits available.

        Args:
            required_credits: Number of credits required

        Returns:
            True if enough credits are available, False otherwise
        """
        try:
            # Get user info which includes credit balance
            user_info = await self._client.user.subscription.get()
            available_credits = user_info.character_count

            return available_credits >= required_credits
        except Exception as e:
            logger.warning(f"Could not check credit balance: {e}")
            # If we can't check, assume it's available and let the API handle it
            return True

    def text_to_speech(
        self,
        text: str,
        voice_id: str,
        output_format: str = "mp3_22050_32",
        model_id: str = "eleven_turbo_v2_5",
        **kwargs,
    ) -> AsyncIterator[bytes]:
        """
        Convert text to speech.

        Note: This method returns an async generator directly without retry logic
        because async generators are difficult to retry properly. The underlying
        ElevenLabs client handles retries internally.

        Args:
            text: Text to convert to speech
            voice_id: Voice ID to use
            output_format: Audio output format
            model_id: Model ID to use
            **kwargs: Additional parameters

        Returns:
            Async iterator yielding audio chunks

        Raises:
            ElevenLabsAPIError: If the request fails
            QuotaExceededError: If the quota is exceeded
            ValidationError: If input validation fails
        """
        # Validate input
        if not text.strip():
            raise ValidationError("Text cannot be empty")

        if len(text) > 5000:
            logger.warning(
                f"Text too long ({len(text)} chars), truncating to 5000 chars"
            )
            text = text[:5000]

        # Estimate required credits
        required_credits = self.estimate_credits(text, model_id)
        logger.info(
            f"Estimated credits required: {required_credits} for {len(text)} characters"
        )

        try:
            # Return the async generator directly from the convert method
            return self._client.text_to_speech.convert(
                voice_id=voice_id,
                text=text,
                output_format=output_format,
                model_id=model_id,
                **kwargs,
            )
        except Exception as e:
            logger.error(f"ElevenLabs TTS failed: {e}")

            # Try to parse the error for specific error types
            try:
                # If the error has response data, try to parse it
                if hasattr(e, "response") and e.response is not None:
                    parsed_error = parse_elevenlabs_error(e.response)
                    raise parsed_error
                elif hasattr(e, "body") and e.body is not None:
                    # Some errors might have the response data in the body
                    parsed_error = parse_elevenlabs_error(e.body)
                    raise parsed_error
                else:
                    # Fallback to generic error
                    raise ElevenLabsAPIError(f"Text to speech failed: {e}") from e
            except (QuotaExceededError, RateLimitError, AuthenticationError):
                # Re-raise specific errors
                raise
            except Exception:
                # If parsing fails, raise the original error
                raise ElevenLabsAPIError(f"Text to speech failed: {e}") from e

    @retry_with_rate_limit_awareness(
        max_attempts=3, base_delay=1.0, max_delay=30.0, log_retries=True
    )
    async def get_voices(self) -> List[Dict[str, Any]]:
        """
        Get available voices with retry logic.

        Returns:
            List of available voices

        Raises:
            ElevenLabsAPIError: If the request fails after all retries
        """
        try:
            result = await self.call_with_retry()

            if not result.success:
                raise ElevenLabsAPIError(
                    f"ElevenLabs voices call failed: {result.error}"
                )

            return result.data

        except Exception as e:
            logger.error(f"Error getting voices: {e}")
            raise ElevenLabsAPIError(f"Get voices failed: {e}") from e


# Factory functions for easy client creation
def create_openai_client(
    api_key: str, config: Optional[RetryConfig] = None
) -> RobustOpenAIClient:
    """
    Create a robust OpenAI client.

    Args:
        api_key: OpenAI API key
        config: Optional retry configuration

    Returns:
        RobustOpenAIClient instance
    """
    return RobustOpenAIClient(api_key, config)


def create_elevenlabs_client(
    api_key: str, config: Optional[RetryConfig] = None
) -> RobustElevenLabsClient:
    """
    Create a robust ElevenLabs client.

    Args:
        api_key: ElevenLabs API key
        config: Optional retry configuration

    Returns:
        RobustElevenLabsClient instance
    """
    return RobustElevenLabsClient(api_key, config)


# Utility functions for common operations
async def check_api_connection(client: BaseAPIClient) -> bool:
    """
    Check if an API client can connect successfully.

    Args:
        client: API client to test

    Returns:
        True if connection successful, False otherwise
    """
    try:
        # Check client type by class name to handle mocks
        client_type = client.__class__.__name__

        if client_type == "RobustOpenAIClient":
            # Test with a simple completion
            await client.chat_completion(
                messages=[{"role": "user", "content": "Hello"}],
                model="gpt-4.1-mini",
                max_tokens=5,
            )
        elif client_type == "RobustElevenLabsClient":
            # Test by getting voices
            await client.get_voices()
        else:
            return False

        return True

    except Exception as e:
        logger.error(f"API connection test failed: {e}")
        return False


async def get_elevenlabs_credit_info(api_key: str) -> Dict[str, Any]:
    """
    Get ElevenLabs credit information for the user.

    Args:
        api_key: ElevenLabs API key

    Returns:
        Dictionary with credit information
    """
    try:
        from elevenlabs import AsyncElevenLabs

        client = AsyncElevenLabs(api_key=api_key)
        user_info = await client.user.subscription.get()

        return {
            "available_credits": user_info.character_count,
            "plan_type": user_info.tier,
            "next_reset": user_info.next_character_count_reset_unix,
            "status": "success",
        }
    except Exception as e:
        logger.error(f"Failed to get ElevenLabs credit info: {e}")
        return {
            "available_credits": None,
            "plan_type": None,
            "next_reset": None,
            "status": "error",
            "error": str(e),
        }


def get_client_status(client: BaseAPIClient) -> Dict[str, Any]:
    """
    Get status information about an API client.

    Args:
        client: API client to get status for

    Returns:
        Dictionary with status information
    """
    return {
        "type": client.__class__.__name__,
        "api_key_configured": bool(client.api_key),
        "retry_config": {
            "max_attempts": client.config.max_attempts,
            "base_delay": client.config.base_delay,
            "max_delay": client.config.max_delay,
            "jitter": client.config.jitter,
        },
    }
