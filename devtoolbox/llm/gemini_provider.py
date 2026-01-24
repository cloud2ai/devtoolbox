"""Google Gemini provider implementation using LangChain.

This module provides an implementation of the Google Gemini provider using LangChain.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log
)

from devtoolbox.llm.provider import (
    BaseLLMConfig,
    BaseLLMProvider,
    register_provider,
    register_config,
)

logger = logging.getLogger(__name__)


@register_config('gemini')
@dataclass
class GeminiConfig(BaseLLMConfig):
    """Google Gemini configuration settings.

    This class automatically loads configuration from environment variables
    if not provided during initialization. Required parameters must be set
    either through constructor or environment variables.
    """

    api_key: str = field(
        default_factory=lambda: os.environ.get('GOOGLE_API_KEY')
    )
    model: str = field(
        default_factory=lambda: os.environ.get('GEMINI_MODEL', 'gemini-pro')
    )
    temperature: float = field(
        default_factory=lambda: float(
            os.environ.get('GEMINI_TEMPERATURE', '0.7')
        )
    )
    max_tokens: int = field(
        default_factory=lambda: int(
            os.environ.get('GEMINI_MAX_TOKENS', '60000')
        )
    )
    top_p: float = field(
        default_factory=lambda: float(
            os.environ.get('GEMINI_TOP_P', '1.0')
        )
    )
    top_k: int = field(
        default_factory=lambda: int(
            os.environ.get('GEMINI_TOP_K', '40')
        )
    )

    def __post_init__(self):
        """Validate configuration and log loading process."""
        self._log_config_loading()
        self._validate_config()

    def _log_config_loading(self):
        """Log configuration loading process."""
        if not self.api_key and not os.environ.get('GOOGLE_API_KEY'):
            logger.error(
                "Google Gemini API key not found in constructor or environment"
            )

        logger.debug(
            f"Google Gemini initialized: model={self.model}, "
            f"temperature={self.temperature}"
        )

    def _validate_config(self):
        """Validate Google Gemini configuration."""
        if not self.api_key:
            raise ValueError(
                "Google Gemini API key is required. Set it either in constructor "
                "or through GOOGLE_API_KEY environment variable"
            )

    @classmethod
    def from_env(cls) -> 'GeminiConfig':
        """Create Google Gemini configuration from environment variables.

        This method is kept for backward compatibility.
        """
        logger.warning(
            "from_env() is deprecated. Configuration is now automatically "
            "loaded during initialization."
        )
        return cls()


class GeminiError(Exception):
    """Base exception for Google Gemini-related errors."""
    pass


class GeminiRateLimitError(GeminiError):
    """Raised when Google Gemini rate limit is exceeded."""
    pass


@register_provider('GeminiProvider')
class GeminiProvider(BaseLLMProvider):
    """Google Gemini provider implementation using LangChain.

    This implementation uses LangChain's ChatGoogleGenerativeAI class which
    provides integration with Google's Gemini models.
    """

    def __init__(self, config: GeminiConfig):
        """Initialize Google Gemini provider with LangChain."""
        logger.debug(
            f"Initializing Google Gemini provider: model={config.model}, "
            f"temperature={config.temperature}"
        )
        super().__init__(config)
        self.config = config
        if not config.api_key:
            config.api_key = os.environ.get('GOOGLE_API_KEY')
        if not config.api_key:
            raise ValueError(
                "Google Gemini API key is required. Set it either in constructor "
                "or through GOOGLE_API_KEY environment variable"
            )

        # Initialize LangChain ChatGoogleGenerativeAI client
        self.llm = ChatGoogleGenerativeAI(
            model=config.model,
            temperature=config.temperature,
            max_output_tokens=config.max_tokens,
            google_api_key=config.api_key,
            top_p=config.top_p,
            top_k=config.top_k
        )

    def _convert_messages(self, messages: List[Dict[str, str]]) -> List[Any]:
        """Convert message dictionaries to LangChain message objects.

        Args:
            messages: List of message dictionaries with 'role' and 'content'

        Returns:
            List of LangChain message objects
        """
        converted_messages = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                converted_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                converted_messages.append(AIMessage(content=content))
            elif role == "system":
                converted_messages.append(SystemMessage(content=content))
            else:
                logger.warning(f"Unknown message role: {role}")
                converted_messages.append(HumanMessage(content=content))
        return converted_messages

    @retry(
        retry=retry_if_exception_type(GeminiRateLimitError),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=30),
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )
    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        *args,
        **kwargs
    ) -> str:
        """Chat with Google Gemini API using LangChain.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            str: Model's response text with full metadata

        Raises:
            GeminiRateLimitError: If rate limit is exceeded
            GeminiError: If any other error occurs
        """
        try:
            # Convert messages to LangChain format
            langchain_messages = self._convert_messages(messages)

            # Create a new instance with updated parameters if specified
            llm = self.llm
            if max_tokens is not None or temperature is not None:
                llm = ChatGoogleGenerativeAI(
                    model=self.config.model,
                    temperature=temperature if temperature is not None else self.config.temperature,
                    max_output_tokens=max_tokens if max_tokens is not None else self.config.max_tokens,
                    google_api_key=self.config.api_key,
                    top_p=self.config.top_p,
                    top_k=self.config.top_k
                )

            # Get response from LangChain and return directly
            response = llm.invoke(langchain_messages)
            return response

        except Exception as e:
            if "rate_limit" in str(e).lower() or "quota" in str(e).lower():
                raise GeminiRateLimitError("Rate limit exceeded")
            raise GeminiError(f"Google Gemini API error: {str(e)}")

    def complete(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        *args,
        **kwargs
    ) -> str:
        """Complete text using chat API.

        This is a compatibility method that converts a simple prompt to
        chat format and uses the chat API.

        Args:
            prompt: Text prompt to complete
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            str: Completed text

        Raises:
            GeminiError: If completion fails
        """
        logger.debug(f"Text completion: {len(prompt)} chars")

        messages = [{"role": "user", "content": prompt}]
        try:
            response = self.chat(
                messages,
                max_tokens=max_tokens,
                temperature=temperature,
                *args,
                **kwargs
            )
            logger.debug(f"Response: {len(response)} chars")
            return response
        except Exception as e:
            logger.error(f"Text completion failed: {str(e)}")
            logger.error(f"Failed prompt: {prompt}")
            logger.error(
                f"Parameters - max_tokens: {max_tokens}, "
                f"temperature: {temperature}"
            )
            raise GeminiError(f"Google Gemini API error: {str(e)}")

    def embed(self, text: str) -> List[float]:
        """Get embeddings from Google Gemini API.

        Note: Google Gemini does not have a dedicated embedding endpoint.
        This method raises NotImplementedError as embeddings are typically
        handled through other Google services like Vertex AI.

        Args:
            text: Text to generate embeddings for

        Returns:
            List[float]: The embeddings for the text

        Raises:
            NotImplementedError: Embeddings not directly supported
        """
        raise NotImplementedError(
            "Google Gemini does not provide a direct embedding endpoint. "
            "Please use Vertex AI Embeddings API or other embedding services."
        )

    def list_models(self) -> List[str]:
        """List available Google Gemini models.

        Returns:
            List[str]: List of available model names

        Raises:
            GeminiError: If model listing fails
        """
        # Common Gemini models
        common_models = [
            "gemini-pro",
            "gemini-pro-vision",
            "gemini-1.5-pro",
            "gemini-1.5-flash",
            "gemini-2.0-flash-exp",
        ]
        try:
            logger.info("Returning common Gemini models")
            return common_models
        except Exception as e:
            raise GeminiError(f"Failed to list models: {str(e)}")
