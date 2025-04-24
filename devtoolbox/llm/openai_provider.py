"""OpenAI provider implementation using LangChain.

This module provides an implementation of the OpenAI provider using LangChain.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import os

from langchain_openai import ChatOpenAI
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


@register_config('openai')
@dataclass
class OpenAIConfig(BaseLLMConfig):
    """OpenAI configuration settings.

    This class automatically loads configuration from environment variables
    if not provided during initialization. Required parameters must be set
    either through constructor or environment variables.
    """

    api_key: str = field(
        default_factory=lambda: os.environ.get('OPENAI_API_KEY')
    )
    api_base: Optional[str] = field(
        default_factory=lambda: os.environ.get('OPENAI_API_BASE')
    )
    model: str = field(
        default_factory=lambda: os.environ.get('OPENAI_MODEL', 'gpt-4o-mini')
    )
    temperature: float = field(
        default_factory=lambda: float(
            os.environ.get('OPENAI_TEMPERATURE', '0.7')
        )
    )
    max_tokens: int = field(
        default_factory=lambda: int(
            os.environ.get('OPENAI_MAX_TOKENS', '8000')
        )
    )
    top_p: float = field(
        default_factory=lambda: float(
            os.environ.get('OPENAI_TOP_P', '1.0')
        )
    )
    frequency_penalty: float = field(
        default_factory=lambda: float(
            os.environ.get('OPENAI_FREQUENCY_PENALTY', '0.0')
        )
    )
    presence_penalty: float = field(
        default_factory=lambda: float(
            os.environ.get('OPENAI_PRESENCE_PENALTY', '0.0')
        )
    )

    def __post_init__(self):
        """Validate configuration and log loading process."""
        self._log_config_loading()
        self._validate_config()

    def _log_config_loading(self):
        """Log configuration loading process."""
        if self.api_key:
            logger.info("OpenAI API key loaded from constructor")
        elif os.environ.get('OPENAI_API_KEY'):
            logger.info("OpenAI API key loaded from environment variable")
        else:
            logger.error(
                "OpenAI API key not found in constructor or environment"
            )

        if self.api_base:
            logger.info("OpenAI API base loaded from constructor")
        elif os.environ.get('OPENAI_API_BASE'):
            logger.info("OpenAI API base loaded from environment variable")
        else:
            logger.info("OpenAI API base not set, using default")

        logger.info(f"Model: {self.model}")
        logger.info(f"Temperature: {self.temperature}")
        logger.info(f"Max tokens: {self.max_tokens}")
        logger.info(f"Top P: {self.top_p}")
        logger.info(f"Frequency penalty: {self.frequency_penalty}")
        logger.info(f"Presence penalty: {self.presence_penalty}")

    def _validate_config(self):
        """Validate OpenAI configuration."""
        if not self.api_key:
            raise ValueError(
                "OpenAI API key is required. Set it either in constructor "
                "or through OPENAI_API_KEY environment variable"
            )

    @classmethod
    def from_env(cls) -> 'OpenAIConfig':
        """Create OpenAI configuration from environment variables.

        This method is kept for backward compatibility.
        """
        logger.warning(
            "from_env() is deprecated. Configuration is now automatically "
            "loaded during initialization."
        )
        return cls()


class OpenAIError(Exception):
    """Base exception for OpenAI-related errors."""
    pass


class OpenAIRateLimitError(OpenAIError):
    """Raised when OpenAI rate limit is exceeded."""
    pass


@register_provider('OpenAIProvider')
class OpenAIProvider(BaseLLMProvider):
    """OpenAI provider implementation using LangChain.

    This implementation uses LangChain's ChatOpenAI class which handles
    automatic continuation of responses when they are truncated.
    """

    def __init__(self, config: OpenAIConfig):
        """Initialize OpenAI provider with LangChain."""
        logger.info(f"Initializing OpenAI provider with config: {config}")
        super().__init__(config)
        self.config = config
        if not config.api_key:
            config.api_key = os.environ.get('OPENAI_API_KEY')
        if not config.api_key:
            raise ValueError(
                "OpenAI API key is required. Set it either in constructor "
                "or through OPENAI_API_KEY environment variable"
            )

        # Initialize LangChain ChatOpenAI client
        self.llm = ChatOpenAI(
            model_name=config.model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            openai_api_key=config.api_key,
            openai_api_base=config.api_base,
            top_p=config.top_p,
            frequency_penalty=config.frequency_penalty,
            presence_penalty=config.presence_penalty
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
        retry=retry_if_exception_type(OpenAIRateLimitError),
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
        """Chat with OpenAI API using LangChain.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            str: Model's response text

        Raises:
            OpenAIRateLimitError: If rate limit is exceeded
            OpenAIError: If any other error occurs
        """
        try:
            # Convert messages to LangChain format
            langchain_messages = self._convert_messages(messages)

            # Update model parameters if specified
            if max_tokens is not None:
                self.llm.max_tokens = max_tokens
            if temperature is not None:
                self.llm.temperature = temperature

            # Get response from LangChain
            response = self.llm.invoke(langchain_messages)
            return response.content.strip()

        except Exception as e:
            if "rate_limit" in str(e).lower():
                raise OpenAIRateLimitError("Rate limit exceeded")
            raise OpenAIError(f"OpenAI API error: {str(e)}")

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
            OpenAIError: If completion fails
        """
        logger.info("Starting text completion")
        logger.debug(f"Input prompt: {prompt}")
        logger.debug(
            f"Parameters - max_tokens: {max_tokens}, "
            f"temperature: {temperature}"
        )
        logger.debug(f"Additional kwargs: {kwargs}")

        messages = [{"role": "user", "content": prompt}]
        try:
            logger.info("Converting prompt to chat format")
            response = self.chat(
                messages,
                max_tokens=max_tokens,
                temperature=temperature,
                *args,
                **kwargs
            )
            logger.debug(f"Received response: {response}")
            logger.info("Text completion completed successfully")
            return response
        except Exception as e:
            logger.error(f"Text completion failed: {str(e)}")
            logger.error(f"Failed prompt: {prompt}")
            logger.error(
                f"Parameters - max_tokens: {max_tokens}, "
                f"temperature: {temperature}"
            )
            raise OpenAIError(f"OpenAI API error: {str(e)}")

    def embed(self, text: str) -> List[float]:
        """Get embeddings from OpenAI API."""
        try:
            response = self.llm.client.embeddings.create(
                model="text-embedding-ada-002",
                input=text
            )
            return response.data[0].embedding

        except Exception as e:
            raise OpenAIError(f"OpenAI API error: {str(e)}")

    def list_models(self) -> List[str]:
        """List available OpenAI models."""
        try:
            response = self.llm.client.models.list()
            return [model.id for model in response.data]

        except Exception as e:
            raise OpenAIError(f"OpenAI API error: {str(e)}")
