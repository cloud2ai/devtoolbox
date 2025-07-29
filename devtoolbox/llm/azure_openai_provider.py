"""Azure OpenAI provider implementation.

This module provides Azure OpenAI service integration, which is compatible with
OpenAI's API but requires additional Azure-specific configuration.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict
import os

from langchain_openai import AzureChatOpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log
)

from devtoolbox.llm.openai_provider import (
    OpenAIProvider,
    OpenAIConfig,
    OpenAIError,
    OpenAIRateLimitError
)
from devtoolbox.llm.provider import register_provider, register_config

logger = logging.getLogger(__name__)


# Default Azure OpenAI API settings
AZURE_API_VERSION = "2024-10-01-preview"


@register_config('azure_openai')
@dataclass
class AzureOpenAIConfig(OpenAIConfig):
    """Azure OpenAI configuration settings.

    Inherits from OpenAIConfig but adds Azure-specific settings.
    Azure OpenAI requires additional configuration like deployment name
    and API version.
    """

    # Azure specific settings
    api_key: str = field(
        default_factory=lambda: os.environ.get('AZURE_OPENAI_API_KEY')
    )
    api_base: str = field(
        default_factory=lambda: os.environ.get('AZURE_OPENAI_API_BASE')
    )
    deployment: str = field(
        default_factory=lambda: os.environ.get('AZURE_OPENAI_DEPLOYMENT')
    )
    api_version: str = field(
        default_factory=lambda: os.environ.get(
            'AZURE_OPENAI_API_VERSION',
            AZURE_API_VERSION
        )
    )
    model: str = field(
        default_factory=lambda: os.environ.get('AZURE_OPENAI_MODEL', 'gpt-4')
    )
    temperature: float = field(
        default_factory=lambda: float(
            os.environ.get('AZURE_OPENAI_TEMPERATURE', '0.7')
        )
    )
    max_tokens: int = field(
        default_factory=lambda: int(
            os.environ.get('AZURE_OPENAI_MAX_TOKENS', '2000')
        )
    )
    top_p: float = field(
        default_factory=lambda: float(
            os.environ.get('AZURE_OPENAI_TOP_P', '1.0')
        )
    )
    frequency_penalty: float = field(
        default_factory=lambda: float(
            os.environ.get('AZURE_OPENAI_FREQUENCY_PENALTY', '0.0')
        )
    )
    presence_penalty: float = field(
        default_factory=lambda: float(
            os.environ.get('AZURE_OPENAI_PRESENCE_PENALTY', '0.0')
        )
    )

    def __post_init__(self):
        """Validate configuration and log loading process."""
        self._log_config_loading()
        self._validate_config()

    def _log_config_loading(self):
        """Log configuration loading process."""
        if self.api_key:
            logger.info("Azure OpenAI API key loaded from constructor")
        elif os.environ.get('AZURE_OPENAI_API_KEY'):
            logger.info(
                "Azure OpenAI API key loaded from environment variable"
            )
        else:
            logger.error(
                "Azure OpenAI API key not found in constructor or environment"
            )

        if self.api_base:
            logger.info("Azure OpenAI API base loaded from constructor")
        elif os.environ.get('AZURE_OPENAI_API_BASE'):
            logger.info(
                "Azure OpenAI API base loaded from environment variable"
            )
        else:
            logger.error(
                "Azure OpenAI API base not found in constructor or environment"
            )

        if self.deployment:
            logger.info("Azure OpenAI deployment loaded from constructor")
        elif os.environ.get('AZURE_OPENAI_DEPLOYMENT'):
            logger.info(
                "Azure OpenAI deployment loaded from environment variable"
            )
        else:
            logger.error(
                "Azure OpenAI deployment not found in constructor or "
                "environment"
            )

        logger.info(f"API Version: {self.api_version}")
        logger.info(f"Model: {self.model}")
        logger.info(f"Temperature: {self.temperature}")
        logger.info(f"Max tokens: {self.max_tokens}")
        logger.info(f"Top P: {self.top_p}")
        logger.info(f"Frequency penalty: {self.frequency_penalty}")
        logger.info(f"Presence penalty: {self.presence_penalty}")

    def _validate_config(self):
        """Validate Azure OpenAI configuration."""
        if not self.api_key:
            raise ValueError(
                "Azure OpenAI API key is required. Set it either in "
                "constructor or through AZURE_OPENAI_API_KEY environment "
                "variable"
            )
        if not self.api_base:
            raise ValueError(
                "Azure OpenAI endpoint URL is required. Set it either in "
                "constructor or through AZURE_OPENAI_API_BASE environment "
                "variable"
            )
        if not self.deployment:
            raise ValueError(
                "Azure OpenAI deployment name is required. Set it either in "
                "constructor or through AZURE_OPENAI_DEPLOYMENT environment "
                "variable"
            )

    @classmethod
    def from_env(cls) -> 'AzureOpenAIConfig':
        """Create Azure OpenAI configuration from environment variables.

        This method is kept for backward compatibility.
        """
        logger.warning(
            "from_env() is deprecated. Configuration is now automatically "
            "loaded during initialization."
        )
        return cls()


@register_provider('AzureOpenAIProvider')
class AzureOpenAIProvider(OpenAIProvider):
    """Azure OpenAI provider implementation using LangChain.

    This implementation uses LangChain's AzureChatOpenAI class which handles
    automatic continuation of responses when they are truncated.
    """

    def __init__(self, config: AzureOpenAIConfig):
        """Initialize Azure OpenAI provider with LangChain."""
        if not isinstance(config, AzureOpenAIConfig):
            raise ValueError("Config must be an instance of AzureOpenAIConfig")

        # Initialize base provider with Azure configuration
        super().__init__(config)

        # Initialize LangChain AzureChatOpenAI client
        self.llm = AzureChatOpenAI(
            deployment_name=config.deployment,
            openai_api_version=config.api_version,
            azure_endpoint=config.api_base,
            openai_api_key=config.api_key,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            top_p=config.top_p,
            frequency_penalty=config.frequency_penalty,
            presence_penalty=config.presence_penalty
        )

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
        """Chat with Azure OpenAI API using LangChain.

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
            # Convert messages to LangChain format using inherited method
            langchain_messages = super()._convert_messages(messages)

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
            raise OpenAIError(f"Azure OpenAI API error: {e}")

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
        messages = [{"role": "user", "content": prompt}]
        return self.chat(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            *args,
            **kwargs
        )