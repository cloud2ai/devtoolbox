"""LLM provider module.

This module provides the base classes and interfaces for LLM providers.
It defines the common interface that all LLM providers must implement.

To extend this module with a new LLM provider, follow these steps:

1. Create a new provider module:
   - Create a new file named `{provider_name}_provider.py` in this directory
   - The file should contain both the provider class and its config class

2. Implement the config class:
   - Create a dataclass that extends LLMConfig
   - Add provider-specific configuration options
   - Example:
     @dataclass
     class NewProviderConfig(LLMConfig):
         base_url: str = "https://api.newprovider.com/v1"
         default_model: str = "new-model"
         custom_option: str = "default_value"

3. Implement the provider class:
   - Create a class that extends BaseProvider
   - Implement all abstract methods
   - Add provider-specific methods if needed
   - Example:
     class NewProvider(BaseProvider):
         def __init__(self, config: NewProviderConfig):
             super().__init__(config)
             self.client = NewProviderClient(config)

         def complete(self, prompt: str, **kwargs) -> str:
             # Implementation here
             pass

         def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
             # Implementation here
             pass

         def embed(self, text: str, **kwargs) -> List[float]:
             # Implementation here
             pass

         def list_models(self) -> List[str]:
             # Implementation here
             pass

4. Error handling:
   - Use the logger for error logging
   - Raise appropriate exceptions
   - Example:
     try:
         response = self.client.chat(messages)
     except Exception as e:
         self.logger.error(f"Chat failed: {str(e)}")
         raise

5. Testing:
   - Create tests in the tests directory
   - Test both success and error cases
   - Mock external API calls

6. Documentation:
   - Add docstrings to all classes and methods
   - Document configuration options
   - Provide usage examples

The provider will be automatically discovered and registered by the
LLMService when the module is imported.

Base provider for multiple LLM engines.

This module provides the base classes and interfaces for implementing
different LLM providers. To create a new provider:

1. Create a configuration class that inherits from BaseLLMConfig
2. Create a provider class that inherits from BaseLLMProvider
3. Implement the required methods
4. Use the register_provider and register_config decorators to register them

Example:
    @register_config('openai')
    class OpenAIConfig(BaseLLMConfig):
        api_key: str
        model: str = "gpt-3.5-turbo"

    @register_provider('OpenAIProvider')
    class OpenAIProvider(BaseLLMProvider):
        def __init__(self, config: OpenAIConfig):
            super().__init__(config)
            self.config = config
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any, Type
import os
import json
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# Provider registry
_registered_providers: Dict[str, type] = {}
_registered_configs: Dict[str, type] = {}

def register_provider(name: str):
    """Decorator to register an LLM provider.

    Args:
        name: The name of the provider to register.
    """
    def decorator(cls):
        _registered_providers[name] = cls
        return cls
    return decorator

def register_config(name: str):
    """Decorator to register an LLM config.

    Args:
        name: The name of the config to register.
    """
    def decorator(cls):
        _registered_configs[name] = cls
        return cls
    return decorator

@dataclass
class BaseLLMConfig:
    """Base configuration for LLM providers.

    This class defines the common configuration options for all LLM
    providers. Each provider can extend this class to add its own
    specific configuration options.

    Attributes:
        api_key (Optional[str]): API key for authentication.
        timeout (int): Request timeout in seconds.
        max_retries (int): Maximum number of retries for failed requests.
        proxy (Optional[str]): Proxy URL if needed.
        verify_ssl (bool): Whether to verify SSL certificates.
        extra_params (Dict[str, Any]): Additional provider-specific
            parameters.
    """
    api_key: Optional[str] = None
    timeout: int = 30
    max_retries: int = 3
    proxy: Optional[str] = None
    verify_ssl: bool = True
    extra_params: Dict[str, Any] = None

    def __post_init__(self):
        """Initialize extra parameters."""
        if self.extra_params is None:
            self.extra_params = {}


class BaseLLMProvider(ABC):
    """Base class for LLM providers.

    This class defines the interface that all LLM providers must implement.
    It provides a common interface for different LLM services like OpenAI,
    DeepSeek, etc.

    The implementation should handle:
    - Text completion
    - Chat completion
    - Embedding generation
    - Model management

    When implementing a new provider, make sure to:
    1. Extend this class
    2. Implement all abstract methods
    3. Handle errors appropriately
    4. Use the logger for logging
    5. Document all methods and parameters
    """

    def __init__(self, config: BaseLLMConfig):
        """Initialize the provider.

        Args:
            config (BaseLLMConfig): Configuration for the provider.
        """
        self.config = config
        self.logger = logger

    @abstractmethod
    def complete(self, prompt, **kwargs):
        """Generate text completion for the given prompt.

        This method should:
        1. Handle API calls to the LLM service
        2. Process and return the completion
        3. Handle errors and retries
        4. Log important events

        Args:
            prompt (str): The prompt to complete.
            **kwargs: Additional arguments for the completion.

        Returns:
            str: The completed text.

        Raises:
            Exception: If the completion fails.
        """
        raise NotImplementedError(
            "complete() method needs to be implemented"
        )

    @abstractmethod
    def chat(self, messages, **kwargs):
        """Generate chat completion for the given messages.

        This method should:
        1. Handle API calls to the LLM service
        2. Process and return the chat response
        3. Handle errors and retries
        4. Log important events

        Args:
            messages (list): List of message dictionaries with 'role' and
                'content' keys.
            **kwargs: Additional arguments for the chat completion.

        Returns:
            str: The chat completion response.

        Raises:
            Exception: If the chat completion fails.
        """
        raise NotImplementedError("chat() method needs to be implemented")

    @abstractmethod
    def embed(self, text, **kwargs):
        """Generate embeddings for the given text.

        This method should:
        1. Handle API calls to the LLM service
        2. Process and return the embeddings
        3. Handle errors and retries
        4. Log important events

        Args:
            text (str): The text to generate embeddings for.
            **kwargs: Additional arguments for the embedding generation.

        Returns:
            List[float]: The embeddings for the text.

        Raises:
            Exception: If the embedding generation fails.
        """
        raise NotImplementedError("embed() method needs to be implemented")

    @abstractmethod
    def list_models(self):
        """List available models for the provider.

        This method should:
        1. Handle API calls to the LLM service
        2. Process and return the list of models
        3. Handle errors and retries
        4. Log important events

        Returns:
            List[str]: List of available model names.

        Raises:
            Exception: If the model listing fails.
        """
        raise NotImplementedError("list_models() method needs to be implemented")