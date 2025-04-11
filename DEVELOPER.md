# Developer Guide

This document provides guidelines and best practices for developing and
extending the DevToolbox project.

## Driver Pattern Implementation

The DevToolbox project uses a driver pattern to implement various services
(LLM, Speech, etc.). This pattern provides a flexible and extensible way to
support multiple providers while maintaining a consistent interface.

### Core Components

1. **Base Classes**
   - `BaseConfig`: Abstract base class for provider configurations
   - `BaseProvider`: Abstract base class for provider implementations
   - `Service`: High-level service class that manages providers

2. **Registration Mechanism**
   - Decorators for automatic provider and config registration
   - Dynamic discovery of available providers
   - Environment-based configuration loading

### Implementation Steps

1. **Create Base Classes**
   ```python
   from abc import ABC, abstractmethod
   from dataclasses import dataclass, field
   from typing import Dict, Type, Optional, Any
   import logging
   import os

   logger = logging.getLogger(__name__)

   # Provider registry
   _registered_providers: Dict[str, Type] = {}
   _registered_configs: Dict[str, Type] = {}

   def register_provider(name: str):
       def decorator(cls):
           _registered_providers[name] = cls
           return cls
       return decorator

   def register_config(name: str):
       def decorator(cls):
           _registered_configs[name] = cls
           return cls
       return decorator

   @dataclass
   class BaseConfig:
       """Base configuration class with common options.
       
       This class provides the foundation for all provider configurations.
       It supports loading values from environment variables and provides
       common configuration options.
       """
       api_key: Optional[str] = field(
           default_factory=lambda: os.environ.get('API_KEY')
       )
       timeout: int = field(
           default_factory=lambda: int(
               os.environ.get('TIMEOUT', '30')
           )
       )
       max_retries: int = field(
           default_factory=lambda: int(
               os.environ.get('MAX_RETRIES', '3')
           )
       )
       proxy: Optional[str] = field(
           default_factory=lambda: os.environ.get('PROXY')
       )
       verify_ssl: bool = field(
           default_factory=lambda: os.environ.get(
               'VERIFY_SSL', 'True'
           ).lower() == 'true'
       )
       extra_params: Dict[str, Any] = None

       def __post_init__(self):
           """Initialize and validate configuration."""
           if self.extra_params is None:
               self.extra_params = {}
           self._log_config_loading()
           self._validate_config()

       def _log_config_loading(self):
           """Log configuration loading process."""
           if self.api_key:
               logger.info("API key loaded from constructor")
           elif os.environ.get('API_KEY'):
               logger.info("API key loaded from environment variable")
           else:
               logger.error("API key not found in constructor or environment")

           logger.info(f"Timeout: {self.timeout}")
           logger.info(f"Max retries: {self.max_retries}")
           logger.info(f"Verify SSL: {self.verify_ssl}")

       def _validate_config(self):
           """Validate configuration."""
           if not self.api_key:
               raise ValueError(
                   "API key is required. Set it either in constructor "
                   "or through API_KEY environment variable"
               )

   class BaseProvider(ABC):
       def __init__(self, config: BaseConfig):
           self.config = config
           self.logger = logger
   ```

2. **Implement Provider**
   ```python
   @register_config('myprovider')
   @dataclass
   class MyProviderConfig(BaseConfig):
       """Configuration for MyProvider.
       
       Environment Variables:
           MYPROVIDER_API_KEY: API key for authentication (required)
           MYPROVIDER_MODEL: Model name (default: 'default')
           MYPROVIDER_TEMPERATURE: Sampling temperature (default: 0.7)
       """
       model: str = field(
           default_factory=lambda: os.environ.get(
               'MYPROVIDER_MODEL', 'default'
           )
       )
       temperature: float = field(
           default_factory=lambda: float(
               os.environ.get('MYPROVIDER_TEMPERATURE', '0.7')
           )
       )

       def __post_init__(self):
           super().__post_init__()
           self._validate_config()

       def _log_config_loading(self):
           """Log configuration loading process."""
           super()._log_config_loading()
           if self.model == "default":
               logger.warning("Using default model")
           logger.info(f"Model: {self.model}")
           logger.info(f"Temperature: {self.temperature}")

       def _validate_config(self):
           """Validate configuration."""
           super()._validate_config()
           if not self.model:
               raise ValueError("Model name is required")

   @register_provider('MyProvider')
   class MyProvider(BaseProvider):
       """MyProvider implementation."""
       def __init__(self, config: MyProviderConfig):
           super().__init__(config)
           self.client = self._init_client()

       def _init_client(self):
           """Initialize provider client."""
           try:
               return MyProviderClient(self.config.api_key)
           except Exception as e:
               self.logger.error(f"Failed to initialize client: {e}")
               raise
   ```

3. **Create Service**
   ```python
   class MyService:
       """Service layer for managing providers.
       
       The service layer provides a high-level interface for interacting with
       different providers. It handles:
       1. Provider initialization and management
       2. Error handling and retries
       3. Rate limiting and resource management
       4. Common functionality across providers
       """
       def __init__(self, config: BaseConfig):
           self.config = config
           self.provider = self._init_provider()
           self._setup_common_handlers()

       def _setup_common_handlers(self):
           """Setup common handlers for all providers."""
           # Setup rate limiter
           self.rate_limiter = RateLimiter(
               max_requests=self.config.max_requests,
               time_window=self.config.time_window
           )
           
           # Setup retry handler
           self.retry_handler = RetryHandler(
               max_retries=self.config.max_retries,
               backoff_factor=self.config.backoff_factor
           )
           
           # Setup error handler
           self.error_handler = ErrorHandler(
               retry_on_errors=self.config.retry_on_errors,
               fallback_provider=self.config.fallback_provider
           )

       def _init_provider(self) -> BaseProvider:
           """Initialize provider based on config.
           
           The provider initialization follows a specific naming convention:
           
           1. Config Class Name -> Provider Class Name:
              - Example: `OpenAIConfig` -> `OpenAIProvider`
              - Rule: Remove 'Config' suffix and add 'Provider' suffix
              
           2. Config Class Name -> Module Name:
              - Example: `OpenAIConfig` -> `openai_provider`
              - Rule: Convert to lowercase and add '_provider' suffix
              
           3. Special Cases (PROVIDER_MODULE_NAMES mapping):
              - Example: `OpenAIConfig` -> `openai_provider`
              - This mapping handles cases where the module name doesn't follow
                the standard lowercase conversion rule
           """
           config_class_name = self.config.__class__.__name__.replace(
               'Config',
               ''
           )
           
           # Use mapping for special cases, otherwise just lowercase
           module_name = PROVIDER_MODULE_NAMES.get(
               config_class_name,
               config_class_name.lower()
           )
           
           try:
               module = importlib.import_module(
                   f'devtoolbox.mymodule.{module_name}_provider'
               )
               provider_class = getattr(module, f'{config_class_name}Provider')
               return provider_class(self.config)
           except (ImportError, AttributeError) as e:
               raise ValueError(
                   f"Failed to initialize provider for config {self.config}: {e}"
               )

       def some_method(self, *args, **kwargs):
           """Example method showing common service layer patterns."""
           try:
               # Apply rate limiting
               with self.rate_limiter:
                   # Apply retry logic
                   return self.retry_handler.execute(
                       lambda: self.provider.some_method(*args, **kwargs)
                   )
           except Exception as e:
               # Handle errors with fallback if configured
               return self.error_handler.handle_error(e)

       def another_method(self, *args, **kwargs):
           """Another example method with different error handling."""
           try:
               with self.rate_limiter:
                   return self.retry_handler.execute(
                       lambda: self.provider.another_method(*args, **kwargs)
                   )
           except Exception as e:
               # Log error but don't retry
               self.logger.error(f"Error in another_method: {e}")
               raise

       def cleanup(self):
           """Cleanup resources."""
           if hasattr(self.provider, 'cleanup'):
               self.provider.cleanup()
           self.rate_limiter.cleanup()
   ```

### Example Usage

```python
from devtoolbox.mymodule import get_config, MyService

# Get configuration
config = get_config('myprovider')

# Create service instance
service = MyService(config)

# Use the service
result = service.some_method()
```
