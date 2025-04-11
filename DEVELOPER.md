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
   from typing import Dict, Type

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

   class BaseConfig(ABC):
       @classmethod
       @abstractmethod
       def from_env(cls) -> 'BaseConfig':
           pass

       @abstractmethod
       def validate(self):
           pass

   class BaseProvider(ABC):
       def __init__(self, config: BaseConfig):
           self.config = config
   ```

2. **Implement Provider**
   ```python
   @register_config('myprovider')
   class MyProviderConfig(BaseConfig):
       api_key: str
       model: str = "default"

       @classmethod
       def from_env(cls) -> 'MyProviderConfig':
           return cls(
               api_key=os.environ.get('MYPROVIDER_API_KEY'),
               model=os.environ.get('MYPROVIDER_MODEL', 'default')
           )

       def validate(self):
           if not self.api_key:
               raise ValueError("API key is required")

   @register_provider('MyProvider')
   class MyProvider(BaseProvider):
       def __init__(self, config: MyProviderConfig):
           super().__init__(config)
           self.api_key = config.api_key
   ```

3. **Create Service**
   ```python
   class MyService:
       def __init__(self, config: BaseConfig):
           self.config = config
           self.provider = self._init_provider()

       def _init_provider(self) -> BaseProvider:
           provider_class = _registered_providers.get(
               self.config.__class__.__name__.replace('Config', 'Provider')
           )
           if not provider_class:
               raise ValueError("No provider found for config")
           return provider_class(self.config)
   ```

4. **Update __init__.py**
   ```python
   from .provider import (
       BaseConfig,
       BaseProvider,
       _registered_providers,
       _registered_configs,
   )
   from .service import MyService

   def get_config(provider_name: str) -> BaseConfig:
       config_class = _registered_configs.get(provider_name.lower())
       if not config_class:
           raise ValueError(f"Invalid provider: {provider_name}")
       return config_class.from_env()

   __all__ = [
       'BaseConfig',
       'BaseProvider',
       'MyService',
       'get_config',
       *_registered_providers.keys(),
       *_registered_configs.keys(),
   ]
   ```

### Best Practices

1. **Configuration**
   - Use environment variables for sensitive data
   - Provide sensible defaults
   - Validate required parameters

2. **Error Handling**
   - Define custom exceptions for provider-specific errors
   - Implement retry mechanisms for transient failures
   - Provide clear error messages

3. **Documentation**
   - Document all public methods and classes
   - Include usage examples
   - Explain configuration options

4. **Testing**
   - Mock external API calls
   - Test error handling
   - Verify configuration validation

### Example Usage

```python
from devtoolbox.mymodule import get_config, MyService

# Get configuration from environment variables
config = get_config('myprovider')

# Create service instance
service = MyService(config)

# Use the service
result = service.some_method()
```

### Adding New Providers

1. Create a new config class with `@register_config`
2. Create a new provider class with `@register_provider`
3. Implement required methods
4. Add tests
5. Update documentation

This pattern allows for easy addition of new providers while maintaining a
consistent interface and reducing code duplication.