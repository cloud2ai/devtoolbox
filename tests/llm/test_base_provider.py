"""Unit tests for base LLM provider.

This module contains comprehensive tests for the base LLM provider
implementation, focusing on testing the core functionality and common
behaviors that all providers should follow.
"""

import pytest

from devtoolbox.llm.provider import (
    BaseLLMConfig,
    BaseLLMProvider
)


class MockProvider(BaseLLMProvider):
    """Mock provider for testing base class functionality."""
    
    def complete(self, prompt, **kwargs):
        """Mock complete implementation."""
        return "Mock completion"
    
    def chat(self, messages, **kwargs):
        """Mock chat implementation."""
        return "Mock chat response"
    
    def embed(self, text, **kwargs):
        """Mock embed implementation."""
        return [0.1, 0.2, 0.3]
    
    def list_models(self):
        """Mock list_models implementation."""
        return ["mock-model-1", "mock-model-2"]


@pytest.fixture
def mock_provider():
    """Create a mock provider instance."""
    config = BaseLLMConfig(api_key="test-key")
    return MockProvider(config)


class TestBaseLLMConfig:
    """Tests for BaseLLMConfig class."""

    def test_create_config_with_defaults(self):
        """Test creating config with default values."""
        config = BaseLLMConfig()
        
        assert config.api_key is None
        assert config.timeout == 30
        assert config.max_retries == 3
        assert config.proxy is None
        assert config.verify_ssl is True
        assert config.extra_params == {}

    def test_create_config_with_custom_values(self):
        """Test creating config with custom values."""
        config = BaseLLMConfig(
            api_key="test-key",
            timeout=60,
            max_retries=5,
            proxy="http://proxy.example.com",
            verify_ssl=False,
            extra_params={"custom": "value"}
        )
        
        assert config.api_key == "test-key"
        assert config.timeout == 60
        assert config.max_retries == 5
        assert config.proxy == "http://proxy.example.com"
        assert config.verify_ssl is False
        assert config.extra_params == {"custom": "value"}

    def test_extra_params_initialization(self):
        """Test extra_params initialization."""
        config = BaseLLMConfig()
        assert config.extra_params == {}
        
        config = BaseLLMConfig(extra_params={"test": "value"})
        assert config.extra_params == {"test": "value"}


class TestBaseLLMProvider:
    """Tests for BaseLLMProvider class."""

    def test_init_with_config(self):
        """Test initialization with config."""
        config = BaseLLMConfig(api_key="test-key")
        provider = MockProvider(config)
        
        assert provider.config == config
        assert provider.logger is not None

    def test_complete_implementation(self, mock_provider):
        """Test complete method implementation."""
        response = mock_provider.complete("Test prompt")
        assert response == "Mock completion"

    def test_chat_implementation(self, mock_provider):
        """Test chat method implementation."""
        messages = [
            {"role": "user", "content": "Hello"}
        ]
        response = mock_provider.chat(messages)
        assert response == "Mock chat response"

    def test_embed_implementation(self, mock_provider):
        """Test embed method implementation."""
        embedding = mock_provider.embed("Test text")
        assert embedding == [0.1, 0.2, 0.3]

    def test_list_models_implementation(self, mock_provider):
        """Test list_models method implementation."""
        models = mock_provider.list_models()
        assert models == ["mock-model-1", "mock-model-2"]

    def test_abstract_methods(self):
        """Test that abstract methods are properly defined."""
        # 检查基类是否包含所有必要的抽象方法
        assert hasattr(BaseLLMProvider, 'complete')
        assert hasattr(BaseLLMProvider, 'chat')
        assert hasattr(BaseLLMProvider, 'embed')
        assert hasattr(BaseLLMProvider, 'list_models')
        
        # 检查这些方法是否被标记为抽象方法
        assert getattr(
            BaseLLMProvider.complete, '__isabstractmethod__', False
        )
        assert getattr(
            BaseLLMProvider.chat, '__isabstractmethod__', False
        )
        assert getattr(
            BaseLLMProvider.embed, '__isabstractmethod__', False
        )
        assert getattr(
            BaseLLMProvider.list_models, '__isabstractmethod__', False
        ) 