"""Unit tests for DeepSeek provider.

This module contains comprehensive tests for the DeepSeek provider
implementation, focusing on testing the logic of each method using mocks.
"""

from unittest.mock import MagicMock, patch
import pytest
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from tenacity import RetryError

from devtoolbox.llm.deepseek_provider import (
    DeepSeekConfig,
    DeepSeekProvider
)
from devtoolbox.llm.openai_provider import (
    OpenAIError,
    OpenAIRateLimitError
)


@pytest.fixture
def mock_retry():
    """Mock the retry decorator to simulate a single retry attempt."""
    with patch('devtoolbox.llm.openai_provider.retry') as mock:
        mock.side_effect = (
            lambda *args, **kwargs: lambda f: f
        )
        yield mock


@pytest.fixture
def mock_chat_openai():
    """Mock the ChatOpenAI class."""
    with patch('devtoolbox.llm.openai_provider.ChatOpenAI') as mock:
        mock_instance = MagicMock()
        mock.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def deepseek_provider(mock_chat_openai):
    """Create a DeepSeekProvider instance with mocked dependencies."""
    config = DeepSeekConfig(api_key="test-key")
    return DeepSeekProvider(config)


class TestDeepSeekConfig:
    """Tests for DeepSeekConfig class."""

    def test_create_config_with_defaults(self, monkeypatch):
        """Test creating config with default values from environment"""
        monkeypatch.setenv('DEEPSEEK_API_KEY', 'test-key')
        monkeypatch.setenv('DEEPSEEK_MODEL', 'deepseek-chat')
        monkeypatch.setenv('DEEPSEEK_TEMPERATURE', '0.5')
        
        config = DeepSeekConfig()
        
        assert config.api_key == 'test-key'
        assert config.model == 'deepseek-chat'
        assert config.temperature == 0.5
        assert config.max_tokens == 2000  # default value

    def test_config_validation_error(self):
        """Test validation error when API key is missing"""
        with pytest.raises(ValueError, match="DeepSeek API key is required"):
            DeepSeekConfig(api_key="")._validate_config()

    def test_deprecated_from_env_warning(self, caplog):
        """Test from_env() deprecation warning"""
        DeepSeekConfig.from_env()
        assert "from_env() is deprecated" in caplog.text


class TestDeepSeekProvider:
    """Tests for DeepSeekProvider class using LangChain."""

    def test_convert_messages(self, deepseek_provider):
        """Test message conversion to LangChain format"""
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
            {"role": "unknown", "content": "Test"}
        ]
        
        converted = deepseek_provider._convert_messages(messages)
        
        assert len(converted) == 4
        assert isinstance(converted[0], SystemMessage)
        assert isinstance(converted[1], HumanMessage)
        assert isinstance(converted[2], AIMessage)
        assert isinstance(
            converted[3], HumanMessage
        )  # unknown roles default to HumanMessage

    def test_chat_with_parameters(self, deepseek_provider, mock_chat_openai):
        """Test chat with custom parameters"""
        mock_response = AIMessage(content="Test response")
        deepseek_provider.llm.invoke.return_value = mock_response

        messages = [{"role": "user", "content": "Hello"}]
        response = deepseek_provider.chat(
            messages,
            max_tokens=100,
            temperature=0.8
        )

        assert response == "Test response"
        assert deepseek_provider.llm.max_tokens == 100
        assert deepseek_provider.llm.temperature == 0.8

    def test_chat_rate_limit_retry(
        self, deepseek_provider, mock_chat_openai
    ):
        """Test chat rate limit with retry mechanism"""
        error = OpenAIRateLimitError("rate_limit exceeded")
        success_response = AIMessage(content="Success")
        
        deepseek_provider.llm.invoke.side_effect = [
            error,
            success_response
        ]

        messages = [{"role": "user", "content": "Hello"}]
        response = deepseek_provider.chat(messages)
        
        assert response == "Success"
        assert deepseek_provider.llm.invoke.call_count == 2

    def test_complete_converts_to_chat(
        self, deepseek_provider, mock_chat_openai
    ):
        """Test complete method converts to chat format"""
        mock_response = AIMessage(content="Completion")
        deepseek_provider.llm.invoke.return_value = mock_response

        response = deepseek_provider.complete(
            "Test prompt",
            max_tokens=50,
            temperature=0.9
        )

        assert response == "Completion"
        assert deepseek_provider.llm.max_tokens == 50
        assert deepseek_provider.llm.temperature == 0.9

    def test_init_with_api_key(self):
        """Test initialization with API key."""
        config = DeepSeekConfig(api_key="test-key")
        provider = DeepSeekProvider(config)
        assert provider.config.api_key == "test-key"

    def test_init_with_env_api_key(self, monkeypatch):
        """Test initialization with environment API key."""
        monkeypatch.setenv('DEEPSEEK_API_KEY', 'env-key')
        config = DeepSeekConfig()
        provider = DeepSeekProvider(config)
        assert provider.config.api_key == 'env-key'

    def test_init_missing_api_key(self, monkeypatch):
        """Test initialization with missing API key."""
        monkeypatch.setenv('DEEPSEEK_API_KEY', '')
        with pytest.raises(ValueError, match="API key is required"):
            DeepSeekConfig(api_key="")

    def test_chat_success(self, deepseek_provider, mock_chat_openai):
        """Test successful chat completion."""
        mock_response = AIMessage(content="Test response")
        deepseek_provider.llm.invoke.return_value = mock_response

        messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello"}
        ]

        response = deepseek_provider.chat(messages)

        assert response == "Test response"
        deepseek_provider.llm.invoke.assert_called_once()

    def test_chat_rate_limit(
        self, deepseek_provider, mock_chat_openai, mock_retry
    ):
        """Test handling of rate limit error with single retry."""
        error = OpenAIRateLimitError("rate_limit exceeded")
        deepseek_provider.llm.invoke.side_effect = error

        messages = [{"role": "user", "content": "Hello"}]
        with pytest.raises(
            RetryError,
            match="OpenAIRateLimitError"
        ):
            deepseek_provider.chat(messages)

    def test_chat_api_error(self, deepseek_provider, mock_chat_openai):
        """Test handling of general API error."""
        error = OpenAIError("API error")
        deepseek_provider.llm.invoke.side_effect = error

        messages = [{"role": "user", "content": "Hello"}]
        with pytest.raises(
            OpenAIError,
            match="OpenAI API error: API error"
        ):
            deepseek_provider.chat(messages)