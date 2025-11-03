"""Unit tests for Azure OpenAI provider.

This module contains comprehensive tests for the Azure OpenAI provider
implementation, focusing on testing the logic of each method using mocks.
"""

from unittest.mock import MagicMock, patch
import pytest
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from tenacity import RetryError

from devtoolbox.llm.azure_openai_provider import (
    AzureOpenAIConfig,
    AzureOpenAIProvider
)
from devtoolbox.llm.openai_provider import (
    OpenAIError,
    OpenAIRateLimitError
)


@pytest.fixture
def mock_retry():
    """Mock the retry decorator to simulate a single retry attempt."""
    with patch('devtoolbox.llm.azure_openai_provider.retry') as mock:
        mock.side_effect = (
            lambda *args, **kwargs: lambda f: f
        )
        yield mock


@pytest.fixture
def mock_azure_chat_openai():
    """Mock the AzureChatOpenAI class."""
    with patch('devtoolbox.llm.azure_openai_provider.AzureChatOpenAI') as mock:
        mock_instance = MagicMock()
        mock.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def azure_provider(mock_azure_chat_openai):
    """Create an AzureOpenAIProvider instance with mocked dependencies."""
    config = AzureOpenAIConfig(
        api_key="test-key",
        api_base="https://test.openai.azure.com/",
        deployment="test-deployment"
    )
    return AzureOpenAIProvider(config)


class TestAzureOpenAIConfig:
    """Tests for AzureOpenAIConfig class."""

    def test_create_config_with_defaults(self, monkeypatch):
        """Test creating config with default values from environment"""
        monkeypatch.setenv('AZURE_OPENAI_API_KEY', 'test-key')
        monkeypatch.setenv('AZURE_OPENAI_API_BASE', 'https://test.openai.azure.com/')
        monkeypatch.setenv('AZURE_OPENAI_DEPLOYMENT', 'test-deployment')
        monkeypatch.setenv('AZURE_OPENAI_MODEL', 'gpt-4')
        monkeypatch.setenv('AZURE_OPENAI_TEMPERATURE', '0.5')

        config = AzureOpenAIConfig()

        assert config.api_key == 'test-key'
        assert config.api_base == 'https://test.openai.azure.com/'
        assert config.deployment == 'test-deployment'
        assert config.model == 'gpt-4'
        assert config.temperature == 0.5
        assert config.max_tokens == 2000  # default value

    def test_config_validation_error(self):
        """Test validation error when required fields are missing"""
        with pytest.raises(ValueError, match="Azure OpenAI API key is required"):
            AzureOpenAIConfig(api_key="")._validate_config()

        with pytest.raises(ValueError, match="Azure OpenAI endpoint URL is required"):
            AzureOpenAIConfig(
                api_key="test-key",
                api_base=""
            )._validate_config()

        with pytest.raises(ValueError, match="Azure OpenAI deployment name is required"):
            AzureOpenAIConfig(
                api_key="test-key",
                api_base="https://test.openai.azure.com/",
                deployment=""
            )._validate_config()

    def test_deprecated_from_env_warning(self, caplog):
        """Test from_env() deprecation warning"""
        AzureOpenAIConfig.from_env()
        assert "from_env() is deprecated" in caplog.text


class TestAzureOpenAIProvider:
    """Tests for AzureOpenAIProvider class using LangChain."""

    def test_convert_messages(self, azure_provider):
        """Test message conversion to LangChain format"""
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
            {"role": "unknown", "content": "Test"}
        ]

        converted = azure_provider._convert_messages(messages)

        assert len(converted) == 4
        assert isinstance(converted[0], SystemMessage)
        assert isinstance(converted[1], HumanMessage)
        assert isinstance(converted[2], AIMessage)
        assert isinstance(
            converted[3], HumanMessage
        )  # unknown roles default to HumanMessage

    def test_chat_with_parameters(self, azure_provider, mock_azure_chat_openai):
        """Test chat with custom parameters returns raw AIMessage"""
        mock_response = AIMessage(content="Test response")
        azure_provider.llm.invoke.return_value = mock_response

        messages = [{"role": "user", "content": "Hello"}]
        response = azure_provider.chat(
            messages,
            max_tokens=100,
            temperature=0.8
        )

        # Provider returns raw AIMessage (transparent pass-through)
        assert isinstance(response, AIMessage)
        assert response.content == "Test response"
        assert azure_provider.llm.max_tokens == 100
        assert azure_provider.llm.temperature == 0.8

    def test_chat_rate_limit_retry(
        self, azure_provider, mock_azure_chat_openai
    ):
        """Test chat rate limit with retry mechanism"""
        error = OpenAIRateLimitError("rate_limit exceeded")
        success_response = AIMessage(content="Success")

        azure_provider.llm.invoke.side_effect = [
            error,
            success_response
        ]

        messages = [{"role": "user", "content": "Hello"}]
        response = azure_provider.chat(messages)

        # Provider returns raw AIMessage
        assert isinstance(response, AIMessage)
        assert response.content == "Success"
        assert azure_provider.llm.invoke.call_count == 2

    def test_complete_converts_to_chat(
        self, azure_provider, mock_azure_chat_openai
    ):
        """Test complete method converts to chat format and returns raw AIMessage"""
        mock_response = AIMessage(content="Completion")
        azure_provider.llm.invoke.return_value = mock_response

        response = azure_provider.complete(
            "Test prompt",
            max_tokens=50,
            temperature=0.9
        )

        # Provider returns raw AIMessage
        assert isinstance(response, AIMessage)
        assert response.content == "Completion"
        assert azure_provider.llm.max_tokens == 50
        assert azure_provider.llm.temperature == 0.9

    def test_init_with_api_key(self):
        """Test initialization with API key."""
        config = AzureOpenAIConfig(
            api_key="test-key",
            api_base="https://test.openai.azure.com/",
            deployment="test-deployment"
        )
        provider = AzureOpenAIProvider(config)
        assert provider.config.api_key == "test-key"
        assert provider.config.api_base == "https://test.openai.azure.com/"
        assert provider.config.deployment == "test-deployment"

    def test_init_with_env_api_key(self, monkeypatch):
        """Test initialization with environment API key."""
        monkeypatch.setenv('AZURE_OPENAI_API_KEY', 'env-key')
        monkeypatch.setenv('AZURE_OPENAI_API_BASE', 'https://env.openai.azure.com/')
        monkeypatch.setenv('AZURE_OPENAI_DEPLOYMENT', 'env-deployment')
        config = AzureOpenAIConfig()
        provider = AzureOpenAIProvider(config)
        assert provider.config.api_key == 'env-key'
        assert provider.config.api_base == 'https://env.openai.azure.com/'
        assert provider.config.deployment == 'env-deployment'

    def test_init_missing_api_key(self, monkeypatch):
        """Test initialization with missing API key."""
        monkeypatch.setenv('AZURE_OPENAI_API_KEY', '')
        with pytest.raises(ValueError, match="API key is required"):
            AzureOpenAIConfig(api_key="")

    def test_chat_success(self, azure_provider, mock_azure_chat_openai):
        """Test successful chat completion returns raw AIMessage."""
        mock_response = AIMessage(content="Test response")
        azure_provider.llm.invoke.return_value = mock_response

        messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello"}
        ]

        response = azure_provider.chat(messages)

        # Provider returns raw AIMessage (transparent pass-through)
        assert isinstance(response, AIMessage)
        assert response.content == "Test response"
        azure_provider.llm.invoke.assert_called_once()

    def test_chat_rate_limit(
        self, azure_provider, mock_azure_chat_openai, mock_retry
    ):
        """Test handling of rate limit error with single retry."""
        error = OpenAIRateLimitError("rate_limit exceeded")
        azure_provider.llm.invoke.side_effect = error

        messages = [{"role": "user", "content": "Hello"}]
        with pytest.raises(
            RetryError,
            match="OpenAIRateLimitError"
        ):
            azure_provider.chat(messages)

    def test_chat_api_error(self, azure_provider, mock_azure_chat_openai):
        """Test handling of general API error."""
        error = OpenAIError("API error")
        azure_provider.llm.invoke.side_effect = error

        messages = [{"role": "user", "content": "Hello"}]
        with pytest.raises(
            OpenAIError,
            match="Azure OpenAI API error: API error"
        ):
            azure_provider.chat(messages)
