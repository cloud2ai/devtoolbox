"""Unit tests for OpenAI provider.

This module contains comprehensive tests for the OpenAI provider
implementation, focusing on testing the logic of each method using mocks.
"""

import os
from unittest.mock import MagicMock, patch
import pytest
from openai import RateLimitError, APIError
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from tenacity import RetryError
import warnings

from devtoolbox.llm.openai_provider import (
    OpenAIConfig,
    OpenAIProvider,
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
def openai_provider(mock_chat_openai):
    """Create an OpenAIProvider instance with mocked dependencies."""
    config = OpenAIConfig(api_key="test-key")
    return OpenAIProvider(config)


class TestOpenAIConfig:
    """Tests for OpenAIConfig class."""

    def test_create_config_with_defaults(self, monkeypatch):
        """Test creating config with default values from environment"""
        monkeypatch.setenv('OPENAI_API_KEY', 'test-key')
        monkeypatch.setenv('OPENAI_MODEL', 'gpt-4')
        monkeypatch.setenv('OPENAI_TEMPERATURE', '0.5')
        
        config = OpenAIConfig()
        
        assert config.api_key == 'test-key'
        assert config.model == 'gpt-4'
        assert config.temperature == 0.5
        assert config.max_tokens == 2000  # default value

    def test_config_validation_error(self):
        """Test validation error when API key is missing"""
        with pytest.raises(ValueError, match="OpenAI API key is required"):
            OpenAIConfig(api_key="")._validate_config()

    def test_deprecated_from_env_warning(self, caplog):
        """Test from_env() deprecation warning"""
        OpenAIConfig.from_env()
        assert "from_env() is deprecated" in caplog.text


class TestOpenAIProvider:
    """Tests for OpenAIProvider class using LangChain."""

    def test_convert_messages(self, openai_provider):
        """Test message conversion to LangChain format"""
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
            {"role": "unknown", "content": "Test"}
        ]
        
        converted = openai_provider._convert_messages(messages)
        
        assert len(converted) == 4
        assert isinstance(converted[0], SystemMessage)
        assert isinstance(converted[1], HumanMessage)
        assert isinstance(converted[2], AIMessage)
        assert isinstance(converted[3], HumanMessage)  # unknown roles default to HumanMessage

    def test_chat_with_parameters(self, openai_provider, mock_chat_openai):
        """Test chat with custom parameters"""
        mock_response = AIMessage(content="Test response")
        openai_provider.llm.invoke.return_value = mock_response

        messages = [{"role": "user", "content": "Hello"}]
        response = openai_provider.chat(
            messages,
            max_tokens=100,
            temperature=0.8
        )

        assert response == "Test response"
        assert openai_provider.llm.max_tokens == 100
        assert openai_provider.llm.temperature == 0.8

    def test_chat_rate_limit_retry(self, openai_provider, mock_chat_openai):
        """Test chat rate limit with retry mechanism"""
        error = OpenAIRateLimitError("rate_limit exceeded")
        success_response = AIMessage(content="Success")
        
        openai_provider.llm.invoke.side_effect = [
            error,
            success_response
        ]

        messages = [{"role": "user", "content": "Hello"}]
        response = openai_provider.chat(messages)
        
        assert response == "Success"
        assert openai_provider.llm.invoke.call_count == 2

    def test_embed_with_langchain(self, openai_provider, mock_chat_openai):
        """Test embedding generation using LangChain client"""
        mock_embedding = [0.1, 0.2, 0.3]
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=mock_embedding)]
        
        openai_provider.llm.client.embeddings.create.return_value = mock_response
        
        result = openai_provider.embed("Test text")
        
        assert result == mock_embedding
        openai_provider.llm.client.embeddings.create.assert_called_once_with(
            model="text-embedding-ada-002",
            input="Test text"
        )

    def test_complete_converts_to_chat(self, openai_provider, mock_chat_openai):
        """Test complete method converts to chat format"""
        mock_response = AIMessage(content="Completion")
        openai_provider.llm.invoke.return_value = mock_response

        response = openai_provider.complete(
            "Test prompt",
            max_tokens=50,
            temperature=0.9
        )

        assert response == "Completion"
        assert openai_provider.llm.max_tokens == 50
        assert openai_provider.llm.temperature == 0.9

    def test_init_with_api_key(self):
        """Test initialization with API key."""
        config = OpenAIConfig(api_key="test-key")
        provider = OpenAIProvider(config)
        assert provider.config.api_key == "test-key"

    def test_init_with_env_api_key(self, monkeypatch):
        """Test initialization with environment API key."""
        monkeypatch.setenv('OPENAI_API_KEY', 'env-key')
        config = OpenAIConfig()
        provider = OpenAIProvider(config)
        assert provider.config.api_key == 'env-key'

    def test_init_missing_api_key(self, monkeypatch):
        """Test initialization with missing API key."""
        # Ensure the environment variable is set to an empty string
        monkeypatch.setenv('OPENAI_API_KEY', '')
        with pytest.raises(ValueError, match="API key is required"):
            OpenAIConfig(api_key="")

    def test_chat_success(self, openai_provider, mock_chat_openai):
        """Test successful chat completion."""
        mock_response = AIMessage(content="Test response")
        openai_provider.llm.invoke.return_value = mock_response

        messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello"}
        ]

        response = openai_provider.chat(messages)

        assert response == "Test response"
        openai_provider.llm.invoke.assert_called_once()

    def test_chat_rate_limit(
        self, openai_provider, mock_chat_openai, mock_retry
    ):
        """Test handling of rate limit error with single retry."""
        error = RateLimitError(
            message="rate_limit exceeded",
            response=MagicMock(status_code=429),
            body={"error": {"message": "rate_limit exceeded"}}
        )
        openai_provider.llm.invoke.side_effect = error

        messages = [{"role": "user", "content": "Hello"}]
        with pytest.raises(
            RetryError,
            match="OpenAIRateLimitError"
        ):
            openai_provider.chat(messages)

    def test_chat_api_error(self, openai_provider, mock_chat_openai):
        """Test handling of general API error."""
        error = APIError(
            message="API error",
            request=MagicMock(),
            body={"error": {"message": "API error"}}
        )
        openai_provider.llm.invoke.side_effect = error

        messages = [{"role": "user", "content": "Hello"}]
        with pytest.raises(
            OpenAIError,
            match="OpenAI API error: API error"
        ):
            openai_provider.chat(messages)

    def test_complete_success(self, openai_provider, mock_chat_openai):
        """Test successful text completion."""
        mock_response = AIMessage(content="Test completion")
        openai_provider.llm.invoke.return_value = mock_response

        response = openai_provider.complete("Test prompt")

        assert response == "Test completion"
        openai_provider.llm.invoke.assert_called_once()

    def test_embed_success(self, openai_provider, mock_chat_openai):
        """Test successful embedding generation."""
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]
        embeddings_client = openai_provider.llm.client.embeddings
        embeddings_client.create.return_value = mock_response

        embedding = openai_provider.embed("Test text")

        assert embedding == [0.1, 0.2, 0.3]
        embeddings_client.create.assert_called_once_with(
            model="text-embedding-ada-002",
            input="Test text"
        )

    def test_list_models_success(self, openai_provider, mock_chat_openai):
        """Test successful model listing."""
        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(id="gpt-4"),
            MagicMock(id="gpt-3.5-turbo")
        ]
        models_client = openai_provider.llm.client.models
        models_client.list.return_value = mock_response

        models = openai_provider.list_models()

        assert models == ["gpt-4", "gpt-3.5-turbo"]
        models_client.list.assert_called_once()