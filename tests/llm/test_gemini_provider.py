"""Unit tests for Gemini provider.

This module contains comprehensive tests for the Gemini provider
implementation, focusing on testing the logic of each method using mocks.
"""

from unittest.mock import MagicMock, patch
import pytest
from tenacity import RetryError

from devtoolbox.llm.gemini_provider import (
    GeminiConfig,
    GeminiProvider,
    GeminiError,
    GeminiRateLimitError,
)


@pytest.fixture
def mock_retry():
    """Mock the retry decorator to simulate a single retry attempt."""
    with patch('devtoolbox.llm.gemini_provider.retry') as mock:
        mock.side_effect = lambda *args, **kwargs: lambda f: f
        yield mock


@pytest.fixture
def mock_genai_client():
    """Mock the google.genai.Client class."""
    with patch('devtoolbox.llm.gemini_provider.google_genai.Client') as mock:
        mock_instance = MagicMock()
        mock.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_genai_types():
    """Mock the google.genai.types module."""
    with patch('devtoolbox.llm.gemini_provider.genai_types') as mock:
        mock.Content = MagicMock
        mock.Part = MagicMock()
        mock.Part.from_text = MagicMock(return_value=MagicMock())
        yield mock


@pytest.fixture
def gemini_provider(mock_genai_client, mock_genai_types):
    """Create a GeminiProvider instance with mocked dependencies."""
    config = GeminiConfig(api_key="test-key")
    return GeminiProvider(config)


class TestGeminiConfig:
    """Tests for GeminiConfig class."""

    def test_create_config_with_defaults(self, monkeypatch):
        """Test creating config with default values from environment."""
        monkeypatch.setenv('GOOGLE_API_KEY', 'test-key')
        monkeypatch.setenv('GEMINI_MODEL', 'gemini-2.5-pro')
        monkeypatch.setenv('GEMINI_TEMPERATURE', '0.5')

        config = GeminiConfig()

        assert config.api_key == 'test-key'
        assert config.model == 'gemini-2.5-pro'
        assert config.temperature == 0.5
        assert config.max_tokens == 80000

    def test_config_default_model(self):
        """Test default model is gemini-2.5-flash-lite."""
        config = GeminiConfig(api_key="test-key")
        assert config.model == 'gemini-2.5-flash-lite'

    def test_config_validation_error(self):
        """Test validation error when API key is missing."""
        with pytest.raises(
            ValueError,
            match="Google Gemini API key is required",
        ):
            GeminiConfig(api_key="")._validate()

    def test_deprecated_from_env_warning(self, caplog, monkeypatch):
        """Test from_env() deprecation warning."""
        monkeypatch.setenv('GOOGLE_API_KEY', 'test-key')
        GeminiConfig.from_env()
        assert "from_env() deprecated" in caplog.text


class TestGeminiProvider:
    """Tests for GeminiProvider class."""

    def test_to_contents_single_user(self, gemini_provider):
        """Test message conversion for single user message."""
        messages = [{"role": "user", "content": "Hello"}]
        result = gemini_provider._to_contents(messages)
        assert result == "Hello"

    def test_to_contents_multi_turn(self, gemini_provider, mock_genai_types):
        """Test message conversion for multi-turn conversation."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
            {"role": "user", "content": "How are you?"},
        ]
        result = gemini_provider._to_contents(messages)
        assert isinstance(result, list)
        assert len(result) == 3

    def test_to_contents_system_message(
        self,
        gemini_provider,
        mock_genai_types,
    ):
        """Test system message is converted to user role."""
        messages = [{"role": "system", "content": "You are helpful"}]
        result = gemini_provider._to_contents(messages)
        assert isinstance(result, list)
        assert len(result) == 1

    def test_config_build_basic(self, gemini_provider):
        """Test building config with defaults."""
        cfg = gemini_provider._config(None, None, False, None)
        assert "temperature" in cfg
        assert "max_output_tokens" in cfg

    def test_config_build_json_mode(self, gemini_provider):
        """Test building config with JSON mode."""
        cfg = gemini_provider._config(None, None, True, None)
        assert cfg["response_mime_type"] == "application/json"

    def test_config_build_with_schema(self, gemini_provider):
        """Test building config with response schema."""
        schema = {"type": "OBJECT", "properties": {}}
        cfg = gemini_provider._config(None, None, False, schema)
        assert cfg["response_mime_type"] == "application/json"
        assert cfg["response_schema"] == schema

    def test_config_build_invalid_schema(self, gemini_provider):
        """Test building config with invalid schema raises TypeError."""
        with pytest.raises(TypeError, match="must be a dict"):
            gemini_provider._config(None, None, False, "not-a-dict")

    def test_chat_success(
        self,
        gemini_provider,
        mock_genai_client,
        mock_genai_types,
    ):
        """Test successful chat completion."""
        mock_response = MagicMock()
        mock_response.text = "Test response"
        mock_genai_client.models.generate_content.return_value = (
            mock_response
        )

        messages = [{"role": "user", "content": "Hello"}]
        response = gemini_provider.chat(messages)

        assert response == "Test response"
        mock_genai_client.models.generate_content.assert_called_once()

    def test_chat_with_json_mode(
        self,
        gemini_provider,
        mock_genai_client,
        mock_genai_types,
    ):
        """Test chat with JSON mode enabled."""
        mock_response = MagicMock()
        mock_response.text = '{"result": "test"}'
        mock_genai_client.models.generate_content.return_value = (
            mock_response
        )

        messages = [{"role": "user", "content": "Return JSON"}]
        response = gemini_provider.chat(messages, json_mode=True)

        assert response == '{"result": "test"}'
        call_args = (
            mock_genai_client.models.generate_content.call_args
        )
        assert call_args[1]["config"]["response_mime_type"] == (
            "application/json"
        )

    def test_chat_rate_limit_error(
        self,
        gemini_provider,
        mock_genai_client,
        mock_genai_types,
        mock_retry,
    ):
        """Test handling of rate limit error with retry."""
        error = Exception("rate_limit exceeded")
        mock_genai_client.models.generate_content.side_effect = error

        messages = [{"role": "user", "content": "Hello"}]
        with pytest.raises(RetryError):
            gemini_provider.chat(messages)

    def test_chat_general_error(
        self,
        gemini_provider,
        mock_genai_client,
        mock_genai_types,
    ):
        """Test handling of general API error."""
        error = Exception("API error occurred")
        mock_genai_client.models.generate_content.side_effect = error

        messages = [{"role": "user", "content": "Hello"}]
        with pytest.raises(GeminiError, match="Gemini API error"):
            gemini_provider.chat(messages)

    def test_complete_success(
        self,
        gemini_provider,
        mock_genai_client,
        mock_genai_types,
    ):
        """Test successful text completion."""
        mock_response = MagicMock()
        mock_response.text = "Completion text"
        mock_genai_client.models.generate_content.return_value = (
            mock_response
        )

        response = gemini_provider.complete("Test prompt")

        assert response == "Completion text"
        mock_genai_client.models.generate_content.assert_called_once()

    def test_embed_not_implemented(self, gemini_provider):
        """Test embed method raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            gemini_provider.embed("test text")

    def test_list_models(self, gemini_provider):
        """Test list_models returns current recommended models."""
        models = gemini_provider.list_models()
        assert isinstance(models, list)
        assert len(models) > 0
        assert "gemini-2.5-flash-lite" in models
        assert "gemini-3-pro-preview" in models
        assert "gemini-pro" not in models

    def test_init_with_api_key(self, mock_genai_client, mock_genai_types):
        """Test initialization with API key."""
        config = GeminiConfig(api_key="test-key")
        provider = GeminiProvider(config)
        assert provider.config.api_key == "test-key"

    def test_init_with_env_api_key(
        self,
        monkeypatch,
        mock_genai_client,
        mock_genai_types,
    ):
        """Test initialization with environment API key."""
        monkeypatch.setenv('GOOGLE_API_KEY', 'env-key')
        config = GeminiConfig()
        provider = GeminiProvider(config)
        assert provider.config.api_key == 'env-key'

    def test_init_missing_api_key(self, monkeypatch):
        """Test initialization with missing API key."""
        monkeypatch.setenv('GOOGLE_API_KEY', '')
        with pytest.raises(ValueError, match="API key is required"):
            GeminiConfig(api_key="")

    def test_chat_with_custom_params(
        self,
        gemini_provider,
        mock_genai_client,
        mock_genai_types,
    ):
        """Test chat with custom max_tokens and temperature."""
        mock_response = MagicMock()
        mock_response.text = "Response"
        mock_genai_client.models.generate_content.return_value = (
            mock_response
        )

        messages = [{"role": "user", "content": "Hello"}]
        gemini_provider.chat(
            messages,
            max_tokens=1000,
            temperature=0.9,
        )

        call_args = (
            mock_genai_client.models.generate_content.call_args
        )
        cfg = call_args[1]["config"]
        assert cfg["max_output_tokens"] == 1000
        assert cfg["temperature"] == 0.9
