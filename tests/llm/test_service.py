"""Unit tests for LLM service layer.

This module contains comprehensive tests for the LLM service layer
implementation, focusing on testing the high-level interface and
advanced features like context management and fallback handling.
"""

import pytest
from unittest.mock import MagicMock, patch

from devtoolbox.llm.service import LLMService
from devtoolbox.llm.provider import BaseLLMProvider


class MockConfig:
    """Mock configuration class."""
    def __init__(self, api_key="test-key"):
        self.api_key = api_key


class MockProvider(BaseLLMProvider):
    """Mock provider for testing service layer."""
    
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
    return MockProvider(MockConfig())


@pytest.fixture
def llm_service():
    """Create an LLMService instance with mocked dependencies."""
    config = MockConfig()
    with patch(
        'devtoolbox.llm.service.importlib.import_module'
    ) as mock_import:
        mock_import.return_value = MagicMock(
            MockProvider=MockProvider
        )
        service = LLMService(config)
        # Create a mock provider with all methods configured
        mock_provider = MagicMock(spec=MockProvider)
        mock_provider.chat.return_value = "Mock chat response"
        mock_provider.complete.return_value = "Mock completion"
        mock_provider.embed.return_value = [0.1, 0.2, 0.3]
        mock_provider.list_models.return_value = ["mock-model-1", "mock-model-2"]
        service.provider = mock_provider
        return service


class TestLLMService:
    """Tests for LLMService class."""

    def test_init_with_config(self, llm_service):
        """Test initialization with config."""
        assert llm_service.config.api_key == "test-key"
        assert isinstance(llm_service.provider, MagicMock)

    def test_chat(self, llm_service):
        """Test basic chat functionality."""
        messages = [
            {"role": "user", "content": "Hello"}
        ]
        response = llm_service.chat(messages)
        assert response == "Mock chat response"

    def test_chat_with_parameters(self, llm_service):
        """Test chat with custom parameters."""
        messages = [{"role": "user", "content": "Hello"}]
        response = llm_service.chat(
            messages,
            max_tokens=100,
            temperature=0.8
        )
        assert response == "Mock chat response"

    def test_chat_with_context(self, llm_service):
        """Test chat with context."""
        messages = [{"role": "user", "content": "Hello"}]
        context = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Previous message"}
        ]
        response = llm_service.chat_with_context(
            messages,
            context
        )
        assert response == "Mock chat response"

    def test_chat_with_fallback_success(self, llm_service):
        """Test chat with fallback when main chat succeeds."""
        messages = [{"role": "user", "content": "Hello"}]
        fallback = [{"role": "user", "content": "Fallback"}]
        response = llm_service.chat_with_fallback(
            messages,
            fallback
        )
        assert response == "Mock chat response"

    def test_chat_with_fallback_error(self, llm_service):
        """Test chat with fallback when main chat fails."""
        messages = [{"role": "user", "content": "Hello"}]
        fallback = [{"role": "user", "content": "Fallback"}]
        
        # Configure the mock to raise an exception on first call
        llm_service.provider.chat.side_effect = [
            Exception("Chat failed"),
            "Mock chat response"
        ]
        
        response = llm_service.chat_with_fallback(
            messages,
            fallback
        )
        assert response == "Mock chat response"
        assert llm_service.provider.chat.call_count == 2

    def test_complete(self, llm_service):
        """Test text completion."""
        response = llm_service.complete("Test prompt")
        assert response == "Mock completion"

    def test_complete_with_parameters(self, llm_service):
        """Test text completion with custom parameters."""
        response = llm_service.complete(
            "Test prompt",
            max_tokens=50,
            temperature=0.9
        )
        assert response == "Mock completion"

    def test_embed(self, llm_service):
        """Test embedding generation."""
        embedding = llm_service.embed("Test text")
        assert embedding == [0.1, 0.2, 0.3]

    def test_provider_initialization_error(self):
        """Test provider initialization error handling."""
        config = MockConfig()
        with patch(
            'devtoolbox.llm.service.importlib.import_module'
        ) as mock_import:
            mock_import.side_effect = ImportError("Module not found")
            with pytest.raises(
                ValueError,
                match="Failed to initialize provider"
            ):
                LLMService(config) 