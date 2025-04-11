import unittest
from unittest.mock import patch

from devtoolbox.llm.service import LLMService
from devtoolbox.llm.openai_provider import OpenAIConfig


class TestLLMService(unittest.TestCase):
    """Test suite for LLMService."""

    def setUp(self):
        """Set up test environment."""
        self.config = OpenAIConfig(
            api_key="test-openai-key",
            model="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=2000,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
        self.service = LLMService(self.config)

    def test_init_with_config(self):
        """Test service initialization with config."""
        self.assertIsNotNone(self.service.provider)
        self.assertEqual(
            self.service.provider.config.api_key,
            "test-openai-key"
        )

    @patch('devtoolbox.llm.openai_provider.OpenAIProvider.chat')
    def test_chat(self, mock_chat):
        """Test chat functionality."""
        mock_chat.return_value = "Test response"

        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"}
        ]
        result = self.service.chat(messages)
        self.assertEqual(result, "Test response")

        mock_chat.assert_called_once_with(
            messages,
            max_tokens=None,
            temperature=None
        )

    @patch('devtoolbox.llm.openai_provider.OpenAIProvider.chat')
    def test_chat_with_fallback_success(self, mock_chat):
        """Test chat with fallback messages when first attempt succeeds."""
        mock_chat.return_value = "Test response"

        messages = [{"role": "user", "content": "Hello"}]
        fallback_messages = [{"role": "user", "content": "Hi"}]
        result = self.service.chat_with_fallback(messages, fallback_messages)
        self.assertEqual(result, "Test response")

        mock_chat.assert_called_once_with(
            messages,
            max_tokens=None,
            temperature=None
        )

    @patch('devtoolbox.llm.openai_provider.OpenAIProvider.chat')
    def test_chat_with_fallback_failure(self, mock_chat):
        """Test chat with fallback messages when first attempt fails.

        This test verifies that when the main chat fails, the service
        falls back to the alternative messages. Since retry logic is
        handled at the provider level, we expect exactly two calls:
        1. One for the main messages (which fails)
        2. One for the fallback messages (which succeeds)
        """
        def mock_chat_side_effect(*args, **kwargs):
            if args[0] == messages:
                raise Exception("First attempt failed")
            return "Fallback response"

        mock_chat.side_effect = mock_chat_side_effect

        messages = [{"role": "user", "content": "Hello"}]
        fallback_messages = [{"role": "user", "content": "Hi"}]
        result = self.service.chat_with_fallback(messages, fallback_messages)
        self.assertEqual(result, "Fallback response")

        # Expect exactly two calls since retries are handled at provider level
        self.assertEqual(mock_chat.call_count, 2)
        mock_chat.assert_any_call(
            messages,
            max_tokens=None,
            temperature=None
        )
        mock_chat.assert_any_call(
            fallback_messages,
            max_tokens=None,
            temperature=None
        )

    @patch('devtoolbox.llm.openai_provider.OpenAIProvider.chat')
    def test_chat_with_context(self, mock_chat):
        """Test chat with context."""
        mock_chat.return_value = "Test response"

        messages = [{"role": "user", "content": "Hello"}]
        context = [
            {"role": "system", "content": "You are a helpful assistant"}
        ]
        result = self.service.chat_with_context(messages, context)
        self.assertEqual(result, "Test response")

        mock_chat.assert_called_once_with(
            context + messages,
            max_tokens=None,
            temperature=None
        )

    @patch('devtoolbox.llm.openai_provider.OpenAIProvider.complete')
    def test_complete(self, mock_complete):
        """Test text completion."""
        mock_complete.return_value = "Test completion"

        result = self.service.complete("Test prompt")
        self.assertEqual(result, "Test completion")

        mock_complete.assert_called_once_with(
            "Test prompt",
            max_tokens=None,
            temperature=None
        )

    @patch('devtoolbox.llm.openai_provider.OpenAIProvider.embed')
    def test_embed(self, mock_embed):
        """Test embedding generation."""
        mock_embed.return_value = [0.1, 0.2, 0.3]

        result = self.service.embed("Test text")
        self.assertEqual(result, [0.1, 0.2, 0.3])

        mock_embed.assert_called_once_with("Test text")

    @patch('devtoolbox.llm.openai_provider.OpenAIProvider.list_models')
    def test_list_models(self, mock_list_models):
        """Test model listing."""
        mock_list_models.return_value = ["model1", "model2"]

        result = self.service.provider.list_models()
        self.assertEqual(result, ["model1", "model2"])

        mock_list_models.assert_called_once()


if __name__ == '__main__':
    unittest.main()