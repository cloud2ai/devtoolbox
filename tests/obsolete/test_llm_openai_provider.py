import unittest
from unittest.mock import patch, MagicMock
import logging
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from devtoolbox.llm.openai_provider import (
    OpenAIConfig,
    OpenAIProvider,
    OpenAIError,
    OpenAIRateLimitError
)
from tests.utils.test_logging import setup_test_logging
from tenacity.wait import wait_exponential


class TestOpenAIConfig(unittest.TestCase):
    """Test suite for OpenAIConfig."""

    def test_from_env(self):
        """Test configuration loading from environment variables."""
        with patch.dict('os.environ', {
            'OPENAI_API_KEY': 'test-key',
            'OPENAI_MODEL': 'gpt-4',
            'OPENAI_TEMPERATURE': '0.8',
            'OPENAI_MAX_TOKENS': '1000',
            'OPENAI_TOP_P': '0.9',
            'OPENAI_FREQUENCY_PENALTY': '0.5',
            'OPENAI_PRESENCE_PENALTY': '0.5'
        }):
            config = OpenAIConfig.from_env()
            self.assertEqual(config.api_key, 'test-key')
            self.assertEqual(config.model, 'gpt-4')
            self.assertEqual(config.temperature, 0.8)
            self.assertEqual(config.max_tokens, 1000)
            self.assertEqual(config.top_p, 0.9)
            self.assertEqual(config.frequency_penalty, 0.5)
            self.assertEqual(config.presence_penalty, 0.5)

    def test_validate(self):
        """Test configuration validation."""
        config = OpenAIConfig(
            api_key='test-key',
            model='gpt-4o-mini',
            temperature=0.7,
            max_tokens=2000,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
        self.assertTrue(config.validate())

        with self.assertRaises(ValueError):
            OpenAIConfig(
                api_key='',
                model='gpt-3.5-turbo',
                temperature=0.7,
                max_tokens=2000,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0
            ).validate()


class TestOpenAIProvider(unittest.TestCase):
    """Test cases for OpenAIProvider.

    This test suite verifies the functionality of the OpenAIProvider
    class, including initialization, text completion, chat completion,
    and error handling.
    """

    def setUp(self):
        """Set up test fixtures."""
        self.config = OpenAIConfig(
            api_key='test-key',
            model='gpt-4',
            temperature=0.7,
            max_tokens=2000,
            top_p=0.9,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
        setup_test_logging()
        self.logger = logging.getLogger("devtoolbox.llm.openai_provider")

    @patch('devtoolbox.llm.openai_provider.ChatOpenAI')
    @patch('tenacity.nap.sleep')  # Mock sleep to avoid real waiting
    @patch('devtoolbox.llm.openai_provider.retry')  # Mock the retry decorator
    def test_chat_rate_limit_error(self, mock_retry, mock_sleep, mock_chat_openai):
        """Test rate limit error handling and retry logic."""
        # Setup mock to raise rate limit error first, then succeed
        mock_llm = MagicMock()
        mock_chat_openai.return_value = mock_llm
        mock_llm.invoke.side_effect = [
            OpenAIRateLimitError("Rate limit exceeded"),
            AIMessage(content="Success after retry")
        ]

        # Mock the retry decorator with shorter wait times for testing
        def mock_retry_decorator(*args, **kwargs):
            # Override wait_exponential parameters for testing
            if 'wait' in kwargs:
                kwargs['wait'] = wait_exponential(
                    multiplier=0.1,  # Much smaller multiplier
                    min=0.1,  # Much smaller minimum wait
                    max=1.0  # Much smaller maximum wait
                )
            return lambda f: f
        mock_retry.side_effect = mock_retry_decorator

        provider = OpenAIProvider(self.config)
        messages = [{"role": "user", "content": "Hi"}]

        # Test chat method with retry
        response = provider.chat(messages)

        # Verify retry behavior
        self.assertEqual(mock_llm.invoke.call_count, 2)
        self.assertEqual(response, "Success after retry")
        # Verify sleep was called with shorter delay
        mock_sleep.assert_called_once_with(0.1)  # Match the shorter sleep time

    @patch('devtoolbox.llm.openai_provider.ChatOpenAI')
    @patch('tenacity.nap.sleep')  # Mock sleep to avoid real waiting
    @patch('devtoolbox.llm.openai_provider.retry')  # Mock the retry decorator
    def test_chat_rate_limit_error_retry_failure(
        self, mock_retry, mock_sleep, mock_chat_openai
    ):
        """Test rate limit error handling when retry fails."""
        # Setup mock to always raise rate limit error
        mock_llm = MagicMock()
        mock_chat_openai.return_value = mock_llm
        mock_llm.invoke.side_effect = OpenAIRateLimitError(
            "Rate limit exceeded"
        )

        # Mock the retry decorator to just call the function
        def mock_retry_decorator(*args, **kwargs):
            return lambda f: f
        mock_retry.side_effect = mock_retry_decorator

        provider = OpenAIProvider(self.config)
        messages = [{"role": "user", "content": "Hi"}]

        # Test chat method with retry failure
        with self.assertRaises(OpenAIRateLimitError) as context:
            provider.chat(messages)

        # Verify the error message
        self.assertEqual(str(context.exception), "Rate limit exceeded")
        # Verify the function was called only once since retry is disabled
        self.assertEqual(mock_llm.invoke.call_count, 1)
        # Verify sleep was not called since retry is disabled
        mock_sleep.assert_not_called()

    @patch('devtoolbox.llm.openai_provider.ChatOpenAI')
    def test_provider_initialization(self, mock_chat_openai):
        """Test provider initialization with LangChain."""
        OpenAIProvider(self.config)  # Initialize provider but don't store it

        # Verify ChatOpenAI was initialized with correct parameters
        mock_chat_openai.assert_called_once_with(
            model_name=self.config.model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            openai_api_key=self.config.api_key,
            openai_api_base=None,  # Add this parameter to match actual call
            top_p=self.config.top_p,
            frequency_penalty=self.config.frequency_penalty,
            presence_penalty=self.config.presence_penalty
        )

    @patch('devtoolbox.llm.openai_provider.ChatOpenAI')
    def test_message_conversion(self, mock_chat_openai):
        """Test message conversion to LangChain format."""
        provider = OpenAIProvider(self.config)

        messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "unknown", "content": "Test"}
        ]

        # Convert messages
        converted = provider._convert_messages(messages)

        # Verify message types
        self.assertIsInstance(converted[0], SystemMessage)
        self.assertIsInstance(converted[1], HumanMessage)
        self.assertIsInstance(converted[2], AIMessage)
        self.assertIsInstance(converted[3], HumanMessage)

        # Verify content
        self.assertEqual(converted[0].content, "You are a helpful assistant")
        self.assertEqual(converted[1].content, "Hello")
        self.assertEqual(converted[2].content, "Hi there!")
        self.assertEqual(converted[3].content, "Test")

    @patch('devtoolbox.llm.openai_provider.ChatOpenAI')
    def test_chat_with_parameters(self, mock_chat_openai):
        """Test chat completion with custom parameters."""
        mock_llm = MagicMock()
        mock_chat_openai.return_value = mock_llm
        mock_llm.invoke.return_value = AIMessage(content="Response")

        provider = OpenAIProvider(self.config)
        messages = [{"role": "user", "content": "Hi"}]

        # Test with custom parameters
        provider.chat(messages, max_tokens=100, temperature=0.5)

        # Verify parameters were updated
        self.assertEqual(mock_llm.max_tokens, 100)
        self.assertEqual(mock_llm.temperature, 0.5)

    @patch('devtoolbox.llm.openai_provider.ChatOpenAI')
    def test_complete_method(self, mock_chat_openai):
        """Test the complete method (compatibility method)."""
        mock_response = AIMessage(content="Completion response")
        mock_chat_openai.return_value.invoke.return_value = mock_response

        provider = OpenAIProvider(self.config)
        prompt = "Complete this"

        # Test complete method
        response = provider.complete(prompt)

        # Verify response
        self.assertEqual(response, "Completion response")

        # Verify message conversion
        mock_chat_openai.return_value.invoke.assert_called_once()
        args = mock_chat_openai.return_value.invoke.call_args[0][0]
        self.assertEqual(len(args), 1)
        self.assertIsInstance(args[0], HumanMessage)
        self.assertEqual(args[0].content, prompt)

    @patch('devtoolbox.llm.openai_provider.ChatOpenAI')
    def test_error_handling(self, mock_chat_openai):
        """Test error handling."""
        mock_llm = MagicMock()
        mock_chat_openai.return_value = mock_llm
        mock_llm.invoke.side_effect = Exception("API Error")

        provider = OpenAIProvider(self.config)
        messages = [{"role": "user", "content": "Hi"}]

        with self.assertRaises(OpenAIError):
            provider.chat(messages)

    @patch('devtoolbox.llm.openai_provider.ChatOpenAI')
    def test_error_logging(self, mock_chat_openai):
        """Test error logging."""
        mock_llm = MagicMock()
        mock_chat_openai.return_value = mock_llm
        mock_llm.invoke.side_effect = Exception("API Error")

        provider = OpenAIProvider(self.config)
        messages = [{"role": "user", "content": "Hi"}]

        with self.assertRaises(OpenAIError):
            provider.chat(messages)


if __name__ == "__main__":
    unittest.main()